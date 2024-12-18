"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities import rank_zero_only
import torch.distributions as dist

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.losses.customloss import MaskBalanceMSELoss, SmoothMSELoss

from functools import wraps

import pdb


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 smooth_loss_step=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type
        if self.loss_type == 'balance':
            self.criterion = MaskBalanceMSELoss(eta=1)
        elif self.loss_type == 'smooth':
            self.criterion = SmoothMSELoss(step=smooth_loss_step)

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    # new function
    def params_freeze_model(self):
        params_list = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith("model.diffusion_model.input_blocks.0.0."):
                    params_list.append(param)
                elif name.startswith("model_ema.diffusion_modelinput_blocks00"):
                    params_list.append(param)
                elif name.startswith("model.diffusion_model.context_proj_embed."):
                    params_list.append(param)
                elif name.startswith("model_ema.diffusion_modelcontext_proj_embed"):
                    params_list.append(param)
                else:
                    continue
        return params_list

    # Modified function
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        own_state = self.state_dict()
        own_keys = list(own_state.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    if "load_maximally" not in ignore_keys or "diffusion_model" not in k:
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
                    else:
                        if k in own_keys and 'proj' not in k:
                            own_param, param = own_state[k], sd[k]
                            diff_idx = []
                            for idx, shape in enumerate(own_param.shape):
                                if shape != param.shape[idx]:
                                    diff_idx.append(idx)
                            if len(diff_idx) == 0:
                                own_param.copy_(param)
                            elif len(diff_idx) == 1:
                                assert(len(param.shape) == 4)
                                assert(diff_idx[0] == 1)
                                own_param[:, :param.shape[1], :, :].copy_(param)
                                own_param[:, param.shape[1]:, :, :].copy_(own_param[:, param.shape[1]:, :, :] * 0)
                            else:
                                raise AssertionError("Pre-trained model doesn't differ at only one channel!")
                        del sd[k]
                        if 'proj' not in k:
                            sd[k] = own_param
        new_keys = list(sd.keys())
        for k in keys:
            if 'model.diffusion_model' in k and k in new_keys and k[6:] in self.model_ema.m_name2s_name.keys():
                sd['model_ema.' + self.model_ema.m_name2s_name[k[6:]]] = sd[k]
        # init part of the input layer and output layer
        if self.model.diffusion_model.in_channels != 9:
            my_sd = {}
            for k, v in sd.items():
                if "input_blocks.0.0" not in k and "modelinput_blocks00" not in k:
                    my_sd.update({k: v})
                else:
                    if 'weight' in k:
                        if self.model.diffusion_model.in_channels > 9:
                            own_state[k][:, -v.shape[1]:, :, :].copy_(v)
                        else:
                            assert self.model.diffusion_model.in_channels == 8, \
                            'Unsupport channel num!'
                            own_state[k][:, :4, :, :].copy_(v[:,:4,:,:])
                            own_state[k][:, -4:, :, :].copy_(v[:,-4:,:,:])
                        # v_mean = v.mean(dim=1, keepdim=True)
                        # own_state[k][:, :1, :, :].copy_(v_mean)
                        # own_state[k][:, :5, :, :] *= 4.0/5.0
                    elif 'bias' in k:
                        own_state[k].copy_(v)
                    else:
                        raise ValueError('Unexpected key')
                    print(k)
                    
            sd = my_sd
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        if self.model.diffusion_model.input_branch_block_num != 0:
            # copy weight from the main branch to initialize the side branch, skip conv_in
            for i in range(1, self.model.diffusion_model.input_branch_block_num+1, 1):
                module = self.model.diffusion_model.input_blocks[i]
                # copy the weight
                self.model.diffusion_model.pos_input_blocks[i].load_state_dict(module.state_dict())


        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # modify here for loss
    def get_loss(self, pred, target, mean=True, mask=None, balance_mask=None):
        
        if mask is not None:
            mask = (mask + 1) / 2
            target = target * mask
            pred = pred * mask

        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == 'balance':
            if balance_mask is None:
                raise ValueError("balance_mask not provided")
            # criterion = MaskBalanceMSELoss(eta=0.5)
            criterion = MaskBalanceMSELoss(eta=1)
            loss = criterion(pred, target, balance_mask)
        else:
            if balance_mask is None:
                raise ValueError("balance_mask not provided")
            loss = self.criterion(pred, target, balance_mask, self.global_step)
            # raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")


        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        if type(k) == str:
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            # print('x.shape:', x.shape)
            x = x.to(memory_format=torch.contiguous_format).float()
            return x
        else:
            if isinstance(k, list):
                res = {}
                for _kdx in k:
                    x = batch[_kdx]
                    if len(x.shape) == 3:
                        x = x[..., None]
                    # print('x.shape:', x.shape)
                    x = x.to(memory_format=torch.contiguous_format).float()
                    res[_kdx] = x
            else:
                res = {}
                for curr_key in k:
                    curr_res = {}
                    for _kdx in k[curr_key]:
                        x = batch[_kdx]
                        print('dict k:', x)
                        if isinstance(x, torch.Tensor):
                            if len(x.shape) == 3:
                                x = x[..., None]
                            # print('x.shape:', x.shape)
                            x = x.to(memory_format=torch.contiguous_format).float()
                        curr_res[_kdx] = x
                    res[curr_key] = curr_res
            return res

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    # TODO: rewrite here for new training logic
    def training_step(self, batch, batch_idx):
        # print('batch idx:', batch_idx)
        # if batch_idx > 165:
        #     pdb.set_trace()
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)


        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

# decorator to sample prompt from `get_input`
def sample_prompt_according_to_cond(sample_cond):
    
    print("sample_cond:", sample_cond)
    def sample_prompt(get_input_fn):
        @wraps(get_input_fn)
        def sample_from_input(*args, **kwargs):
            out = get_input_fn(*args, **kwargs)
            bs = out[0].shape[0]
            c = out[1]
            
            # sample here
            type_to_sample = [ty['key'] for ty in sample_cond]
            probabilities = [ty['p'] for ty in sample_cond]
            """
            {'concat': {'background': ..., 'point':...(bs, ...), 'box':...(bs, ...)}}
            """
            
            # sample within a batch
            type_sampled_idx = torch.multinomial(torch.tensor(probabilities,dtype=float), bs, replacement=True)
            mask_sampled = []
            for sample_id, idx in enumerate(type_sampled_idx):
                mask_sampled.append(c['concat'][type_to_sample[idx]][sample_id])
            mask_sampled = torch.stack(mask_sampled)
            c['concat'] = {'background': c['concat']['background'], 'mask': mask_sampled, 'org_mask': c['concat']['org_mask']}
            out[1] = c
            # print(type_sampled)
            if kwargs.get('return_original_cond'):
                xc = out[-1]
                xc['concat'] = {'background': xc['concat']['background'], 'mask': mask_sampled}
                out[-1] = xc
            return out

        return sample_from_input
    return sample_prompt

class LatentDiffusion(DDPM):
    """main class"""
    # ! remember to uncomment
    # modify here for sampling. TODO: move it to dataset
    sample_cond = [{'key': 'point', 'p': 0.33}, 
                   {'key': 'bbox', 'p': 0.33}, 
                   {'key': 'mask', 'p': 0.34},
                   {'key': 'blank', 'p': 0}]  # probability to sample, used in prompt sample

    # # affordance setting
    # sample_cond = [{'key': 'point', 'p': 0}, 
    #                {'key': 'bbox', 'p': 1}, 
    #                {'key': 'mask', 'p': 0},
    #                {'key': 'blank', 'p': 0}]  # probability to sample, used in prompt sample

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 rescale_stage_config,
                 heatmap_stage_config,
                 coor2area_stage_config,
                 blank_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 mask_loss=False,
                 finetune_head=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 sample_cond=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 use_gauss_sample=False,
                 gauss_mean=None,
                 gauss_std=None,
                 dual_diffusion=False,
                 change_order=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.mask_loss = mask_loss
        self.finetune_head = finetune_head
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        LatentDiffusion.sample_cond = sample_cond
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_rescale_stage(rescale_stage_config)
        self.instantiate_heatmap_stage(heatmap_stage_config)
        self.instantiate_coor2area_stage(coor2area_stage_config)
        self.instantiate_blank_stage(blank_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
        if finetune_head:
            self.params_list = self.params_freeze_model()

        self.use_gauss_sample = use_gauss_sample
        if use_gauss_sample:
            assert gauss_mean and gauss_std, "You need to provide mean&std to sample!"
            mean = gauss_mean
            std = gauss_std
            x = torch.arange(0, self.num_timesteps)
            normal_dist = dist.Normal(mean, std)
            pdf = torch.exp(normal_dist.log_prob(x))
            self.gauss_pdf = pdf/pdf.sum()

        self.dual_diffusion = dual_diffusion
        self.change_order = change_order
        if self.change_order:
            assert self.dual_diffusion, "Order can only be specified when using dual diffusion"


    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    # new function
    def instantiate_rescale_stage(self, config):
        print('rescale config:', config)
        model = instantiate_from_config(config)
        self.rescale_stage_model = model.eval()
        self.rescale_stage_model.train = disabled_train
        for param in self.rescale_stage_model.parameters():
            param.requires_grad = False
    
    def instantiate_heatmap_stage(self, config):
        print('heatmap config:', config)
        model = instantiate_from_config(config)
        self.heatmap_stage_model = model.eval()
        self.heatmap_stage_model.train = disabled_train
        for param in self.heatmap_stage_model.parameters():
            param.requires_grad = False

    def instantiate_coor2area_stage(self, config):
        print('coor2area config:', config)
        model = instantiate_from_config(config)
        self.coor2area_stage_model = model.eval()
        self.coor2area_stage_model.train = disabled_train
        for param in self.coor2area_stage_model.parameters():
            param.requires_grad = False

    def instantiate_blank_stage(self, config):
        print('blank config:', config)
        model = instantiate_from_config(config)
        self.blank_stage_model = model.eval()
        self.blank_stage_model.train = disabled_train
        for param in self.blank_stage_model.parameters():
            param.requires_grad = False

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        if self.dual_diffusion:
            denoise_row = []
            denoise_row_pos = []
            for zd in tqdm(samples, desc=desc):
                if not self.change_order:
                    zd, zp = torch.split(zd, split_size_or_sections=4, dim=1)
                else:
                    zp, zd = torch.split(zd, split_size_or_sections=[1, 4], dim=1)
                denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                                force_not_quantize=force_no_decoder_quantization))
                denoise_row_pos.append(self.rescale_stage_model.decode(zp))
            n_imgs_per_row = len(denoise_row)
            denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)

            denoise_row_pos = torch.stack(denoise_row_pos)  # n_log_step, n_row, C, H, W
            denoise_pos_grid = rearrange(denoise_row_pos, 'n b c h w -> b n c h w')
            denoise_pos_grid = rearrange(denoise_pos_grid, 'b n c h w -> (b n) c h w')
            denoise_pos_grid = make_grid(denoise_pos_grid, nrow=n_imgs_per_row)
            return denoise_grid, denoise_pos_grid
        else:
            denoise_row = []
            for zd in tqdm(samples, desc=desc):
                denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                                force_not_quantize=force_no_decoder_quantization))
            n_imgs_per_row = len(denoise_row)
            denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
            return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # print((self.scale_factor * z).mean(), (self.scale_factor * z).std())
        return self.scale_factor * z


    def get_learned_conditioning(self, c, encoder='cond'):
        if encoder == 'cond':
            if self.cond_stage_forward is None:
                if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                    c = self.cond_stage_model.encode(c)
                    if isinstance(c, DiagonalGaussianDistribution):
                        c = c.mode()
                else:
                    c = self.cond_stage_model(c)
            else:
                assert hasattr(self.cond_stage_model, self.cond_stage_forward)
                c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        elif encoder == 'rescale':
            c = self.rescale_stage_model.encode(c)
        elif encoder == 'heatmap':
            c = self.heatmap_stage_model.encode(c)
        elif encoder == 'coor2area':
            c = self.coor2area_stage_model.encode(c)
        elif encoder == 'blank':
            c = self.blank_stage_model.encode(c)
        elif encoder == 'first':
            if hasattr(self.first_stage_model, 'encode') and callable(self.first_stage_model.encode):
                c = self.first_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.first_stage_model(c)
        elif encoder == 'null':
            
            return c.squeeze(-1)
        else:
            assert ValueError
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting
    

    # TODO: modify here
    @torch.no_grad()
    @sample_prompt_according_to_cond(sample_cond)
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=True,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['point', 'bbox', 'mask']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = {}
                    for mode in cond_key.keys():
                        curr_dict = {}
                        for elem in cond_key[mode]:
                            key = elem["key"]
                            if key != 'blank':
                                curr_xc = super().get_input(batch, key)
                            else:
                                curr_xc = torch.tensor(x.shape[0])
                            curr_dict[key] = curr_xc.to(self.device)
                        xc[mode] = curr_dict
            # xc = {'concat': {'background': tensor, 'mask': tensor},
            #       'crossattn': 'foreground': tensor}
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict):
                    c = {}
                    for mode in cond_key.keys():
                        curr_dict = {}
                        for elem in cond_key[mode]:
                            key = elem["key"]
                            encoder = elem["encoder"]
                            curr_dict[key] = \
                                self.get_learned_conditioning(xc[mode][key], encoder=encoder)
                            if mode == 'crossattn' and key == 'refer_person' and encoder == 'first':
                                curr_dict[key] = rearrange(curr_dict[key], 'b c h w -> b (h w) c')
                        c[mode] = curr_dict
                    if 'uncond_mask' in batch:
                        c['uncond_mask'] = batch['uncond_mask'].to(self.device)
                elif isinstance(xc, list):
                    c = [self.get_learned_conditioning(elem) for elem in xc]
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc  # condition
            if bs is not None:
                if isinstance(c, dict):
                    for key in c:
                        if isinstance(c[key], dict):
                            for inner_key in c[key]:
                                c[key][inner_key] = c[key][inner_key][:bs]
                        else:
                            c[key] = c[key][:bs]
                elif isinstance(c, list):
                    c = [elem[:bs] for elem in c]
                else:
                    c = c[:bs]

            # default false
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        # for visualization
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        
        x, c = self.get_input(batch, self.first_stage_key)
        
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        # Guassian sampling here when training
        if self.use_gauss_sample and self.training:
            t = torch.multinomial(self.gauss_pdf, x.shape[0]).to(self.device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            
            # if self.cond_stage_trainable:
            #     c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        if self.dual_diffusion:
            org_mask = c['concat']['org_mask']
            if not self.change_order:
                x = torch.concat([x, org_mask], dim=1)
            else:
                x = torch.concat([org_mask, x], dim=1)
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        
        if isinstance(cond, dict):
            if self.model.conditioning_key == 'hybrid':
                cond = {'c_concat': cond['concat'], 'c_crossattn': cond['crossattn'], 'uncond_mask': cond['uncond_mask']}
            else:
                model_cond_key = self.model.conditioning_key
                key = 'c_concat' if model_cond_key == 'concat' else 'c_crossattn'
                cond = {key: cond[model_cond_key], 'uncond_mask': cond['uncond_mask']}
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    # TODO: modify here for loss calculation
    def p_losses(self, x_start, cond, t, noise=None, separate=True):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # TODO: change here
        
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        if isinstance(cond, dict) and 'mask' in cond and self.mask_loss:
            loss_simple = self.get_loss(model_output, target, mean=False, mask=cond['mask']).mean([1, 2, 3])
        elif self.loss_type == 'balance' or self.loss_type == 'smooth':
            loss_simple, log = self.get_loss(model_output, target, mean=False, balance_mask=cond['concat']['org_mask'])
            loss_dict.update(log)
        else:
            if self.dual_diffusion and separate:
                pos_loss = self.get_loss(model_output[:,:1,...], target[:,:1,...], mean=False).mean(dim=[1, 2, 3])
                img_loss = self.get_loss(model_output[:,1:,...], target[:,1:,...], mean=False).mean(dim=[1, 2, 3])
                loss_simple = pos_loss + img_loss
                log_prefix = 'train' if self.training else 'val'
                loss_dict.update({f'{log_prefix}/pos_loss': pos_loss.mean()})
                loss_dict.update({f'{log_prefix}/img_loss': img_loss.mean()})
            else:
                loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if isinstance(cond, dict) and 'mask' in cond and self.mask_loss:
            loss_vlb = self.get_loss(model_output, target, mean=False, mask=cond['mask']).mean(dim=(1, 2, 3))
        elif self.loss_type == 'balance' or self.loss_type == 'smooth':
            loss_vlb, _ = self.get_loss(model_output, target, mean=False, balance_mask=cond['concat']['org_mask'])
        else:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
            print('log_every_t:', log_every_t)
        timesteps = self.num_timesteps
        print('timesteps:', timesteps)
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        print('b:', b)
        if x_T is None:
            img = torch.randn(shape, device=self.device)    # noise map
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if self.model.conditioning_key == 'hybrid':
                for key in cond:
                    if key != 'uncond_mask':
                        cond[key] = {in_key: cond[key][in_key][:batch_size] for in_key in cond[key]}
                    else:
                        cond[key] = {key: cond[key][:batch_size]}
            elif isinstance(cond, dict):
                for key in cond:
                    if key != 'uncond_mask':
                        cond[key] = {in_key: cond[key][in_key][:batch_size] for in_key in cond[key]}
                    else:
                        cond[key] = {key: cond[key][:batch_size]}
                # cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                # list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        # print('cond:', cond)

        if start_T is not None:
            # set biggest T
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        timestep_output = []
        timestep_output.append(timesteps)
        intermediates.append(img)
        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                print('time =', i, 'appended.')
                timestep_output.append(i)
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        print('timestep_output:', timestep_output)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            print('return inter, len(img):', len(img))
            return img, intermediates
        print('return img only, len(img):', len(img))
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    # new function
    @torch.no_grad()
    def unfeat_extract(self, imgs):
        imgs = imgs.transpose(1, 3)
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(imgs.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).to(imgs.device)
        imgs = imgs * std + mean
        imgs = 2 * imgs - 1
        imgs = imgs.transpose(1, 3)
        return imgs

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if self.model.conditioning_key == 'hybrid':
                for key in c['concat']:
                    if key == 'background':
                        xc['concat'][key] = self.first_stage_model.decode(c['concat'][key])
                        log[f'conditioning_concat_{key}'] = xc['concat'][key]
                    elif key == 'point':
                        xc['concat'][key] = self.heatmap_stage_model.decode(c['concat'][key])
                        log[f'conditioning_concat_{key}'] = c['concat'][key]
                    elif key == 'bbox':
                        xc['concat'][key] = self.coor2area_stage_model.decode(c['concat'][key])
                        log[f'conditioning_concat_{key}'] = c['concat'][key]
                    elif key == 'blank':
                        xc['concat'][key] = self.blank_stage_model.decode(c['concat'][key])
                        log[f'conditioning_concat_{key}'] = c['concat'][key]
                    else:
                        xc['concat'][key] = self.rescale_stage_model.decode(c['concat'][key])
                        log[f'conditioning_concat_{key}'] = xc['concat'][key]    
                for key in c['crossattn']:
                    log[f'conditioning_crossattn_{key}'] = self.unfeat_extract(xc['crossattn'][key])
            elif hasattr(self.cond_stage_model, "decode"):
                if isinstance(c, dict):
                    xc = {}
                    for key in c:
                        if key != 'mask':
                            xc[key] = self.cond_stage_model.decode(c[key])
                    if 'mask' in c:
                        all_keys = list(c.keys())
                        all_keys.remove('mask')
                        xc['mask'] = torch.nn.functional.interpolate(c['mask'], \
                                        size=xc[all_keys[0]].shape[-2:])
                    for key in xc:
                        log[f'conditioning_{key}'] = xc[key]
                else: 
                    xc = self.cond_stage_model.decode(c)
                    log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            # with self.ema_scope("Plotting"):
            cfg_scale = 4.0
            uc_cond = {'concat': {}, 'crossattn': {}}
            uc_cond['concat']['background'] = self.first_stage_model.encode(batch["zero_bg"][:N]).mode()
            uc_cond['concat']['mask'] = self.rescale_stage_model.encode(batch["mask"][:N] * 0 + 1)
            uc_cond['concat']['org_mask'] = c['concat']['org_mask'][:N]
            uc_cond['crossattn']['foreground'] = self.cond_stage_model.encode(batch["zero_fg"][:N])
            uc_cond['uncond_mask'] = batch['uncond_mask'][:N] * 0 + 1
            
            samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta,
                                                        unconditional_guidance_scale=cfg_scale,
                                                        unconditional_conditioning=uc_cond)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            if self.dual_diffusion:
                if not self.change_order:
                    x_samples, pos_samples = torch.split(samples, split_size_or_sections=4, dim=1)
                else:
                    pos_samples, x_samples = torch.split(samples, split_size_or_sections=[1,4], dim=1)
                x_samples = self.decode_first_stage(x_samples)
                pos_samples = self.rescale_stage_model.decode(pos_samples)
                log["samples"] = x_samples
                log["samples_pos"] = pos_samples
            else:
                x_samples = self.decode_first_stage(samples)
                log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                # with self.ema_scope("Plotting Quantized Denoised"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                            ddim_steps=ddim_steps,eta=ddim_eta,
                                                            quantize_denoised=True)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if plot_progressive_rows:
            # with self.ema_scope("Plotting Progressives"):
            img, progressives = self.progressive_denoising(c,
                                                            shape=(self.channels, self.image_size, self.image_size),
                                                            batch_size=N)
            if self.dual_diffusion:
                prog_row, pos_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
                log["progressive_row"] = prog_row
                log["progressive_pos_row"] = pos_row
            else:
                prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
                log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


    def configure_optimizers(self):
        lr = self.learning_rate
        if self.finetune_head:
            params = list(self.params_list)
        else:
            params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
            
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            
            print('scheduler:', self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt


    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    # modified function
    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, uncond_mask = None):
        
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            if isinstance(c_concat, dict):
                xc = torch.cat([x] + [c_concat[key] for key in sorted(c_concat)], dim=1)
            else:
                xc = torch.cat([x] + c_concat, dim=1)
            # out = self.diffusion_model(xc, t, uncond_mask=uncond_mask)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            # out = self.diffusion_model(x, t, context=cc, uncond_mask=uncond_mask)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            if isinstance(c_concat, dict):
                concat_keys = c_concat.keys()
                for prompt_type in ['point', 'bbox', 'mask', 'blank']:
                    if prompt_type in concat_keys:
                        xc = torch.cat([x] + [c_concat[prompt_type], c_concat['background']], dim=1)
                        break
            else:
                xc = torch.cat([x] + c_concat, dim=1)
            # xc = torch.cat([x] + c_concat, dim=1)
            if isinstance(c_crossattn, dict):
                cc = torch.cat([c_crossattn[key] for key in sorted(c_crossattn)], dim=1)
            else:
                cc = torch.cat(c_crossattn, dim=1)
            # cc = torch.cat(c_crossattn, 1)
            if isinstance(uncond_mask, dict):
                uncond_mask = uncond_mask['uncond_mask']
            out = self.diffusion_model(xc, t, context=cc, uncond_mask=uncond_mask)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs
