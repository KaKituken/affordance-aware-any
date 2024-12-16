import argparse, os, sys, datetime, glob, importlib, csv
from typing import Any
import numpy as np
import time
import re
import json
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset, default_collate
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from subprocess import getoutput

import copy

import pdb
import torch.distributed as dist

from pytorch_fid.fid_score import calculate_fid_given_paths


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--fid", action="store_true", help="if sampling images for FID"
    )
    ### extra args for distributed
    parser.add_argument(
        "--distributed", action="store_true", help="if using distributed training"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="total number of nodes",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="total number of GPUs across all nodes, needs to be set if number of GPUs is not equal across all nodes",
    )
    parser.add_argument(
        "--node-id", type=int, default=0, help="ID of the current node [0...world-size]"
    )
    parser.add_argument(
        "--nodes-info",
        type=str,
        default=None,
        help="list of IP addresses for each node",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

# def collate_fn(examples):
#         foreground = torch.stack([example[0] for example in examples])
#         foreground = foreground.to(memory_format=torch.contiguous_format).float()
#         background = torch.stack([example[1] for example in examples])
#         background = background.to(memory_format=torch.contiguous_format).float()
#         groundtruth = torch.stack([example[2] for example in examples])
#         groundtruth = groundtruth.to(memory_format=torch.contiguous_format).float()
#         point = torch.stack([example[3]['point'] for example in examples])
#         point = point.to(memory_format=torch.contiguous_format).float()
#         bbox = torch.stack([example[3]['bbox'] for example in examples])
#         bbox = bbox.to(memory_format=torch.contiguous_format).float()
#         mask = torch.stack([example[3]['mask'] for example in examples])
#         mask = mask.to(memory_format=torch.contiguous_format).float()
#         # name_num = torch.stack([example[4] for example in examples])
#         # name_num = name_num.to(memory_format=torch.contiguous_format).float()
#         # bbox, point = example[3]['bbox'], example[3]['point']
#         return {
#             "foreground": foreground,
#             "background": background,
#             "groundtruth": groundtruth,
#             "point": point,
#             "bbox": bbox,
#             "mask": mask    # long()
#             # "name_num": name_num
#         }

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None, fid_cond=None, fid_hall=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.fid_name = re.sub('[^a-zA-Z0-9]', '', str(validation))             
            # self.img_size = validation['params']['resolution']
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        if fid_cond is not None:
            self.dataset_configs["fid_cond"] = fid_cond
            self.fid_cond_dataloader = self._fid_cond_dataloader
        if fid_hall is not None:
            self.dataset_configs["fid_hall"] = fid_hall
            self.fid_hall_dataloader = self._fid_hall_dataloader
        self.wrap = wrap

    def prepare_data(self):
        return
        # for data_cfg in self.dataset_configs.values():
        #     instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
    
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)
    
    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, 
                          shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        print('current rank:', trainer.global_rank)
        if trainer.global_rank == 0:
            # pdb.set_trace()
            # Create logdirs and save configs
            # self.logdir = f'{self.logdir}_{trainer.global_rank}'
            # self.ckptdir = f'{self.ckptdir}_{trainer.global_rank}'
            # self.cfgdir = f'{self.cfgdir}_{trainer.global_rank}'
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            print('rank 0 config dir:', self.cfgdir)
            print('now:', self.now)

            print('check if exist', self.cfgdir, os.path.exists(self.cfgdir))
            OmegaConf.save(self.config,
                        os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # # ModelCheckpoint callback created log directory --- remove it
            # if not self.resume and os.path.exists(self.logdir):
            #     dst, name = os.path.split(self.logdir)
            #     dst = os.path.join(dst, "child_runs", name)
            #     os.makedirs(os.path.split(dst)[0], exist_ok=True)
            #     try:
            #         os.rename(self.logdir, dst)
            #     except FileNotFoundError:
            #         pass
            pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_val_img_max_num = 1

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                # TODO buggy diffusion row/progressive row vis if logging < 3 images
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx >= self.log_val_img_max_num:
            return
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class FIDCallback(Callback):
    def __init__(self, dataset_cfg, batch_size, batch_freq, num_workers,
                 gt_dir, pd_dir, ddim_steps, max_batch, cfg_scales, calc_fid_score):
        super().__init__()
        self.dataset = instantiate_from_config(dataset_cfg)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_freq = batch_freq
        self.gt_dir = gt_dir
        self.pd_dir = pd_dir
        self.ddim_steps = ddim_steps
        self.max_batch = max_batch
        self.cfg_scales = cfg_scales
        self.calc_fid_score = calc_fid_score
        self.dataloader = DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        check_idx = pl_module.global_step
        device = pl_module.parameters().__next__().device
        log_dict = {}
        if (self.check_frequency(check_idx)):
            gt_save_path = os.path.join(self.gt_dir, str(check_idx), 'gt')
            pd_save_path = os.path.join(self.pd_dir, str(check_idx), 'pd')
            os.makedirs(os.path.join(gt_save_path), exist_ok=True)
            os.makedirs(os.path.join(pd_save_path, 'w_cfg'), exist_ok=True)
            if pl_module.dual_diffusion:
                pos_save_path = os.path.join(self.pd_dir, str(check_idx), 'pos')
                os.makedirs(os.path.join(pos_save_path, 'w_cfg'), exist_ok=True)
            if len(self.cfg_scales) == 0:
                os.makedirs(os.path.join(pd_save_path, 'wo_cfg'), exist_ok=True)
            with torch.no_grad():
                sampler = DDIMSampler(pl_module)

                for idx, b in enumerate(self.dataloader):
                    cond_dict_sdm = {'concat': {}, 'crossattn': {}}
                    cond_dict_sdm['concat']['background'] = pl_module.first_stage_model.encode(b['background'].to(device)).mode()
                    cond_dict_sdm['concat']['bbox'] = pl_module.coor2area_stage_model.encode(b["bbox"].to(device))    # ??? why does it work in baseline ???
                    cond_dict_sdm['crossattn']['foreground'] = pl_module.cond_stage_model.encode(b['foreground'].to(device))
                    cond_dict_sdm['uncond_mask'] = b['uncond_mask'].to(device) * 0
                    c_sdm = copy.deepcopy(cond_dict_sdm)

                    cond_dict_sdm['concat']['background'] = pl_module.first_stage_model.encode(b["zero_bg"].to(device)).mode()
                    cond_dict_sdm['concat']['bbox'] = pl_module.rescale_stage_model.encode(b["mask"].to(device) * 0 + 1)
                    cond_dict_sdm['crossattn']['foreground'] = pl_module.cond_stage_model.encode(b["zero_fg"].to(device))
                    cond_dict_sdm['uncond_mask'] = b['uncond_mask'].to(device) * 0 + 1
                    uc_sdm = copy.deepcopy(cond_dict_sdm)

                    # pdb.set_trace()
                    shape_sdm = (pl_module.channels,)+c_sdm['concat']['background'].shape[2:]

                    c_cond = c_sdm
                    uc_cond = uc_sdm
                    batch_size = c_sdm['concat']['background'].shape[0]                        
                    shape = shape_sdm

                    groundtruth = ((b['groundtruth'] + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
                    gt_mask = b['org_mask'].permute(0, 2, 3, 1).cpu().numpy()
                    # run wo_cfg only if len(self.cfg_scales) == 0
                    if len(self.cfg_scales) == 0:
                        wo_cfg_sample_ddim, _ = sampler.sample(S=self.ddim_steps,
                                                        conditioning=c_cond,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        verbose=False)
                        if pl_module.dual_diffusion:
                            if not pl_module.change_order:
                                wo_cfg_sample_ddim, pos_samples = torch.split(wo_cfg_sample_ddim, split_size_or_sections=4, dim=1)
                            else:
                                pos_samples, wo_cfg_sample_ddim = torch.split(wo_cfg_sample_ddim, split_size_or_sections=[1,4], dim=1)
                            pos_samples_ = pl_module.rescale_stage_model.decode(pos_samples).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                            pos_diff = pos_samples_ - gt_mask
                            pos_l1 = torch.sum(torch.abs(pos_diff))
                            if "mse_score/wo_cfg_pos_l1" in log_dict:
                                log_dict["mse_score/wo_cfg_pos_l1"] += pos_l1 / self.max_batch
                            else:
                                log_dict["mse_score/wo_cfg_pos_l1"] = pos_l1 / self.max_batch
                        wo_cfg_x_samples_ddim = pl_module.decode_first_stage(wo_cfg_sample_ddim)
                        wo_cfg_predicted = ((wo_cfg_x_samples_ddim + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                        pos_samples = pos_samples.clip(0, 1).squeeze(1).cpu().numpy()
                        diff = wo_cfg_predicted - groundtruth
                        mse_distance = np.mean(diff*diff)
                        if "mse_score/wo_cfg_mse" in log_dict:
                            log_dict["mse_score/wo_cfg_mse"] += mse_distance / self.max_batch
                        else:
                            log_dict["mse_score/wo_cfg_mse"] = mse_distance / self.max_batch
                        for i in range(batch_size):
                            gt = (groundtruth[i] * 255).astype(np.uint8)
                            pos = (pos_samples[i] * 255).astype(np.uint8)
                            wo_cfg_pd = (wo_cfg_predicted[i] * 255).astype(np.uint8)
                            Image.Image.save(Image.fromarray(gt), os.path.join(gt_save_path, f'{idx*self.batch_size+i}.jpg'))
                            Image.Image.save(Image.fromarray(wo_cfg_pd), os.path.join(pd_save_path, 'wo_cfg', f'{idx*self.batch_size+i}.jpg'))
                            Image.Image.save(Image.fromarray(pos), os.path.join(pos_save_path, 'wo_cfg', f'{idx*self.batch_size+i}.jpg'))

                    else:
                        # run cfg
                        for cfg_scale in self.cfg_scales:
                            os.makedirs(os.path.join(pd_save_path, 'w_cfg', f'{cfg_scale}'), exist_ok=True)
                            if pl_module.dual_diffusion:
                                os.makedirs(os.path.join(pos_save_path, 'w_cfg', f'{cfg_scale}'), exist_ok=True)
                            cfg_samples_ddim, _ = sampler.sample(S=self.ddim_steps,
                                                            conditioning=c_cond,
                                                            batch_size=batch_size,
                                                            shape=shape,
                                                            unconditional_guidance_scale=cfg_scale,
                                                            unconditional_conditioning=uc_cond,
                                                            verbose=False)
                            # clean a part of cache
                            del b
                            garbage_collection_cuda()
                            if pl_module.dual_diffusion:
                                if not pl_module.change_order:
                                    cfg_samples_ddim, pos_samples = torch.split(cfg_samples_ddim, split_size_or_sections=4, dim=1)
                                else:
                                    pos_samples, cfg_samples_ddim = torch.split(cfg_samples_ddim, split_size_or_sections=[1,4], dim=1)
                                pos_samples_ = pl_module.rescale_stage_model.decode(pos_samples).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                                # pdb.set_trace()
                                pos_diff = pos_samples_ - gt_mask
                                pos_l1 = np.mean(np.abs(pos_diff))
                                if f"mse_score/cfg_{cfg_scale}_pos_l1" in log_dict:
                                    log_dict[f"mse_score/cfg_{cfg_scale}_pos_l1"] += pos_l1 / self.max_batch
                                else:
                                    log_dict[f"mse_score/cfg_{cfg_scale}_pos_l1"] = pos_l1 / self.max_batch
                                pos_samples = pl_module.rescale_stage_model.decode(pos_samples).clip(0, 1).squeeze(1).cpu().numpy()
                            cfg_x_samples_ddim = pl_module.decode_first_stage(cfg_samples_ddim)
                            cfg_predicted = ((cfg_x_samples_ddim + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                            diff = groundtruth - cfg_predicted
                            mse_distance = np.mean(diff*diff)
                            if f"mse_score/cfg_{cfg_scale}_mse" in log_dict:
                                log_dict[f"mse_score/cfg_{cfg_scale}_mse"] += mse_distance / self.max_batch
                            else:
                                log_dict[f"mse_score/cfg_{cfg_scale}_mse"] = mse_distance / self.max_batch
                        
                            for i in range(batch_size):
                                gt = (groundtruth[i] * 255).astype(np.uint8)
                                cfg_pd = (cfg_predicted[i] * 255).astype(np.uint8)
                                Image.Image.save(Image.fromarray(gt), os.path.join(gt_save_path, f'{idx*self.batch_size+i}.jpg'))
                                Image.Image.save(Image.fromarray(cfg_pd), 
                                                os.path.join(pd_save_path,
                                                            'w_cfg', 
                                                            f'{cfg_scale}', 
                                                            f'{idx*self.batch_size+i}.jpg'))
                                if pl_module.dual_diffusion:
                                    pos = (pos_samples[i] * 255).astype(np.uint8)
                                    Image.Image.save(Image.fromarray(pos), 
                                                    os.path.join(pos_save_path,
                                                                'w_cfg', 
                                                                f'{cfg_scale}', 
                                                                f'{idx*self.batch_size+i}.jpg'))

                    # calc fid
                    if idx >= self.max_batch - 1:
                        for key, val in log_dict.items():
                            trainer.logger.log_metrics({key: val}, step=check_idx)
                        # calc fid score
                        if self.calc_fid_score:
                            fid_log_dict = {}
                            # calc wo_cfg fid
                            # pdb.set_trace()
                            if len(self.cfg_scales) == 0:
                                wo_cfg_fid = calculate_fid_given_paths(paths=[gt_save_path, os.path.join(pd_save_path, 'wo_cfg')],
                                                                    batch_size=64, dims=2048, device=device)
                                fid_log_dict.update({"fid_score/wo_cfg_fid": wo_cfg_fid})
                            # calc w_cfg fid
                            for cfg in self.cfg_scales:
                                cfg_fid = calculate_fid_given_paths(paths=[gt_save_path, os.path.join(pd_save_path, 'w_cfg', f'{cfg}')],
                                                                    batch_size=64, dims=2048, device=device)
                                fid_log_dict.update({f"fid_score/cfg_{cfg}_fid": cfg_fid})
                            for key, val in fid_log_dict.items():
                                trainer.logger.log_metrics({key: val}, step=check_idx)
                        break
        if is_train:
            pl_module.train()
        return 
    
    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0 and check_idx > 0:
            return True
        return False
    
if __name__ == "__main__":
    # print('rank:', os.environ['SLURM_PROCID'])
    # print('local rank:', os.environ['SLURM_LOCALID'])
    # print('rank:', dist.get_rank())
    # print('world_size:', dist.get_world_size())
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    print("you don't print this?")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    print('before parser')
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    print('num-nodes:', opt.num_nodes)

    if opt.distributed:
        # set environment variables for 'env://'
        import re
        def extract_strings(s):
            match = re.match(r'(\w+)\[([\w,]+)\]', s)
            if not match:
                return []

            prefix = match.group(1)
            items = match.group(2).split(',')
            
            return [prefix + item for item in items]
        
        node_list = os.getenv('SLURM_JOB_NODELIST',
                              os.environ['SLURM_NODELIST']) # holygpu2c[0901,0913]
        # if ',' not in node_list:
        #     nodes_info = extract_strings(node_list)
        # else:
        #     nodes_info = node_list.split(',')
        # print('nodes info:', nodes_info)
        # if len(nodes_info[0]) == len(nodes_info[1]):
        #     opt.nodes_info = nodes_info
        # else:
        #     opt.nodes_info = extract_strings(node_list)
        master_addr = getoutput('scontrol show hostname {} | head -n1'.format(node_list))
        print("PARSED SUCCESSFULLY")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(29500)

        # Set up dist configrations
        opt.gpus = torch.cuda.device_count()
        if opt.world_size is None:
            # infer the world size; assuming all instances have the same number of gpus
            opt.world_size = opt.num_nodes * opt.gpus
            # opt.world_size = os.environ['SLURM_NTASKS']
        # os.environ["WORLD_SIZE"] = str(opt.world_size)
        node_name = os.environ['SLURMD_NODENAME']
        # os.environ["NODE_RANK"] = str(opt.nodes_info.index(node_name))
        # opt.node_id = os.environ["NODE_RANK"]
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ["WORLD_SIZE"] = str(opt.world_size)
        # os.environ["NODE_RANK"] = str(opt.node_id)

        print("Distributed training info:")
        print("Nodes info:", opt.nodes_info)
        print("Number of nodes: {}".format(opt.num_nodes))
        print("Current node: {}".format(opt.node_id))
        print("World size: {}".format(opt.world_size))

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            if "trainstep_checkpoints" in opt.resume:
                logdir = "/".join(paths[:-3])
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            else:
                logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        if not opt.fid:
            opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        print('Did I provide logdir???', opt.logdir)
        logdir = os.path.join(opt.logdir, nowname)
    

    print('now you get the logdir from nowhere', logdir)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        print(config.data)

        ### extra distributed logistics
        # pdb.set_trace()
        if opt.gpus == "-1" or opt.gpus == -1:  # uses all available GPUs
            num_of_gpus = torch.cuda.device_count()
            devices = [int(d) for d in range(num_of_gpus)]
        else:
            print('opt.gpus:', opt.gpus)
            try:
                devices = [
                    int(d) for d in range(int(opt.gpus))
                ]  # integer specifies how many GPUs to use
            except:
                devices = opt.gpus.strip(",").split(",")  # use only specified GPUs
                devices = [int(d) for d in devices]

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        # devices = [1]
        print('devices:', devices)
        trainer_config["gpus"] = devices
        trainer_config["num_nodes"] = opt.num_nodes
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        print('config.model:', config.model)
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        print('logger_cfg:', logger_cfg)
        # pdb.set_trace()
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        # !!!! Turn it off when tuning params
        # if hasattr(model, "monitor"):
        #     print(f"Monitoring {model.monitor} as checkpoint metric.")
        #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
        #     default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,  # Modify here for the frequency to save checkpoint
                         'save_weights_only': True
                    }
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        if opt.fid:
            print(
                'EVAL MODE: Sampling images for FID computation.')
            print(f"Loading {ckpt} for FID samples.")
            model.init_from_ckpt(ckpt)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        print('trainer_opt:', trainer_opt)
        print('trainer_kwargs:', trainer_kwargs)
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###
        print('trainer:', trainer)

        # data
        data = instantiate_from_config(config.data)
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus)
            # ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        # pdb.set_trace()
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")
   
        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception as e:
        print('wtf')
        print(e)
        if opt.debug and trainer.global_rank == 0:
            try:
                import ipdb as debugger
            except ImportError:
                import ipdb as debugger
            debugger.set_trace()
            # debugger.post_mortem()
        raise
    finally:
        print('qwq')
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())