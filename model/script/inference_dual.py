import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import copy
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import CLIPProcessor, AutoImageProcessor
from torchvision import transforms
import pdb

def unfeat_extract(imgs):
    imgs = imgs.transpose(1, 3)
    mean = torch.Tensor([0.481, 0.458, 0.408]).to(imgs.device)
    std = torch.Tensor([0.269, 0.261, 0.276]).to(imgs.device)
    imgs = imgs * std + mean
    imgs = 2 * imgs - 1
    imgs = imgs.transpose(1, 3)
    return imgs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        '--blend', 
        action='store_true',
        help="blend the pred image with masked image"
    )
    parser.add_argument(
        '-b',
        '--background',
        type=str,
        help="path of background image"
    )
    parser.add_argument(
        '-f',
        '--foreground',
        type=str,
        help="path of foreground image"
    )
    parser.add_argument(
        '-s',
        '--save-dir',
        type=str,
        help="path to save image"
    )
    parser.add_argument(
        '--num-predict',
        type=int,
        default=50,
        help="path of foreground image"
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help="path of config"
    )
    parser.add_argument(
        '-k',
        '--checkpoint',
        type=str,
        help="path of checkpoint"
    )
    parser.add_argument(
        '-m',
        '--mask',
        type=str,
        help="path of mask"
    )
    parser.add_argument(
        '--num-repeats',
        type=int, 
        default=10
    )
    opt = parser.parse_args()

    save_path = opt.save_dir
    os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pos'), exist_ok=True)

    config = OmegaConf.load(opt.config)
    print('instantiating model...')
    model = instantiate_from_config(config.model)
    print('loading check point...')
    model.load_state_dict(torch.load(opt.checkpoint)["state_dict"], strict=False)
    attn_maps = []
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    bg_processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])

    fg = Image.open(opt.foreground).convert('RGB').resize((224, 224))
    bg = Image.open(opt.background).convert('RGB')
    fg = torch.from_numpy(processor(images=[fg])['pixel_values'][0])
    fg = torch.stack([fg] * opt.num_predict)
    bg = bg_processor(bg)
    bg = torch.stack([bg] * opt.num_predict)

    # point = torch.stack([torch.tensor([-1, -1])] * opt.num_predict)
    # point = torch.stack([torch.tensor([0.5, 0.5])] * opt.num_predict)
    mask = Image.open(opt.mask).convert('L').resize((256, 256))
    mask = np.array(mask) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0)
    mask = torch.stack([mask] * opt.num_predict)

    print(fg.shape)
    print(bg.shape)
    
    with torch.no_grad():

        cond_dict_sdm = {'concat': {}, 'crossattn': {}}
        cond_dict_sdm['concat']['background'] = model.first_stage_model.encode(bg.to(device)).mode()
        # cond_dict_sdm['concat']['point'] = model.heatmap_stage_model.encode(point.to(device))
        # cond_dict_sdm['concat']['bbox'] = model.coor2area_stage_model.encode(bbox.to(device))
        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(mask.to(device))
        cond_dict_sdm['crossattn']['foreground'] = model.cond_stage_model.encode(fg.to(device))
        cond_dict_sdm['uncond_mask'] = torch.zeros(size=(opt.num_predict,), device=device)
        c_sdm = copy.deepcopy(cond_dict_sdm)

        zero_fg = torch.zeros_like(fg)
        zero_bg = torch.zeros_like(bg)
        cond_dict_sdm['concat']['background'] = model.first_stage_model.encode(zero_bg.to(device)).mode()
        # cond_dict_sdm['concat']['point'] = model.heatmap_stage_model.encode(torch.stack([torch.tensor([-1, -1])] * opt.num_predict).to(device))
        cond_dict_sdm['concat']['bbox'] = model.heatmap_stage_model.encode(torch.stack([torch.tensor([-1, -1])] * opt.num_predict).to(device))
        cond_dict_sdm['crossattn']['foreground'] = model.cond_stage_model.encode(zero_fg.to(device))
        cond_dict_sdm['uncond_mask'] = torch.ones(size=(opt.num_predict,), device=device)
        uc_sdm = copy.deepcopy(cond_dict_sdm)


        shape_sdm = (model.channels,)+c_sdm['concat']['background'].shape[2:]
        # pdb.set_trace()

        c_cond = c_sdm
        uc_cond = uc_sdm
        batch_size = c_sdm['concat']['background'].shape[0]                        
        shape = shape_sdm
        sampler = DDIMSampler(model)
        cfg_scale = 4.0
        for r_idx in range(opt.num_repeats):
            cfg_samples_ddim, _ = sampler.sample(S=opt.steps,
                                            conditioning=c_cond,
                                            batch_size=batch_size,
                                            shape=shape,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc_cond,
                                            eta=1,
                                            verbose=False)
            if model.dual_diffusion:
                if not model.change_order:
                    cfg_samples_ddim, pos_samples = torch.split(cfg_samples_ddim, split_size_or_sections=4, dim=1)
                else:
                    pos_samples, cfg_samples_ddim = torch.split(cfg_samples_ddim, split_size_or_sections=[1,4], dim=1)
            pos_samples = model.rescale_stage_model.decode(pos_samples)
            cfg_x_samples_ddim = model.decode_first_stage(cfg_samples_ddim)
            background = ((bg + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
            foreground = ((unfeat_extract(fg) + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            # point = c_sdm['concat']['point'].permute(0, 2, 3, 1).cpu().numpy()
            predicted = ((cfg_x_samples_ddim + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            pos_samples = pos_samples.clip(0, 1).squeeze(1).cpu().numpy()

            # pdb.set_trace()


            for i in range(opt.num_predict):
                pd = (predicted[i] * 255).astype(np.uint8)
                pos = (pos_samples[i] * 255).astype(np.uint8)
                save_idx = r_idx*opt.num_predict+i
                Image.Image.save(Image.fromarray(pd), os.path.join(save_path, 'img', f'{save_idx}.jpg'))
                Image.Image.save(Image.fromarray(pos), os.path.join(save_path, 'pos', f'{save_idx}.jpg'))