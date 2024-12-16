import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import copy

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
from transformers import AutoImageProcessor
from torchvision import transforms
import pdb

# image_resolution = 256


# config = OmegaConf.load('/data/birth/lmx/work/Class_projects/course2/blurdiff/insert-anything/sd_style/configs/compress.yaml')
config = OmegaConf.load('/data/birth/lmx/work/Class_projects/course2/blurdiff/insert-anything/sd_style/configs/compress512.yaml')
model = instantiate_from_config(config.model)

print('loading checkpoint...')
# model.load_state_dict(torch.load('/data/birth/lmx/data/datasets/SAM-FB/modify_compress_epoch=000005-step=000050000.ckpt')["state_dict"], strict=False)
model.load_state_dict(torch.load('/data/birth/lmx/data/datasets/SAM-FB/logs/modify_full_512/2024-05-16T14-09-18_dual_input_branch_compress/checkpoints/epoch=000007.ckpt')["state_dict"], strict=False)


model = model.cuda()
ddim_sampler = DDIMSampler(model)
device = model.parameters().__next__().device

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

def unfeat_extract(imgs):
    imgs = imgs.transpose(1, 3)
    mean = torch.Tensor([0.481, 0.458, 0.408]).to(imgs.device)
    std = torch.Tensor([0.269, 0.261, 0.276]).to(imgs.device)
    imgs = imgs * std + mean
    imgs = 2 * imgs - 1
    imgs = imgs.transpose(1, 3)
    return imgs

def process(input_fg, point, bbox, mask_input, num_samples, image_resolution, ddim_steps, img_scale, seed, eta):

    bg_processor = transforms.Compose([
        transforms.Resize(image_resolution),
        transforms.CenterCrop(image_resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    
    print(point)
    print(bbox)
    print(mask_input)
    fg = input_fg
    bg = mask_input['background']
    mask = mask_input["layers"][0]
    mask = np.array(mask)

    # torch.manual_seed(seed)
    # np.random.seed(seed)

    with torch.no_grad():

        # call the model from ./inference_dual.py
        fg = fg.convert('RGB').resize((224, 224))
        fg = torch.from_numpy(processor(images=[fg])['pixel_values'][0])
        fg = torch.stack([fg] * num_samples)
        bg = bg_processor(bg)
        bg = torch.stack([bg] * num_samples)


        print(fg.shape)
        print(bg.shape)

        cond_dict_sdm = {'concat': {}, 'crossattn': {}}
        cond_dict_sdm['concat']['background'] = model.first_stage_model.encode(bg.to(device)).mode()
        cond_dict_sdm['crossattn']['foreground'] = model.cond_stage_model.encode(fg.to(device))
        cond_dict_sdm['uncond_mask'] = torch.zeros(size=(num_samples,), device=device)
        if np.any(mask == 255):
            # input mask
            mask = np.array(mask) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask = torch.stack([mask] * num_samples)
            cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(mask.to(device))
        elif len(bbox):
            # input bbox
            bbox = np.array([float(b) for b in bbox.split(' ')], dtype=np.float32)
            bbox = torch.stack([torch.from_numpy(bbox)] * num_samples)
            cond_dict_sdm['concat']['bbox'] = model.coor2area_stage_model.encode(bbox.to(device))
        elif len(point):
            # input point
            point = np.array([float(p) for p in point.split(' ')], dtype=np.float32)
            point = torch.stack([torch.from_numpy(point)] * num_samples)
            cond_dict_sdm['concat']['point'] = model.heatmap_stage_model.encode(point.to(device))
        else:
            # input null
            point = torch.stack([torch.tensor([-1, -1])] * num_samples)
            cond_dict_sdm['concat']['point'] = model.heatmap_stage_model.encode(point.to(device))
        
        c_sdm = copy.deepcopy(cond_dict_sdm)

        zero_fg = torch.zeros_like(fg)
        zero_bg = torch.zeros_like(bg)
        cond_dict_sdm['concat']['background'] = model.first_stage_model.encode(zero_bg.to(device)).mode()
        # cond_dict_sdm['concat']['point'] = model.heatmap_stage_model.encode(torch.stack([torch.tensor([-1, -1])] * num_samples).to(device))
        cond_dict_sdm['concat']['bbox'] = model.heatmap_stage_model.encode(torch.stack([torch.tensor([-1, -1])] * num_samples).to(device))
        cond_dict_sdm['crossattn']['foreground'] = model.cond_stage_model.encode(zero_fg.to(device))
        cond_dict_sdm['uncond_mask'] = torch.ones(size=(num_samples,), device=device)
        uc_sdm = copy.deepcopy(cond_dict_sdm)


        shape_sdm = (model.channels,)+c_sdm['concat']['background'].shape[2:]
        # pdb.set_trace()

        c_cond = c_sdm
        uc_cond = uc_sdm
        batch_size = c_sdm['concat']['background'].shape[0]                        
        shape = shape_sdm
        sampler = DDIMSampler(model)
        cfg_scale = img_scale
        cfg_samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c_cond,
                                        batch_size=batch_size,
                                        shape=shape,
                                        unconditional_guidance_scale=img_scale,
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
        # background = ((bg + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
        # foreground = ((unfeat_extract(fg) + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        # point = c_sdm['concat']['point'].permute(0, 2, 3, 1).cpu().numpy()
        predicted = ((cfg_x_samples_ddim + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        pos_samples = pos_samples.clip(0, 1).squeeze(1).cpu().numpy()

        pd_list = []
        pos_list = []
        for i in range(num_samples):
            pd = (predicted[i] * 255).astype(np.uint8)
            pos = (pos_samples[i] * 255).astype(np.uint8)
            pd_list.append(pd)
            pos_list.append(pos)
    
    return pd_list


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Insert anything into any scene")
    with gr.Row():
        with gr.Column():
            input_fg = gr.Image(label="Foreground", type="pil", image_mode="RGB")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=128, maximum=768, value=512, step=64)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                img_scale = gr.Slider(label="Image Guidance Scale", minimum=0.1, maximum=30.0, value=4.0, step=0.1)
                # text_scale = gr.Slider(label="Text Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=1.0)
        with gr.Column():
            mask = gr.ImageMask(label="Mask", image_mode="RGB", sources="upload", type="pil", crop_size="1:1",)
            bbox = gr.Textbox(label="Bounding Box")
            point = gr.Textbox(label="Point")
            run_button = gr.Button(value="Run")
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", height=image_resolution)
    ips = [input_fg, point, bbox, mask, num_samples, image_resolution, ddim_steps, img_scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(share=True, server_name='0.0.0.0', debug=True)