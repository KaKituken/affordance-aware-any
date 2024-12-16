from __future__ import annotations

__all__ = ["FBPImageDataset", "FBPVideoDataset"]

import os
import json
import yaml
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import pdb
import random
from omegaconf import OmegaConf

from ldm.data.data_fbp.utils.parser import TransformParser
from main import instantiate_from_config
import copy


class FBPImageDataset(Dataset):

    def __init__(
            self, 
            fbp_path,
            gt_path,
            num_folder=5,
            mode='train',
            transform=None,
            off_line=False) -> None:
        super().__init__()
        self.fbp_path = fbp_path
        self.gt_path = gt_path
        self.num_folder = num_folder
        self.mode = mode
        self.off_line = off_line

        parser = TransformParser()
        if transform is None:
            self.fg_transform = None
            self.bg_gt_transform = None
            print('transform not get:', transform)
        else:
            self.fg_transform = parser.parse(transform.get('foreground'))
            self.bg_gt_transform = parser.parse(transform.get('background_with_groundtruth'))
            print('transform get:', transform)

        self.data = []  # (fg, folder)
        self.annotations = {}
        
        folder_names = [d for d in os.listdir(self.fbp_path) if os.path.isdir(os.path.join(self.fbp_path, d))]
        folder_names.sort()
        for folder_name in folder_names[:num_folder]:
            ann_path = os.path.join(self.fbp_path, folder_name, 'prompt/prompt_new.json')
            if not os.path.exists(ann_path):
                continue
            if len(os.listdir(os.path.join(self.fbp_path, folder_name, 'background'))) != len(os.listdir(os.path.join(self.fbp_path, folder_name, 'foreground'))):
                continue
            print('folder_name:', folder_name)
            with open(os.path.join(self.fbp_path, folder_name, 'prompt/prompt_new.json')) as f:
                local_prompt = json.load(f)
                self.annotations.update(local_prompt)
            # fg_folder = os.path.join(self.fbp_path, folder_name, 'foreground')
            
            # fg_imgs = [img for img in os.listdir(fg_folder) if img.endswith('.jpg')]
            fg_imgs = list(local_prompt.keys())
            
            for img_name in fg_imgs:
                self.data.append((img_name, folder_name))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # pdb.set_trace()
        fg_img_name, folder_name = self.data[index]
        bg_img_name = fg_img_name
        mask_name = fg_img_name
        gt_img_name = '_'.join(fg_img_name.split('_')[:-1]) + '.jpg'
        fg = Image.open(os.path.join(self.fbp_path, folder_name, 'foreground', fg_img_name))
        bg = Image.open(os.path.join(self.fbp_path, folder_name, 'background', bg_img_name))
        final_path = os.path.join(self.fbp_path, folder_name, 'groundtruth', bg_img_name)
        if not os.path.exists(final_path):
            try:
                gt = Image.open(os.path.join(self.gt_path, folder_name, gt_img_name))
            except:
                print(f'{folder_name}/{gt_img_name} missing, using bg as gt')
                gt = copy.copy(bg)
        else:
            try:
                gt = Image.open(final_path)
            except:
                print(f'{final_path} missing, using bg as gt')
                gt = copy.copy(bg)
        mask = Image.open(os.path.join(self.fbp_path, folder_name, 'mask', mask_name))
        org_mask = copy.copy(mask) # mask without augmentation
        fg = fg.convert("RGB")
        bg = bg.convert("RGB")
        gt = gt.convert("RGB")
        ann = self.annotations[fg_img_name]
        bg_gt_with_ann = {'image_bg': bg, 'image_gt': gt, 'mask': mask, 'org_mask': org_mask, **ann}
        if self.fg_transform:
            fg = self.fg_transform(fg)
        if self.bg_gt_transform:
            bg_gt_with_ann = self.bg_gt_transform(bg_gt_with_ann)
        
        point = bg_gt_with_ann['point']
        bbox = bg_gt_with_ann['bbox']
        mask = bg_gt_with_ann['mask']
        org_mask = bg_gt_with_ann['org_mask']
        bg = bg_gt_with_ann['image_bg']
        gt = bg_gt_with_ann['image_gt']

        # unconditional signal for CFG
        is_uncond = 0
        zero_fg = torch.zeros_like(fg)
        zero_bg = torch.zeros_like(bg)
        drop_type = random.random()

        # To support classifier-free guidance, randomly drop out only fg conditioning 5%, only bg conditioning 5%, only mask condition 5%, and all 5%.
        if drop_type <= 0.05 and self.mode == 'train':
            # drop only fg
            fg = torch.zeros_like(fg)
            is_uncond = 1
        elif drop_type > 0.05 and drop_type <= 0.10 and self.mode == 'train':
            # drop only bg
            bg = torch.zeros_like(bg)
            is_uncond = 1 
        elif drop_type > 0.10 and drop_type <= 0.15 and self.mode == 'train':
            # drop only mask
            mask = torch.ones_like(mask)
            bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])
            point = torch.tensor([-1.0, -1.0])
            is_uncond = 1
        elif drop_type > 0.15 and drop_type <= 0.20 and self.mode == 'train':
            # drop all
            fg = torch.zeros_like(fg)
            bg = torch.zeros_like(bg)
            mask = torch.ones_like(mask)
            bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])
            point = torch.tensor([-1.0, -1.0])
            is_uncond = 1
        

        if self.off_line:
            fg_feature_name = '.'.join([fg_img_name.split('.')[0], 'pt'])
            fg_feature = torch.load(os.path.join('/n/holyscratch01/pfister_lab/jixuan/dataset/FBP_fg_feature', folder_name, fg_feature_name))
            return {'foreground':fg, 'background':bg, 'groundtruth': gt, 
                'point': point, 'bbox': bbox, 'mask': mask, 'org_mask': org_mask,
                'zero_fg': zero_fg, 'zero_bg': zero_bg, 'fg_feature': fg_feature,
                'uncond_mask': is_uncond}
        
        return {'foreground':fg, 'background':bg, 'groundtruth': gt, 
                'point': point, 'bbox': bbox, 'mask': mask, 'org_mask': org_mask,
                'zero_fg': zero_fg, 'zero_bg': zero_bg,
                'uncond_mask': is_uncond}

class FBPVideoDataset(Dataset):

    def __init__(
            self, 
            fbp_path,
            gt_path,
            num_folder=5,
            mode='train',
            transform=None) -> None:
        
        torch.manual_seed(2023)
        parser = TransformParser()
        self.fbp_path = fbp_path
        self.gt_path = gt_path
        self.num_folder = num_folder
        self.mode = mode
 
        if transform is None:
            self.fg_transform = None
            self.bg_gt_transform = None
        else:
            self.fg_transform = parser.parse(transform.get('foreground'))
            self.bg_gt_transform = parser.parse(transform.get('background_with_groundtruth'))

        self.data = []  # (fg, folder)
        self.annotations = {}
        self.video_name_map = {} # {video: (begin_idx:end_idx)}

        
        folder_names = [d for d in os.listdir(self.fbp_path) if os.path.isdir(os.path.join(self.fbp_path, d))]
        folder_names.sort()
        for folder_name in folder_names[:num_folder]:
            print('folder_name:', folder_name)
            ann_path = os.path.join(self.fbp_path, folder_name, 'prompt/prompt.json')
            if not os.path.exists(ann_path):
                continue
            with open(os.path.join(self.fbp_path, folder_name, 'prompt/prompt.json')) as f:
                local_propmt = json.load(f)
                self.annotations.update(local_propmt)
            
            fg_imgs = list(local_propmt.keys())
            
            begin_idx = len(self.data)
            for img_name in fg_imgs:
                self.data.append((img_name, folder_name))
            end_idx = len(self.data)
            self.video_name_map[folder_name] = (begin_idx, end_idx)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        fg_img_name, folder_name = self.data[index]
        start, end = self.video_name_map[folder_name]
        bg_idx = torch.randint(start, end, (1,))
        bg_img_name, bg_folder = self.data[bg_idx]
        assert bg_folder == folder_name, f"How did you implement this??? fg: {folder_name}, bg: {bg_folder}"
        fg = Image.open(os.path.join(self.fbp_path, folder_name, 'foreground', fg_img_name))
        bg = Image.open(os.path.join(self.fbp_path, folder_name, 'background', bg_img_name))
        gt = Image.open(os.path.join(self.fbp_path, folder_name, 'groundtruth', bg_img_name))
        mask = Image.open(os.path.join(self.fbp_path, folder_name, 'mask', bg_img_name))
        org_mask = copy.copy(mask) # mask without augmentation
        fg = fg.convert("RGB")
        bg = bg.convert("RGB")
        gt = gt.convert("RGB")
        ann = self.annotations[bg_img_name]
        bg_gt_with_ann = {'image_bg': bg, 'image_gt': gt, 'mask': mask, 'org_mask': org_mask, **ann}
        if self.fg_transform:
            fg = self.fg_transform(fg)
        if self.bg_gt_transform:
            bg_gt_with_ann = self.bg_gt_transform(bg_gt_with_ann)

        # range: [-1, 1]

        point = bg_gt_with_ann['point']
        bbox = bg_gt_with_ann['bbox']
        mask = bg_gt_with_ann['mask']
        org_mask = bg_gt_with_ann['org_mask']
        bg = bg_gt_with_ann['image_bg']
        gt = bg_gt_with_ann['image_gt']

        is_uncond = 0
        zero_fg = torch.zeros_like(fg)
        zero_bg = torch.zeros_like(bg)
        drop_type = random.random()

        # To support classifier-free guidance, randomly drop out only fg conditioning 5%, only bg conditioning 5%, only mask condition 5%, and all 5%.
        if drop_type <= 0.05 and self.mode == 'train':
            # drop only fg
            fg = torch.zeros_like(fg)
            is_uncond = 1
        elif drop_type > 0.05 and drop_type <= 0.10 and self.mode == 'train':
            # drop only bg
            bg = torch.zeros_like(bg)
            is_uncond = 1 
        elif drop_type > 0.10 and drop_type <= 0.15 and self.mode == 'train':
            # drop only mask
            mask = torch.ones_like(mask)
            bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])
            point = torch.tensor([-1.0, -1.0])
            is_uncond = 1
        elif drop_type > 0.15 and drop_type <= 0.20 and self.mode == 'train':
            # drop all
            fg = torch.zeros_like(fg)
            bg = torch.zeros_like(bg)
            mask = torch.ones_like(mask)
            bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])
            point = torch.tensor([-1.0, -1.0])
            is_uncond = 1

        return {'foreground':fg, 'background':bg, 'groundtruth': gt, 
                'point': point, 'bbox': bbox, 'mask': mask, 'org_mask': org_mask,
                'zero_fg': zero_fg, 'zero_bg': zero_bg,
                'uncond_mask': is_uncond}

if __name__ == '__main__':
    config = OmegaConf.load('/n/home11/jxhe/insert-any/insert_anything/sd_style/configs/modify_video/modify_dual_input_branch_compress.yaml')
    dataset = instantiate_from_config(config.data.params.train)
    for i in range(10):
        print(dataset[i]['uncond_mask'])