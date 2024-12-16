"""
Call Lama to inpaint background & get foreground for sa1b annotation format.
Perform filter logic here for better foreground quality.
Read the `.json` mask annotation file, and inpaint the background.
Input: Image + mask
Output: /save-path
        |_ background
        |_ foreground
        |_ mask (mask for foreground)
        |_ groundtruth
        |_ prompt
"""

import os
import cv2
import json
import torch
import numpy as np
from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config
from pycocotools import mask as coco_mask
import sys
from classification.mini_net import Model
from torchvision import transforms

from nncore.engine import load_checkpoint

sys.path.append('..')
import utils.filters as filters

import argparse
from tqdm import tqdm

def parse_argments():
    parser = argparse.ArgumentParser(description='Input process arguments')

    parser.add_argument('-a', '--annotation-path', type=str, help='Path of mask annotations')
    parser.add_argument('-i', '--image-path', type=str, help='Path of images')
    parser.add_argument('-s', '--save-path', type=str, help='Path to save FBP')
    parser.add_argument('-m', '--model-path', type=str, help='The path of the classification model')
    args = parser.parse_args()
    
    return args



def enlarge_mask(mask, enlarge_val):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*enlarge_val+1, 2*enlarge_val+1))
    enlarged_mask = cv2.dilate(mask, kernel)
    return enlarged_mask

def get_bbox_size(mask):
    rows, cols = np.where(mask != 0)[:2]

    if len(rows) == 0:
        row_min, row_max = 0, mask.shape[0]
    else:
        row_min, row_max = rows.min(), rows.max()
    if len(cols) == 0:
        col_min, col_max = 0, mask.shape[1]
    else:
        col_min, col_max = cols.min(), cols.max()
    w = col_max - col_min
    h = row_max - row_min

    return col_min, row_min, w, h

def apply_gaussian_blur(image, sigma=1):
    size = int(2*round(3*sigma)+1)
    blurred_image = cv2.GaussianBlur(image, (size, size), sigma)
    return blurred_image

def adjust_contrast(image, gamma=1.0):
    # In OpenCV, adjust contrast can be done using alpha and beta parameters. 
    # Where alpha controls the contrast and beta controls the brightness.
    contrast = gamma
    brightness = 0
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image

def process_fg(fg, target_size):
    h, w = fg.shape[:2]
    if h > w:
        new_h, new_w = target_size, int(target_size * w / h)
    else:
        new_h, new_w = int(target_size * h / w), target_size
    # print(fg.shape)
    white = np.full((target_size, target_size, 3), fill_value=255, dtype=np.uint8)
    fg = cv2.resize(fg, (new_w, new_h))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    white[top:top+new_h, left:left+new_w] = fg
    # print(white.shape)
    return white

def get_fg(image, bi_mask):
    fg_mask = np.array(bi_mask)
    fg = np.array(image)
    fg[bi_mask == 0] = 255
    fg_mask = apply_gaussian_blur(fg_mask)
    fg_mask = adjust_contrast(fg_mask)
    x, y ,w, h = get_bbox_size(fg_mask)
    fg = fg[y:y+h, x:x+w]
    fg_mask = fg_mask.astype(np.uint8)
    return fg, fg_mask, [int(x), int(y), int(w), int(h)]

def get_bgmask(bi_mask):
    return enlarge_mask(bi_mask, 20)


def inpaint_by_ann(img_path, anns, inpaint_model, cus_filter):
    # load image
    image_name = anns['image']['file_name']
    print(image_name)
    image = cv2.imread(os.path.join(img_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fg_list = []
    bg_list = []
    gt_list = []
    bbox_list = []
    fg_mask_list = []

    ng_fg_list = []
    
    count = 0
    for ann in anns['annotations']:
        seg = ann['segmentation']
        
        count += 1
        bi_mask = coco_mask.decode(seg) # fg 255, bg 0
        bi_mask[bi_mask==1] = 255
        fg, fg_mask, bbox = get_fg(image, bi_mask)

        if not cus_filter.apply(fg, image, is_mask=False) or \
            not cus_filter.apply(bi_mask, image, is_mask=True):
            fg = process_fg(fg, target_size=224)
            fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
            ng_fg_list.append(fg)
            continue

        fg = process_fg(fg, target_size=224)
        bg_mask = get_bgmask(bi_mask)
        h, w = bg_mask.shape[:2]

        # resize the long edge to 720
        tar_size = 720
        if max(h, w) == h:  
            new_h, new_w = tar_size, int(w * tar_size / h)
        else:
            new_h, new_w = int(h * tar_size / w), tar_size
        bg_mask = cv2.resize(bg_mask, (new_w, new_h))

        inpainted_bg = inpaint_model(cv2.resize(image, (new_w, new_h)), 
                                     bg_mask, 
                                     Config(hd_strategy="Original", 
                                            ldm_steps=20,
                                            hd_strategy_crop_margin=128,
                                            hd_strategy_crop_trigger_size=800,
                                            hd_strategy_resize_limit=800))
        fg_list.append(cv2.cvtColor(fg, cv2.COLOR_RGB2BGR))
        bg_list.append(inpainted_bg)
        fg_mask_list.append(fg_mask)
        gt_list.append(image)
        bbox_list.append(bbox)
        if count > 14:
            break
    return fg_list, bg_list, bbox_list, fg_mask_list, ng_fg_list, gt_list


def main():
    args = parse_argments()
    ann_path = args.annotation_path
    img_path = args.image_path
    save_path = args.save_path
    model_path = args.model_path
    print('save_path:', save_path)
    foreground_path = os.path.join(save_path, 'foreground')
    background_path = os.path.join(save_path, 'background')
    groundtruth_path = os.path.join(save_path, 'groundtruth')
    prompt_path = os.path.join(save_path, 'prompt')
    mask_path = os.path.join(save_path, 'mask')
    os.makedirs(foreground_path, exist_ok=True)
    os.makedirs(background_path, exist_ok=True)
    os.makedirs(prompt_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(groundtruth_path, exist_ok=True)

    ann_list = []
    for file in os.listdir(ann_path):
        if file[-4:] != 'json':
            continue
        if not os.path.exists(os.path.join(img_path, file[:-4]+'jpg')):
            # filter the ones without image accordingly
            continue
        with open(os.path.join(ann_path, file)) as f:
            ann_json = json.load(f)
        ann_list.append(ann_json)

    inpaint_model = LaMa('cuda:0')  # LAMA: fg 255, bg 0
    model = Model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    load_checkpoint(model, model_path)
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    shape_filter_list = [
        # filters.FilterByAbsSize(64, 400),
        filters.FilterByRelSize(0.25, 0.75),
        filters.FilterByShape(3),
        filters.FilterByModel(model, image_size=224, transform=transform, threshold=0.7)
    ]
    mask_filter_list = [
        filters.FilterByConnect(4),
        filters.FilterByColor(45)
    ]
    cus_filter = filters.Filter(shape_filter_list,
                                mask_filter_list)
    
    pos_prompt = {}
    # fg saved for binary classifier
    valid_fg = []
    invalid_fg = []
    for fig_idx, ann in enumerate(tqdm(ann_list)):
        fg_list, bg_list, bbox_list, fg_mask_list, ng_fg_list, gt_list = \
            inpaint_by_ann(img_path, ann, inpaint_model, cus_filter)
        image_name = ann['image']['file_name'][:-4]
        print('saving fig ', fig_idx)
        for idx, (fg, bg, gt, fg_mask, bbox) in \
                enumerate(zip(fg_list, bg_list, gt_list, fg_mask_list, bbox_list)):
            save_name = f'{image_name}_{idx}.jpg'
            cv2.imwrite(os.path.join(foreground_path, save_name), fg)
            cv2.imwrite(os.path.join(background_path, save_name), bg)
            cv2.imwrite(os.path.join(mask_path, save_name), fg_mask)
            cv2.imwrite(os.path.join(groundtruth_path, save_name), gt)
            pos_prompt[save_name] = {'bbox': bbox, 
                                     'point': [bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2],
                                     'meta': ann['image']}
        valid_fg += fg_list
        invalid_fg += ng_fg_list
    with open(os.path.join(prompt_path, 'prompt.json'), 'w') as f:
        json.dump(pos_prompt, f)


if __name__ == "__main__":
    main()