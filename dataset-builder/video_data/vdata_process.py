"""
This file is used to process vdata of YTB-VOS.
Read the .jpg mask annotation file, crop the foreground and inpaint the background.
Perform filter here.
Save the result into save-path.
Structure is /save-path/
                |_ video_name/
                    |_ background
                    |_ foreground
                    |_ groundtruth (hard to retrieve, so save them using soft link)
                    |_ prompt

p.s. In the annotation .jpg file, pixel value represents the index of the mask. e.g. mask == 1 will get the first mask
"""

import os
import cv2
import json
import argparse
import numpy as np
from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config
import shutil
from PIL import Image
import pdb
from tqdm import tqdm
import copy

global_index = 0

def parse_argments():
    parser = argparse.ArgumentParser(description='Input process arguments')

    parser.add_argument('-a', '--annotation-path', type=str, help='Path of mask annotations')
    parser.add_argument('-i', '--image-path', type=str, help='Path of src images')
    parser.add_argument('-s', '--save-path', type=str, help='Path to save FBP')
    parser.add_argument('--start-index', type=int, help='index to start')
    args = parser.parse_args()
    
    return args


def enlarge_mask(mask, enlarge_val):
    # create a (2*enlarge_val+1, 2*enlarge_val+1) kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*enlarge_val+1, 2*enlarge_val+1))
    # enlarge mask
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
    print(fg.shape)
    white = np.full((target_size, target_size, 3), fill_value=255, dtype=np.uint8)
    fg = cv2.resize(fg, (new_w, new_h))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    white[top:top+new_h, left:left+new_w] = fg
    print(white.shape)
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
    return enlarge_mask(bi_mask, 10)


def inpaint_by_mask(img_path, mask_path, inpaint_model, video_name):
    # load image
    global global_index
    image_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = copy.copy(image)
    h, w = image.shape[:2]
    org_mask = np.array(Image.open(mask_path))
    instance_num = org_mask.max()
    if instance_num == 0:
        return
    res = {}
    for ins in range(1, instance_num+1):
        mask = copy.copy(org_mask)
        if not np.any(mask==ins):
            continue
        mask[mask!=ins] = 0
        mask[mask==ins] = 255
        fg, fg_mask, bbox = get_fg(image, mask)
        # filter objects that are too small
        fg = process_fg(fg, target_size=224)

        bg_mask = get_bgmask(mask)
        
        # resize the long edge to 720
        tar_size = 720
        if max(h, w) == h:  
            new_h, new_w = tar_size, int(w * tar_size / h)
        else:
            new_h, new_w = int(h * tar_size / w), tar_size
        bg_mask = cv2.resize(bg_mask, (new_w, new_h))

        inpainted_bg = inpaint_model(cv2.resize(image, (new_w, new_h)), 
                                    bg_mask, 
                                    Config(hd_strategy="Original", ldm_steps=20,
                                            hd_strategy_crop_margin=128,
                                            hd_strategy_crop_trigger_size=800,
                                            hd_strategy_resize_limit=800))
        print(fg.shape)
        fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
        meta = {"image_id": global_index, "width": w, "height": h, "file_name": video_name + '_' + str(ins) + '_' + image_name}
        global_index += 1
        res[ins] = (gt, fg, inpainted_bg, bbox, fg_mask, meta)
    return res
    


def main():
    args = parse_argments()
    ann_path = args.annotation_path
    img_path = args.image_path
    save_path = args.save_path
    start_index = args.start_index

    print('Loading LaMa...')
    inpaint_model = LaMa('cuda:0')  # LAMA: fg 255, bg 0
    # inpaint_model = None
    videos = os.listdir(img_path)
    videos.sort()

    print('Start process...')
    for video_name in tqdm(videos[start_index:start_index+200]):
        video_ann = os.path.join(ann_path, video_name)
        mask_frames = os.listdir(video_ann)
        res_to_save = {}
        for mask_frame in mask_frames:
            print('processing new frame')
            image_frame = mask_frame.rstrip('png') + 'jpg'
            image_frame_path = os.path.join(img_path, video_name, image_frame)
            mask_frame_path = os.path.join(video_ann, mask_frame)
            res = inpaint_by_mask(image_frame_path, mask_frame_path, inpaint_model,
                                  video_name) # {1: (fg, bg, bbox, fg_mask, meta), 2: (fg, bg, bbox, fg_mask, meta)}
            if res is None:
                continue
            for ins in res:
                if ins in res_to_save:
                    res_to_save[ins].append(res[ins])
                else:
                    res_to_save[ins] = [res[ins]]

        for ins in res_to_save:
            # save_video_name = str(save_video_idx).zfill(6)
            pos_prompt = {}
            save_video_name = video_name
            foreground_path = os.path.join(save_path, f'{save_video_name}_{ins}', 'foreground')
            background_path = os.path.join(save_path, f'{save_video_name}_{ins}', 'background')
            groundtruth_path = os.path.join(save_path, f'{save_video_name}_{ins}', 'groundtruth')
            mask_path = os.path.join(save_path, f'{save_video_name}_{ins}', 'mask')
            prompt_path = os.path.join(save_path, f'{save_video_name}_{ins}', 'prompt')
            os.makedirs(foreground_path, exist_ok=True)
            os.makedirs(background_path, exist_ok=True)
            os.makedirs(groundtruth_path, exist_ok=True)
            os.makedirs(mask_path, exist_ok=True)
            os.makedirs(prompt_path, exist_ok=True)

            for gt, fg, bg, bbox, fg_mask, meta in res_to_save[ins]:
                save_name = meta["file_name"]
                cv2.imwrite(os.path.join(foreground_path, save_name), fg)
                cv2.imwrite(os.path.join(background_path, save_name), bg)
                cv2.imwrite(os.path.join(mask_path, save_name), fg_mask)
                cv2.imwrite(os.path.join(groundtruth_path, save_name), gt)
                pos_prompt[save_name] = {'bbox': bbox,
                                        'point': [bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2],
                                        'meta': meta}
                with open(os.path.join(prompt_path, 'prompt.json'), 'w') as f:
                    json.dump(pos_prompt, f)



if __name__ == "__main__":
    main()