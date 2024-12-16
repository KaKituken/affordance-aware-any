import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from .vis_utils import show_anns

sys.path.append("../..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
    

def load_sam(sam_checkpoint):
    model_type = "vit_h"

    device = torch.device("cuda:0")

    print('registering...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('to cuda...')
    sam.to(device=device)
    return sam


def set_sam_mask_generator(sam):
    print('setting mask generator...')
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def set_sam_predictor(sam):
    print('setting predictor...')
    predictor = SamPredictor(sam)
    return predictor

def get_resize_transform(sam):
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    return resize_transform


if __name__ == '__main__':
    sam_checkpoint = "../../segment-anything/sam_vit_h_4b8939.pth"
    mask_generator = set_sam_mask_generator(sam_checkpoint)
    image = cv2.imread('./coco2017/selected_images/000000000632.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('mask generating...')
    masks = mask_generator.generate(image)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('./tmp/mask.jpg')
