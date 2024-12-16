import os
import cv2
import json
import numpy as np
from skimage import measure
from PIL import Image
import torch

class Filter():
    def __init__(self, 
                 shape_filter_list=None, 
                 mask_filter_list=None) -> None:
        self.shape_filter_list = shape_filter_list
        self.mask_filter_list = mask_filter_list

    def apply(self, fg, image, is_mask):
        if not is_mask and self.shape_filter_list is not None:
            for f in self.shape_filter_list:
                if not f.apply(fg, image):
                    return False
        elif is_mask and self.mask_filter_list is not None:
            for f in self.mask_filter_list:
                if not f.apply(fg, image):
                    return False
        return True


#################
# shape related #
#################

class FilterByAbsSize():
    def __init__(self, min_size, max_size) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def apply(self, fg, image):
        h, w = fg.shape[:2]
        return self.min_size < h < self.max_size and\
               self.min_size < w < self.max_size


class FilterByRelSize():
    def __init__(self, min_ratio, max_ratio) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply(self, fg, image):
        f_h, f_w = fg.shape[:2]
        h, w = image.shape[:2]
        h_ratio, w_ratio = f_h / h, f_w / w
        valid = self.min_ratio < h_ratio < self.max_ratio and\
               self.min_ratio < w_ratio < self.max_ratio
        # if not valid:
        #     print('FilterByRelSize works, got h_ratio={}, w_ratio={}, expect min_ratio={}, max_ratio={}'.format(
        #         h_ratio, w_ratio, self.min_ratio, self.max_ratio
        #     ))
        return valid
    
class FilterByShape():
    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def apply(self, fg, image):
        f_h, f_w = fg.shape[:2]
        ratio = max(f_h/f_w, f_w/f_h)
        valid = ratio < self.threshold
        # if not valid:
        #     print('FilterByShape works, got ratio={}, expect threshold={}'.format(
        #         ratio, self.threshold
        #     ))
        return valid
    
class FilterByModel():
    """
    Take a binary classification model. 0 for negative, 1 for positive.
    """
    def __init__(self, model, image_size, transform, threshold) -> None:
        self.model = model
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.device = next(model.parameters()).device

    # BUG: take the according transforms
    @torch.no_grad()
    def apply(self, fg, image):
        # put into a 224 * 224 white background
        h, w = fg.shape[:2]
        if h > w:
            new_h, new_w = self.image_size, int(self.image_size * w / h)
        else:
            new_h, new_w = int(self.image_size * h / w), self.image_size
        white = np.full((self.image_size, self.image_size, 3), fill_value=255, dtype=np.uint8)
        fg = cv2.resize(fg, (new_w, new_h))
        left = (self.image_size - new_w) // 2
        top = (self.image_size - new_h) // 2
        white[top:top+new_h, left:left+new_w] = fg
        fg = Image.fromarray(white)
        fg_tensor = self.transform(fg)
        # print(fg_tensor.shape)
        logits = self.model.backbone(fg_tensor.unsqueeze(0).to(self.device)).softmax(dim=1)
        pos_score = logits[:, 1]    # (B,)
        valid = pos_score.cpu().numpy() > self.threshold
        # if not valid:
        #     print('FilterByModel works')
        return valid

################
# mask related #
################

class FilterByConnect():
    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def apply(self, fg_mask, image):
        labels = measure.label(fg_mask, connectivity=2)
        num = labels.max()
        valid = num < self.threshold
        # if not valid:
        #     print('FilterByConnect works, got connect num={}, expect threshold={}'.format(
        #         num, self.threshold
        #     ))
        return valid
    
class FilterByColor():
    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def apply(self, fg_mask, image):
        """
        Filter by color std
        params:
        - fg_mask [np.array]: 0 for bg, non-0 for fg
        """
        region_pixels = image[fg_mask == 255]
        std_dev = np.std(region_pixels, axis=0)    # HWC
        valid = np.mean(std_dev) > self.threshold
        # valid = np.max(std_dev) > self.threshold
        # if not valid:
        #     print('FilterByColor works, got color std={}, expect threshold={}'.format(
        #         std_dev, self.threshold
        #     ))
        return valid
    
if __name__ == '__main__':
    test_img = cv2.imread('../img_data/SA1B_mini/sa_3.jpg')
    print(test_img.shape)
    std_dev = np.std(test_img, axis=(0,1))
    print(std_dev.shape)
    print(np.mean(std_dev))