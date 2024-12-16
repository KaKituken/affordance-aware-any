import torch
from typing import Any
import numpy as np
from transformers import CLIPProcessor, AutoImageProcessor
import torchvision.transforms.functional as F
from scipy import ndimage
from PIL import Image

import pdb

class CLIPImageProcess():
    def __init__(
            self, 
            path: str,
            return_tensors: str = 'pt',
            padding: bool = True
        ) -> None:
        self.processor = CLIPProcessor.from_pretrained(path)
        self.return_tensors = return_tensors
        self.padding = padding

    def __call__(self, images) -> Any:
        ts = self.processor(images=images, 
                              return_tensors=self.return_tensors, 
                              padding=self.padding)['pixel_values'][0]
        return ts

class DinoImageProcess():
    def __init__(
            self,
            path:str,
            return_tensors: str = 'pt',
            padding: bool = True
        ) -> None:
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.return_tensors = return_tensors
        self.padding = padding

    def __call__(self, images) -> Any:
        ts = self.processor(images=images,
                            return_tensors=self.return_tensors)['pixel_values'][0]
        return ts



class RescaleProcess():
    def __init__(
            self, 
            low, 
            high
        ) -> None:
        self.low = low
        self.high = high

    def __call__(self, images) -> Any:
        """
        [img_low, img_high] -> [low, high]
        """
        img_low = torch.min(images)
        img_high = torch.max(images)
        img_range = img_high - img_low if img_high - img_low > 1e-4 else 1 # might devide 0!
        scale = (self.high - self.low) / img_range   
        return (images - img_low) * scale + self.low
    
class CropToMultiple():
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, images) -> Any:
        if isinstance(images, Image.Image):
            w, h = images.size
        elif isinstance(images, torch.tensor):
            h, w = images.shape[1:]
        else:
            raise BaseException('Invalid image format')
        new_width = w - (w % self.factor)
        new_height = h - (h % self.factor)
        return F.crop(images, 0, 0, new_height, new_width)
    
class RandomResize():
    def __init__(self, factor, isotropic) -> None:
        self.factor = factor
        self.isotropic = isotropic

    def __call__(self, images) -> Any:
        # print("RandomResize called")
        if isinstance(images, Image.Image):
            img_w, img_h = images.size
        else:
            raise BaseException('Invalid image format')
        l, h = self.factor
        if self.isotropic:
            w_factor_sample = l + (h - l) * torch.rand((1,))
            h_factor_sample = w_factor_sample
        else:
            w_factor_sample = l + (h - l) * torch.rand((1,))
            h_factor_sample = l + (h - l) * torch.rand((1,))
        new_width = int(img_w * w_factor_sample)
        new_height = int(img_h * h_factor_sample)
        # print('current size:', (new_width, new_height))
        return F.resize(images, (new_width, new_height))

class RandomCutout():
    def __init__(self, cut_factor):
        self.cut_factor = cut_factor

    def __call__(self, images):
        # print("RandomCutout called")
        if isinstance(images, Image.Image):
            img_w, img_h = images.size
        else:
            raise BaseException('Invalid image format')
        # print('image size:', (img_w, img_h))
        mask = np.ones((img_h, img_w), np.float32)
        y = np.random.randint(img_h)
        x = np.random.randint(img_w)
        l, h = self.cut_factor
        cut_factor_sample = l + (h - l) * np.random.rand()
        # print('cut_factor_sample:', cut_factor_sample)
        cut_length = int(min(img_w, img_h) * cut_factor_sample)
        # print('cut_length:', cut_length)

        y1 = np.clip(y - cut_length // 2, 0, img_h)
        y2 = np.clip(y + cut_length // 2, 0, img_h)
        x1 = np.clip(x - cut_length // 2, 0, img_w)
        x2 = np.clip(x + cut_length // 2, 0, img_w)

        # print('(y1, y2, x1, x2):', (y1, y2, x1, x2))
        mask[y1: y2, x1: x2] = 0
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask, 'L') 
        images = Image.composite(images, Image.new('RGB', images.size, (255, 255, 255)), mask)
        return images
    
class CropOrPad():
    def __init__(self, size):
        self.size = size

    def __call__(self, images) -> Any:
        # print("CropOrPad called")
        if isinstance(images, Image.Image):
            w, h = images.size
        else:
            raise BaseException('Invalid image format')
        target_w, target_h = self.size
        max_x = max(0, w - target_w)
        max_y = max(0, h - target_h)
        x = np.random.randint(max_x+1)
        y = np.random.randint(max_y+1)
        cropped_img = F.crop(images, 
                             top=y, 
                             left=x, 
                             height=min(h, target_h),
                             width=min(w, target_w))
        padded_img = F.pad(cropped_img, 
                           padding=[0, 0, max(0, target_w-w), max(0, target_h-h)], 
                           fill=255)
        return padded_img
    
# TODO: reconstruct former functions
####################
# Annotation Related
####################
class AlignAnn():
    def __init__(self) -> None:
        pass

    def __call__(self, image_with_ann):
        bbox = image_with_ann['bbox']
        point = image_with_ann['point']
        meta = image_with_ann['meta']
        cur_w, cur_h = image_with_ann['image'].size
        org_w = meta['width']
        ratio = cur_w / org_w
        point = [corr * ratio for corr in point]
        bbox = [corr * ratio for corr in bbox]
        image_with_ann['bbox'] = bbox
        image_with_ann['point'] = point
        return image_with_ann


class ResizeWithAnn():
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, image_with_ann):
        # ! The size of bg and gt might be different. But it's fine, cuz bbox and point are corresponding to gt
        # !! why the meta is not correct
        bg = image_with_ann['image_bg']
        gt = image_with_ann['image_gt']
        point = image_with_ann['point']
        bbox = image_with_ann['bbox']
        mask = image_with_ann['mask']
        org_mask = image_with_ann['org_mask']
        meta = image_with_ann['meta']
        org_w, org_h = meta['width'], meta['height']
        bg = F.resize(bg, self.size)
        gt = F.resize(gt, self.size)
        mask = F.resize(mask, self.size)
        org_mask = F.resize(org_mask, self.size)
        ratio = self.size / min(org_w, org_h)
        point = [int(corr * ratio) for corr in point]
        bbox = [int(corr * ratio) for corr in bbox]
        # check here
        box_x, box_y, box_w, box_h = bbox
        box_x_center = box_x + box_w // 2
        box_y_center = box_y + box_h // 2
        assert box_x_center < bg.width, \
        f"Impossible! \n\
        img_size: {bg.size}\n\
        bbox: {bbox}\n\
        image_with_ann['bbox']: {image_with_ann['bbox']}\n\
        org_w, org_h: {meta['width'], meta['height']}\n\
        image_id: {meta['image_id']}"
        image_with_ann['image_bg'] = bg
        image_with_ann['image_gt'] = gt
        image_with_ann['org_mask'] = org_mask
        image_with_ann['mask'] = mask
        image_with_ann['point'] = point
        image_with_ann['bbox'] = bbox
        return image_with_ann
    
class CropWithAnn():
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, image_with_ann):
        bg = image_with_ann['image_bg']
        gt = image_with_ann['image_gt']
        org_mask = image_with_ann['org_mask']
        mask = image_with_ann['mask']
        point = image_with_ann['point']
        bbox = image_with_ann['bbox']       # xywh
        img_w, img_h = bg.size
        assert img_w >= self.size and img_h >= self.size, 'Cannot crop to a larger size!'
        
        box_x, box_y, box_w, box_h = bbox
        box_x_center = box_x + box_w // 2
        box_y_center = box_y + box_h // 2
        assert max(0, box_x_center - self.size) < min(box_x_center, img_w - self.size)+1, \
        f"Impossible! The weird case is \n\
            bbox: {bbox}\n\
            box_x_center: {box_x_center}\n\
            self.size: {self.size}\n\
            img_w: {img_w}\n\
            img_h: {img_h}\n"
        crop_x = np.random.randint(max(0, box_x_center - self.size), min(box_x_center, img_w - self.size)+1)
        crop_y = np.random.randint(max(0, box_y_center - self.size), min(box_y_center, img_h - self.size)+1)
        bg = F.crop(bg, top=crop_y, left=crop_x, height=self.size, width=self.size)
        gt = F.crop(gt, top=crop_y, left=crop_x, height=self.size, width=self.size)
        mask = F.crop(mask, top=crop_y, left=crop_x, height=self.size, width=self.size)
        org_mask = F.crop(org_mask, top=crop_y, left=crop_x, height=self.size, width=self.size)
        box_w = min(crop_x + self.size, box_x + box_w) - max(box_x, crop_x)

        bbox[0] = max(0, box_x - crop_x)
        bbox[2] = box_w
        point[0] -= crop_x
        box_h = min(crop_y + self.size, box_y + box_h) - max(box_y, crop_y)
        bbox[1] = max(0, box_y - crop_y)
        bbox[3] = box_h
        point[1] -= crop_y
        # TODO: check valid here
        image_with_ann['image_bg'] = bg
        image_with_ann['image_gt'] = gt
        image_with_ann['org_mask'] = org_mask
        image_with_ann['mask'] = mask
        image_with_ann['point'] = point
        image_with_ann['bbox'] = bbox
        return image_with_ann
    
class CropToMultipleWithAnn():
    def __init__(self, factor) -> None:
        self.crop_to_mul = CropToMultiple(factor)

    def __call__(self, image_with_ann):
        bg = image_with_ann['image_bg']
        gt = image_with_ann['image_gt']
        mask = image_with_ann['mask']
        org_mask = image_with_ann['org_mask']
        bg = self.crop_to_mul(bg)
        gt = self.crop_to_mul(gt)
        mask = self.crop_to_mul(mask)
        org_mask = self.crop_to_mul(org_mask)
        image_with_ann['mask'] = mask
        image_with_ann['org_mask'] = org_mask
        image_with_ann['image_bg'] = bg
        image_with_ann['image_gt'] = gt
        return image_with_ann
    
class JitterPoint():
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, image_with_ann):
        point = image_with_ann['point']
        img_w, img_h = image_with_ann['image_bg'].size
        offset_x = np.random.randint(-int(self.factor*img_w), int(self.factor*img_w)+1)
        offset_y = np.random.randint(-int(self.factor*img_h), int(self.factor*img_h)+1)
        point[0] += offset_x
        point[1] += offset_y
        point[0] = np.clip(point[0], 0, img_w)
        point[1] = np.clip(point[1], 0, img_h)
        image_with_ann['point'] = point
        return image_with_ann

class CenterEnlargeBbox():
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, image_with_ann):
        bg_w, bg_h = image_with_ann['image_bg'].size
        bbox = image_with_ann['bbox']
        box_x, box_y, box_w, box_h = bbox
        box_center_x = box_x + box_w // 2
        box_center_y = box_y + box_h // 2
        min_f, max_f = self.factor
        f = np.random.uniform(min_f, max_f)
        w = max(int(box_w * f), 2)
        h = max(int(box_h * f), 2)
        x1 = box_center_x - w // 2
        y1 = box_center_y - h // 2
        x2 = box_center_x + w // 2
        y2 = box_center_y + h // 2
        # check bound
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bg_w, x2)
        y2 = min(bg_h, y2)
        w = x2 - x1
        h = y2 - y1
        assert w > 0 and h > 0,\
        f"Impossible to get null bbox!\n\
        meta: {image_with_ann['meta']}\b\
        bbox: {bbox}"
        image_with_ann['bbox'] = [x1, y1, w, h]
        return image_with_ann

class EnlargeMask():
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, image_with_ann):
        mask = image_with_ann['mask']
        # pdb.set_trace()
        # img_w, img_h = image_with_ann['image_bg'].size
        min_f, max_f = self.factor
        f = np.random.uniform(min_f, max_f)

        box_x, box_y, box_w, box_h = image_with_ann['bbox']
        box_center_x = box_x + box_w // 2
        box_center_y = box_y + box_h // 2

        cropped_mask = mask.crop([box_x, box_y, box_x+box_w, box_y+box_h])
        
        width, height = cropped_mask.size
        enlarged_cropped_mask = cropped_mask.resize((int(width * f), int(height * f)), Image.BICUBIC)
        
        new_width, new_height = enlarged_cropped_mask.size
        paste_x, paste_y = box_center_x - new_width // 2, box_center_y - new_height // 2

        mask.paste(enlarged_cropped_mask, (paste_x, paste_y))
        image_with_ann['mask'] = mask
        return image_with_ann

class FeatherMask():
    def __init__(self, sigma) -> None:
        self.sigma = sigma

    def __call__(self, image_with_ann):
        image_with_ann['mask'] = ndimage.gaussian_filter(image_with_ann['mask'], sigma=self.sigma)
        return image_with_ann
    
class ToTensorWithAnn():
    def __init__(self) -> None:
        pass

    def __call__(self, image_with_ann):
        bg = image_with_ann['image_bg']
        gt = image_with_ann['image_gt']
        mask = image_with_ann['mask']
        org_mask = image_with_ann['org_mask']
        point = np.array(image_with_ann['point'], dtype=np.float32)
        bbox = np.array(image_with_ann['bbox'], dtype=np.float32)
        w, h = bg.size
        bg = F.to_tensor(bg)
        gt = F.to_tensor(gt)
        mask = F.to_tensor(mask)
        org_mask = F.to_tensor(org_mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            org_mask = org_mask.unsqueeze(0)
        assert mask.shape[0] == 1 and org_mask.shape[0] == 1, 'Mask must have only one channel!'
        point[0] /= w
        point[1] /= h
        for i in range(0, 4, 2):
            bbox[i] /= w
        for i in range(1, 4, 2):
            bbox[i] /= h
        image_with_ann['image_bg'] = bg
        image_with_ann['image_gt'] = gt
        image_with_ann['org_mask'] = org_mask
        image_with_ann['mask'] = mask
        image_with_ann['point'] = torch.tensor(point)
        image_with_ann['bbox'] = torch.tensor(bbox)
        return image_with_ann
    
class RescaleWithAnn():
    def __init__(self, low, high) -> None:
        self.rescale = RescaleProcess(low, high)

    def __call__(self, image_with_ann):
        bg = image_with_ann['image_bg']
        gt = image_with_ann['image_gt']
        bg = self.rescale(bg)
        gt = self.rescale(gt)
        image_with_ann['image_bg'] = bg
        image_with_ann['image_gt'] = gt
        return image_with_ann

class SamplePoint():
    def __init__(self) -> None:
        pass

    def __call__(self, image_with_ann):
        # pdb.set_trace()
        meta = image_with_ann['meta']
        org_w, org_h = meta['width'], meta['height']
        org_mask = F.resize(image_with_ann['org_mask'], [org_h, org_w])
        non_zero_indices = torch.nonzero(torch.from_numpy(np.array(org_mask)))
        if non_zero_indices.nelement() == 0:
            return
        point = non_zero_indices[torch.randint(len(non_zero_indices), (1,))][0]
        image_with_ann['point'] = point
        return image_with_ann

class JitterBox():
    def __init__(self, std, max_shift) -> None:
        self.std = std
        self.max_shift = max_shift

    def __call__(self, image_with_ann):
        bbox = image_with_ann['bbox']
        meta = image_with_ann['meta']
        org_w, org_h = meta['width'], meta['height']
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        x1 += torch.clip(torch.normal(0, self.std * w, size=(1,)),
                         -self.max_shift,
                         self.max_shift)
        x2 += torch.clip(torch.normal(0, self.std * w, size=(1,)),
                         -self.max_shift,
                         self.max_shift)
        y1 += torch.clip(torch.normal(0, self.std * h, size=(1,)),
                         -self.max_shift,
                         self.max_shift)
        y2 += torch.clip(torch.normal(0, self.std * h, size=(1,)),
                         -self.max_shift,
                         self.max_shift)
        if x1 >= x2:
            x1 = max(0, x1 - w // 2)
            x2 = min(x2 + w // 2, org_w)
        if y1 >= y2:
            y1 = max(0, y1 - h // 2)
            y2 = min(y2 + h // 2, org_h)
        image_with_ann['bbox'] = [x1, y1, x2-x1, y2-y1]
        return image_with_ann
        

if __name__ == '__main__':
    jitter_box = JitterBox(0.1, 20)
    gray = torch.ones((1, 128, 128))
    img = {'meta': {'width': 128, 'height': 128}, 'org_mask': gray, 'bbox': [10, 26, 50, 60]}
    img = jitter_box(img)
    print(img)