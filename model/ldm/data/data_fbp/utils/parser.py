import torch
from torchvision import transforms
from typing import Dict, List, Optional

from ldm.data.data_fbp.utils.custom_transforms import CLIPImageProcess, RescaleProcess, CropToMultiple, \
    RandomResize, RandomCutout, CropOrPad, ResizeWithAnn, CropWithAnn, \
    CropToMultipleWithAnn, ToTensorWithAnn, RescaleWithAnn, AlignAnn, \
    DinoImageProcess, JitterPoint, CenterEnlargeBbox, EnlargeMask, FeatherMask, \
    SamplePoint, JitterBox

class TransformParser():
    def __init__(self) -> None:
        pass

    def process_ts(self, ts_type: str, params: Dict):
        if ts_type == 'Resize':
            size = params.get('size')
            fix_size = params.get('fix_size')
            if fix_size is not None:
                op = transforms.Resize(fix_size)
            elif size is not None:
                op = transforms.Resize(size)
            else:
                raise BaseException('missing arguments')
        elif ts_type == 'RandomResize':
            factor = params.get('factor')
            isotropic = params.get('isotropic')
            prob = params.get('prob', 1)
            assert factor is not None, 'missing arguments factor'
            assert isotropic is not None, 'missing arguments isotropic'
            op = transforms.RandomApply([RandomResize(factor, isotropic)], p=prob)
        elif ts_type == 'RandomHorizonFlip':
            prob = params.get('prob', 1)
            op = transforms.RandomHorizontalFlip(p=prob)
        elif ts_type == 'CenterCrop':
            size = params.get('size')
            assert size is not None, 'missing arguments'
            op = transforms.CenterCrop(size)
        elif ts_type == 'RandomRotation':
            angle = params.get('angle')
            assert angle is not None, 'missing arguments'
            prob = params.get('prob', 1)
            op = transforms.RandomApply([transforms.RandomRotation(angle, fill=255)], p=prob)
        elif ts_type == 'RandomCutout':
            cut_factor = params.get('cut_factor')
            assert cut_factor is not None, 'missing arguments'
            prob = params.get('prob', 1)
            op = transforms.RandomApply([RandomCutout(cut_factor)], p=prob)
        elif ts_type == 'CropOrPad':
            fix_size = params.get('fix_size')
            assert fix_size is not None, 'missing arguments fix_size'
            op = CropOrPad(fix_size)
        elif ts_type == 'ToTensor':
            assert len(params) == 0, 'illeagal arguments'
            op = transforms.ToTensor()
        elif ts_type == 'Brightness':
            brightness = params.get('brightness')
            assert brightness is not None, "missing arguments 'brightness'"
            prob = params.get('prob', 1)
            op = transforms.RandomApply([transforms.ColorJitter(brightness=brightness)], p=prob)
        elif ts_type == 'Contrast':
            contrast = params.get('contrast')
            assert contrast is not None, "missing arguments 'contrast'"
            prob = params.get('prob', 1)
            op = transforms.RandomApply([transforms.ColorJitter(contrast=contrast)], p=prob)
        elif ts_type == 'Saturation':
            saturation = params.get('saturation')
            assert saturation is not None, "missing arguments 'saturation'"
            prob = params.get('prob', 1)
            op = transforms.RandomApply([transforms.ColorJitter(saturation=saturation)], p=prob)
        elif ts_type == 'GaussianBlur':
            kernel_size = params.get('kernel_size')
            assert kernel_size is not None, "missing arguments 'kernel_size'"
            prob = params.get('prob', 1)
            op = transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=prob)
        elif ts_type == 'AddNoise':
            noise = params.get('noise')
            assert noise is not None, "missing arguments 'noise'"
            prob = params.get('prob', 1)
            op = transforms.RandomApply([transforms.Lambda(lambda x: (1-noise)*x+noise*torch.randn_like(x))], p=prob)
        elif ts_type == 'CLIPImage':
            path = params.get('path')
            assert path is not None, 'missing arguments'
            return_tensors = params.get('return_tensors', 'pt')
            padding = params.get('padding', True)
            op = CLIPImageProcess(path, return_tensors, padding)
        elif ts_type == 'DinoImage':
            path = params.get('path')
            assert path is not None, 'missing arguments'
            return_tensors = params.get('return_tensors', 'pt')
            padding = params.get('padding', True)
            op = DinoImageProcess(path, return_tensors, padding)
        elif ts_type == 'Rescale':
            scale_range = params.get('scale_range')
            assert scale_range is not None, 'missing arguments'
            l, h = scale_range
            op = RescaleProcess(l, h)
        elif ts_type == 'CropToMultiple':
            factor = params.get('factor')
            assert factor is not None, 'missing arguments'
            op = CropToMultiple(factor)
        elif ts_type == 'AlignAnn':
            assert len(params) == 0, 'illeagal arguments'
            op = AlignAnn()
        elif ts_type == 'ResizeWithAnn':
            size = params.get('size')
            fix_size = params.get('fix_size')
            if fix_size is not None:
                op = ResizeWithAnn(fix_size)
            elif size is not None:
                op = ResizeWithAnn(size)
            else:
                raise BaseException('missing arguments')
        elif ts_type == 'CropWithAnn':
            size = params.get('size')
            assert size is not None, 'missing arguments'
            op = CropWithAnn(size)
        elif ts_type == 'CropToMultipleWithAnn':
            factor = params.get('factor')
            assert factor is not None, 'missing arguments'
            op = CropToMultipleWithAnn(factor)
        elif ts_type == 'JitterPoint':
            factor = params.get('factor')
            assert factor is not None, 'missing arguments'
            op = JitterPoint(factor)
        elif ts_type == 'CenterEnlargeBbox':
            factor = params.get('factor')
            assert factor is not None, 'missing arguments'
            op = CenterEnlargeBbox(factor)
        elif ts_type == 'FeatherMask':
            print('params:', params)
            sigma = params.get('sigma')
            assert sigma is not None, 'missing arguments'
            op = FeatherMask(sigma)
        elif ts_type == 'EnlargeMask':
            factor = params.get('factor')
            assert factor is not None, 'missing arguments'
            op = EnlargeMask(factor)
        elif ts_type == 'ToTensorWithAnn':
            op = ToTensorWithAnn()
        elif ts_type == 'RescaleWithAnn':
            scale_range = params.get('scale_range')
            assert scale_range is not None, 'missing arguments'
            l, h = scale_range
            op = RescaleWithAnn(l, h)
        elif ts_type == 'SamplePoint':
            op = SamplePoint()
        elif ts_type == 'JitterBox':
            std = params.get('std')
            max_shift = params.get('max_shift')
            op = JitterBox(std, max_shift)
        else:
            raise NotImplementedError(f'{ts_type} not emplement yet')
        return op


    def parse(self, tfs: Optional[List[Dict]]):
        if not tfs or len(tfs) == 0:
            return None
        pipeline = []
        for tf in tfs:
            assert len(tf) == 1, "illeagal transforms input!"
            ts_type = list(tf.keys())[0]
            params = tf[ts_type]
            pipeline.append(self.process_ts(ts_type, params))
        return transforms.Compose(pipeline)
