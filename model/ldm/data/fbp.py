import torch
from torch.utils.data import Dataset
from ldm.data.data_fbp.FBPDatsets import FBPImageDataset, FBPVideoDataset
from omegaconf import OmegaConf
from main import instantiate_from_config

class ImageVideoWrapper(Dataset):
    def __init__(self, 
                 image_cfg, 
                 video_cfg, 
                 ratio=0.5):
        """
        arguments:
            - ratio [float]: The probability to sample from image dataset
        """
        self.image_dataset = FBPImageDataset(**image_cfg)
        self.video_dataset = FBPVideoDataset(**video_cfg)
        self.ratio = ratio
        self.len1 = len(self.image_dataset)
        self.len2 = len(self.video_dataset)

    def __getitem__(self, index):
        if torch.rand(1).item() < self.ratio:  
            # sample from dataset1 according to ratio
            return self.image_dataset[index % self.len1]
        else:  
            # else, sample from dataset2
            return self.video_dataset[index % self.len2]

    def __len__(self):
        return self.len1 + self.len2

if __name__ == '__main__':
    config = OmegaConf.load('/n/home11/jxhe/insert-any/insert_anything/sd_style/configs/modify_video/modify_dual_input_branch_compress.yaml')
    dataset = instantiate_from_config(config.data.params.train)
    for i in range(10):
        print(dataset[i]['uncond_mask'])