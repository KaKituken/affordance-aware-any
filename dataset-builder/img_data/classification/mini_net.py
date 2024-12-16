import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import transforms

from nncore.engine import load_checkpoint


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Linear(2048, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data, **kwargs):
        x, y = data[0].cuda(), data[1].cuda()

        x = self.backbone(x).softmax(dim=1)

        out = torch.argmax(x, dim=1)
        acc = torch.eq(out, y).sum().float() / x.size(0)

        loss = self.loss(x, y)

        return dict(_avg_factor=x.size(0),
                    _out=dict(x=x[:, 1], y=y),
                    acc=acc,
                    loss=loss)

if __name__ == '__main__':
    model = Model()
    device = torch.device('cuda:0')
    load_checkpoint(model, './epoch_2000.pth')
    model.to(device)
    print(next(model.parameters()).device)