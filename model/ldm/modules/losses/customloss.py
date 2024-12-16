import torch
import torch.nn as nn
import torch.nn.functional as F

class BalanceMSELoss(nn.Module):

    def __init__(self, eta):
        super().__init__()
        self.eta = eta

    def forward(self, input, target, boxes, split="train"):
        """
        Compute balanced MSE loss with a foreground box. Foreground and background equally contribute 
        to the loss.

        Args:
            - input (Tensor): the input tensor (predictions of your model).
            - target (Tensor): the target tensor (ground-truth labels).
            - boxes (Tensor): the relative coordinate of the center of box. Using xywh format.
                (B, 4)

        Returns:
            - Tensor: the loss value.
        """
        B, C, H, W = input.shape
        device = input.device

        # meshgrid coordinate
        x = torch.arange(W).view(1, 1, 1, W).expand(B, C, H, W).to(device)  # (B, C, H, W)
        y = torch.arange(H).view(1, 1, H, 1).expand(B, C, H, W).to(device)  # (B, C, H, W)

        # convert to absolute coordinate on feature map
        x1, y1, box_w, box_h = boxes.split(1, dim=1)  # (B, 1)
        x1 = (x1 * W).to(torch.long)
        y1 = (y1 * H).to(torch.long)
        box_w = (box_w * W).to(torch.long)
        box_h = (box_h * H).to(torch.long)

        # broadcast
        inside_x = (x >= x1.unsqueeze(-1).unsqueeze(-1)) & (x <= (x1 + box_w).unsqueeze(-1).unsqueeze(-1))
        inside_y = (y >= y1.unsqueeze(-1).unsqueeze(-1)) & (y <= (y1 + box_h).unsqueeze(-1).unsqueeze(-1))
        inside_box = inside_x & inside_y    # (B, C, H, W)
        
        inside_loss = F.mse_loss(input[inside_box], target[inside_box], reduction='mean')
        outside_loss = F.mse_loss(input[~inside_box], target[~inside_box], reduction='mean')

        log = {
            "{}/inside_loss".format(split): inside_loss.clone().detach(),
            "{}/outside_loss".format(split): outside_loss.clone().detach()
        }

        # Multiply eta for loss consistency. Hyper param

        return (0.5 * inside_loss + 0.5 * outside_loss) * self.eta, log


class MaskBalanceMSELoss(nn.Module):

    def __init__(self, eta):
        super().__init__()
        self.eta = eta

    def forward(self, input, target, masks, split="train"):
        """
        The mask version of balance_mse_loss.

        Args:
            - input (Tensor): the input tensor (predictions of your model).
            - target (Tensor): the target tensor (ground-truth labels).
            - masks (Tensor): the mask of foreground. shape=(B, 1, H, W). 1 for foreground 0 for background.

        Returns:
            - Tensor: the loss value.
        """
        # down sample the mask
        B, C, H, W = input.shape
        device = input.device
        mask_h, mask_w = masks.shape[-2:]
        assert mask_h % H == 0, f'The shape of mask must be multiple of {H}'
        masks = masks.repeat(1, C, 1, 1)
        max_pooling = torch.nn.MaxPool2d(kernel_size=mask_h//H, stride=mask_h//H)
        masks = max_pooling(masks)
        masks = masks.detach()

        # to avoid NaN, we use reduction='sum', then divide len
        input_inside = input[masks>=0.5]
        target_inside = target[masks>=0.5]
        inside_len = max(len(input_inside), 1)
        input_outside = input[masks<0.5]
        target_outside = target[masks<0.5]
        outside_len = max(len(input_outside), 1)
        inside_loss = F.mse_loss(input_inside, target_inside, reduction='sum') / inside_len
        outside_loss = F.mse_loss(input_outside, target_outside, reduction='sum') / outside_len

        log = {
            "{}/inside_loss".format(split): inside_loss.clone().detach(),
            "{}/outside_loss".format(split): outside_loss.clone().detach(),
        }

        return (0.5 * inside_loss + 0.5 * outside_loss) * self.eta, log


class SmoothMSELoss(nn.Module):

    def __init__(self, step, eta=1):
        super().__init__()
        self.step = step
        self.balance_loss = MaskBalanceMSELoss(eta=eta)
        self.mse_loss = F.mse_loss

    def forward(self, input, target, masks, global_step, split="train"):
        """
        The mask version of balance_mse_loss.

        Args:
            - input (Tensor): the input tensor (predictions of your model).
            - target (Tensor): the target tensor (ground-truth labels).
            - masks (Tensor): the mask of foreground. shape=(B, 1, H, W). 1 for foreground 0 for background.

        Returns:
            - Tensor: the loss value.
        """
        mse = self.mse_loss(input, target)
        balance, balance_log = self.balance_loss(input, target, masks, split)

        ratio = max(global_step * 1.0 / self.step, 1)
        loss = mse * (1 - ratio) + balance * ratio

        log = {
            "{}/mse_loss".format(split): mse.clone().detach(),
            "{}/balance_loss".format(split): balance.clone().detach(),
            "{}/smooth_loss".format(split): loss.clone().detach()
        }

        log.update(balance_log)

        return loss, log