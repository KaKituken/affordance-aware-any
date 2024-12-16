import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPVisionModel
from transformers import Dinov2Model

from PIL import Image
import numpy as np
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from typing import Tuple
import torch.nn.functional as F
from ldm.modules.attention import CrossAttention
import pdb

prompt_mean = 0.129
prompt_std = 0.332
rgb_mean = 0.0108
rgb_std = 0.768


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)

# TODO: handel the range of pixel value
class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False,
                 augment:bool = False,
                 normalize:bool = False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area','nearest-exact']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        self.augment = augment
        self.normalize = normalize
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def augment_mask(self, mask_list):
        scale_range=(0.8, 1.2)
        shift_range=(-30, 30)
        bs, _, h, w = mask_list.size()

        # random choose scaler
        scale = torch.rand(bs) * (scale_range[1] - scale_range[0]) + scale_range[0]

        # random choose shifter
        shift_x = (torch.rand(bs) * (shift_range[1] - shift_range[0]) + shift_range[0]).int()
        shift_y = (torch.rand(bs) * (shift_range[1] - shift_range[0]) + shift_range[0]).int()

        augmented_masks = []

        for i in range(bs):
            # use affine grid to augment
            theta = torch.tensor([
                [scale[i], 0, shift_x[i]/w],
                [0, scale[i], shift_y[i]/h]
            ], dtype=torch.float).unsqueeze(0)
            grid = F.affine_grid(theta, size=(1, 1, h, w))
            grid = grid.to(mask_list.device)
            new_mask = F.grid_sample(mask_list[i].unsqueeze(0), grid)
            augmented_masks.append(new_mask)

        return torch.cat(augmented_masks, dim=0)

    def forward(self,x):
        if self.augment:
            x = self.augment_mask(x)
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        pos = self(x)
        # print('before norm range:', pos.min(), pos.max())
        if self.normalize:
            # pos -= prompt_mean
            # pos /= prompt_std
            # pos *= rgb_std
            # pos += rgb_mean
            pos *= 2
            pos -= 1
            pos *= rgb_std
        # print('after norm range:', pos.min(), pos.max())
        return pos

    def decode(self, x):
        if self.normalize:
            # x -= rgb_mean
            # x *= rgb_std
            # x /= prompt_std
            # x += prompt_mean
            x /= rgb_std
            x += 1
            x /= 2
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=1/self.multiplier)
        return x
    
class Coor2Area(nn.Module):
    def __init__(
            self,
            target_size: Tuple[int],
            augment:bool = False,
            normalize:bool = False) -> None:
        super().__init__()
        self.hidden_size = (256, 256)   # for scale to [0,1]
        self.target_size = target_size
        self.normalize = normalize
        if target_size[0] == 32:
            n_stages = 3
        elif target_size[0] == 16:
            n_stages = 4
        elif target_size[0] == 8:
            n_stages = 5
        elif target_size[0] == 64:
            n_stages = 2
        else:
            raise NotImplementedError("target_size must be 64 or 32 or 16 or 8")
        self.augment = augment

        self.spatial_rescaler = SpatialRescaler(n_stages=n_stages, method='bilinear', augment=False)

    def forward(self, boxes):
        """
        Params:
        - bboxes (torch.Tensor): The coordinater of box vertex. Using xywh format.
        
        Returns:
        - area (torch.Tensor): 1 for in the bbox, 0 for out of the bbox.
        """
        device = boxes.device
        if self.augment:
            jitter = torch.rand_like(boxes, device=device) * 0.2 - 0.1  # [0, 1] -> [-0.1, 0.1]
            boxes = boxes + jitter
        boxes = torch.clip(boxes, 0, 1)
        x = torch.arange(0, self.hidden_size[0], dtype=torch.float32) # width dimension
        y = torch.arange(0, self.hidden_size[1], dtype=torch.float32) # height dimension
        y, x = torch.meshgrid(y, x)

        x = x[None, ...]        # (1, H, W)
        y = y[None, ...]        # (1, H, W)
        x = x.to(device)
        y = y.to(device)

        W, H = self.hidden_size

        x0 = boxes[:, 0:1].unsqueeze(-1).unsqueeze(-1) * W      # (B, 1, 1, 1)
        y0 = boxes[:, 1:2].unsqueeze(-1).unsqueeze(-1) * H      # (B, 1, 1, 1)
        box_w = boxes[:, 2:3].unsqueeze(-1).unsqueeze(-1) * W   # (B, 1, 1, 1)
        box_h = boxes[:, 3:4].unsqueeze(-1).unsqueeze(-1) * H   # (B, 1, 1, 1)

        x = x.unsqueeze(0)      # (1, 1, H, W)
        y = y.unsqueeze(0)      # (1, 1, H, W)

        mask_x = (x >= x0) & (x <= x0 + box_w)
        mask_y = (y >= y0) & (y <= y0 + box_h)

        box_mask = torch.zeros_like(mask_x, dtype=torch.long, device=device, requires_grad=False)
        box_mask[mask_x & mask_y] = 1
        return self.spatial_rescaler(box_mask.float())         # (B, 1, H, W)
    
    def encode(self, x):
        pos = self(x)
        # print('before norm range:', pos.min(), pos.max())
        if self.normalize:
            # pos -= prompt_mean
            # pos /= prompt_std
            # pos *= rgb_std
            # pos += rgb_mean
            pos *= 2
            pos -= 1
            pos *= rgb_std
        # print('after norm range:', pos.min(), pos.max())
        return pos
    
    def decode(self, x):
        """
        Params:
        - x: bbox embedding. (B, 1, H, W)

        Return: 
        - bboxes (torch.Tensor): The coordinater of box vertex. Using xywh format. (B, 4)
        """
        B = x.shape[0]
        W, H = self.target_size
        bboxes = torch.zeros((B, 4))
        for idx, tensor_2d in enumerate(x.squeeze(1)):
            ones_y, ones_x = torch.where(tensor_2d == 1)
            
            if len(ones_y) == 0:  # Handle the case where there are no ones in the tensor.
                continue

            bboxes[idx][0] = ones_x.min()
            bboxes[idx][1] = ones_y.min()
            bboxes[idx][2] = ones_x.max()
            bboxes[idx][3] = ones_y.max()

        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        bboxes = bboxes.float()
        bboxes[:, 0] /= W
        bboxes[:, 2] /= W
        bboxes[:, 1] /= H
        bboxes[:, 3] /= H
        return bboxes


class GuassHeatmapper(nn.Module):
    def __init__(
            self,
            target_size: Tuple[int],    # (w, h)
            sigma: float,
            augment:bool = False,
            normalize:bool = False
            ) -> None:
        super().__init__()
        self.target_size = target_size
        self.normalize = normalize
        if isinstance(sigma, int) or isinstance(sigma, float):
            self.sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=False)
        else:
            self.sigma = sigma.detach()
        self.augment = augment

    @staticmethod
    def gaussian_2d(x, y, x0, y0, sigma, augment):
        """
        x0: (B,)
        y0: (B,)
        """
        if not augment:
            x0 = x0.unsqueeze(-1).unsqueeze(-1)
            y0 = y0.unsqueeze(-1).unsqueeze(-1)
            sigma = sigma.unsqueeze(-1).unsqueeze(-1).to(x0.device)
        else:
            # TODO: implement augment here
            B = x0.shape[0]
            disturb_sigma = torch.tensor(4)
            x_disturb = torch.randn((B,)) * torch.sqrt(disturb_sigma)
            y_disturb = torch.randn((B,)) * torch.sqrt(disturb_sigma)
            x_disturb = x_disturb.to(x0.device)
            y_disturb = y_disturb.to(y0.device)
            x0 = (x0 + x_disturb).unsqueeze(-1).unsqueeze(-1).clip(0)
            y0 = (y0 + y_disturb).unsqueeze(-1).unsqueeze(-1).clip(0)
        return torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    @ torch.no_grad()
    def generate_heatmap(self, center):
        """
        Generate a 2D Gaussian heatmap using PyTorch.

        Params:
        - center (torch.Tensor): the center of the Gaussian distribution [[x,y],[x,y]], relative coordinate.
        
        Returns:
        - heatmap (torch.Tensor): the generated heatmap with shape (1, h, w)
        """
        device = center.device
        x = torch.arange(0, self.target_size[0], dtype=torch.float32) # width dimension
        y = torch.arange(0, self.target_size[1], dtype=torch.float32) # height dimension
        y, x = torch.meshgrid(y, x)

        x = x[None, ...]
        y = y[None, ...]
        x = x.to(device)
        y = y.to(device)
        # print('x.device', x.device)

        W, H = self.target_size
        heatmaps = GuassHeatmapper.gaussian_2d(x, y, 
                                               center[:, 0]*W, center[:, 1]*H, 
                                               self.sigma, self.augment)
        
        return heatmaps.unsqueeze(1).to(device).detach()    # B 1 H W
    
    def forward(self, points):
        """
        Use Heatmap embedding.
        Params:
        - points [list(np.array)]: points to encode. [[x1, y1], [x2, y2], ...]  B * 2
        """
        if isinstance(points, list):
            points = np.stack(points)
            device = next(self.parameters()).device
            points = torch.tensor(points, dtype=torch.float).to(device)     # (B, 2)

        # 1. Generate heatmaps for all points
        point_embedding = self.generate_heatmap(points)       # (B, 1, H, W)

        # 2. Check which coordinates in the 'points' are negative
        negative_mask = (points < 0).any(dim=1)

        # 3. Replace the heatmap for those points with a white heatmap
        white_heatmap = torch.ones((1, self.target_size[1], self.target_size[0])).to(points.device)
        point_embedding[negative_mask] = white_heatmap

        return point_embedding
    
    def encode(self, x):
        pos = self(x)
        # print('before norm range:', pos.min(), pos.max())
        if self.normalize:
            # pos -= prompt_mean
            # pos /= prompt_std
            # pos *= rgb_std
            # pos += rgb_mean
            pos *= 2
            pos -= 1
            pos *= rgb_std
        # print('after norm range:', pos.min(), pos.max())
        return pos

    def decode(self, x):
        max_indices = torch.argmax(x.view(x.size(0), -1), dim=1)
        max_coords = torch.stack((torch.div(max_indices, x.size(3), rounding_mode='trunc'), 
                                  max_indices % x.size(3)), dim=1)
        return max_coords

class BlankGenerator(nn.Module):
    """Generate blank image, fill it with 1"""
    def __init__(self,
                 target_size):
        super().__init__()
        self.target_size = target_size 

    def forward(self, bs):
        device = bs.device
        B = bs
        W, H = self.target_size
        return torch.ones((B, 1, H, W), requires_grad=False).float().to(device)
    
    def encode(self, bs):
        return self(bs)
    
    def decode(self, x):
        return torch.tensor(x.shape[0])


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenHFCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for image (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        # self.feat_extract = CLIPFeatureExtractor.from_pretrained(version)
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        # batch_encoding = self.feat_extract(img, return_tensors="pt")
        # tokens = batch_encoding["pixel_values"].to(self.device)
        outputs = self.transformer(pixel_values=img)

        z = outputs.last_hidden_state
        return z.contiguous()

    def encode(self, img):
        return self(img)

class FrozenHFCLIPImageEmbedderSmall(AbstractEncoder):
    """Uses the CLIP transformer encoder for image (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-base-patch16", device="cuda"):
        super().__init__()
        # self.feat_extract = CLIPFeatureExtractor.from_pretrained(version)
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        # batch_encoding = self.feat_extract(img, return_tensors="pt")
        # tokens = batch_encoding["pixel_values"].to(self.device)
        outputs = self.transformer(pixel_values=img)

        z = outputs.last_hidden_state
        return z.contiguous()

    def encode(self, img):
        return self(img)

class FrozenHFDinoV2ImageEmbedder(AbstractEncoder):
    """Uses the DinoV2 transformer encoder for image (from Hugging Face)"""
    def __init__(self, version="facebook/dinov2-large", device="cuda", multi_scale=False, num_layer=None, 
                 compress=False, token_nums=None, compress_method='group'):
        super().__init__()
        # self.feat_extract = CLIPFeatureExtractor.from_pretrained(version)
        self.transformer = Dinov2Model.from_pretrained(version) # output shape: [B, N, C]
        self.device = device
        self.multi_scale = multi_scale
        self.compress = compress

        # self.test_linear = nn.Linear(in_features=self.transformer.config.hidden_size, out_features=self.transformer.config.hidden_size)
        
        if self.multi_scale:
            # add a Linear Layer to project
            if num_layer is None:
                self.num_layer = len(self.transformer.config.stage_names)
            else:
                self.num_layer = num_layer
            hidden_dim = self.transformer.config.hidden_size
            self.linear_proj = nn.Sequential(
                nn.Linear(in_features=self.num_layer*hidden_dim, out_features=hidden_dim),
                nn.LayerNorm(normalized_shape=hidden_dim)
            )
        if self.compress:
            if token_nums is None:
                # only use cls token
                self.token_nums = 0
            else:
                assert 256 % token_nums == 0, 'token number must be factor of 256!'
                self.token_nums = token_nums
            self.compress_method = compress_method
            if compress_method == 'pool':
                hidden_dim = self.transformer.config.hidden_size
                self.compress_module = nn.AdaptiveAvgPool1d(output_size=token_nums)
            elif compress_method == 'conv':
                hidden_dim = self.transformer.config.hidden_size
                self.compress_module = nn.Sequential(
                    nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=256//token_nums, stride=256//token_nums),
                    nn.ReLU(),
                    # nn.BatchNorm1d(hidden_dim)
                )
            elif compress_method == 'cross-attn':
                hidden_dim = self.transformer.config.hidden_size
                self.patch_query_tokens = nn.Parameter(torch.randn(token_nums, hidden_dim))
                self.compress_module = CrossAttention(query_dim=hidden_dim,
                                                      context_dim=hidden_dim,
                                                      dropout=0.2)
            else:
                raise NotImplementedError("compress method doesn't implemented")
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        # if self.multi_scale:
        #     for param in self.linear_proj.parameters():
        #         param.requires_grad = True
        if self.compress:
            for param in self.compress_module.parameters():
                param.requires_grad = True  
            if hasattr(self, 'patch_query_tokens'):
                self.patch_query_tokens.requires_grad = True

    def forward(self, img):
        if not self.multi_scale:
            outputs = self.transformer(pixel_values=img)
            z = outputs.last_hidden_state
        else:
            outputs = self.transformer(pixel_values=img, output_hidden_states=True)
            z = outputs.hidden_states[-self.num_layer:]
            z = torch.concat(z, dim=-1)
            z = self.linear_proj(z)
        if self.compress:
            cls_token = z[:,:1,...]
            if self.token_nums == 0:
                z = cls_token
            else:
                if self.compress_method == 'pool':
                    patch_tokens = rearrange(z[:,1:,...], 'b n d -> b d n')  # (b, d, 256)
                    patch_tokens = self.compress_module(patch_tokens)   # (b, 256, num_tokens)
                    patch_tokens = rearrange(patch_tokens, 'b d n -> b n d')  # (b, num_tokens, 256)
                elif self.compress_method == 'cross-attn':
                    bs = z.shape[0]
                    patch_tokens = z[:,1:,...]
                    query = torch.stack([self.patch_query_tokens] * bs)
                    patch_tokens = self.compress_module(query, patch_tokens)
                    print('self.patch_query_tokens[0,0]:', self.patch_query_tokens[0,0])
                    print('self.patch_query_tokens[0,0].grad:', self.patch_query_tokens[0,0].grad)
                elif self.compress_method == 'conv':
                    patch_tokens = rearrange(z[:,1:,...], 'b n d -> b d n')  # (b, d, 256)
                    patch_tokens = self.compress_module(patch_tokens)   # (b, 256, num_tokens)
                    patch_tokens = rearrange(patch_tokens, 'b d n -> b n d')  # (b, num_tokens, 256)
                z = torch.cat([cls_token, patch_tokens], dim=1)
            # z = self.test_linear(z)
        return z.contiguous()
    
    def encode(self, img):
        return self(img)

if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
