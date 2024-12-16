from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint
import pdb


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 vis_attn_map=False # whether return the attention map. For visualization.
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.vis_attn_map = vis_attn_map

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)    # b n d
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # (b h) nq nk

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class GatedAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 vis_attn_map=False # whether return the attention map. For visualization.
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_ = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.vis_attn_map = vis_attn_map

    def forward(self, x, context=None, mask=None):
        # pdb.set_trace()
        h = self.heads

        q = self.to_q(x)    # b n d
        context = default(context, x)
        k = self.to_k(context)
        v_cross = self.to_v(context)
        v_self = self.to_v_(x).detach()  # TODO: whether to detach?

        gated_token_len = q.shape[1]
        # concat querries and keys
        q = torch.concat([q, k], dim=1)
        k = q.clone()
        v = torch.concat([v_self, v_cross], dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # (b h) nq nk

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        gated_out = out[:, :gated_token_len, ...]
        gated_out = rearrange(gated_out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(gated_out)


class ResidualAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 vis_attn_map=False # whether return the attention map. For visualization.
                 ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_ = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v_ = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.vis_attn_map = vis_attn_map

    def forward(self, x, context=None, mask=None):
        # pdb.set_trace()
        h = self.heads

        q = self.to_q(x)    # b n d
        pooled_x = torch.mean(x, dim=1, keepdim=True).detach()   # cannot detach, for we need to train self.to_v_ & self.to_k_
        # pooled_x = torch.mean(x, dim=1, keepdim=True)

        context = default(context, x)
        k_cross = self.to_k(context)
        v_cross = self.to_v(context)
        k_bg = self.to_k_(pooled_x)
        v_bg = self.to_v_(pooled_x)
        
        # concat querries and keys
        k = torch.concat([k_cross, k_bg], dim=1)
        v = torch.concat([v_cross, v_bg], dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # (b h) nq nk

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class HierarchicalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 vis_attn_map=False, # whether return the attention map. For visualization.
                 keep_bg_token=True
                 ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_ = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v_ = nn.Linear(query_dim, inner_dim, bias=False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.vis_attn_map = vis_attn_map
        self.keep_bg_token = keep_bg_token

    def forward(self, x, context=None, mask=None):
        # pdb.set_trace()
        fg_cls_token = context[:, :1, ...].clone().detach()
        h = self.heads

        q = self.to_q(x)    # b n d
        pooled_x = torch.mean(x, dim=1, keepdim=True).detach()

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        bg_k = self.to_k_(pooled_x)     # (b 1 d)
        fg_k = self.to_k(fg_cls_token)  # (b 1 d)
        bg_v = self.to_v_(pooled_x)
        fg_v = self.to_v(fg_cls_token)

        first_hie_k = torch.concat([fg_k, bg_k], dim=1)
        first_hie_v = torch.concat([fg_v, bg_v], dim=1)

        q, k, v, first_hie_k, first_hie_v, bg_v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), 
                                                (q, k, v, first_hie_k, first_hie_v, bg_v))
        
        bg_fg_sim = einsum('b i d, b j d -> b i j', q, first_hie_k) * self.scale # b n d, b 2 d -> b n 2
        bg_fg_attn = bg_fg_sim.softmax(dim=-1)  # b n 2. [fg, bg]

        # second hierarchy attention
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # (b h) nq nk

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)  # b n d

        # first hierarchy attention
        query_cond_on_fg = out * bg_fg_attn[...,:1] # (b n d) * (b n 1)
        if self.keep_bg_token:
            query_cond_on_bg = einsum('b i j, b j d -> b i d', bg_fg_attn[...,-1:], bg_v)
            out = query_cond_on_fg + query_cond_on_bg
        else:
            out = query_cond_on_fg
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MaskAwareAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 vis_attn_map=False, # whether return the attention map. For visualization.
                 ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(query_dim, 1, bias=False),
            nn.Tanh()   # try different activation
        )
        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.vis_attn_map = vis_attn_map

    def forward(self, x, context=None, mask=None):
        # x.shape: b (hgt wid) d
        h = self.heads

        q = self.to_q(x)    # b n d
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # (b h) nq nk

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)  # (b h) nq nk

        out = einsum('b i j, b j d -> b i d', attn, v)  # (b h) nq d
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # add gate here
        gate = self.to_gate(x)  # b n 1, range(-1,1) if use tanh
        return self.to_out(out) * gate


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., 
                 context_dim=None, gated_ff=True, checkpoint=True, # !! has been changed
                 vis_attn_map=False, attn_type='cross', keep_bg_token=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn_type = attn_type
        if attn_type == 'cross':
            self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=dropout,
                                        vis_attn_map=vis_attn_map)  # is self-attn if context is none
        elif attn_type == 'hierachy':
            self.attn2 = HierarchicalAttention(query_dim=dim, context_dim=context_dim,
                                               heads=n_heads, dim_head=d_head, dropout=dropout,
                                               vis_attn_map=vis_attn_map, keep_bg_token=keep_bg_token)
        elif attn_type == 'gated':
            self.attn2 = GatedAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=dropout,
                                        vis_attn_map=vis_attn_map)
        elif attn_type == 'residual':
            self.attn2 = ResidualAttention(query_dim=dim, context_dim=context_dim,
                                           heads=n_heads, dim_head=d_head, dropout=dropout,
                                           vis_attn_map=vis_attn_map)
        elif attn_type == 'mask_aware':
            self.attn2 = MaskAwareAttention(query_dim=dim, context_dim=context_dim,
                                           heads=n_heads, dim_head=d_head, dropout=dropout,
                                           vis_attn_map=vis_attn_map)
        else:
            raise NotImplementedError(f'Unsupported attention type {attn_type}')
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.vis_attn_map = vis_attn_map

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # pdb.set_trace()
        x = self.attn1(self.norm1(x)) + x           # self attention
        x = self.attn2(self.norm2(x), context) + x  # cross attention
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 vis_attn_map=False, attn_type='cross', keep_bg_token=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   vis_attn_map=vis_attn_map, attn_type=attn_type, keep_bg_token=keep_bg_token)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            # pdb.set_trace()
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in