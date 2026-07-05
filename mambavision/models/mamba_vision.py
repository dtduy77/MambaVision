#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from .registry import register_pip_model
from pathlib import Path


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'mamba_vision_T': _cfg(url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_T2': _cfg(url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
                            crop_pct=0.98,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_S': _cfg(url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
                           crop_pct=0.93,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L2': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_L2_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar',
                            crop_pct=0.93,
                            input_size=(3, 512, 512),
                            crop_mode='squash'),
    'mamba_vision_L3_256_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 256, 256),
                            crop_mode='center'),
    'mamba_vision_L3_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar',
                            crop_pct=0.93,
                            input_size=(3, 512, 512),
                            crop_mode='squash'),
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    # Calculate batch size B:
    # - (H * W / window_size / window_size) is the number of windows per image:
    #   * Stage 3: 28*28 / 14 / 14 = 4 windows per image
    #   * Stage 4: 14*14 / 7 / 7 = 4 windows per image
    # - windows.shape[0] is the total number of windows across the batch (e.g. B * 4 = 32 windows)
    # - We divide total windows by windows per image to get B: B = 32 / 4 = 8
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    # Reshape from (num_windows * B, window_size * window_size, C) -> (B, num_windows_h, num_windows_w, window_size, window_size, C)
    #   * Stage 3: (B * 4, 196, 160) -> (B, 2, 2, 14, 14, 160)
    #   * Stage 4: (B * 4, 49, 320)  -> (B, 2, 2, 7, 7, 320)
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)

    # permute(0, 5, 1, 3, 2, 4): rearranges dimensions to (B, C, num_windows_h, window_size_h, num_windows_w, window_size_w)
    #   * Stage 3: (B, 2, 2, 14, 14, 160) -> (B, 160, 2, 14, 2, 14)
    #   * Stage 4: (B, 2, 2, 7, 7, 320)   -> (B, 320, 2, 7, 2, 7)
    # reshape(B, C, H, W): merges (num_windows_h * window_size_h) into H, and (num_windows_w * window_size_w) into W
    #   * Stage 3: (B, 160, 2, 14, 2, 14) -> (B, 160, 28, 28)
    #   * Stage 4: (B, 320, 2, 7, 2, 7)   -> (B, 320, 14, 14)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')


    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim

        # Noted that: kernel_size=3, stride=2, padding=1 -> H, W down by 2, C update by 2 in line 247 if keep_dim=False.
        # This layer is used to reduce the resolution of the feature map and increase the number of channels.
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False), # Noted that: kernel_size=3, stride=2, padding=1 -> H, W down by 2. Ex: for mamba_vision_T: (B, 3, 224, 224) -> (B, 32, 112, 112); for mamba_vision_S: (B, 3, 224, 224) -> (B, 64, 112, 112).
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False), # Noted that: kernel_size=3, stride=2, padding=1 -> H, W down by 2. Ex: for mamba_vision_T: (B, 32, 112, 112) -> (B, 80, 56, 56); for mamba_vision_S: (B, 64, 112, 112) -> (B, 96, 56, 56).
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        # output shape: (B, dim, H/4, W/4)
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1) # Noted that: kernel_size=3, stride=1, padding=1 -> H, W keep the same as input. Ex: for mamba_vision_T in Stage 1: (B, 80, 56, 56) -> (B, 80, 56, 56); in Stage 2: (B, 160, 28, 28) -> (B, 160, 28, 28).
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1) # Noted that: kernel_size=3, stride=1, padding=1 -> H, W keep the same as input. Ex: for mamba_vision_T in Stage 1: (B, 80, 56, 56) -> (B, 80, 56, 56); in Stage 2: (B, 160, 28, 28) -> (B, 160, 28, 28).
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) # Shape: (dim,)
            self.layer_scale = True
        else:
            self.layer_scale = False
        # Layer Scale (Optional) is a technique used to improve the training stability of deep neural networks, especially Transformers.
        # Note: In default configurations, this is only enabled in the hybrid blocks (Block class) of mamba_vision_B and all large variants (L, L2, L3).
        # In ConvBlock, it is always None (disabled) by default.

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # DropPath (Stochastic Depth) randomly drops entire paths/blocks to prevent overfitting and improve training stability.
        # Only active during the training phase, behaves as an Identity layer during inference.
        # Note: The feature tensors of randomly selected images in a batch are set to zero, but the overall shape remains unchanged.


    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        # Residual Connection for output avoiding vanishing/exploding gradient
        # Shape x: (B, C, H, W) the same as input. Ex: for mamba_vision_T: (B, 80, 56, 56) -> (B, 80, 56, 56)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model # Number of input channels
        self.d_state = d_state # SSM state dimension
        self.d_conv = d_conv # Kernel size of the local convolution
        self.expand = expand # Expansion factor for the intermediate dimension
        self.d_inner = int(self.expand * self.d_model) # Intermediate dimension
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # Rank for time-step dependent projection
        self.use_fast_path = use_fast_path # Use fast path for SSM
        self.layer_idx = layer_idx # Index of the layer in the model
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs) # Input projection
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        ) # Receive input x in one branch with shape (B, D/2, L) and project to (B, D/2, dt_rank + d_state*2) to prepare for splitting into dt, B, C

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        # dt_proj is the learnable parameter that controls the time-step dependent projection
        # dt_proj has shape (d_inner//2, dt_rank)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        # Init matrix A with value in matrix start 1-> d_state. Shape (d_inner/2, d_state)
        A_log = torch.log(A)
        # Log of SSM Matrix A.
        self.A_log = nn.Parameter(A_log)
        # Log of SSM Matrix A. Shape (d_inner/2, d_state). For model to learn
        self.A_log._no_weight_decay = True
        # Disable weight decay for A_log
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D) with B is batch size, L is sequence length, D is feature size and it was a Chanel.
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        # Shape (B, L, D) -> (B, L, D). Mean apply linear for both branch SSM and Symetric
        xz = rearrange(xz, "b l d -> b d l")
        # Shape (B, L, D) -> (B, D, L)
        x, z = xz.chunk(2, dim=1)
        # Split into 2 branch with x-SSM branch and z-Symetric branch have both shape: (B, D/2, L)
        A = -torch.exp(self.A_log.float())
        # SSM Matrix A use (-) for get make sure A matrix always negative value because the torch.exp(self.A_log.float()) will be positive value and A matrix must be negative value.
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        # x branch after apply conv1d and silu. Shape: (B, D/2, L)
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        # z branch after apply conv1d and silu. Shape: (B, D/2, L)

        # Start SSM
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        # 1. use rearrange for get x to get shape (B * L, D/2)
        # 2. use x_proj to get shape (B * L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Split x_dbl of shape (B * L, dt_rank + d_state * 2) into:
        # - dt shape: (B * L, dt_rank)
        # - B shape:  (B * L, d_state)
        # - C shape:  (B * L, d_state)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        # t shape: (B, D/2, L)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # B shape: (B, d_state, L)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # C shape: (B, d_state, L)
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        # Concat SSM branch and Symetric branch
        # Shape (B, D/2, L) + (B, D/2, L) -> (B, D, L)
        y = rearrange(y, "b d l -> b l d")
        # Shape (B, D, L) -> (B, L, D)
        out = self.out_proj(y)
        # Linear projection
        # Shape (B, L, D) -> (B, L, D)
        return out


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # mean sqrt(h) where h is head_dimension
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Noted that B means Batchsize, N is number of sequence lenght, C is Chanel or Dimension

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # 1. x shape (B, N, C) -> self.qkv(x) -> (B, N, 3C)
        # 2. (B, N, 3C) -> reshape(B, N, 3, head_number, head_dimension)
        # 3. (B, N, 3, head_number, head_dimension) -> permute(2, 0, 3, 1, 4) -> (3, B, head_number, N, head_dimension)

        q, k, v = qkv.unbind(0)
        # q, k, v has shape (B, head_number, N, head_dimension)

        q, k = self.q_norm(q), self.k_norm(k)
        # Normalize q and k by using layer norm

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p) # Shape: (B, head_number, N, head_dimension)
        else:
            q = q * self.scale # q / sqrt(head_dimension). Shape: (B, head_number, N, head_dimension)
            attn = q @ k.transpose(-2, -1) # BMM: (B, head_number, N, head_dimension) @ (B, head_number, head_dimension, N) -> (B, head_number, N, N)
            attn = attn.softmax(dim=-1) # Softmax over sequence length. Shape: (B, head_number, N, N)
            attn = self.attn_drop(attn) # Apply dropout on attention weights. Shape: (B, head_number, N, N)
            x = attn @ v # BMM: (B, head_number, N, N) @ (B, head_number, N, head_dimension) -> (B, head_number, N, head_dimension)

        x = x.transpose(1, 2).reshape(B, N, C)
        # 1. x.transpose(1, 2) -> (B, N, head_number, head_dimension)
        # 2. reshape(B, N, C) -> (B, N, C)
        x = self.proj(x) # Shape: (B, N, C)
        x = self.proj_drop(x) # Shape: (B, N, C)
        return x

# Block in Step 3, 4 use MambaVision Mixer / Attention + MLP
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Shape (B, L, D) -> (B, L, D)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # Shape (B, L, D) -> (B, L, D)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # mlp_hidden_din means hidden_features of mlp
        # For example, if dim = 320 and mlp_ratio = 4.0, then mlp_hidden_dim = 320 * 4.0 = 1280 you can see in https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # Shape (B, L, D) -> (B, L, D)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        # Shape of gamma_1: (D,)
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        # Shape of gamma_2: (D,)
        # Benefit of layer scale : stability of training.
        # In the early stage of training, the weight of the network is initialized randomly, so the gradient is large.
        # The layer scale can reduce the gradient and prevent the model from diverging.
        # In the later stage of training, the layer scale can increase the gradient and prevent the model from overfitting.
        # In addition, layer scale can help the model to learn long-range dependencies and reduce the model's sensitivity to noise.
        # In summary, layer scale is a useful technique for training deep neural networks.

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        # x: input tensor of shape (B, L, D). L is the number of tokens, D is the feature dimension.
        # self.norm1(x): LayerNorm. Shape (B, L, D) -> (B, L, D)
        # self.mixer(...): Mixer. Shape (B, L, D) -> (B, L, D)
        # self.gamma_1: Scalar. Shape (D,). Multiplies with the output of the mixer.
        # self.drop_path(...): DropPath. Shape (B, L, D) -> (B, L, D)
        # x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x))): Residual connection with layer scaling. Shape (B, L, D)

        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        # self.norm2(x): LayerNorm. Shape (B, L, D) -> (B, L, D)
        # self.mlp(...): MLP. Shape (B, L, D) -> (B, L, D)
        # self.gamma_2: Scalar. Shape (D,). Multiplies with the output of the mlp.
        # self.drop_path(...): DropPath. Shape (B, L, D) -> (B, L, D)
        # x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))): Residual connection with layer scaling. Shape (B, L, D)
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normaization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            # Stage 1 and 2: Create a list of ConvBlocks.
            # - i is the block index generated by Python's list comprehension: for i in range(depth) (e.g. 0 to depth-1).
            # - drop_path[i] retrieves the specific drop path rate allocated for the i-th block in this stage.
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            # Stage 3 and 4: Create a list of hybrid Blocks.
            # - i is the block index generated by: for i in range(depth) (e.g. 0 to depth-1).
            # - counter=i is passed to the Block to identify its position in the stage (to choose Mamba vs Attention).
            #   * Example (Tiny Model):
            #     - Stage 3 (depth=8, transformer_blocks=[4, 5, 6, 7]):
            #       i = 0 to 3 -> counter not in transformer_blocks -> MambaVisionMixer.
            #       i = 4 to 7 -> counter in transformer_blocks -> Attention.
            #     - Stage 4 (depth=4, transformer_blocks=[2, 3]):
            #       i = 0 to 1 -> counter not in transformer_blocks -> MambaVisionMixer.
            #       i = 2 to 3 -> counter in transformer_blocks -> Attention.
            # - drop_path[i] selects the specific drop path rate for this block from the pre-computed schedule list.
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            # Calculate the amount of padding needed to make the width divisible by the window size.
            # Example (Stage 3): W = 20, window_size = 14 -> W % window_size = 6 -> pad_r = (14 - 6) % 14 = 8 columns
            # Example (Stage 4): W = 10, window_size = 7  -> W % window_size = 3 -> pad_r = (7 - 3) % 7 = 4 columns
            # If W is already divisible (e.g. W = 28 for Stage 3 or W = 14 for Stage 4): pad_r = 0 (no padding)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            # Same calculation for height padding

            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                # pad right by pad_r and bottom by pad_b
                _, _, Hp, Wp = x.shape
                # Save padded height and width for reverse window function
                # Example (Stage 3): if H, W = 20, window_size = 14 -> Hp, Wp = 28, 28
                # Example (Stage 4): if H, W = 10, window_size = 7  -> Hp, Wp = 14, 14
            else:
                Hp, Wp = H, W
                # no padding needed
            x = window_partition(x, self.window_size)
            # Partition the padded feature map into non-overlapping windows:
            # - Example (Stage 3): Hp, Wp = 28, 28, ws = 14 -> (28/14)*(28/14) = 4 windows. Shape: (B * 4, 196, C)
            # - Example (Stage 4): Hp, Wp = 14, 14, ws = 7  -> (14/7)*(14/7)   = 4 windows. Shape: (B * 4, 49, C)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
            # Process each block (ConvBlock or Block)
            # ConvBlock: (when self.transformer_block = False)
            # - Input/Output shape: (B, C, H, W) -> e.g. Stage 1: (B, 80, 56, 56)
            # Block: (when self.transformer_block = True)
            # - Input/Output shape: (num_windows * B, window_size * window_size, C)
            #   * Stage 3: (B * 4, 196, 160)
            #   * Stage 4: (B * 4, 49, 320)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
                # Remove the padding back to original size:
                # - Example (Stage 3): Hp, Wp = 28, 28 -> crop back to (B, C, 20, 20)
                # - Example (Stage 4): Hp, Wp = 14, 14 -> crop back to (B, C, 10, 10)
            # Reverse window partition: Reshape from (num_windows*B, window_size*window_size, C) back to (B, C, H, W)
            # * Stage 3: (B * 4, 196, 160) -> (B, 160, 28, 28)
            # * Stage 4: (B * 4, 49, 320)  -> (B, 320, 14, 14)
        if self.downsample is None:
            return x
            # After stage 4, self.downsample is None
            # so return x
        return self.downsample(x) # ALL 3 stages 1, 2, 3

class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        # num_features represents the output dimension of the final stage (Stage 4) before classification.
        # - The initial dimension is dim (e.g. 80 for Tiny).
        # - The number of channels doubles at the end of Stage 1, Stage 2, and Stage 3 (3 times total).
        # - Hence: num_features = dim * 2^(number_of_stages - 1) = 80 * 2^(4-1) = 80 * 8 = 640.
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        # Create patch embedding (stem block) to convert raw RGB images to patch tokens.
        # For Tiny: input shape (B, 3, 224, 224) -> (B, 80, 56, 56)

        # Stochastic Depth Rate Schedule (DropPath):
        # - Shallow blocks extract basic features and should rarely be dropped (rate starts at 0).
        # - Deeper blocks extract abstract semantic features and can be dropped more often to avoid overfitting (up to drop_path_rate).
        # - linspace(0, drop_path_rate, sum(depths)) generates a list of rates increasing linearly across all sum(depths) blocks.
        # - E.g. for Tiny, sum(depths) = 16 blocks. linspace(0, 0.2, 16) yields [0.0, 0.0133, ..., 0.2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            # Stage 1 (i=0) and Stage 2 (i=1) are CNN-based (conv=True)
            # Stage 3 (i=2) and Stage 4 (i=3) are Hybrid Mamba/Attention (conv=False)
            conv = True if (i == 0 or i == 1) else False
            
            # The dimension dim doubles at each subsequent stage: 
            # - i=0: dim=80
            # - i=1: dim=160
            # - i=2: dim=320
            # - i=3: dim=640 (for Tiny)
            # downsample=(i < 3) means downsampling is applied at the end of Stages 1, 2, and 3.
            # transformer_blocks specifies the list of block indices in Hybrid stages that run Attention instead of Mamba.
            # Formula logic:
            # - If D = depths[i] is Even (e.g. D = 8 in Stage 3 of Tiny):
            #   D % 2 != 0 is False -> else branch is chosen: list(range(D // 2, D)) = list(range(4, 8)) = [4, 5, 6, 7]
            #   (Blocks 0-3 run Mamba, blocks 4-7 run Attention)
            # - If D = depths[i] is Odd (e.g. D = 11 in Stage 3 of Tiny2):
            #   D % 2 != 0 is True -> if branch is chosen: list(range(D // 2 + 1, D)) = list(range(6, 11)) = [6, 7, 8, 9, 10]
            #   (Blocks 0-5 run Mamba, blocks 6-10 run Attention)
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
            self.levels.append(level)
            
        # Final normalization layer before classification
        self.norm = nn.BatchNorm2d(num_features)
        
        # Global Average Pooling: pools spatial H x W resolution to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head: projects final num_features channels to num_classes logits
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Custom weights initialization for training stability
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        # 1. Stem Patch Embedding: (B, 3, 224, 224) -> (B, 80, 56, 56)
        x = self.patch_embed(x)
        
        # 2. Sequential Stage processing (Levels 1 to 4):
        # - level 0 (Stage 1): (B, 80, 56, 56) -> (B, 160, 28, 28)
        # - level 1 (Stage 2): (B, 160, 28, 28) -> (B, 320, 14, 14)
        # - level 2 (Stage 3): (B, 320, 14, 14) -> (B, 640, 7, 7)
        # - level 3 (Stage 4): (B, 640, 7, 7)   -> (B, 640, 7, 7)
        for level in self.levels:
            x = level(x)
            
        # 3. Final normalization and pooling:
        # - norm: (B, 640, 7, 7) -> (B, 640, 7, 7)
        # - avgpool: (B, 640, 7, 7) -> (B, 640, 1, 1)
        # - flatten: (B, 640, 1, 1) -> (B, 640)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # Forward pass: Extract features and project to class logits
        # - input x: (B, 3, 224, 224)
        # - forward_features(x): (B, 640)
        # - head: (B, 640) -> (B, 1000)
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def _load_state_dict(self,
                         pretrained,
                         strict: bool = False):
        # Load weights from checkpoint file
        _load_checkpoint(self,
                         pretrained,
                         strict=strict)


@register_pip_model
@register_model
def mamba_vision_T(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
    depths = kwargs.pop("depths", [1, 3, 8, 4])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_T2(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T2.pth.tar")
    depths = kwargs.pop("depths", [1, 3, 11, 4])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T2').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_S(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_S.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 7, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 96)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_S').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_B(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_B_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L2(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 12, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L2_512_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2_512_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 12, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 32, 16])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 512)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2_512_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L3_256_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L3_256_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 20, 10])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 16, 8])
    dim = kwargs.pop("dim", 256)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 256)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L3_256_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model


@register_pip_model
@register_model
def mamba_vision_L3_512_21k(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L3_512_21k.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 20, 10])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 32, 16])
    dim = kwargs.pop("dim", 256)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 512)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L3_512_21k').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model
