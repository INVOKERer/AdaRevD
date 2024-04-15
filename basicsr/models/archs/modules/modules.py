"""
    Modified from https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/common/sr_backbone_utils.py
"""
import torch.nn as nn
import torch
# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import imp
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
# from basicsr.models.archs.fftformer_arch import TransformerBlock as fftformerblock
def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)



class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= scale


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer)
        self.down_sampling = conv5x5(in_channels, 4 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out

# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out





class UpSampleConvnext(nn.Module):
    def __init__(self, ratio, inchannel, outchannel):
        super().__init__()
        self.ratio = ratio
        self.channel_reschedule = nn.Sequential(
            # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
            nn.Linear(inchannel, outchannel),
            LayerNorm(outchannel, eps=1e-6, data_format="channels_last"))
        self.upsample = nn.Upsample(scale_factor=2 ** ratio, mode='nearest')

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = x = x.permute(0, 3, 1, 2)

        return self.upsample(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                groups=in_channel)  # depthwise conv
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, hidden_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # print(f"x min: {x.min()}, x max: {x.max()}, input min: {input.min()}, input max: {input.max()}, x mean: {x.mean()}, x var: {x.var()}, ratio: {torch.sum(x>8)/x.numel()}")
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2], dim=[112, 72, 40, 24], block_type=None, kernel_size=3) -> None:
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.block_type = block_type
        self._build_decode_layer(dim, depth, kernel_size)
        self.projback = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=3, kernel_size=3, padding=1), # 4 ** 2 * 3
            # nn.PixelShuffle(4),
        )

    def _build_decode_layer(self, dim, depth, kernel_size):
        normal_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        proj_layers = nn.ModuleList()

        norm_layer = LayerNorm

        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            proj_layers.append(nn.Sequential(
                nn.Conv2d(dim[i - 1], dim[i], 1, 1),
                norm_layer(dim[i]),
                nn.GELU()
            ))
        self.normal_layers = normal_layers
        self.upsample_layers = upsample_layers
        self.proj_layers = proj_layers

    def _forward_stage(self, stage, x):
        x = self.proj_layers[stage](x)
        x = self.upsample_layers[stage](x)
        return self.normal_layers[stage](x)

    def forward(self, c3):
        x = self._forward_stage(0, c3)  # 14
        x = self._forward_stage(1, x)  # 28
        x = self._forward_stage(2, x)  # 56
        x = self.projback(x)
        return x
class DecoderX(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2], dim=[112, 72, 40, 24], block_type=None, kernel_size=3) -> None:
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.block_type = block_type
        self._build_decode_layer(dim, depth, kernel_size)
        self.projback = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=3, kernel_size=3, padding=1), # 4 ** 2 * 3
            # nn.PixelShuffle(4),
        )

    def _build_decode_layer(self, dim, depth, kernel_size):
        normal_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        proj_layers = nn.ModuleList()

        norm_layer = LayerNorm

        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            proj_layers.append(nn.Sequential(
                nn.Conv2d(dim[i - 1], dim[i], 1, 1),
                norm_layer(dim[i]),
                nn.GELU()
            ))
        self.normal_layers = normal_layers
        self.upsample_layers = upsample_layers
        self.proj_layers = proj_layers

    def _forward_stage(self, stage, x):
        x = self.proj_layers[stage](x)
        x = self.upsample_layers[stage](x)
        return self.normal_layers[stage](x)

    def forward(self, c0, c1, c2, c3):
        x = self._forward_stage(0, c3)+c2  # 14
        x = self._forward_stage(1, x)+c1  # 28
        x = self._forward_stage(2, x)+c0  # 56
        x = self.projback(x)
        return x


class SimDecoder(nn.Module):
    def __init__(self, in_channel, encoder_stride) -> None:
        super().__init__()
        self.projback = nn.Sequential(
            LayerNorm(in_channel),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, c3):
        return self.projback(c3)