import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange
from torch.nn.init import xavier_uniform_, constant_
from basicsr.utils import get_root_logger
# from basicsr.models.archs.dct_util import *
from basicsr.models.archs.win_util import *
from basicsr.models.archs.norm_util import *
# from basicsr.models.archs.attn_util import *
from torch.nn.parameter import Parameter
import numpy as np

from basicsr.utils import get_root_logger

# try:
#     from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     # print('Cannot import dcn. Ignore this warning if dcn is not used. '
#     #       'Otherwise install BasicSR with compiling dcn.')
#

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)
def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))

def norm_(x):
    import numpy as np
    return (1 - np.exp(-x)) / (1 + np.exp(-x))
class channel_shuffle(nn.Module):
    def __init__(self, groups=4):
        super(channel_shuffle, self).__init__()
        self.groups=groups
    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
               channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x
def Seg():
    dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                 9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                 17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                 25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                 33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                 41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                 49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                 57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i+32], 0, 0] = 1

    return a


class PAM(nn.Module):

    def __init__(self, in_dim):

        super(PAM, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv( torch.cat( (rgb,freq),1) )
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:,0:1,:,:] * rgb * self.v_rgb
        freq = attmap[:,1:,:,:] * freq * self.v_freq
        out = rgb + freq
        return out


# DWT DCT FFT
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, relu_method=nn.ReLU, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))

        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class BasicConv_Phase(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, relu_method=nn.ReLU, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv_Phase, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                PhaseConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))

        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class conv_bench(nn.Module):
    def __init__(self, n_feat, kernel_size=3, act_method=nn.ReLU, bias=False):
        super(conv_bench, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        self.act = act_method()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))
class res_phaseconv_bench(nn.Module):
    def __init__(self, n_feat, kernel_size=3, act_method=nn.ReLU):
        super().__init__()
        # self.norm = LayerNorm2d_mean_var(n_feat)
        self.norm = LayerNorm2d(n_feat)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.conv2 = PhaseConv(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False, groups=n_feat)
        self.act = act_method(inplace=True) #nn.Softshrink(lambd=0.1) #
        # self.conv2 = PhaseConv(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
    def forward(self, x):
        # x_norm, mean, var = self.norm(x)
        # print(var.max(), mean.max())
        return x + self.conv2(self.act(self.conv1(self.norm(x)))) # * var # )

class Window_Local:
    def __init__(self, overlap_size=0, window_size=(8,8), qkv=True):
        self.overlap_size = (overlap_size, overlap_size)
        self.window_size = window_size
        self.original_size = None
        self.qkv = qkv
    def grids(self, x):
        b, c, h, w = x.shape
        if self.qkv:
            self.original_size = (b, c // 3, h, w)
        else:
            self.original_size = (b, c, h, w)
        x = x.unsqueeze(0)
        # assert b == 1
        k1, k2 = self.window_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size# (64, 64)
        stride = (k1-overlap_size[0], k2-overlap_size[1])
        num_row = (h - 1) // stride[0] + 1
        num_col = (w - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, : , :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts
    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.window_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[:, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :, :]
            count_mt[:, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
class fourier_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False):
        super(fourier_complex_conv, self).__init__()
        # torch 1.12+
        self.act_fft = act_method()

        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, 1, bias=bias, dtype=torch.complex64)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, 1, bias=bias, dtype=torch.complex64)
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        H, W = x.shape[-2:]
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1

        y = self.complex_conv1(y)
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = self.complex_conv2(y)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fourier_real_conv3x3(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False):
        super(fourier_real_conv3x3, self).__init__()
        # torch 1.12+
        self.act_fft = act_method()

        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, 3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, 3, padding=1, bias=bias)
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        H, W = x.shape[-2:]
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 0
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.complex_conv1(y)
        y = self.act_fft(y)
        y = self.complex_conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fourier_conv_in_spatial(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, kernel_size=1, bias=True):
        super(fourier_conv_in_spatial, self).__init__()
        # torch 1.12+
        self.act_fft = act_method()

        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_conv1 = spatial_fourier_fastconv(dim, hid_dim, kernel_size=kernel_size, bias=bias)
        self.complex_conv2 = spatial_fourier_fastconv(hid_dim, dim, kernel_size=kernel_size, bias=bias)
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.complex_conv1(x)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.complex_conv2(y)
        return y
class FFTAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='spatial', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.norm = LayerNorm(dim)
        # self.norm_k = LayerNorm(dim)
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.project_out.weight.data, 0.)
        constant_(self.project_out.bias.data, 0.)
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):
        # qkv = self.qkv(self.norm(x))
        qkv = self.qkv_dwconv(self.qkv(self.norm(x)))
        # qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out) + x
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
class spatial_block_in_fourier(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, kernel_size=1, bias=True, num_heads=1, fft=False):
        super(spatial_block_in_fourier, self).__init__()
        # torch 1.12+
        self.act_fft = act_method()

        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.conv1 = fourier_spatial_fastconv(dim, hid_dim, kernel_size=kernel_size, bias=bias)
        self.conv2 = fourier_spatial_fastconv(hid_dim, dim, kernel_size=kernel_size, bias=bias)
        self.norm = norm
        # self.norm1 = LayerNorm2d(hid_dim * 2)
        self.attn = FFTAttention(hid_dim, num_heads=num_heads, bias=bias)
        # self.min = inf
        self.fft = fft
    def forward(self, x):
        if self.fft:
            x = torch.fft.rfft2(x, norm=self.norm)
        H, W = x.shape[-2:]
        x = self.conv1(x)

        dim = 0
        y = torch.cat([x.real, x.imag], dim=dim)

        y = self.attn(y)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        y = self.conv2(y)
        if self.fft:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y

def fourier_kernel_to_spatial(ker, h, w, x_device):
    in_c, out_c, h_k, w_k = ker.shape
    psf = torch.zeros([in_c, out_c, h, w], device=x_device, dtype=torch.cfloat)
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    centre_h = h_k // 2 + 1
    centre_w = w_k // 2 + 1
    psf[:, :, :centre_h, :centre_w] = ker[:, :, (centre_h - 1):, (centre_w - 1):]
    if h_k > 1 and w_k > 1:
        psf[:, :, -(centre_h - 1):, -(centre_w - 1):] = ker[:, :, :(centre_h - 1), :(centre_w - 1)]
    if w_k > 1:
        psf[:, :, :centre_h, -(centre_w - 1):] = ker[:, :, (centre_h - 1):, :(centre_w - 1)]
    if h_k > 1:
        psf[:, :, -(centre_h - 1):, :centre_w] = ker[:, :, : (centre_h - 1), (centre_w - 1):]

    return torch.fft.ifft2(psf) # .real
class spatial_fourier_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, bias=True):
        super(spatial_fourier_conv, self).__init__()

        self.complex_kernel_weight_real = nn.Parameter(torch.randn(
            (in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        self.complex_kernel_weight_imag = nn.Parameter(
            torch.randn((in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        init.kaiming_uniform_(self.complex_kernel_weight_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_kernel_weight_imag, a=math.sqrt(16))
        if bias:
            self.b = nn.Parameter(torch.zeros((1, out_channel, 1, 1), dtype=torch.float), requires_grad=True)

        self.bias = bias
    def forward(self, x):
        h, w = x.shape[-2:]
        kernel_weight = torch.complex(self.complex_kernel_weight_real, self.complex_kernel_weight_imag)
        spatial_kernel = fourier_kernel_to_spatial(kernel_weight, h, w, x.device)
        # print(spatial_kernel)

        y = torch.einsum('bchw,cjhw->bjhw', x, spatial_kernel.real)
        if self.bias:
            y = y + self.b
        return y
class spatial_fourier_fastconv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, bias=True):
        super(spatial_fourier_fastconv, self).__init__()
        self.kernel_weight = nn.Parameter(torch.randn(
            (in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(16))
        self.inc = in_channel
        self.outc = out_channel
        # self.complex_kernel_weight_real = nn.Parameter(torch.randn(
        #     (in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        # self.complex_kernel_weight_imag = nn.Parameter(
        #     torch.randn((in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        # init.kaiming_uniform_(self.complex_kernel_weight_real, a=math.sqrt(16))
        # init.kaiming_uniform_(self.complex_kernel_weight_imag, a=math.sqrt(16))

        if bias:
            self.b = nn.Parameter(torch.zeros((1, out_channel, 1, 1), dtype=torch.float), requires_grad=True)
        self.base_images = None # torch.zeros((H, W, kernel_size, kernel_size))
        self.bias = bias
        self.kernel_size = (kernel_size, kernel_size)
    def get_base_image(self, h, w):
        self.base_images = torch.zeros((h, w, self.kernel_size[0], self.kernel_size[1]), dtype=torch.cfloat)
        centre_h = self.kernel_size[0] // 2 + 1
        centre_w = self.kernel_size[1] // 2 + 1
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                ker = torch.zeros((self.kernel_size[0], self.kernel_size[1]), dtype=torch.cfloat)
                ker[i, j] = 1+0j
                psf = torch.zeros((h, w), dtype=torch.cfloat)
                # centre_h = h_k // 2 + 1
                # centre_w = w_k // 2 + 1
                psf[:centre_h, :centre_w] = ker[(centre_h - 1):, (centre_w - 1):]
                if self.kernel_size[0] > 1 and self.kernel_size[1] > 1:
                    psf[-(centre_h - 1):, -(centre_w - 1):] = ker[:(centre_h - 1), :(centre_w - 1)]
                if self.kernel_size[1] > 1:
                    psf[:centre_h, -(centre_w - 1):] = ker[(centre_h - 1):, :(centre_w - 1)]
                if self.kernel_size[0] > 1:
                    psf[-(centre_h - 1):, :centre_w] = ker[: (centre_h - 1), (centre_w - 1):]
                self.base_images[:, :, i, j] = torch.fft.ifft2(psf, norm='forward')
        # print(self.base_images[..., 0, 0])
        self.base_images = self.base_images.view(h, w, -1).real
    def forward(self, x):
        h, w = x.shape[-2:]
        if self.base_images is None:
            self.get_base_image(h, w)

        self.base_images = self.base_images.to(x.device)
        # kernel_weight = torch.complex(self.complex_kernel_weight_real, self.complex_kernel_weight_imag)
        kernel_weight = torch.fft.ifftshift(self.kernel_weight, dim=[-2, -1])
        kernel_weight = kernel_weight.view(self.inc, self.outc, -1)
        # print(kernel_weight)
        # print(self.base_images)
        # spatial_kernel = torch.einsum('cjn,hwn->cjhw', kernel_weight, self.base_images)
        # x_complex = torch.complex(x, torch.zeros_like(x))
        # y = torch.einsum('bchw,cjhw->bjhw', x, spatial_kernel.real)
        y = torch.einsum('bchw,cjn,hwn->bjhw', x, kernel_weight, self.base_images)
        # y = y.real
        if self.bias:
            y = y + self.b
        return y
class fourier_spatial_fastconv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, bias=True):
        super(fourier_spatial_fastconv, self).__init__()
        # self.kernel_weight = nn.Parameter(torch.randn(
        #     (in_channel, out_channel, kernel_size * kernel_size), dtype=torch.float), requires_grad=True)
        # init.kaiming_uniform_(self.kernel_weight, a=math.sqrt(16))
        self.inc = in_channel
        self.outc = out_channel
        self.complex_kernel_weight_real = nn.Parameter(torch.randn(
            (in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        self.complex_kernel_weight_imag = nn.Parameter(
            torch.randn((in_channel, out_channel, kernel_size, kernel_size), dtype=torch.float), requires_grad=True)
        init.kaiming_uniform_(self.complex_kernel_weight_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_kernel_weight_imag, a=math.sqrt(16))
        # torch.fft.fftshift
        if bias:
            self.b_real = nn.Parameter(torch.zeros((1, out_channel, 1, 1), dtype=torch.float), requires_grad=True)
            self.b_imag = nn.Parameter(torch.zeros((1, out_channel, 1, 1), dtype=torch.float), requires_grad=True)
        self.base_images = None # torch.zeros((H, W, kernel_size, kernel_size))
        self.bias = bias
        self.kernel_size = (kernel_size, kernel_size)

    def get_base_image(self, h, w):
        self.base_images = torch.zeros((h, w, self.kernel_size[0], self.kernel_size[1]), dtype=torch.cfloat)
        centre_h = self.kernel_size[0] // 2 + 1
        centre_w = self.kernel_size[1] // 2 + 1
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                ker = torch.zeros((self.kernel_size[0], self.kernel_size[1]), dtype=torch.cfloat)
                ker[i, j] = 1 + 0j
                psf = torch.zeros((h, w), dtype=torch.cfloat)
                psf[:centre_h, :centre_w] = ker[(centre_h - 1):, (centre_w - 1):]
                if self.kernel_size[0] > 1 and self.kernel_size[1] > 1:
                    psf[-(centre_h - 1):, -(centre_w - 1):] = ker[:(centre_h - 1), :(centre_w - 1)]
                if self.kernel_size[1] > 1:
                    psf[:centre_h, -(centre_w - 1):] = ker[(centre_h - 1):, :(centre_w - 1)]
                if self.kernel_size[0] > 1:
                    psf[-(centre_h - 1):, :centre_w] = ker[: (centre_h - 1), (centre_w - 1):]
                self.base_images[:, :, i, j] = torch.fft.fft2(psf, norm='forward')
        # print(self.base_images[..., 0, 0])
        self.base_images = self.base_images.view(h, w, -1) # .real
    def forward(self, x):
        h, w = x.shape[-2:]
        if self.base_images is None:
            self.get_base_image(h, w)
            # print()

        self.base_images = self.base_images.to(x.device)
        kernel_weight = torch.complex(self.complex_kernel_weight_real, self.complex_kernel_weight_imag)
        kernel_weight = torch.fft.ifftshift(kernel_weight, dim=[-2, -1])
        kernel_weight = kernel_weight.view(self.inc, self.outc, -1)
        # spatial_kernel = torch.einsum('cjn,hwn->cjhw', kernel_weight, self.base_images)
        # x_complex = torch.complex(x, torch.zeros_like(x))
        # y = torch.einsum('bchw,cjhw->bjhw', x, spatial_kernel.real)
        y = torch.einsum('bchw,cjn,hwn->bjhw', x, kernel_weight, self.base_images)
        # y = y.real
        if self.bias:
            bias = torch.complex(self.b_real, self.b_imag)
            y = y + bias
        return y
class fourier_complex_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=True):
        super(fourier_complex_mlp, self).__init__()
        self.act_fft = act_method()
        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_weight1 = nn.Parameter(torch.randn(dim, hid_dim, dtype=torch.cfloat), requires_grad=True)

        self.complex_weight2 = nn.Parameter(torch.randn(hid_dim, dim, dtype=torch.cfloat), requires_grad=True)
        init.kaiming_uniform_(self.complex_weight1, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2, a=math.sqrt(16))

        if bias:
            self.b1 = nn.Parameter(torch.zeros((1, 1, 1, hid_dim), dtype=torch.cfloat), requires_grad=True)
            self.b2 = nn.Parameter(torch.zeros((1, 1, 1, hid_dim), dtype=torch.cfloat), requires_grad=True)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)

        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ self.complex_weight1
        if self.bias:
            y = y + self.b1
        y = torch.cat([y.real, y.imag], dim=-1)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=-1)
        y = torch.complex(y_real, y_imag)
        y = y @ self.complex_weight2
        if self.bias:
            y = y + self.b2
        y = rearrange(y, 'b h w c -> b c h w')

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False, fft=True):
        super(fft_bench_complex_mlp, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        self.fft = fft
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        if self.fft:
            x = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(x, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            if self.fft:
                y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            if self.fft:
                y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_spatial3x3(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False, fft=True, kernel_size=3):
        super(fft_bench_spatial3x3, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        self.fft = fft
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim, hid_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv1(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        if self.fft:
            x = torch.fft.rfft2(x, norm=self.norm)
        dim = 1

        x = torch.cat([x.real, x.imag], dim=dim)

        y = self.act_fft(x)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            if self.fft:
                y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            if self.fft:
                y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.conv2(y)
        return y
class fft_bench_complex_mlp_dyrelu(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False):
        super(fft_bench_complex_mlp_dyrelu, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.thi = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        thi = torch.cos(self.thi) + 1j*torch.sin(self.thi)
        y = y * thi
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y / thi
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_globalphase(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=64, bias=False):
        super(fft_bench_complex_mlp_globalphase, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        hid_dim = dim * dw
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.gphase = nn.Parameter(torch.zeros((1, window_size, window_size//2+1, hid_dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        # print(x.shape)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        thi = torch.cos(self.gphase) + 1j*torch.sin(self.gphase)
        y = y * thi
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = y / thi
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_flops(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=0, bias=False):
        super(fft_bench_complex_mlp_flops, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1 = nn.Conv2d(dim*2, hid_dim*2, 1, bias=bias)
        self.complex_weight2 = nn.Conv2d(hid_dim*2, dim*2, 1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.complex_weight1(y)
        y = self.act_fft(y)
        y = self.complex_weight2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp2(nn.Module):
    def __init__(self, dim, out_dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=0, bias=False):
        super(fft_bench_complex_mlp2, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, out_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, out_dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))

        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        # _, _, H, W = x.shape
        # if self.window_size > 0 and (H != self.window_size or W != self.window_size):
        #     x, batch_list = window_partitionx(x, self.window_size)
        # y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            # b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(x, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # if self.window_size > 0 and (H != self.window_size or W != self.window_size):
        #     y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
        #     y = window_reversex(y, self.window_size, H, W, batch_list)
        # else:
        #     y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_phase(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=0, bias=False):
        super().__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, C, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        weight1 = torch.exp(1j * torch.angle(weight1)) / C
        weight2 = torch.exp(1j * torch.angle(weight2)) / C
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_phase2(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=0, bias=False):
        super().__init__()
        # self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1_phase = nn.Parameter(torch.Tensor(dim, hid_dim))
        # self.complex_weight2_phase = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_phase, a=math.sqrt(16))
        # init.kaiming_uniform_(self.complex_weight2_phase, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            # self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            # self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, C, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        mag = torch.abs(y)
        y = torch.exp(1j * torch.angle(y))
        weight1 = torch.exp(1j * self.complex_weight1_phase) # / C
        
        # weight2 = torch.exp(1j * self.complex_weight2_phase) # / C
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            # b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = torch.sin(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = y @ weight2
        # if self.bias:
        #     y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        y = mag * y
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_Phase(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=0, bias=False):
        super(fft_bench_complex_mlp_Phase, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.exp(1j * torch.angle(y))
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResBlock, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        # self.main_fft = fft_bench_complex_mlp(n_feat, norm)

    def forward(self, x):
        res = self.main(x) # + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_complex(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward', num_heads=1): # 'backward'
        super(ResFourier_complex, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm=norm)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class LPFAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
class ResFourier_LPF(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward', num_heads=1): # 'backward'
        super(ResFourier_LPF, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm=norm)
        # cs = 'global_channel_nodct'
        cs = 'local_channel_dct'
        self.norm1 = LayerNorm2d(n_feat)
        self.lpf = LPFAttention(n_feat, num_heads=num_heads, cs=cs, bias=True)
        self.dw = nn.Conv2d(n_feat, n_feat, kernel_size=3,
                                           stride=1, padding=1, groups=n_feat, bias=True, padding_mode='zeros')
        if 'nodct' in cs:
            self.dct = nn.Identity()
            self.idct = nn.Identity()
        elif 'dct_torch' in cs:
            self.dct = DCT2_torch()
            self.idct = IDCT2_torch()
        else:
            self.dct = DCT2x()
            self.idct = IDCT2x()
    def forward(self, x):
        res = self.main(x) + self.main_fft(x) + x

        res = res + self.idct(self.dw(self.lpf(self.norm1(self.dct(res)))))
        # res += x
        return res
class ResFourier_dual3x3(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward', num_heads=1): # 'backward'
        super(ResFourier_dual3x3, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_spatial3x3(n_feat, norm=norm)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_conv1x1_in_spatial(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_conv1x1_in_spatial, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fourier_conv_in_spatial(n_feat, norm=norm, kernel_size=1, bias=True)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_conv3x3_in_spatial(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_conv3x3_in_spatial, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fourier_conv_in_spatial(n_feat, norm=norm, kernel_size=3, bias=True)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_spatialattn_in_fourier(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward', num_heads=1): # 'backward'
        super(ResFourier_spatialattn_in_fourier, self).__init__()
        self.main = spatial_block_in_fourier(n_feat, kernel_size=kernel_size, num_heads=num_heads, fft=False)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm=norm, bias=True, fft=False)
        self.norm = norm
    def forward(self, x):
        x = torch.fft.rfft2(x, norm=self.norm)
        x = self.main(x) + x
        x = self.main_fft(x) + x
        out = torch.fft.irfft2(x, norm=self.norm)
        # res += x
        return out
class ResFourier_real_conv3x3(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_real_conv3x3, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fourier_real_conv3x3(n_feat, norm=norm)

    def forward(self, x):
        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_complex_gelu(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super().__init__()
        act_method = nn.GELU
        self.main = conv_bench(n_feat, kernel_size=kernel_size, act_method=act_method)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm, act_method=act_method)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x

class SimpleGate(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class ResFourier_complex_LN1(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super().__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)
        self.norm1 = LayerNorm2d(n_feat)
    def forward(self, x):
        y = self.norm1(x)
        res = self.main(y) + self.main_fft(y)
        # res += x
        return res + x
class ResFourier_complex_LN2(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super().__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)
        self.norm1 = LayerNorm2d(n_feat)
    def forward(self, x):
        y = self.norm1(x)
        res = self.main(y) + self.main_fft(x)
        # res += x
        return res + x
# def get_dctMatrix(m, n):
#     N = n
#     C_temp = np.zeros([m, n])
#     C_temp[0, :] = 1 * np.sqrt(1 / N)
#
#     for i in range(1, m):
#         for j in range(n):
#             C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)
#                                   ) * np.sqrt(2 / N)
#     return torch.tensor(C_temp, dtype=torch.float)
#
# def dct1d(feature, dctMat):
#     feature = dctMat @ feature #
#     return feature  # torch.tensor(x, device=feature.device)
#
# def idct1d(feature, dctMat):
#     feature = dctMat.T @ feature # .T
#     return feature  # torch.tensor(x, device=feature.device)
# def dct2d(feature, dctMat):
#     # print(dctMat.shape, feature.shape)
#     feature = dctMat @ feature
#     # print(dctMat.shape, feature.shape)
#     feature = feature @ dctMat.T
#     return feature # torch.tensor(x, device=feature.device)
#
# def idct2d(feature, dctMat):
#     feature = dctMat.T @ feature
#     feature = feature @ dctMat
#     return feature # torch.tensor(x, device=feature.device)
#
# def dct2dx(feature, dctMat1, dctMat2):
#     # print(dctMat.shape, feature.shape)
#     feature = dctMat1 @ feature
#     # print(dctMat.shape, feature.shape)
#     feature = feature @ dctMat2.T
#     return feature # torch.tensor(x, device=feature.device)
#
# def idct2dx(feature, dctMat1, dctMat2):
#     feature = dctMat1.T @ feature
#     feature = feature @ dctMat2
#     return feature # torch.tensor(x, device=feature.device)
def get_Coord(ins_feat):
    x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
    y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([ins_feat.shape[0], 1, -1, -1])
    x = x.expand([ins_feat.shape[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    ins_feat = torch.cat([ins_feat, coord_feat], 1)
    return ins_feat

def window_pad(x, window_size):
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    b, c, h, w = x.shape
    h_ = h + window_size[0]
    w_ = w + window_size[1]
    y = torch.zeros([b, c, h_, w_], device=x.device)
    y[:, :, :h, :w] = x
    y[:, :, h:, :w] = x[:, :, h - 2 * window_size[0]:h - window_size[0], :]
    y[:, :, :h, w:] = x[:, :, :, w - 2 * window_size[1]:w - window_size[1]]
    y[:, :, h:, w:] = x[:, :, h - 2 * window_size[0]:h - window_size[0], w - 2 * window_size[1]:w - window_size[1]]
    return y
# class DCT1d(nn.Module):
#     def __init__(self, window_size=64):
#         super(DCT1d, self).__init__()
#         self.dctMat = get_dctMatrix(window_size, window_size)
#     def forward(self, x):
#         self.dctMat = self.dctMat.to(x.device)
#         # print(x.shape, self.dctMat.shape)
#         x = dct1d(x, self.dctMat)
#         return x
# class IDCT1d(nn.Module):
#     def __init__(self, window_size=64):
#         super(IDCT1d, self).__init__()
#         self.dctMat = get_dctMatrix(window_size, window_size)
#     def forward(self, x):
#         self.dctMat = self.dctMat.to(x.device)
#         x = idct1d(x, self.dctMat)
#         return x
# class DCT2(nn.Module):
#     def __init__(self, window_size=8):
#         super(DCT2, self).__init__()
#         self.dctMat = get_dctMatrix(window_size, window_size)
#     def forward(self, x):
#         self.dctMat = self.dctMat.to(x.device)
#         # print(x.shape, self.dctMat.shape)
#         x = dct2d(x, self.dctMat)
#         return x
# class IDCT2(nn.Module):
#     def __init__(self, window_size=8):
#         super(IDCT2, self).__init__()
#         self.dctMat = get_dctMatrix(window_size, window_size)
#     def forward(self, x):
#         self.dctMat = self.dctMat.to(x.device)
#         x = idct2d(x, self.dctMat)
#         return x
# class DCT2x(nn.Module):
#     def __init__(self, H=8, W=8, norm='backward'):
#         super(DCT2x, self).__init__()
#         self.dctMatH = get_dctMatrix(H, H)
#         self.dctMatW = get_dctMatrix(W, W)
#         self.H = H
#         self.W = W
#         self.norm = norm
#
#     def forward(self, x):
#         _, _, h, w = x.shape
#         if h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
#             self.dctMatH = get_dctMatrix(h, h)
#             self.dctMatW = get_dctMatrix(w, w)
#         elif h != self.dctMatH.shape[-1]:
#             self.dctMatH = get_dctMatrix(h, h)
#             # self.dctMatH = self.dctMatH.to(x.device)
#         elif w != self.dctMatW.shape[-1]:
#             self.dctMatW = get_dctMatrix(w, w)
#             # self.dctMatW = self.dctMatW.to(x.device)
#         # else:
#         #     self.dctMatH = self.dctMatH.to(x.device)
#         #     self.dctMatW = self.dctMatW.to(x.device)
#         self.dctMatH = self.dctMatH.to(x.device)
#         self.dctMatW = self.dctMatW.to(x.device)
#         # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
#         x = dct2dx(x, self.dctMatH, self.dctMatW)
#         if self.norm == 'hw':
#             return x / (h * w)
#         else:
#             return x
# class IDCT2x(nn.Module):
#     def __init__(self, H=8, W=8, norm='backward'):
#         super(IDCT2x, self).__init__()
#         self.dctMatH = get_dctMatrix(H, H)
#         self.dctMatW = get_dctMatrix(W, W)
#         self.H = H
#         self.W = W
#         self.norm = norm
#
#     def forward(self, x):
#         _, _, h, w = x.shape
#         if h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
#             self.dctMatH = get_dctMatrix(h, h)
#             self.dctMatW = get_dctMatrix(w, w)
#         elif h != self.dctMatH.shape[-1]:
#             self.dctMatH = get_dctMatrix(h, h)
#             # self.dctMatH = self.dctMatH.to(x.device)
#         elif w != self.dctMatW.shape[-1]:
#             self.dctMatW = get_dctMatrix(w, w)
#             # self.dctMatW = self.dctMatW.to(x.device)
#         # else:
#         #     self.dctMatH = self.dctMatH.to(x.device)
#         #     self.dctMatW = self.dctMatW.to(x.device)
#         self.dctMatH = self.dctMatH.to(x.device)
#         self.dctMatW = self.dctMatW.to(x.device)
#         x = idct2dx(x, self.dctMatH, self.dctMatW)
#         if self.norm == 'hw':
#             return x * (h * w)
#         else:
#             return x
class GridGmlpLayer(nn.Module):
    def __init__(self, dim, grid_size=8, factor=2, bias=True):
        self.factor = factor
        self.grid_size = grid_size
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.gelu = nn.GELU()
        self.project_in = nn.Conv2d(in_channels=dim, out_channels=dim * factor, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)
        self.grid_mlp = nn.Linear(grid_size**2, grid_size**2, bias=bias)
        self.project_out = nn.Conv2d(in_channels=dim * factor, out_channels=dim * dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)
    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        x = self.project_in(self.norm1(x))
        x = self.gelu(x)
        # x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', k1=self.grid_size, k2=self.grid_size)
        u, v = torch.chunk(x, 2, dim=1)
        v = self.norm2(v)
        v = rearrange(v, 'b c (k1 h) (k2 w) -> b c (h w) (k1 k2)', k1=self.grid_size, k2=self.grid_size)
        v = self.grid_mlp(v)
        v = rearrange(v, 'b c (h w) (k1 k2) -> b c (k1 h) (k2 w)', k1=self.grid_size, k2=self.grid_size,
                      h=H//self.grid_size, w=W//self.grid_size)
        x = u * (v + 1.)
        x = self.project_out(x)
        return x + short_cut
class BlockGmlpLayer(nn.Module):
    def __init__(self, dim, block_size=8, factor=2, bias=True):
        self.factor = factor
        self.block_size = block_size
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.gelu = nn.GELU()
        self.project_in = nn.Conv2d(in_channels=dim, out_channels=dim * factor, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)
        self.block_mlp = nn.Linear(block_size**2, block_size**2, bias=bias)
        self.project_out = nn.Conv2d(in_channels=dim * factor, out_channels=dim * dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)
    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        x = self.project_in(self.norm1(x))
        x = self.gelu(x)
        # x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', k1=self.grid_size, k2=self.grid_size)
        u, v = torch.chunk(x, 2, dim=1)
        v = self.norm2(v)
        v = rearrange(v, 'b c (k1 h) (k2 w) -> b c (k1 k2) (h w)', k1=self.block_size, k2=self.block_size)
        v = self.block_mlp(v)
        v = rearrange(v, 'b c (k1 k2) (h w) -> b c (k1 h) (k2 w)', h=self.block_size, w=self.block_size,
                      k1=H//self.block_size, k2=W//self.block_size)
        x = u * (v + 1.)
        x = self.project_out(x)
        return x + short_cut
class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    def __init__(self, dim, grid_size=8, block_size=8, factor=2, bias=True):
        self.factor = factor
        self.block_size = block_size
        self.norm1 = LayerNorm2d(dim)

        self.gelu = nn.GELU()
        self.project_in = nn.Conv2d(in_channels=dim, out_channels=dim * factor, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)
        self.grid_blk = GridGmlpLayer(dim, grid_size, factor=2, bias=bias)
        self.block_blk = BlockGmlpLayer(dim, block_size, factor=2, bias=bias)
        self.project_out = nn.Conv2d(in_channels=dim * factor, out_channels=dim * dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)
    def forward(self, x):
        short_cut = x
        x = self.project_in(self.norm1(x))
        x = self.gelu(x)
        # x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', k1=self.grid_size, k2=self.grid_size)
        u, v = torch.chunk(x, 2, dim=1)
        u = self.grid_blk(u)
        v = self.block_blk(v)
        x = torch.cat([u, v], dim=1)
        x = self.project_out(x)
        return x + short_cut
class FFT_ReLU(nn.Module):
    def __init__(self, window_size=None, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class FFT_Convblock(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_Convblock_Phase_act(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.act_fft = Phase_Act3()
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class Dropout_on_phase(nn.Module):
    def __init__(self, drop_out_rate=0.1):
        super().__init__()
        self.norm = 'backward'
        # self.dropout_real = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout_imag = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        mag = torch.abs(y)
        phase = torch.exp(1j * torch.angle(y))

        # phase_real = self.dropout_real(phase.real)
        # phase_imag = self.dropout_imag(phase.imag)
        phase_real, phase_imag = self.dropout(phase.real), self.dropout(phase.imag)
        phase = torch.complex(phase_real, phase_imag)
        y = mag * phase
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
    
class FFT_DConvblock(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.dconv = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1, stride=1,
                               groups=dim * 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.dconv(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_noact(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        # self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.dconv = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1, stride=1,
                               groups=dim * 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.dconv(y)
        # y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_Phase(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        self.conv_mag = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True)])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_mag = self.conv_mag(y_mag)
        y_phase = torch.angle(y)
        y = torch.fft.irfft2(torch.exp(1j * y_phase), s=(H, W), norm=self.norm)
        # y_phase = y_phase / self.pi
        y = self.conv_phase(y)
        y = torch.tanh(y)
        y = torch.fft.rfft2(y, norm=self.norm)
        y_ir_angle = torch.log(y) / 1j
        y_ir_angle = y_ir_angle.real # + y_phase
        y_ir_angle = torch.clamp(y_ir_angle, -2.*self.pi, 2.*self.pi)

        y = torch.fft.irfft2(y_mag * torch.exp(1j * y_ir_angle), s=(H, W), norm=self.norm)
        return torch.sin(y)
class Sin_ACT(nn.Module):
    def __init__(self, alpha=1, beta=3.1415926536):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        return torch.sin(x*self.beta) * self.beta
class FFT_DConvblock_mag(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_mag = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1,groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,groups=1, bias=False),
            nn.ReLU(inplace=True)
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        y_mag = self.conv_mag(y_mag)
        y = y_mag * torch.exp(1j * y_phase)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_mag_phase(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        # self.conv_mag = nn.Sequential(*[
        #     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,groups=1, bias=False),
        #     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1,groups=dim, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,groups=1, bias=False),
        #     nn.ReLU(inplace=True)
        # ])
        self.conv_mag1 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
        ])
        self.conv_mag2 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True)
        ])
        # self.conv_phase = nn.Sequential(*[
        #     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        #     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
        #     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        # ])
        self.conv_phase1 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.ReLU(inplace=True)
        ])
        self.conv_phase2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),

        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        y_mag = self.conv_mag1(y_mag)
        y_phase = y_phase / self.pi
        # y_phase = self.conv_phase(y_phase)
        y_phase = self.conv_phase1(y_phase)
        # y_phase, y_phase2 = torch.chunk(y_phase, 2, dim=1)
        # y_mag = y_mag * y_phase2
        y_mag = self.conv_mag2(y_mag)
        y_phase = self.conv_phase2(y_phase)
        y_phase = torch.tanh(y_phase)*self.pi
        y = y_mag * torch.exp(1j * y_phase)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_phase(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        # y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)
        # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
        y_phase = torch.tanh(y_phase) * self.pi
        # y = y_mag * torch.exp(1j * y_phase)
        y = torch.exp(1j * y_phase)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_phase_sin(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        # y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        # y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase) + y_phase
        # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
        # y_phase = torch.sin(y_phase) * self.pi
        # y_phase = y_phase +
        # y = y_mag * torch.exp(1j * y_phase)
        y = torch.exp(1j * y_phase)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)


class FFT_DConvblock_mag_phase_sin(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        # self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        # self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, stride=1, groups=dim, bias=False)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        # y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        # y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        # y_mag = self.conv_mag(y_mag)
        # y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)  # + y_phase
        # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
        # y_phase = torch.sin(y_phase) * self.pi
        # y_phase = y_phase +
        # y = y_mag * torch.exp(1j * y_phase)
        y = torch.exp(1j * y_phase)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_phase_2pi(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        # y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)
        # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
        y_phase = torch.tanh(y_phase) * 2 * self.pi
        # y = y_mag * torch.exp(1j * y_phase)
        y = torch.exp(1j * y_phase)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
# class FFT_DConvblock_phase_v2(nn.Module):
#     def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
#         super().__init__()
#         self.pi = 3.141592653589793
#         # self.act_mag = nn.ReLU(inplace=True)
#         self.conv_phase = nn.Sequential(*[
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
#         ])
#         # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
#         self.norm = norm
#         self.window_size = window_size
#
#     def forward(self, x):
#         _, _, H, W = x.shape
#         if self.window_size > 0 and (H != self.window_size or W != self.window_size):
#             x, batch_list = window_partitionx(x, self.window_size)
#         y = torch.fft.rfft2(x, norm=self.norm)
#         # dim = 1
#         # y_mag = torch.abs(y)
#         y_phase = torch.angle(y)
#         y_phase = y_phase / self.pi
#         y_phase = self.conv_phase(y_phase)
#         # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
#         y_phase = torch.tanh(y_phase) * self.pi
#         # y = y_mag * torch.exp(1j * y_phase)
#         y = torch.exp(1j * y_phase)
#         if self.window_size > 0 and (H != self.window_size or W != self.window_size):
#             y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
#             y = window_reversex(y, self.window_size, H, W, batch_list)
#         else:
#             y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
#         return torch.sin(y)
class FFT_DConvblock_phase_v2(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x1, x2):
        _, _, H, W = x1.shape
        y1 = torch.fft.rfft2(x1, norm=self.norm)
        y2 = torch.fft.rfft2(x2, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y2)
        y_phase = torch.angle(y1)
        y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)
        # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
        y_phase = torch.sin(y_phase) * self.pi * 2
        y = y_mag * torch.exp(1j * y_phase)
        # y = torch.exp(1j * y_phase)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_DConvblock_phase_v3(nn.Module):
    def __init__(self, dim, window_size=0, shift_size=0, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.conv_phase = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        self.conv_spatial = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        ])
        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        x2 = self.conv_spatial(x)
        y1 = torch.fft.rfft2(x, norm=self.norm)
        y2 = torch.fft.rfft2(x2, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y2)
        y_phase = torch.angle(y1)
        y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)
        # y_phase = torch.clamp(y_phase, -1.*self.pi, self.pi)
        y_phase = torch.tanh(y_phase) * self.pi * 2
        y = y_mag * torch.exp(1j * y_phase)
        # y = torch.exp(1j * y_phase)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return torch.sin(y)
class FFT_Convblock_shift(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=0, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        self.shift_size = shift_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            if self.shift_size > 0:
                x = F.pad(x, (self.shift_size, 0, self.shift_size, 0), mode='reflect')
            _, _, H, W = x.shape
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
            if self.shift_size > 0:
                y = y[:, :, self.shift_size:, self.shift_size:]
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        return torch.sin(y)
class FFT_Convblock_nosin(nn.Module):
    def __init__(self, dim, window_size=0, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y # torch.sin(y)
class FFT_Convblock_LN(nn.Module):
    def __init__(self, dim, window_size=None, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        self.layer_norm1 = LayerNorm2d(dim)
        self.layer_norm2 = LayerNorm2d(dim)
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.layer_norm1(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = torch.sin(y)
        y = self.layer_norm2(y)
        y = self.conv3(y)
        y1, y2 = torch.chunk(y, 2, dim=1)
        y = y1 * y2
        y = self.conv4(y)
        return y #
class FFT_Convblock_pad(nn.Module):
    def __init__(self, dim, window_size=None, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, stride=1,
                              groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size
        # self.top = nn.Parameter(torch.ones(1))
    def forward(self, x):
        _, _, H, W = x.shape
        padh, padw = self.window_size // 4, self.window_size // 4
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        x = F.pad(x, (padh, padh, padw, padw), mode='reflect')
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size+2*padh, self.window_size+2*padw), norm=self.norm)
            y = y[:, :, padh:-padh, padw:-padw]
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H+2*padh, W+2*padw), norm=self.norm)
            y = y[:, :, padh:-padh, padw:-padw]
        return torch.sin(y)
class FFT_Gridblock(nn.Module):
    def __init__(self, dim, window_size=None, norm='backward'):
        super().__init__()
        self.act_fft = nn.ReLU(inplace=True)
        dil = [window_size, window_size//2+1]
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=2, padding=0, dilation=dil
                               , stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)
        self.norm = norm
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.pad(x, (self.window_size, 0, self.window_size, 0), mode='reflect')
        if self.window_size is not None:
            x = window_partitions(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)

        if self.window_size is not None:
            y_w = y.shape[-1]
            W_ = y.shape[2] * y_w
            y = window_reverses(y, [self.window_size, y_w], H, W_)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv1(y)
        y = self.act_fft(y)
        y = self.conv2(y)

        if self.window_size is not None:
            y = window_partitions(y, [self.window_size, y_w])
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reverses(y, self.window_size, H, W)
        else:
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class DCT_block(nn.Module):
    def __init__(self, dim, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.act = nn.ReLU(inplace=True)
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=2, padding=0, dilation=window_size
                               , stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        if H != self.window_size or W != self.window_size:
            x = window_partitions(x, self.window_size)
        y = self.dct(x)
        if H != self.window_size or W != self.window_size:
            y = window_reverses(y, self.window_size, H, W)
        y = window_pad(y, self.window_size)
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)
        if H != self.window_size or W != self.window_size:
            y = window_partitions(y, self.window_size)
            y = self.idct(y)
            y = window_reverses(y, self.window_size, H, W)
        else:
            y = self.idct(y)
        return y
class DCT_block2(nn.Module):
    def __init__(self, dim, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.act = nn.ReLU(inplace=True)
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.conv1 = nn.Conv2d(in_channels=dim//2, out_channels=dim//2, kernel_size=2, padding=0, dilation=window_size
                               , stride=1, groups=dim//2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim//2, out_channels=dim//2, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        if H != self.window_size or W != self.window_size:
            x = window_partitions(x, self.window_size)
        y = self.dct(x)
        if H != self.window_size or W != self.window_size:
            y = window_reverses(y, self.window_size, H, W)
        y1, y2 = torch.chunk(y, 2, dim=1)
        y1 = window_pad(y1, self.window_size)
        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y = torch.cat([y1, y2], dim=1)
        y = self.act(y)
        y = self.conv3(y)
        if H != self.window_size or W != self.window_size:
            y = window_partitions(y, self.window_size)
            y = self.idct(y)
            y = window_reverses(y, self.window_size, H, W)
        else:
            y = self.idct(y)
        return y
class DCT_block_noact(nn.Module):
    def __init__(self, dim, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        # self.act = nn.ReLU(inplace=True)
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=2, padding=0, dilation=window_size
                               , stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        if H != self.window_size or W != self.window_size:
            x = window_partitions(x, self.window_size)
        y = self.dct(x)
        if H != self.window_size or W != self.window_size:
            y = window_reverses(y, self.window_size, H, W)
        y = window_pad(y, self.window_size)
        y = self.conv1(y)
        # y = self.act(y)
        y = self.conv2(y)
        if H != self.window_size or W != self.window_size:
            y = window_partitions(y, self.window_size)
            y = self.idct(y)
            y = window_reverses(y, self.window_size, H, W)
        else:
            y = self.idct(y)
        return y
class DCT_block_3D(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.act = nn.ReLU(inplace=True)
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.qkv = nn.Conv3d(in_channels=dim, out_channels=dim*3, kernel_size=[1, 3, 3], padding=[0,1,1], dilation=1
                               , stride=1, groups=1, bias=False)
        # self.conv2 = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=[1, 3, 3], padding=[0,1,1], dilation=1
        #                        , stride=1, groups=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape


        x = self.dct(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        q = rearrange(q, 'b (head c) d h w -> b head (c d) (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head (c d) (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        # v = self.act(v)
        # v = self.conv2(v)
        v = rearrange(v, 'b (head c) d h w -> b head (c d) (h w)', head=self.num_heads)
        x = (attn @ v)
        x = rearrange(x, 'b head (c d) (h w) -> b (head c) d h w', head=self.num_heads, c=C//self.num_heads, h=self.window_size,
                        w=self.window_size)
        x = self.idct(x)
        x = rearrange(x, 'b c (h1 w1) h w -> b c (h1 h) (w1 w)', h1=H//self.window_size, w1=W//self.window_size)
        return x # torch.sin(x)
class DCT_block_attn(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.act = nn.ReLU() # inplace=True
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.qkv = nn.Conv2d(in_channels=dim, out_channels=dim*3, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, dilation=1
                             , stride=1, groups=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        x = rearrange(x, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h=self.window_size, w=self.window_size)
        x = self.dct(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        q = rearrange(q, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)
        k = rearrange(k, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        v = self.act(v)
        v = self.conv2(v)
        v = rearrange(v, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)
        out = (attn @ v)
        out = rearrange(out, 'b head (c d) (h w) -> b (head c) d h w', head=self.num_heads, c=C//self.num_heads, h=self.window_size,
                        w=self.window_size)
        out = self.idct(out)
        out = rearrange(out, 'b c (h1 w1) h w -> b c (h1 h) (w1 w)', h1=H//self.window_size, w1=W//self.window_size)
        return torch.sin(out)
class FFT_block_attn(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.act = nn.ReLU() # inplace=True

        self.qkv = nn.Conv2d(in_channels=dim, out_channels=dim*3, kernel_size=1, padding=0, dilation=1
                               , stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=1, padding=0, dilation=1
                             , stride=1, groups=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        x = rearrange(x, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h=self.window_size, w=self.window_size)
        qkv = self.qkv(x)
        qkv = torch.fft.rfft2(qkv, norm=self.norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        dim = 1
        q = torch.cat([q.real, q.imag], dim=dim)
        k = torch.cat([k.real, k.imag], dim=dim)
        v = torch.cat([v.real, v.imag], dim=dim)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        q = rearrange(q, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)
        k = rearrange(k, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        v = self.act(v)
        v = self.conv2(v)
        v = rearrange(v, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)
        out = (attn @ v)
        w_f = self.window_size//2+1
        out = rearrange(out, 'b head (c d) (h w) -> (b d) (head c) h w', head=self.num_heads, c=2*C//self.num_heads,
                        h=self.window_size, w=w_f)
        y_real, y_imag = torch.chunk(out, 2, dim=dim)
        out = torch.complex(y_real, y_imag)
        out = torch.fft.irfft2(out, s=(self.window_size, self.window_size), norm=self.norm)
        out = rearrange(out, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=H//self.window_size, w1=W//self.window_size)
        return torch.sin(out)
class FFT_Conv(nn.Module):
    def __init__(self, dim, window_size=None, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.conv = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=0, stride=1,
                      groups=1, bias=False)
    def forward(self, x):
        _, _, H, W = x.shape

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.conv(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.norm = 'backward'
        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        # x_h = torch.fft.rfft2(x_h, norm=self.norm)
        # x_w = torch.fft.rfft2(x_w, norm=self.norm)
        # x = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        # x_h = torch.cat([x_h.real, x_h.imag], dim=dim)
        #         x_1=self.fc_h(x)
        #         x_2=self.fc_w(x)
        #         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
        #         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        # x = torch.fft.irfft2(x, s=(H, W), norm=self.norm)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PATM_fft(nn.Module):
    def __init__(self, dim, window_size=-1, qkv_bias=False, proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(2 * dim, 2 * dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, 2 * dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, 2 * dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(2 * dim, 2 * dim // 4, 2 * dim * 3)
        # self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.norm = 'backward'
        self.window_size = window_size
        # if mode == 'fc':
        #     self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        #     self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        # else:
        #     self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
        #                                       nn.BatchNorm2d(dim), nn.ReLU())
        #     self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
        #                                       nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        # theta_h = self.theta_h_conv(x)
        # theta_w = self.theta_w_conv(x)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
            B = x.shape[0]
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        # x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        # x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        x_h = torch.fft.rfft2(x_h, norm=self.norm)
        x_w = torch.fft.rfft2(x_w, norm=self.norm)
        x = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        x_h = torch.cat([x_h.real, x_h.imag], dim=dim)
        x_w = torch.cat([x_w.real, x_w.imag], dim=dim)
        x = torch.cat([x.real, x.imag], dim=dim)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C*2, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c * a[2]
        x_real, x_imag = torch.chunk(x, 2, dim=dim)
        x = torch.complex(x_real, x_imag)
        if self.window_size > 0 and (H != self.window_size or W != self.window_size):
            x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), norm=self.norm)
            x = window_reversex(x, self.window_size, H, W, batch_list)
        else:
            x = torch.fft.irfft2(x, s=(H, W), norm=self.norm)
        x = torch.sin(x)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x
class PATM_fft_v2(nn.Module):
    def __init__(self, dim, window_size=-1, window_size_fft=7, qkv_bias=False, proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(2 * dim, 2 * dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, 2 * dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, 2 * dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(2 * dim, 2 * dim // 4, 2 * dim * 3)
        # self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.norm = 'backward'
        self.window_size = window_size
        self.window_size_fft = window_size_fft
        # if mode == 'fc':
        #     self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        #     self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        # else:
        #     self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
        #                                       nn.BatchNorm2d(dim), nn.ReLU())
        #     self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
        #                                       nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        # theta_h = self.theta_h_conv(x)
        # theta_w = self.theta_w_conv(x)
        # if self.window_size_fft > 0 and (H != self.window_size_fft or W != self.window_size_fft):
        #     x, batch_list = window_partitionx(x, self.window_size_fft)
        #     B = x.shape[0]
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        # x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        # x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        x_h = torch.fft.rfft2(x_h, norm=self.norm)
        x_w = torch.fft.rfft2(x_w, norm=self.norm)
        x = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        x_h = torch.cat([x_h.real, x_h.imag], dim=dim)
        x_w = torch.cat([x_w.real, x_w.imag], dim=dim)
        x = torch.cat([x.real, x.imag], dim=dim)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        # y = h + w + c
        # y_real, y_imag = torch.chunk(y, 2, dim=dim)
        # y = torch.complex(y_real, y_imag)
        # if self.window_size_fft > 0 and (H != self.window_size_fft or W != self.window_size_fft):
        #     h = torch.fft.irfft2(h, s=(self.window_size_fft, self.window_size_fft), norm=self.norm)
        #     h = window_reversex(h, self.window_size_fft, H, W, batch_list)
        # else:
        #     h = torch.fft.irfft2(h, s=(H, W), norm=self.norm)
        if self.window_size_fft > 0 and (H != self.window_size_fft or W != self.window_size_fft):
            h, batch_list = window_partitionx(h, self.window_size_fft)
            w, _ = window_partitionx(w, self.window_size_fft)
            c, _ = window_partitionx(c, self.window_size_fft)

        y = h + w + c
        B = y.shape[0]
        a = F.adaptive_avg_pool2d(y, output_size=1)
        a = self.reweight(a).reshape(B, C * 2, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c * a[2]
        x_real, x_imag = torch.chunk(x, 2, dim=dim)
        x = torch.complex(x_real, x_imag)
        if self.window_size_fft > 0 and (H != self.window_size_fft or W != self.window_size_fft):
            x = window_reversex(x, self.window_size_fft, H, W // 2 + 1, batch_list)
            x = torch.fft.irfft2(x, s=(H, W), norm=self.norm)
        else:
            x = torch.fft.irfft2(x, s=(H, W), norm=self.norm)

        x = torch.sin(x)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x
class PATM_fft_act(nn.Module):
    def __init__(self, dim, window_size=-1, window_size_fft=7, qkv_bias=False, proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(2 * dim, 2 * dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, 2 * dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, 2 * dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        # self.reweight = Mlp(dim, dim // 4, dim * 3)
        # self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.norm = 'backward'
        # self.window_size = window_size
        # self.window_size_fft = window_size_fft
        # if mode == 'fc':
        #     self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        #     self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        # else:
        #     self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
        #                                       nn.BatchNorm2d(dim), nn.ReLU())
        #     self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
        #                                       nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        # theta_h = self.theta_h_conv(x)
        # theta_w = self.theta_w_conv(x)
        # if self.window_size_fft > 0 and (H != self.window_size_fft or W != self.window_size_fft):
        #     x, batch_list = window_partitionx(x, self.window_size_fft)
        #     B = x.shape[0]
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        # x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        # x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        hwc = torch.cat([x_h, x_w, x], dim=1)
        hwc = torch.fft.rfft2(hwc, norm=self.norm)
        x_h, x_w, x_c = torch.chunk(hwc, 3, dim=1)
        # x_h = torch.fft.rfft2(x_h, norm=self.norm)
        # x_w = torch.fft.rfft2(x_w, norm=self.norm)
        # x = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        x_h = torch.cat([x_h.real, x_h.imag], dim=dim)
        x_w = torch.cat([x_w.real, x_w.imag], dim=dim)
        x_c = torch.cat([x_c.real, x_c.imag], dim=dim)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x_c)
        h_real, h_imag = torch.chunk(h, 2, dim=dim)
        h = torch.complex(h_real, h_imag)
        # h = torch.fft.irfft2(h, s=(H, W), norm=self.norm)
        w_real, w_imag = torch.chunk(w, 2, dim=dim)
        w = torch.complex(w_real, w_imag)
        # w = torch.fft.irfft2(w, s=(H, W), norm=self.norm)
        c_real, c_imag = torch.chunk(c, 2, dim=dim)
        c = torch.complex(c_real, c_imag)
        hwc = torch.cat([h, w, c], dim=1)
        hwc = torch.fft.irfft2(hwc, s=(H, W), norm=self.norm)
        hwc = torch.sin(hwc)
        h, w, c = torch.chunk(hwc, 3, dim=1)
        x = h * w * c

        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x
class DCT_Conv(nn.Module):
    def __init__(self, dim, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1,
                      groups=dim, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = self.dct(x)
        y = self.conv(y)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = self.idct(y)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = self.idct(y)
        return y
class DCT_PatchMix(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, window_size=8, num_heads=1):
        super(DCT_PatchMix, self).__init__()

        self.window_size = window_size
        self.num_heads = num_heads
        self.in_channels = in_channels # // num_heads
        self.n = window_size ** 2

        self.GELU = nn.GELU()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            # nn.BatchNorm2d(in_channels),
        )

        self.dct = DCT2(window_size)
        self.idct = IDCT2(window_size)
        self.dct_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1, groups=in_channels, padding_mode='reflect'),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            # nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        # _tmp = x
        x = self.fc1(x)
        # B, C, H, W = x.shape

        x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w',
                      h=self.window_size, w=self.window_size)
        # if self.window_size > 0 and (H != self.window_size or W != self.window_size):
        #     x, batch_list = window_partitionx(x, self.window_size)

        x = self.dct(x)
            # print(x.shape)
        # x = torch.cat([x.real, x.imag], dim=1)
        x = rearrange(x, 'b c k1 k2 h w -> (b h w) c k1 k2')

        x = self.dct_conv(x)

        x = rearrange(x, '(b h w) c k1 k2 -> b c k1 k2 h w', h=self.window_size, w=self.window_size)
        x = self.idct(x)
        # print(x)
        # if self.window_size > 0 and (H != self.window_size or W != self.window_size):
        #     x = window_reversex(x, self.window_size, H, W, batch_list)

        # x = rearrange(x, '(b head) c k1 k2 h w -> b (head c) (k1 h) (k2 w)',
        #               head=self.num_heads, h=self.window_size, w=self.window_size)
        x = rearrange(x, 'b c k1 k2 h w -> b c (k1 h) (k2 w)',
                      h=self.window_size, w=self.window_size)
        x = self.GELU(x)
        # print(x)
        x = self.fc2(x)
        return x
class DCT_act(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.window_size = window_size
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = self.dct(x)
        y = self.act(y)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = self.idct(y)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = self.idct(y)
        return y
class Mlp_hwc(nn.Module):
    def __init__(self, dim, window_size=8, bias=False):
        super().__init__()
        self.mlp_c = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=bias)
        self.mlp_h = nn.Linear(window_size, window_size, bias)
        self.mlp_w = nn.Linear(window_size, window_size, bias)

    def forward(self, x):
        x_c = self.mlp_c(x)
        x_w = self.mlp_w(x)
        x_h = rearrange(x, 'b c h w -> b c w h')
        x_h = self.mlp_h(x_h)
        x_h = rearrange(x_h, 'b c w h -> b c h w')
        y = x_c + x_h + x_w
        return y
class DCT_Mlp_hwc(nn.Module):
    def __init__(self, dim, window_size=8, bias=False):
        super().__init__()
        self.window_size = window_size
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.mlp1 = Mlp_hwc(dim, window_size, bias=bias)
        self.mlp2 = Mlp_hwc(dim, window_size, bias=bias)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = self.dct(x)
        y = self.mlp1(y)
        y = self.act(y)
        y = self.mlp2(y)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = self.idct(y)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = self.idct(y)
        return y
class DCT_Mlp_win_a(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=1, bias=False):
        super().__init__()
        self.window_size = window_size
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        self.mlp = Win_Conv(dim, dim, window_size, bias=bias, num_heads=num_heads)

    def forward(self, x):
        _, _, H, W = x.shape
        x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w',
                      h=self.window_size, w=self.window_size)
        x = self.dct(x)
        x = rearrange(x, 'b c k1 k2 h w -> b (c h w) k1 k2')
        x = self.mlp(x)
        x = rearrange(x, 'b (c h w) k1 k2 -> b c k1 k2 h w', h=self.window_size, w=self.window_size)
        x = self.idct(x)
        x = rearrange(x, 'b c k1 k2 h w -> b c (k1 h) (k2 w)',
                      h=self.window_size, w=self.window_size)
        return x
class Win_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, bias=False, num_heads=1):
        super().__init__()
        n = window_size ** 2
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * n, out_channels * n, 1, bias=bias, groups=num_heads),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * n, out_channels * n, 1, bias=bias, groups=num_heads)
        )

    def forward(self, x):
        return self.nn(x)
class DCT_mlp(nn.Module):
    def __init__(self, dim, window_size=8, norm='backward'):
        super().__init__()
        self.norm = norm
        self.window_size = window_size
        self.dct = DCT2(window_size=window_size)
        self.idct = IDCT2(window_size=window_size)
        w2 = window_size * window_size
        self.mlp = nn.Linear(w2, w2, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = self.dct(x)
        b, c, h, w = y.shape
        y = rearrange(y, 'b c h w -> (b c) (h w)')
        y = self.mlp(y)
        y = rearrange(y, '(b c) (h w) -> b c h w', c=c, h=h, w=w)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = self.idct(y)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = self.idct(y)
        return y


# Attn
class Attention_win(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=self.window_size,
                        w=self.window_size)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        out = self.project_out(out)
        return out
class Attention_win_plus(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = rearrange(qkv, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h=self.window_size, w=self.window_size)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, '(b d) (head c) h w -> b head (c d) (h w)', b=b, head=self.num_heads)
        k = rearrange(k, '(b d) (head c) h w -> b head (c d) (h w)', b=b, head=self.num_heads)
        v = rearrange(v, '(b d) (head c) h w -> b head (c d) (h w)', b=b, head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c h1 w1) (h w) -> b (head c) (h1 h) (w1 w)',
                        head=self.num_heads,
                        h1=H//self.window_size, w1=W//self.window_size,
                        h=self.window_size, w=self.window_size)
        out = self.project_out(out)
        return out
class Attention_winx_plus(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8, window_size_local=32):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.window_size_local = window_size_local
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        if self.window_size_local is not None and (h != self.window_size_local or w != self.window_size_local):
            qkv, batch_list = window_partitionx(qkv, self.window_size_local)
        B, C, H, W = qkv.shape
        qkv = rearrange(qkv, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h=self.window_size, w=self.window_size)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)
        k = rearrange(k, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)
        v = rearrange(v, '(b d) (head c) h w -> b head (c d) (h w)', b=B, head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c h1 w1) (h w) -> b (head c) (h1 h) (w1 w)',
                        head=self.num_heads,
                        h1=H//self.window_size, w1=W//self.window_size,
                        h=self.window_size, w=self.window_size)
        if self.window_size_local is not None and (h != self.window_size_local or w != self.window_size_local):
            out = window_reversex(out, self.window_size_local, h, w, batch_list)
        out = self.project_out(out)
        return out
class Attention_winx_plus_nc(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8, window_size_local=32):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.window_size_local = window_size_local
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        if self.window_size_local is not None and (h != self.window_size_local or w != self.window_size_local):
            qkv, batch_list = window_partitionx(qkv, self.window_size_local)
        B, C, H, W = qkv.shape
        qkv = rearrange(qkv, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h=self.window_size, w=self.window_size)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, '(b d) c h w -> (b c) d (h w)', b=B)
        k = rearrange(k, '(b d) c h w -> (b c) d (h w)', b=B)
        v = rearrange(v, '(b d) c h w -> (b c) d (h w)', b=B)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, '(b c) h1 w1) (h w) -> b c (h1 h) (w1 w)',
                        b=B,
                        h1=H//self.window_size, w1=W//self.window_size,
                        h=self.window_size, w=self.window_size)
        if self.window_size_local is not None and (h != self.window_size_local or w != self.window_size_local):
            out = window_reversex(out, self.window_size_local, h, w, batch_list)
        out = self.project_out(out)
        return out
class Attention_win_hw(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()
    def forward(self, x):
        b, c, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=self.window_size,
                        w=self.window_size)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        out = self.act(out)
        out = self.project_out(out)
        return out

class Attention_win_maskpred(nn.Module):
    def __init__(self, dim, mask_dim=1, num_heads=1, bias=False, window_size=8):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1, 1))
        self.window_size = window_size
        self.qk = nn.Conv2d(dim, mask_dim + dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(mask_dim + dim, mask_dim + dim, kernel_size=3, stride=1, padding=1, groups=mask_dim + dim, bias=bias)
        # self.proj_out = nn.Linear(k_dim, 1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.num_heads = num_heads
        self.mask_dim = mask_dim
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        b, c, H, W = x.shape
        # print(x.shape)
        qk = self.qk_dwconv(self.qk(x))
        v = self.v(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qk, _ = window_partitionx(qk, self.window_size)
            v, batch_list = window_partitionx(v, self.window_size)
        # print(qk.shape)
        # q, k = qk.chunk(2, dim=1)
        q = qk[:, :self.mask_dim, :, :]
        k = qk[:, self.mask_dim:, :, :]
        if self.mask_dim == 1:
            q = torch.unsqueeze(q, 1)
        # bs, cq = q.shape[:2]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # print(q.shape, k.shape)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, c_mask, c_in]
        attn = attn.softmax(dim=-1)
        mask = (attn @ v)
        mask = rearrange(mask, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=self.window_size,
                         w=self.window_size)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            mask = window_reversex(mask, self.window_size, H, W, batch_list)
        mask = self.project_out(mask)
        mask = self.act(mask)
        return mask

class Conv_attn_maskpred(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8):
        super().__init__()
        self.window_size = window_size
        mask_dim = dim
        self.mask_pred = Attention_win_maskpred(dim, mask_dim, num_heads, bias, window_size=window_size)
        self.conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim, bias=bias)])
    def forward(self, x):
        mask = self.mask_pred(x)
        y = mask * x
        y = self.conv(y)
        return y
class Attention_win_get_weight(nn.Module):
    def __init__(self, dim, num_heads=1, k_dim=2, bias=False, window_size=8):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1, 1))
        self.window_size = window_size
        self.qk = nn.Conv2d(dim, k_dim + 1, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(k_dim + 1, k_dim + 1, kernel_size=3, stride=1, padding=1, groups=k_dim + 1, bias=bias)
        # self.proj_out = nn.Linear(k_dim, 1)
    def forward(self, x):
        b, c, H, W = x.shape
        # print(x.shape)
        qk = self.qk_dwconv(self.qk(x))

        q = qk[:, :-1, :, :]
        k = qk[:, -1, :, :]
        k = torch.unsqueeze(k, 1)
        # bs, cq = q.shape[:2]
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # print(q.shape, k.shape)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = torch.squeeze(attn, -1)
        # attn = rearrange(attn, 'b head c -> b (head c)', head=self.num_heads)
        attn = attn.softmax(dim=-1)  # b, k
        return attn
class Attention_win_dynamic_conv(nn.Module):
    def __init__(self, dim, k_dim, bias=False, window_size=8):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1, 1) / k_dim)
        self.window_size = window_size
        self.qk = nn.Conv2d(dim, k_dim + 1, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(k_dim + 1, k_dim + 1, kernel_size=3, stride=1, padding=1, groups=k_dim + 1, bias=bias)
        # self.proj_out = nn.Linear(k_dim, 1)
    def forward(self, x):
        b, c, H, W = x.shape
        # print(x.shape)
        qk = self.qk_dwconv(self.qk(x))
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qk, batch_list = window_partitionx(qk, self.window_size)
        # print(qk.shape)
        # q, k = qk.chunk(2, dim=1)
        q = qk[:, :-1, :, :]
        k = qk[:, -1, :, :]
        k = torch.unsqueeze(k, 1)
        # bs, cq = q.shape[:2]
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        # print(q.shape, k.shape)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # print(attn.shape)
        # attn = self.proj_out(attn)
        # print(attn.shape)
        attn = torch.squeeze(attn, -1)
        attn = attn.softmax(dim=-1)
        return attn

class Attention_dynamic(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if (init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if (self.temprature > 1):
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return F.softmax(att / self.temprature, -1)
class Attention_OD(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, groups, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.temprature = temprature
        assert c_in > ratio
        hidden_planes = c_in // ratio
        self.net = nn.Sequential(
                    nn.Conv2d(c_in, hidden_planes, kernel_size=1, bias=False),
                    nn.ReLU()
                    )
        self.fc_n = nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        self.fc_in = nn.Conv2d(hidden_planes, c_in // groups, kernel_size=1, bias=False)
        self.fc_out = nn.Conv2d(hidden_planes, c_out, kernel_size=1, bias=False)
        self.fc_k = nn.Conv2d(hidden_planes, kernel_size*kernel_size, kernel_size=1, bias=False)
        if (init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if (self.temprature > 1):
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att)
        b = x.shape[0]
        w_in = torch.sigmoid(self.fc_in(att).view(b, -1))
        w_out = torch.sigmoid(self.fc_out(att).view(b, -1))
        w_k = torch.sigmoid(self.fc_k(att).view(b, -1))
        w_n = torch.softmax(self.fc_n(att).view(b, -1) / self.temprature, -1)
        return w_in, w_out, w_k, w_n

class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1, grounps=1, bias=False, K=9,
                 init_weight=True, window_size=8):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.window_size = window_size
        self.init_weight = init_weight
        # self.attention = Attention_dynamic(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
        #                            init_weight=init_weight)
        self.attention = Attention_win_dynamic_conv(dim=in_planes, k_dim=K, window_size=window_size)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        b, in_planels, H, W = x.shape
        softmax_att = self.attention(x)  # bs,K
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)

        h, w = x.shape[-2:]
        bs = x.shape[0]
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        # print(weight.shape, softmax_att.shape)
        # aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
        #                                                       self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k
        # aggregate_weight = softmax_att @ weight
        aggregate_weight = torch.einsum('bk,kc->bc', softmax_att, weight)
        # print(aggregate_weight.shape)
        aggregate_weight = aggregate_weight.view(-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        # print(x.shape, aggregate_weight.shape)
        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(-1, self.out_planes, h, w)
        # print(output.shape, bs)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            output = window_reversex(output, self.window_size, H, W, batch_list)

        return output

class PhaseConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 init_weight=True, Phase=True):
        super().__init__()
        self.Phase = Phase
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        groups = in_planes
        self.groups = in_planes # groups
        # self.bias = bias
        # self.window_size = window_size
        self.init_weight = init_weight

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size * kernel_size),
                                   requires_grad=True)
        # if (bias):
        #     self.bias = nn.Parameter(torch.randn(out_planes), requires_grad=True)
        # else:
        #     self.bias = None
        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        # try softmax on kernel!
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            x = torch.fft.irfft2(torch.exp(1j * x))
        # print(x.shape)
        aggregate_weight = torch.softmax(self.weight, dim=-1)
        aggregate_weight = aggregate_weight.view(self.out_planes,
                                                 self.in_planes // self.groups,
                                                 self.kernel_size, self.kernel_size)
        # aggregate_weight_f = torch.fft.rfft2(aggregate_weight) # (self.weight)
        # aggregate_weight_f = torch.angle(aggregate_weight_f)
        # aggregate_weight = torch.fft.irfft2(torch.exp(1j * aggregate_weight_f), s=[self.kernel_size,self.kernel_size])

        # print(aggregate_weight)
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          groups=self.groups, dilation=self.dilation)
        if self.Phase:
            output = torch.fft.rfft2(output)
            output = torch.fft.irfft2(mag * output) # * torch.exp(1j * output))
        return output

class PhaseMagConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 init_weight=True, Phase=True, shift=True):
        super().__init__()
        self.Phase = Phase
        self.in_planes = in_planes*2
        self.out_planes = out_planes*2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.shift = shift
        groups = in_planes
        self.groups = in_planes * 2  # groups
        # self.bias = bias
        # self.window_size = window_size
        self.init_weight = init_weight

        self.weight = nn.Parameter(torch.randn(out_planes*2, in_planes // groups, kernel_size * kernel_size),
                                   requires_grad=True)
        # if (bias):
        #     self.bias = nn.Parameter(torch.randn(out_planes), requires_grad=True)
        # else:
        #     self.bias = None
        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        # try softmax on kernel!
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            mag = torch.fft.irfft2(mag)
            if self.shift:
                mag = torch.fft.fftshift(mag)
            x = torch.cat([torch.fft.irfft2(torch.exp(1j * x)), mag], dim=1)
        # print(x.shape)
        aggregate_weight = torch.softmax(self.weight, dim=-1)
        aggregate_weight = aggregate_weight.view(self.out_planes,
                                                 self.in_planes // self.groups,
                                                 self.kernel_size, self.kernel_size)
        # aggregate_weight_f = torch.fft.rfft2(aggregate_weight) # (self.weight)
        # aggregate_weight_f = torch.angle(aggregate_weight_f)
        # aggregate_weight = torch.fft.irfft2(torch.exp(1j * aggregate_weight_f), s=[self.kernel_size,self.kernel_size])

        # print(aggregate_weight)
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          groups=self.groups, dilation=self.dilation)
        if self.Phase:
            if self.shift:
                phase, mag = torch.chunk(output, 2, dim=1)
                mag = torch.fft.fftshift(mag)
                mag = torch.fft.rfft2(mag)
                phase = torch.fft.rfft2(phase)
            else:
                output = torch.fft.rfft2(output)
                phase, mag = torch.chunk(output, 2, dim=1)
            output = torch.fft.irfft2(mag * phase) # * torch.exp(1j * output))
        return output
class PhaseDyConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 init_weight=True, Phase=True, shift=True):
        super().__init__()
        self.Phase = Phase
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(in_planes, 1, groups=1, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # try softmax on kernel!
        B, C, H, W = x.shape
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            # mag = torch.fft.irfft2(mag)
            # if self.shift:
            #     mag = torch.fft.fftshift(mag)
            # x = torch.cat([torch.fft.irfft2(torch.exp(1j * x)), mag], dim=1)
            x = torch.fft.irfft2(torch.exp(1j * x))
        # print(x.shape)
        aggregate_weight = self.conv_kernel(x)
        pad = (self.kernel_size - 1) // 2
        k2 = self.kernel_size * self.kernel_size
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = F.unfold(x, self.kernel_size)
        aggregate_weight = F.pad(aggregate_weight, (pad, pad, pad, pad), mode="replicate")
        aggregate_weight = F.unfold(aggregate_weight, self.kernel_size)
        x = x.view(B, C, k2, -1)
        # aggregate_weight = aggregate_weight.view(B, C, k2, -1)
        # print(x.shape, aggregate_weight.shape)
        aggregate_weight = aggregate_weight.softmax(1)
        output = torch.einsum('bikj, bkj -> bij', x, aggregate_weight)
        # print(output.shape)
        output = F.fold(output, [H, W], 1)
        # print(output.shape)
        # output = aggregate_weight
        # output = torch.sum(torch.mul(x, aggregate_weight), -1)
        if self.Phase:
            # if self.shift:
            #     phase, mag = torch.chunk(output, 2, dim=1)
            #     mag = torch.fft.fftshift(mag)
            #     mag = torch.fft.rfft2(mag)
            #     phase = torch.fft.rfft2(phase)
            # else:
            #     output = torch.fft.rfft2(output)
            #     phase, mag = torch.chunk(output, 2, dim=1)
            phase = torch.fft.rfft2(output)
            output = torch.fft.irfft2(mag * phase) # * torch.exp(1j * output))
        return output
class PhaseDyConv2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 init_weight=True, Phase=True, shift=True):
        super().__init__()
        self.Phase = Phase
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(in_planes, in_planes, groups=in_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # try softmax on kernel!
        B, C, H, W = x.shape
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            # mag = torch.fft.irfft2(mag)
            # if self.shift:
            #     mag = torch.fft.fftshift(mag)
            # x = torch.cat([torch.fft.irfft2(torch.exp(1j * x)), mag], dim=1)
            x = torch.fft.irfft2(torch.exp(1j * x))
        # print(x.shape)
        aggregate_weight = self.conv_kernel(x)
        pad = (self.kernel_size - 1) // 2
        k2 = self.kernel_size * self.kernel_size
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = F.unfold(x, self.kernel_size)
        aggregate_weight = F.pad(aggregate_weight, (pad, pad, pad, pad), mode="replicate")
        aggregate_weight = F.unfold(aggregate_weight, self.kernel_size)
        x = x.view(B, C, k2, -1)
        aggregate_weight = aggregate_weight.view(B, C, k2, -1)
        # print(x.shape, aggregate_weight.shape)
        aggregate_weight = aggregate_weight.softmax(2)
        output = torch.einsum('bikj, bikj -> bij', x, aggregate_weight)
        # print(output.shape)
        output = F.fold(output, [H, W], 1)
        # print(output.shape)
        # output = aggregate_weight
        # output = torch.sum(torch.mul(x, aggregate_weight), -1)
        if self.Phase:
            # if self.shift:
            #     phase, mag = torch.chunk(output, 2, dim=1)
            #     mag = torch.fft.fftshift(mag)
            #     mag = torch.fft.rfft2(mag)
            #     phase = torch.fft.rfft2(phase)
            # else:
            #     output = torch.fft.rfft2(output)
            #     phase, mag = torch.chunk(output, 2, dim=1)
            phase = torch.fft.rfft2(output)
            output = torch.fft.irfft2(mag * phase) # * torch.exp(1j * output))
        return output
class PhaseDyConv3(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 init_weight=True, Phase=True, shift=True):
        super().__init__()
        self.Phase = Phase
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(in_planes, kernel_size*kernel_size, groups=1, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # try softmax on kernel!
        B, C, H, W = x.shape
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            # mag = torch.fft.irfft2(mag)
            # if self.shift:
            #     mag = torch.fft.fftshift(mag)
            # x = torch.cat([torch.fft.irfft2(torch.exp(1j * x)), mag], dim=1)
            x = torch.fft.irfft2(torch.exp(1j * x))
        # print(x.shape)
        aggregate_weight = self.conv_kernel(x)
        pad = (self.kernel_size - 1) // 2
        k2 = self.kernel_size * self.kernel_size
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = F.unfold(x, self.kernel_size)
        # aggregate_weight = F.pad(aggregate_weight, (pad, pad, pad, pad), mode="replicate")
        aggregate_weight = F.unfold(aggregate_weight, 1)
        x = x.view(B, C, k2, -1)
        # aggregate_weight = aggregate_weight.view(B, k2, -1)
        # print(x.shape, aggregate_weight.shape)
        aggregate_weight = aggregate_weight.softmax(1)
        output = torch.einsum('bikj, bkj -> bij', x, aggregate_weight)
        # print(output.shape)
        output = F.fold(output, [H, W], 1)
        # print(output.shape)
        # output = aggregate_weight
        # output = torch.sum(torch.mul(x, aggregate_weight), -1)
        if self.Phase:
            # if self.shift:
            #     phase, mag = torch.chunk(output, 2, dim=1)
            #     mag = torch.fft.fftshift(mag)
            #     mag = torch.fft.rfft2(mag)
            #     phase = torch.fft.rfft2(phase)
            # else:
            #     output = torch.fft.rfft2(output)
            #     phase, mag = torch.chunk(output, 2, dim=1)
            phase = torch.fft.rfft2(output)
            output = torch.fft.irfft2(mag * phase) # * torch.exp(1j * output))
        return output
class PhaseDyConv4(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,
                 init_weight=True, Phase=True, shift=True):
        super().__init__()
        self.Phase = Phase
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(in_planes, in_planes*kernel_size*kernel_size, groups=in_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # try softmax on kernel!
        B, C, H, W = x.shape
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            # mag = torch.fft.irfft2(mag)
            # if self.shift:
            #     mag = torch.fft.fftshift(mag)
            # x = torch.cat([torch.fft.irfft2(torch.exp(1j * x)), mag], dim=1)
            x = torch.fft.irfft2(torch.exp(1j * x))
        # print(x.shape)
        aggregate_weight = self.conv_kernel(x)
        pad = (self.kernel_size - 1) // 2
        k2 = self.kernel_size * self.kernel_size
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = F.unfold(x, self.kernel_size)
        # aggregate_weight = F.pad(aggregate_weight, (pad, pad, pad, pad), mode="replicate")
        aggregate_weight = F.unfold(aggregate_weight, 1)
        x = x.view(B, C, k2, -1)
        aggregate_weight = aggregate_weight.view(B, C, k2, -1)
        # print(x.shape, aggregate_weight.shape)
        aggregate_weight = aggregate_weight.softmax(2)
        output = torch.einsum('bikj, bikj -> bij', x, aggregate_weight)
        # print(output.shape)
        output = F.fold(output, [H, W], 1)
        # print(output.shape)
        # output = aggregate_weight
        # output = torch.sum(torch.mul(x, aggregate_weight), -1)
        if self.Phase:
            # if self.shift:
            #     phase, mag = torch.chunk(output, 2, dim=1)
            #     mag = torch.fft.fftshift(mag)
            #     mag = torch.fft.rfft2(mag)
            #     phase = torch.fft.rfft2(phase)
            # else:
            #     output = torch.fft.rfft2(output)
            #     phase, mag = torch.chunk(output, 2, dim=1)
            phase = torch.fft.rfft2(output)
            output = torch.fft.irfft2(mag * phase) # * torch.exp(1j * output))
        return output
class PhaseMagConvx(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1,Phase=True):
        super().__init__()
        self.Phase = Phase

        groups = in_planes * 2
        self.conv = nn.Conv2d(in_planes * 2, out_planes * 2, groups=groups, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
    def forward(self, x):
        # try softmax on kernel!
        if self.Phase:
            x_f = torch.fft.rfft2(x)
            mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
            x = torch.angle(x_f)
            x = torch.cat([torch.fft.irfft2(torch.exp(1j * x)), torch.fft.irfft2(mag)], dim=1)
        output = self.conv(x)
        if self.Phase:
            output = torch.fft.rfft2(output)
            phase, mag = torch.chunk(output, 2, dim=1)
            output = torch.fft.irfft2(mag * phase) # * torch.exp(1j * output))
        return output
class ResPhase(nn.Module):
    def __init__(self, n_features, act_layer=nn.ReLU):
        super().__init__()

        self.act = act_layer(inplace=True)

        self.conv1 = PhaseConv(n_features, n_features)
        self.conv2 = PhaseConv(n_features, n_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
class Phase_Act(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x_f = torch.fft.rfft2(x)
        mag = torch.abs(x_f)
        x = torch.angle(x_f)
        x = torch.fft.irfft2(torch.exp(1j * x))
        x = torch.sin(x)
        x = torch.fft.rfft2(x)
        x = torch.fft.irfft2(mag * x)
        return x
class Phase_Act2(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        x = torch.complex(x_real, x_imag)
        mag = torch.abs(x)
        x = torch.angle(x)
        x = torch.fft.irfft2(torch.exp(1j * x))
        x = torch.sin(x)
        x = torch.fft.rfft2(x)
        x = mag * x
        x = torch.cat([x.real, x.imag], dim=1)
        return x
class Phase_Act3(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        x = torch.complex(x_real, x_imag)
        mag = torch.abs(x)
        x = torch.angle(x)
        x = torch.fft.irfft2(torch.exp(1j * x))
        x = torch.sinh(x)
        x = torch.fft.rfft2(x)
        x = mag * x
        x = torch.cat([x.real, x.imag], dim=1)
        return x
class PhaseMLP(nn.Module):
    def __init__(self, n_feature, window_size=8, init_weight=True):
        super().__init__()
        self.n_feature = n_feature
        self.window_size = window_size
        # self.bias = bias
        # self.window_size = window_size
        self.init_weight = init_weight

        self.weight = nn.Parameter(torch.randn(n_feature, window_size * window_size),
                                   requires_grad=True)
        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        # try softmax on kernel!
        _, _, H, W = x.shape
        x = rearrange(x, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h=self.window_size, w=self.window_size)
        x_f = torch.fft.rfft2(x)
        mag = torch.abs(x_f)  # (torch.fft.rfft2(inp, norm=self.fnorm))
        x = torch.angle(x_f)
        # x = torch.fft.irfft2(torch.exp(1j * x))
        # print(x.shape)
        aggregate_weight = torch.softmax(self.weight, dim=-1)
        aggregate_weight = aggregate_weight.view(self.n_feature, self.window_size, self.window_size)
        aggregate_weight_f = torch.fft.rfft2(aggregate_weight) # (self.weight)
        aggregate_weight_f = torch.angle(aggregate_weight_f)
        # aggregate_weight = torch.fft.irfft2(torch.exp(1j * aggregate_weight_f), s=[self.window_size, self.window_size])
        # aggregate_weight = torch.fft.rfft2(aggregate_weight)
        x = torch.exp(1j * (torch.sin(aggregate_weight_f) * x))

        # output = torch.fft.rfft2(output)
        # print(output.shape)
        # output = torch.abs(torch.log(output) / 1j)
        # print(mag.shape, output.shape)
        output = torch.fft.irfft2(mag * x)# * torch.exp(1j * output))
        output = rearrange(output, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=H//self.window_size, w1=W//self.window_size)
        return output
class Phase_Attention_win_hw(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=self.window_size,
                        w=self.window_size)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        out = self.project_out(out)
        return out
class ODConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=False, K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention_OD(in_channels, out_channels, kernel_size=kernel_size, groups=groups,
                                      ratio=ratio, K=K, temprature=temprature,
                                      init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels // groups, kernel_size * kernel_size),
                                   requires_grad=True)
        # if (bias):
        #     self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        # else:
        #     self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        w_in, w_out, w_k, w_n = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight  #n d c k # .view(self.K, -1)  # K,-1
        # print(w_in.shape, w_out.shape, w_k.shape, w_n.shape, weight.shape)
        aggregate_weight = torch.einsum('ndck,bk,bc,bd,bn->bdck', weight, w_k, w_in, w_out, w_n)
        # aggregate_weight = torch.einsum('bn,ndck->bdck', w_n, weight)
        # print(aggregate_weight.shape)
        # # x0, x1, x2, x3 = aggregate_weight.shape # w_in.expand(bs, w_in.shape[1], x2, x3) * aggregate_weight #
        # aggregate_weight = torch.einsum('bc,bdck->bdck', w_in, aggregate_weight)
        # aggregate_weight = torch.einsum('bd,bdck->bdck', w_out, aggregate_weight)
        # aggregate_weight = torch.einsum('bk,bdck->bdck', w_k, aggregate_weight)

        aggregate_weight = aggregate_weight.view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k


        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output
class DynamicConv_ori(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention_dynamic(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                   init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output

# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)
if __name__ == '__main__':
    inp = torch.randn((2, 3, 16, 16))
    # net = DCT_block(64, window_size=8)
    net = spatial_fourier_fastconv(3, 3, kernel_size=1)
    # idctx = IDCT2x(32, 16)
    # dctx = DCT2(32)
    # idctx = IDCT2(32)
    x = net(inp)

    # y = idctx(x)
    print(torch.mean(x - inp))
    # net = FFT_complex_GrapherModule_sgv2(3, 6, m_m='max')# , kernel_size=3, stride=1, padding=1)
    # out = net(inp)
    # out = window_pad(inp, window_size=64)
    # print(out.shape)
