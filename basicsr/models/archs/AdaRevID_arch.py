
import os.path

import kornia
import torch
import torch.nn as nn

from basicsr.models.archs.modules.revcol_function import ReverseFunction

from basicsr.models.archs.baseblock import *
from timm.models.layers import trunc_normal_
from collections import OrderedDict
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.my_module import code_extra_mean_var
from basicsr.models.archs.Flow_arch import KernelPrior
from basicsr.models.archs.UFPNet.UFPNet_code_uncertainty_arch import UFPNet, UFPNetLocal
from basicsr.models.losses.losses import AdaptiveLoss

# from torch.cuda.amp import autocast, GradScaler


class NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., attn_type='SCA'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.attn_type = attn_type
        # Simplified Channel Attention
        if attn_type == 'SCA':
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        elif attn_type == 'grid_dct_SE':
            self.sca = WDCT_SE(dw_channel // 2, num_heads=num_heads, bias=True, window_size=window_size)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        if self.attn_type == 'SCA':
            x = x * self.sca(x)
        elif self.attn_type == 'grid_dct_SE':
            x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class FCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        # x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))
        x = x * self.sca(self.avg(x))
        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))
        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=21):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, inpx):
        inp, kernel = inpx
        x = inp
        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class Fusion_DecoderV0(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, sub_idx=1, fft_bench=False) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # self.down = nn.Sequential(
        #     nn.Conv2d(channels[level + 1], channels[level], kernel_size=2, stride=2),
        #     # LayerNorm(channels[level]),
        # ) if level in [0, 1, 2] else nn.Identity()
        # if not first_col:
        # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
        # self.up = UpsampleX(channels[level], channels[level+1]) if level in [0, 1, 2] else nn.Identity()
        self.up = nn.Sequential(nn.Conv2d(channels[level] * 2, channels[level] * 4, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1, 2, 3] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.level == 3:
            # print(c_down.shape)
            x = self.up(c_down)
        else:
            # print(c_up.shape, c_down.shape)
            x = self.up(c_down)
        return x

class Fusion_DecoderV1(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, sub_idx=1, fft_bench=False) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level + 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level]),
        ) if level in [0, 1, 2] else nn.Identity()
        # if not first_col:
        # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
        # self.up = UpsampleX(channels[level], channels[level+1]) if level in [0, 1, 2] else nn.Identity()
        self.up = nn.Sequential(nn.Conv2d(channels[level] * 2, channels[level] * 4, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1, 2, 3] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.level == 3:
            # print(c_down.shape)
            x = self.up(c_down)
        else:
            # print(c_up.shape, c_down.shape)
            x = self.down(c_up) + self.up(c_down)
        return x


class Level_Decoder(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66, sub_idx=1) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        fft_bench = False
        self.fusion = Fusion_DecoderV1(level, channels, first_col, train_size, num_heads[level],
                                       ffn_expansion_factor=ffn_expansion_factor, sub_idx=sub_idx, fft_bench=fft_bench)
        # self.fusion = Fusion_DecoderV0(level, channels, first_col, train_size, num_heads[level],
        #                                ffn_expansion_factor=ffn_expansion_factor, sub_idx=sub_idx)

        if baseblock == 'naf':
            modules = [NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fcnaf':
            modules = [FCNAFBlock(channels[level], num_heads=num_heads[level], sub_idx=sub_idx) for i in range(layers[level])]

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = in_channels//2
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),  # final norm layer
            # nn.Softmax(-1),
            # nn.Linear(in_channels, num_classes),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_classes)
            # nn.Softmax(-1)
            # nn.Tanh()
        )

        self.alpha = nn.Parameter(torch.ones((1, num_classes)) / math.sqrt(num_classes),
                                  requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        x = self.avgpool(x)
        # x = F.adaptive_avg_pool2d(x, [1, 1])
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.classifier(x) * self.alpha  # * self.alpha0
        # print('ics: ', x)
        return x  # torch.softmax(x, -1)


class UniDecoder(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 2.66  # 1. + 1.66 * sub_idx
        num_heads = [8, 4, 2, 1]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Decoder(0, channels, layers, kernel_size, first_col, dp_rates[0], num_heads, baseblock,
                                    train_size, ffn_expansion_factor, sub_idx)

        self.level1 = Level_Decoder(1, channels, layers, kernel_size, first_col, dp_rates[1], num_heads, baseblock,
                                    train_size, ffn_expansion_factor, sub_idx)

        self.level2 = Level_Decoder(2, channels, layers, kernel_size, first_col, dp_rates[2], num_heads, baseblock,
                                    train_size, ffn_expansion_factor, sub_idx)

        self.level3 = Level_Decoder(3, channels, layers, kernel_size, first_col, dp_rates[3], num_heads, baseblock,
                                    train_size, ffn_expansion_factor, sub_idx)

    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)

        state_dict = checkpoint["params"]
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        encoder3_state_dict = OrderedDict()
        encoder4_state_dict = OrderedDict()
        up1_state_dict = OrderedDict()
        up2_state_dict = OrderedDict()
        up3_state_dict = OrderedDict()
        up4_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)

            # v = repeat(v, )
            if k[:8] == 'decoders':
                name_a = k[9:]  # remove `module.`
                name_1 = int(name_a[0])
                name = k[11:]
                # print(v.shape)
                if name_1 == 0:
                    encoder1_state_dict[name] = v
                elif name_1 == 1:
                    encoder2_state_dict[name] = v
                elif name_1 == 2:
                    encoder3_state_dict[name] = v
                elif name_1 == 3:
                    encoder4_state_dict[name] = v
                    # print(name)
            elif k[:3] == 'ups':
                name_a = k[4:]
                idx = name_a.split('.')
                name = '0' + k[7:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    up1_state_dict[name] = v
                elif name_1 == 1:
                    up2_state_dict[name] = v
                elif name_1 == 2:
                    up3_state_dict[name] = v
                elif name_1 == 3:
                    up4_state_dict[name] = v

        self.level0.blocks.load_state_dict(encoder1_state_dict, strict=True)
        self.level1.blocks.load_state_dict(encoder2_state_dict, strict=True)
        self.level2.blocks.load_state_dict(encoder3_state_dict, strict=True)
        self.level3.blocks.load_state_dict(encoder4_state_dict, strict=True)
        self.level0.fusion.up.load_state_dict(up1_state_dict, strict=True)
        self.level1.fusion.up.load_state_dict(up2_state_dict, strict=True)
        self.level2.fusion.up.load_state_dict(up3_state_dict, strict=True)
        self.level3.fusion.up.load_state_dict(up4_state_dict, strict=True)

        print('-----------load pretrained decoder from ' + state_dict_pth + '----------------')

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign




class NAFNet(nn.Module):

    def __init__(self, width=64, img_channel=3, out_channels=3, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], train_size=None,
                 num_heads_e=[1, 2, 4, 8], num_heads_m=[16], window_size_e=[64, 32, 16, 8], window_size_m=[8],
                 window_size_e_fft=[-1, -1, -1, -1], window_size_m_fft=[-1],
                 return_feat=False, attn_type='SCA'):
        super().__init__()
        self.grid = True
        self.train_size = train_size
        self.overlap_size = (32, 32)
        print(self.overlap_size)
        self.kernel_size = [train_size, train_size]
        self.return_feat = return_feat

        num_heads_d = num_heads_e[::-1]  # [8, 4, 2, 1]
        window_size_d = window_size_e[::-1]
        window_size_d_fft = window_size_e_fft[::-1]

        shift_size_e = [0, 0, 0, -1]
        shift_size_m = [-1]
        # shift_size_d = [-1, -1, -1, -1]
        shift_size_d = shift_size_e[::-1]

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        # self.fft_head = FFT_head()
        self.encoders = nn.ModuleList()

        self.middle_blks = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i in range(len(enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, num_heads_e[i], window_size_e[i], window_size_e_fft[i], shift_size_e[i],
                               attn_type=attn_type) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        # print(NAFBlock)
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, num_heads_m[0], window_size_m[0], window_size_m_fft[0], shift_size_m[0],
                           attn_type=attn_type) for _ in range(middle_blk_num)]
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        # print(x.shape)
        x = self.middle_blks(x)
        # encs.reverse()
        encs.append(x)
        return encs

class AdaRevID(nn.Module):
    def __init__(self, width=64, in_channels=3, out_channels=3,
                 decoder_layers=[1, 1, 1, 1], baseblock=['naf'],
                 kernel_size=19, drop_path=0.0, train_size=[256, 256], overlap_size=[32, 32],  # save_memory=True,
                 save_memory_decoder=False, encoder='UFPNet',
                 pretrain=False,  # stem_method='stem_multi',
                 state_dict_pth_encoder=None,
                 # '/home/ubuntu/106-48t/personal_data/mxt/exp_results/ckpt/UFPNet/train_on_GoPro/net_g_latest.pth'
                 state_dict_pth_classifier=None,
                 state_dict_pth_decoder=None,
                 combine_train=False, test_only=True, imgloss=False, cal_num_decoders=False, dx='fd4_d4', classifier_type='hard_simple', num_classes=5,
                 exit_threshold=0.5, save_feature=False, exit_idx=4, decoder_idx=4, return_cls=False) -> None:
        super().__init__()
        # self.num_subnet = 2
        self.save_feature = save_feature
        self.return_cls = return_cls
        self.classifier_type = classifier_type
        self.imgloss = imgloss
        self.num_subdenet = len(baseblock)
        self.channels = width
        self.test_only = test_only
        self.exit_threshold = exit_threshold
        self.cal_num_decoders = cal_num_decoders
        self.num_decoders = 0
        self.pretrain = pretrain
        # self.use_amp = use_amp
        # self.num_subnet = num_subnet
        self.combine_train = combine_train
        channels = [width, width * 2, width * 4, width * 8]
        middle_blk_num = 1
        chan = 2 * channels[-1]
        # self.downs_mid = nn.Conv2d(channels[-1], chan, 2, 2)
        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock(chan) for _ in range(middle_blk_num)]
        #     )
        self.dx = dx

        if not self.pretrain or state_dict_pth_classifier is not None or return_cls:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.classifier_type == 'hard_simple':
                if dx == 'e4':
                    self.ape = nn.Sequential(
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
                elif dx == 'e3':
                    self.ape = nn.Sequential(
                        nn.Conv2d(channels[-1], chan, 2, 2),
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
            elif self.classifier_type == 'single':
                if dx == 'fd4_d4':
                    self.ape = Classifier(2 * channels[-1], 1)
                elif dx == 'fd4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3':
                    self.ape = Classifier(channels[-2], 1)
                elif dx == 'fd2':
                    self.ape = Classifier(channels[-3], 1)
                elif dx == 'fd1':
                    self.ape = Classifier(channels[0], 1)
                elif dx == '_d4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3_d4':
                    self.ape = Classifier(channels[-2]+channels[-1], 1)
                elif dx == 'fd2_d4':
                    self.ape = Classifier(channels[-3]+channels[-1], 1)
                elif dx == 'fd1_d4':
                    self.ape = Classifier(channels[0]+channels[-1], 1)
            else:
                if dx == 'fd4_d4':
                    self.ape = nn.ModuleList([Classifier(2 * channels[-1], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd3':
                    self.ape = nn.ModuleList([Classifier(channels[-2], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd2':
                    self.ape = nn.ModuleList([Classifier(channels[-3], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd1':
                    self.ape = nn.ModuleList([Classifier(channels[0], 1) for _ in range(self.num_subdenet-1)])
                elif dx == '_d4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd3_d4':
                    self.ape = nn.ModuleList([Classifier(channels[-2]+channels[-1], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd2_d4':
                    self.ape = nn.ModuleList([Classifier(channels[-3]+channels[-1], 1) for _ in range(self.num_subdenet-1)])
                elif dx == 'fd1_d4':
                    self.ape = nn.ModuleList([Classifier(channels[0]+channels[-1], 1) for _ in range(self.num_subdenet-1)])
        if state_dict_pth_classifier is not None:
            self.load_classifier(state_dict_pth_classifier)
        self.subdenets = nn.ModuleList()
        # self.encoder_feature_projs = nn.ModuleList()
        self.conv_features = nn.ParameterList()
        # print(channels)
        h, w = train_size
        # for i in range(len(channels)):
        #     self.encoder_feature_projs.append(MLP_UHD([h, w, channels[i]], 1))
        #     h, w = h//2, w//2
        # self.encoder_feature_projs.append(MLP_UHD([h, w, chan], 4))
        for i in range(len(channels)):
            self.conv_features.append(
                nn.Parameter(torch.ones((1, channels[i], 1, 1)),
                             requires_grad=True))
        self.conv_features.append(
            nn.Parameter(torch.ones((1, chan, 1, 1)),
                         requires_grad=True))

        if not self.pretrain and not self.combine_train:
            # for k, v in self.middle_blks.named_parameters():
            #     v.requires_grad = False
            # for k, v in self.downs_mid.named_parameters():
            #     v.requires_grad = False
            # for k, v in self.prompts.named_parameters():
            #     v.requires_grad = False
            for k, v in self.conv_features.named_parameters():
                v.requires_grad = False
        dp_rate = [x.item() for x in torch.linspace(drop_path, 0, len(decoder_layers))]

        channels.reverse()
        for i in range(self.num_subdenet):
            first_col = False
            self.subdenets.append(UniDecoder(channels=channels, layers=decoder_layers, kernel_size=kernel_size,
                                             first_col=first_col, dp_rates=dp_rate, save_memory=save_memory_decoder,
                                             baseblock=baseblock[i], train_size=train_size, sub_idx=i+1)) # (i+1)/self.num_subdenet
        if state_dict_pth_decoder is not None:
            self.subdenets[0].load_pretain_model(state_dict_pth_decoder)
        if not self.pretrain and not self.combine_train:
            # for k, v in self.subnets.named_parameters():
            #     v.requires_grad = False
            for k, v in self.subdenets.named_parameters():
                v.requires_grad = False
        self.encoder_method = encoder
        if encoder == 'UFPNet':
            self.encoder = UFPNet(width=width)
        # elif encoder == 'FFTformer':
        #     self.encoder = fftformerEncoder()
        #     self.encoderX = fftformerEncoderPro()
        else:
            self.encoder = NAFNet(width=width)
        if state_dict_pth_encoder is not None:
            self.load_pretain_model(state_dict_pth_encoder)

        for k, v in self.encoder.named_parameters():
            v.requires_grad = False
        # for k, v in self.subdenet0.named_parameters():
        #     print(v.requires_grad)
        self.encoder.eval()

        self.outputs = nn.ModuleList()

        for i in range(self.num_subdenet):
            self.outputs.append(nn.Sequential(
                nn.Conv2d(width, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            ))
            # self.outputs.append(SAM(width, chan))
        if not self.pretrain and not self.combine_train:
            if encoder == 'FFTformer':
                for k, v in self.encoderX.named_parameters():
                    v.requires_grad = False
            for k, v in self.outputs.named_parameters():
                v.requires_grad = False
        self.x_num_iter = 0
    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)
        # print(checkpoint.keys())
        if self.encoder_method == 'FFTformer':
            state_dict = checkpoint
        else:
            state_dict = checkpoint["params"]

        self.encoder.load_state_dict(state_dict, strict=False)

        print('-----------load pretrained encoder from ' + state_dict_pth + '----------------')
    def load_classifier(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)
        # print(checkpoint.keys())

        # state_dict = checkpoint["params"]

        self.ape.load_state_dict(checkpoint, strict=False)

        print('-----------load classifier from ' + state_dict_pth + '----------------')
    # def forward(self, img):
    #     if self.use_amp:
    #         with autocast():
    #             return self.forward_main(img)
    #     else:
    #         return self.forward_main(img)
    def forward(self, img):

        with torch.no_grad():
            # if self.encoder_method == 'FFTformer':
            #     e0, e1, e2 = self.encoder(img)
            #
            # else:
            e0, e1, e2, e3, e4 = self.encoder(img)
        # if self.encoder_method == 'FFTformer':
        #     e3, e4 = self.encoderX(e2)
        # print(self.conv_features[0].dtype)
        features = [self.conv_features[0] * (e0),
                    self.conv_features[1] * (e1),
                    self.conv_features[2] * (e2),
                    self.conv_features[3] * (e3),
                    self.conv_features[4] * (e4)]
        # features = [self.encoder_feature_projs[0](e0),
        #             self.encoder_feature_projs[1](e1),
        #             self.encoder_feature_projs[2](e2),
        #             self.encoder_feature_projs[3](e3),
        #             self.encoder_feature_projs[4](e4)]
        # features = [e0, e1, e2, e3, self.conv_features[0] * (e4)]
        if self.pretrain:
            return self._forward_pretrain(img, features)
        else:
            return self._forward(img, features)

    def _forward(self, img, features):
        x_tmp_out_list = []
        e0, e1, e2, e3, e4 = features
        # ics = self.ape(e4)
        b, c, h, w = e0.shape
        ics = []
        num_tmp = 0
        if self.classifier_type == 'hard_simple' and not self.test_only:
            if self.dx == 'e4':
                return self.ape(e4)
            elif self.dx == 'e3':
                return self.ape(e3)
        else:
            e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)

            if not self.test_only:

                if not self.imgloss:
                    x_tmp_out_list.append(self.outputs[0](e0) + img)
                    for i in range(1, self.num_subdenet):
                        e3_pre = e3
                        e3 = (self.subdenets[i].alpha0) * e3 + self.subdenets[i].level0(e4, e2)
                        if self.dx == 'fd4_d4':
                            dc = torch.cat([e3, e3_pre], 1)
                        elif self.dx == 'fd4':
                            dc = e3_pre
                        elif self.dx == 'fd3':
                            dc = e2
                        elif self.dx == 'fd2':
                            dc = e1
                        elif self.dx == 'fd1':
                            dc = e0
                        elif self.dx == '_d4':
                            dc = e3
                        elif self.dx == 'fd3_d4':
                            dc = torch.cat([self.avgpool(e2), self.avgpool(e3)], 1)
                        elif self.dx == 'fd2_d4':
                            dc = torch.cat([self.avgpool(e1), self.avgpool(e3)], 1)
                        elif self.dx == 'fd1_d4':
                            dc = torch.cat([self.avgpool(e0), self.avgpool(e3)], 1)
                        if self.classifier_type == 'single':
                            ics.append(self.ape(dc))
                        else:
                            ics.append(self.ape[i-1](dc))
                        e2 = (self.subdenets[i].alpha1) * e2 + self.subdenets[i].level1(e3, e1)
                        e1 = (self.subdenets[i].alpha2) * e1 + self.subdenets[i].level2(e2, e0)
                        e0 = (self.subdenets[i].alpha3) * e0 + self.subdenets[i].level3(e1, None)
                        # e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
                        # x_tmp_out_list.append(torch.clamp(self.outputs[i](e0) + img, 0., 1.))
                        x_tmp_out_list.append(self.outputs[i](e0) + img)
                        num_tmp += 1

                    return {'img': x_tmp_out_list, 'ics': ics}
                else:
                    exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
                    pass_index = torch.arange(0, e4.shape[0], device=e4.device)
                    # if self.stem_method != 'stem_once':
                    x_img_out = img  # torch.zeros((b, self.out_channels, h, w), device=e0.device)
                    icz = torch.ones([e4.shape[0], 1], device=e4.device) * (-1.)

                    x_tmp_out_list.append(self.outputs[0](e0) + img)
                    for i in range(1, self.num_subdenet):
                        e3_pre = e3
                        e3 = (self.subdenets[i].alpha0) * e3 + self.subdenets[i].level0(e4, e2)
                        if self.dx == 'fd4_d4':
                            dc = torch.cat([e3, e3_pre], 1)
                        elif self.dx == 'fd4':
                            dc = e3_pre
                        elif self.dx == 'fd3':
                            dc = e2
                        elif self.dx == 'fd2':
                            dc = e1
                        elif self.dx == 'fd1':
                            dc = e0
                        elif self.dx == '_d4':
                            dc = e3
                        elif self.dx == 'fd3_d4':
                            dc = torch.cat([self.avgpool(e2), self.avgpool(e3)], 1)
                        elif self.dx == 'fd2_d4':
                            dc = torch.cat([self.avgpool(e1), self.avgpool(e3)], 1)
                        elif self.dx == 'fd1_d4':
                            dc = torch.cat([self.avgpool(e0), self.avgpool(e3)], 1)
                        if self.classifier_type == 'single':
                            ic = self.ape(dc)
                        else:
                            ic = self.ape[i-1](dc)
                        ics.append(ic)
                        e2 = (self.subdenets[i].alpha1) * e2 + self.subdenets[i].level1(e3, e1)
                        e1 = (self.subdenets[i].alpha2) * e1 + self.subdenets[i].level2(e2, e0)
                        e0 = (self.subdenets[i].alpha3) * e0 + self.subdenets[i].level3(e1, None)
                        # e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
                        # x_tmp_out_list.append(torch.clamp(self.outputs[i](e0) + img, 0., 1.))
                        x_tmp_out_list.append(self.outputs[i](e0) + img)
                        num_tmp += 1
                        icz[pass_index, ...] = ic[pass_index, ...]
                        # print('ic: ', ic)
                        # pass_index = torch.where(ic > self.exit_threshold)[0]
                        exit_id = torch.where(icz <= self.exit_threshold)[0]
                        remain_id = torch.where(exit_index < 0.0)[0]
                        intersection_id = []
                        for id in exit_id:
                            if id in remain_id: intersection_id.append(int(id))
                        exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                        if len(intersection_id) > 0:
                            x_img_out[intersection_id, ...] += x_tmp_out_list[i - 1][intersection_id, ...]
                        pass_index = torch.where(exit_index < 0.0)[0]
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += x_tmp_out_list[i][pass_index, ...]
                    x_tmp_out_list.append(x_img_out)
                    return {'img': x_tmp_out_list, 'ics': ics}
            else:
                # self.exit_threshold = 0.
                exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
                pass_index = torch.arange(0, e4.shape[0], device=e4.device)
                # if self.stem_method != 'stem_once':
                x_img_out = img  # torch.zeros((b, self.out_channels, h, w), device=e0.device)
                # print(ics)
                ic = torch.ones([e4.shape[0], 1], device=e4.device) * (-1.)
                # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)  # getattr(self, f'subdenet0')(d4, d0, d1, d2, d3)
                # x_img_out += self.outputs[0](e0)
                # return self.outputs[0](e0)
                for i in range(1, self.num_subdenet):
                    # print(pass_index)
                    e3_pre = e3
                    if len(pass_index) > 0:

                        if self.dx == 'fd4':
                            dc = e3[pass_index, ...]
                        elif self.dx == 'fd3':
                            dc = e2[pass_index, ...]
                        elif self.dx == 'fd2':
                            dc = e1[pass_index, ...]
                        elif self.dx == 'fd1':
                            dc = e0[pass_index, ...]
                        if '_d4' in self.dx:
                            e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                                e4[pass_index, ...], e2[pass_index, ...])
                        if self.dx == '_d4':
                            dc = e3[pass_index, ...]
                        elif self.dx == 'fd4_d4':
                            dc = torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1)
                        elif self.dx == 'fd3_d4':
                            dc = torch.cat([self.avgpool(e2[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                        elif self.dx == 'fd2_d4':
                            dc = torch.cat([self.avgpool(e1[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                        elif self.dx == 'fd1_d4':
                            dc = torch.cat([self.avgpool(e0[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                        ic[pass_index, ...] = self.ape(dc)  # .squeeze(-1)
                        # print('ic: ', ic)
                        # pass_index = torch.where(ic > self.exit_threshold)[0]
                        exit_id = torch.where(ic <= self.exit_threshold)[0]
                        remain_id = torch.where(exit_index < 0.0)[0]
                        intersection_id = []
                        for id in exit_id:
                            if id in remain_id: intersection_id.append(int(id))
                        exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                        if len(intersection_id) > 0:
                            x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                            # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                        pass_index = torch.where(exit_index < 0.0)[0]
                        # print('pass_index: ', pass_index)
                        # print('intersection_id: ', intersection_id)
                        # print('exit_id: ', exit_id)
                        if len(pass_index) > 0:
                            if '_d4' not in self.dx:
                                e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                                    e4[pass_index, ...], e2[pass_index, ...])
                            e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[i].level1(
                                e3[pass_index, ...], e1[pass_index, ...])
                            e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[i].level2(
                                e2[pass_index, ...], e0[pass_index, ...])
                            e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[i].level3(
                                e1[pass_index, ...], None)
                            if self.cal_num_decoders:
                                self.num_decoders += len(pass_index)
                            if i == self.num_subdenet - 1:
                                x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])
                        else:
                            break
                # if self.stem_method == 'stem_once':
                #     x_img_out = self.output(e0) + img
                # print('yyyyyy: ', x_img_out.max(), x_img_out.min())
                # return torch.clamp(x_img_out, 0., 1.)

                return x_img_out
    def _forward_v1(self, img, features):
        x_tmp_out_list = []
        e0, e1, e2, e3, e4 = features
        # with torch.no_grad():
        #     if self.stem_method == 'stem_once':
        #         x = self.stem(img)
        #     for i in range(self.num_subnet):
        #         if self.stem_method != 'stem_once':
        #             x = self.stems[i](img)
        #         e0, e1, e2, e3 = self.subnets[i](x, e0, e1, e2, e3)
        ics = self.ape(e4)
        # ics = list(torch.chunk(ics, self.num_subdenet, -1))
        # with torch.no_grad():
        num_tmp = 0
        if not self.test_only:
            for i in range(self.num_subdenet):
                e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1,
                                                   e0)  # getattr(self, f'subdenet{str(i)}')(d4, d0, d1, d2, d3)
                # if self.stem_method == 'stem_once':
                #     x_tmp_out_list.append(e0)
                # else:
                x_tmp_out_list.append(self.outputs[i](e0) + img)
                num_tmp += 1

            return {'img': x_tmp_out_list, 'ics': ics}
        else:
            ic_max, max_idx = torch.max(ics, dim=-1)
            # exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            # pass_indexs = []
            # for i in range(1, self.num_subdenet):
            #     pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            #     pass_index = torch.where(max_idx <= i)[0]
            #     exit_index = torch.where(max_idx == i)[0]
            #     pass_indexs.append(pass_index)

            # if self.stem_method != 'stem_once':
            x_img_out = img
            # print(self.num_subdenet)
            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)  # getattr(self, f'subdenet0')(d4, d0, d1, d2, d3)
            for i in range(self.num_subdenet):
                # print(pass_index)
                pass_index = torch.where(max_idx >= i)[0]
                exit_id = torch.where(max_idx == i)[0]
                # print(i, max_idx, pass_index, exit_id)
                # print(len(pass_index) > 0)
                if len(pass_index) > 0:
                    e3[pass_index, ...], e2[pass_index, ...], \
                        e1[pass_index, ...], e0[pass_index, ...] = self.subdenets[i](
                        e4[pass_index, ...],
                        e3[pass_index, ...],
                        e2[pass_index, ...],
                        e1[pass_index, ...],
                        e0[pass_index, ...])

                # if self.stem_method != 'stem_once':
                x_img_out[exit_id, ...] += self.outputs[i](e0[exit_id, ...])

            return x_img_out
        # else:
        #     exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
        #     pass_index = torch.arange(0, e4.shape[0], device=e4.device)
        #     # if self.stem_method != 'stem_once':
        #     x_img_out = img
        #     # print(ics)
        #     e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)  # getattr(self, f'subdenet0')(d4, d0, d1, d2, d3)
        #     for i in range(1, self.num_subdenet):
        #         # print(pass_index)
        #         if len(pass_index) > 0:
        #             e3[pass_index, ...], e2[pass_index, ...], \
        #                 e1[pass_index, ...], e0[pass_index, ...] = self.subdenets[i](
        #                 e4[pass_index, ...],
        #                 e3[pass_index, ...],
        #                 e2[pass_index, ...],
        #                 e1[pass_index, ...],
        #                 e0[pass_index, ...])
        #         ic = ics[i - 1]
        #         pass_index = torch.where(ic < self.exit_threshold)[0]
        #
        #         remain_id = torch.where(exit_index < 0.0)[0]
        #         exit_id = torch.where(ic >= self.exit_threshold)[0]
        #         # if self.stem_method != 'stem_once':
        #         x_img_out[exit_id, ...] += self.outputs[i](e0[exit_id, ...])
        #         intersection_id = []
        #         for id in exit_id:
        #             if id in remain_id: intersection_id.append(int(id))
        #         exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
        #     # if self.stem_method == 'stem_once':
        #     #     x_img_out = self.output(e0) + img
        #     return x_img_out

    def _forward_pretrain(self, img, features):
        x_tmp_out_list = []
        e0, e1, e2, e3, e4 = features
        # save_feature = False
        out_dir = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/features_Large_encoder_decoder'
        # os.makedirs(os.path.join(out_dir, ''), exist_ok=True)
        # if self.save_feature:
        #     np_file = os.path.join(out_dir, 'encoder_e0_iter' + str(self.x_num_iter) + '.npy')
        #     data = kornia.tensor_to_image(e0.cpu(), keepdim=True)
        #     with open(np_file, 'wb') as f:
        #         np.save(f, data)
        #     np_file = os.path.join(out_dir, 'encoder_e1_iter' + str(self.x_num_iter) + '.npy')
        #     data = kornia.tensor_to_image(e1.cpu(), keepdim=True)
        #     with open(np_file, 'wb') as f:
        #         np.save(f, data)
        #     np_file = os.path.join(out_dir, 'encoder_e2_iter' + str(self.x_num_iter) + '.npy')
        #     data = kornia.tensor_to_image(e2.cpu(), keepdim=True)
        #     with open(np_file, 'wb') as f:
        #         np.save(f, data)
        #     np_file = os.path.join(out_dir, 'encoder_e3_iter' + str(self.x_num_iter) + '.npy')
        #     data = kornia.tensor_to_image(e3.cpu(), keepdim=True)
        #     with open(np_file, 'wb') as f:
        #         np.save(f, data)
        #     np_file = os.path.join(out_dir, 'encoder_e4_iter' + str(self.x_num_iter) + '.npy')
        #     data = kornia.tensor_to_image(e4.cpu(), keepdim=True)
        #     with open(np_file, 'wb') as f:
        #         np.save(f, data)
        #     cls_feature = self.ape(e3)
        #     np_file = os.path.join(out_dir, 'encoder_classifyfeature_iter' + str(self.x_num_iter) + '.npy')
        #     data = kornia.tensor_to_image(cls_feature.cpu(), keepdim=True)
        #     with open(np_file, 'wb') as f:
        #         np.save(f, data)
        # if self.dx == 'e4':
        #     cls_f = self.ape(e4)
        # elif self.dx == 'e3':
        #     cls_f = self.ape(e3)
        if self.return_cls:
            if self.dx == 'e4':
                cls_f = self.ape(e4)
            elif self.dx == 'e3':
                cls_f = self.ape(e3)
            return torch.argmax(cls_f, dim=1)

        for i in range(self.num_subdenet): #range(1): #
            e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
            # if self.save_feature:
            #     np_file = os.path.join(out_dir, 'sub_decoder_' + str(i)+'_e3_iter'+str(self.x_num_iter)+'.npy')
            #     data = kornia.tensor_to_image(e3.cpu(), keepdim=True)
            #     with open(np_file, 'wb') as f:
            #         np.save(f, data)
            #
            #     np_file = os.path.join(out_dir, 'sub_decoder_' + str(i) + '_e2_iter'+str(self.x_num_iter)+'.npy')
            #     data = kornia.tensor_to_image(e2.cpu(), keepdim=True)
            #     with open(np_file, 'wb') as f:
            #         np.save(f, data)
            #
            #     np_file = os.path.join(out_dir, 'sub_decoder_' + str(i) + '_e1_iter'+str(self.x_num_iter)+'.npy')
            #     data = kornia.tensor_to_image(e1.cpu(), keepdim=True)
            #     with open(np_file, 'wb') as f:
            #         np.save(f, data)
            #
            #     np_file = os.path.join(out_dir, 'sub_decoder_' + str(i) + '_e0_iter'+str(self.x_num_iter)+'.npy')
            #     data = kornia.tensor_to_image(e0.cpu(), keepdim=True)
            #     with open(np_file, 'wb') as f:
            #         np.save(f, data)

            d3_img = self.outputs[i](e0) + img
            # e4, d3_img = self.outputs[i](e0, e4, img)
            # x_tmp_out_list.append(torch.clamp(d3_img, 0., 1.))
            x_tmp_out_list.append(d3_img)
        self.x_num_iter += 1
        if self.test_only:
            return x_tmp_out_list[-1]
        else:

            return x_tmp_out_list


class AdaRevIDSlide(nn.Module):
    def __init__(self, width=64, in_channels=3, out_channels=3,
                 decoder_layers=[1, 1, 1, 1], baseblock=['naf', 'naf', 'naf', 'naf'],
                 kernel_size=19, drop_path=0.0, train_size=[384, 384], overlap_size=[32, 32],  # save_memory=True,
                 save_memory_decoder=True, encoder='UFPNet',
                 pretrain=False, use_amp=False,  # stem_method='stem_multi',
                 state_dict_pth_encoder='/home/ubuntu/106-48t/personal_data/mxt/exp_results/ckpt/UFPNet/train_on_GoPro/net_g_latest.pth',
                 state_dict_pth_classifier=None,
                 state_dict_pth_decoder=None,
                 combine_train=False, test_only=True, cal_num_decoders=False, decoder_idx=4, dx='', classifier_type='single', num_classes=5, exit_idx=[4,3,2,1,0],
                 exit_threshold=0.25) -> None:
        super().__init__()
        self.classifier_type = classifier_type
        self.exit_idx = exit_idx
        self.decoder_idx = decoder_idx
        self.train_size = train_size
        self.overlap_size = overlap_size
        print(self.train_size, self.overlap_size)
        self.kernel_size = self.train_size
        self.inp_channels = 3
        self.out_channels = 3
        self.up_scale = 1
        self.cal_num_decoders = cal_num_decoders
        self.num_decoders = 0
        self.overlap_size_up = (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = [self.train_size[0] * self.up_scale, self.train_size[1] * self.up_scale]
        self.num_subdenet = len(baseblock)
        self.channels = width
        self.test_only = test_only
        thr_scale = 1
        if isinstance(exit_threshold, list):
            self.exit_threshold = [x * thr_scale for x in exit_threshold]
        else:
            self.exit_threshold = [exit_threshold * thr_scale] * (self.num_subdenet-1)
        self.pretrain = pretrain
        self.use_amp = use_amp
        # self.num_subnet = num_subnet
        self.combine_train = combine_train
        channels = [width, width * 2, width * 4, width * 8]
        middle_blk_num = 1
        chan = 2 * channels[-1]
        # self.downs_mid = nn.Conv2d(channels[-1], chan, 2, 2)
        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock(chan) for _ in range(middle_blk_num)]
        #     )
        self.dx = dx
        if not self.pretrain or state_dict_pth_classifier is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.classifier_type == 'hard_simple':
                if dx == 'e4':
                    self.ape = nn.Sequential(
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
                elif dx == 'e3':
                    self.ape = nn.Sequential(
                        nn.Conv2d(channels[-1], chan, 2, 2),
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
            elif self.classifier_type == 'single':
                if dx == 'fd4_d4':
                    self.ape = Classifier(2 * channels[-1], 1)
                elif dx == 'fd4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3':
                    self.ape = Classifier(channels[-2], 1)
                elif dx == 'fd2':
                    self.ape = Classifier(channels[-3], 1)
                elif dx == 'fd1':
                    self.ape = Classifier(channels[0], 1)
                elif dx == '_d4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3_d4':
                    self.ape = Classifier(channels[-2] + channels[-1], 1)
                elif dx == 'fd2_d4':
                    self.ape = Classifier(channels[-3] + channels[-1], 1)
                elif dx == 'fd1_d4':
                    self.ape = Classifier(channels[0] + channels[-1], 1)
            else:
                if dx == 'fd4_d4':
                    self.ape = nn.ModuleList([Classifier(2 * channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd3':
                    self.ape = nn.ModuleList([Classifier(channels[-2], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd2':
                    self.ape = nn.ModuleList([Classifier(channels[-3], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd1':
                    self.ape = nn.ModuleList([Classifier(channels[0], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == '_d4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd3_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[-2] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd2_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[-3] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd1_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[0] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
        if state_dict_pth_classifier is not None:
            self.load_classifier(state_dict_pth_classifier)
        self.subdenets = nn.ModuleList()
        # self.encoder_feature_projs = nn.ModuleList()
        self.conv_features = nn.ParameterList()
        # print(channels)
        h, w = train_size
        # for i in range(len(channels)):
        #     self.encoder_feature_projs.append(MLP_UHD([h, w, channels[i]], 1))
        #     h, w = h//2, w//2
        # self.encoder_feature_projs.append(MLP_UHD([h, w, chan], 4))
        for i in range(len(channels)):
            self.conv_features.append(
                nn.Parameter(torch.ones((1, channels[i], 1, 1)),
                             requires_grad=True))
        self.conv_features.append(
            nn.Parameter(torch.ones((1, chan, 1, 1)),
                         requires_grad=True))


        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(decoder_layers))]

        channels.reverse()
        for i in range(self.num_subdenet):
            first_col = False
            self.subdenets.append(UniDecoder(channels=channels, layers=decoder_layers, kernel_size=kernel_size,
                                             first_col=first_col, dp_rates=dp_rate, save_memory=save_memory_decoder,
                                             baseblock=baseblock[i], train_size=train_size, sub_idx=(i+1)/self.num_subdenet))
        if state_dict_pth_decoder is not None:
            self.subdenets[0].load_pretain_model(state_dict_pth_decoder)

        # if encoder == 'UFPNet':
        # self.encoder = UFPNetLocal(width=width)
        if encoder == 'UFPNet':
            self.encoder = UFPNetLocal(width=width)
        # elif encoder == 'FFTformer':
        #     self.encoder = fftformerEncoder()
        #     self.encoderX = fftformerEncoderPro()
        else:
            self.encoder = NAFNetLocal(width=width)

        self.encoder.eval()

        self.outputs = nn.ModuleList()

        for i in range(self.num_subdenet):
            self.outputs.append(nn.Sequential(
                nn.Conv2d(width, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
                # LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
            ))
    def load_classifier(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)
        # print(checkpoint.keys())

        # state_dict = checkpoint["params"]

        self.ape.load_state_dict(checkpoint, strict=False)

        print('-----------load classifier from ' + state_dict_pth + '----------------')
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = self.kernel_size[0] // n_level, self.kernel_size[1] // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h * self.up_scale, w * self.up_scale)
            self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, img, batch=None, pix_max=1., post_aug=False, out_device=None):
        h, w = img.shape[-2:]
        img_inp = check_image_size(img, 16)
        with torch.no_grad():
            e0, e1, e2, e3, e4 = self.encoder(img_inp)

        features_t = [self.conv_features[0] * (e0),
                      self.conv_features[1] * (e1),
                      self.conv_features[2] * (e2),
                      self.conv_features[3] * (e3),
                      self.conv_features[4] * (e4)]
        features = []
        for i in range(len(features_t)):
            features.append(self.grids(features_t[i], i))
        all_batch = features[0].shape[0]
        # print(all_batch)
        # batch_num = all_batch // batch
        if batch is not None:
            batchs = range(0, all_batch, batch)

            out_parts = []
            # print(inp_img_.shape)
            for batch_z in batchs:
                if batch_z + batch <= all_batch:
                    features_x = [x[batch_z:batch_z + batch, ...] for x in features]
                    if len(features_x[0].shape) == 3:
                        features_x = [x.unsqueeze(0) for x in features_x]
                    out_parts.append(self.forward_feature(features_x))
                else:
                    features_x = [x[batch_z:, ...] for x in features]
                    if len(features_x[0].shape) == 3:
                        features_x = [x.unsqueeze(0) for x in features_x]
                    out_parts.append(self.forward_feature(features_x))
            out_parts = torch.cat(out_parts, dim=0)
        else:
            out_parts = self.forward_feature(features)
        # print(out_parts.shape)

        x = self.grids_inverse(out_parts, out_device=out_device, pix_max=pix_max)
        # print(x.shape)
        if self.cal_num_decoders:
            return x[:, :, :h, :w] + img, self.num_decoders
        else:
            return x[:, :, :h, :w] + img
    def forward_feature(self, features):

        if self.pretrain:
            return self._forward_pretrain(features)
        else:
            return self._forward(features)

    def _forward(self, features):
        x_tmp_out_list = []
        e0, e1, e2, e3, e4 = features
        # ics = self.ape(e4)
        b, c, h, w = e0.shape
        ics = []
        num_tmp = 0
        if self.classifier_type == 'hard_simple':
            if self.dx == 'e4':
                class_hard_simple = self.ape(e4)
            elif self.dx == 'e3':
                class_hard_simple = self.ape(e3)
            # class_hard_simple = self.ape(e4)
            class_hard_simple_idx = torch.argmax(class_hard_simple, dim=-1, keepdim=False)
            out_x_idx = torch.tensor([self.exit_idx[i] for i in class_hard_simple_idx], device=e0.device)

            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)
            # if self.cal_num_decoders:
            #     self.num_decoders += b

            exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            # if self.stem_method != 'stem_once':
            x_img_out = torch.zeros((b, self.out_channels, h, w), device=e0.device)
            # print(ics)
            ic = torch.zeros_like(class_hard_simple_idx)
            for i in range(self.num_subdenet):
                # print(ic, out_x_idx)
                # e3_pre = e3
                if len(pass_index) > 0:
                    exit_id = torch.where(ic == out_x_idx)[0]
                    # print(ic, out_x_idx, exit_id, pass_index)
                    ic = ic + 1
                    remain_id = torch.where(exit_index < 0.0)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                    # if len(intersection_id) > 0:
                    #     x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                        # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                    pass_index = torch.where(exit_index < 0.0)[0]
                    # print('pass_index: ', pass_index)
                    # print('intersection_id: ', intersection_id)
                    # print('exit_id: ', exit_id)
                    # if i == 0:
                    #     if len(exit_id) > 0:
                    #         x_img_out[exit_id, ...] = 0
                    # print(exit_id, pass_index)
                    if len(exit_id) > 0 and i >= 1:
                        x_img_out[exit_id, ...] += self.outputs[i-1](e0[exit_id, ...])
                    if len(pass_index) > 0:
                        e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[
                            i].level0(
                            e4[pass_index, ...], e2[pass_index, ...])
                        e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[
                            i].level1(
                            e3[pass_index, ...], e1[pass_index, ...])
                        e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[
                            i].level2(
                            e2[pass_index, ...], e0[pass_index, ...])
                        e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[
                            i].level3(
                            e1[pass_index, ...], None)
                        if self.cal_num_decoders:
                            self.num_decoders += len(pass_index)
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])

                    else:
                        break
            return x_img_out
        else:
            e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)
            if self.cal_num_decoders:
                self.num_decoders += b
            # if not self.test_only:
            #     e0, e1, e2, e3, e4 = features
            #     # print(e0.shape, e1.shape, e2.shape, e3.shape, e4.shape)
            #     for i in range(1, self.num_subdenet):  # range(1): #
            #         e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
            #     # d3_img = self.outputs[0](e0)
            #     d3_img = self.outputs[-1](e0)
            #     return d3_img
            # else:
            # print('Adaptive')
            # self.exit_threshold = 0.
            exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            # if self.stem_method != 'stem_once':
            x_img_out = torch.zeros((b, self.out_channels, h, w), device=e0.device)
            # print(ics)
            ic = torch.ones([e4.shape[0], 1], device=e4.device) * (-1.)
            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)  # getattr(self, f'subdenet0')(d4, d0, d1, d2, d3)
            # x_img_out += self.outputs[0](e0)
            # return self.outputs[0](e0)

            for i in range(1, self.num_subdenet):
                # print(pass_index)
                e3_pre = e3
                if len(pass_index) > 0:
                    if self.dx == 'fd4':
                        dc = e3[pass_index, ...]
                    elif self.dx == 'fd3':
                        dc = e2[pass_index, ...]
                    elif self.dx == 'fd2':
                        dc = e1[pass_index, ...]
                    elif self.dx == 'fd1':
                        dc = e0[pass_index, ...]
                    if '_d4' in self.dx:
                        e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                            e4[pass_index, ...], e2[pass_index, ...])
                    if self.dx == '_d4':
                        dc = e3[pass_index, ...]
                    elif self.dx == 'fd4_d4':
                        dc = torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1)
                    elif self.dx == 'fd3_d4':
                        dc = torch.cat([self.avgpool(e2[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    elif self.dx == 'fd2_d4':
                        dc = torch.cat([self.avgpool(e1[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    elif self.dx == 'fd1_d4':
                        dc = torch.cat([self.avgpool(e0[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    if self.classifier_type == 'single':
                        ic[pass_index, ...] = self.ape(dc)
                    else:
                        ic[pass_index, ...] = self.ape[i-1](dc)
                     # .squeeze(-1)
                    # e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                    #     e4[pass_index, ...], e2[pass_index, ...])
                    #
                    # ic[pass_index, ...] = self.ape(
                    #     torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1))  # .squeeze(-1)
                    # print('ic: ', ic)
                    # pass_index = torch.where(ic > self.exit_threshold)[0]
                    exit_id = torch.where(ic <= self.exit_threshold[i-1])[0]
                    remain_id = torch.where(exit_index < 0.0)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                    if len(intersection_id) > 0:
                        x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                        # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                    pass_index = torch.where(exit_index < 0.0)[0]
                    # print('pass_index: ', pass_index)
                    # print('intersection_id: ', intersection_id)
                    # print('exit_id: ', exit_id)
                    if len(pass_index) > 0:
                        if '_d4' not in self.dx:
                            e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                                e4[pass_index, ...], e2[pass_index, ...])
                        e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[i].level1(
                            e3[pass_index, ...], e1[pass_index, ...])
                        e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[i].level2(
                            e2[pass_index, ...], e0[pass_index, ...])
                        e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[i].level3(
                            e1[pass_index, ...], None)
                        if self.cal_num_decoders:
                            self.num_decoders += len(pass_index)
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])
                    else:
                        break
            # if self.stem_method == 'stem_once':
            #     x_img_out = self.output(e0) + img
            # print('yyyyyy: ', x_img_out.max(), x_img_out.min())
            return x_img_out

    def _forward_pretrain(self, features):

        e0, e1, e2, e3, e4 = features
        # print(e0.shape, e1.shape, e2.shape, e3.shape, e4.shape)
        # for i in range(self.num_subdenet): #range(1): #
        #     e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
        # # d3_img = self.outputs[0](e0)
        # d3_img = self.outputs[-1](e0)
        for i in range(self.decoder_idx): #range(1): #
            e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)

        d3_img = self.outputs[self.decoder_idx-1](e0)
        return d3_img
class AdaRevIDSlideV2(nn.Module):
    def __init__(self, width=64, in_channels=3, out_channels=3,
                 decoder_layers=[1, 1, 1, 1], baseblock=['naf', 'naf', 'naf', 'naf'],
                 kernel_size=19, drop_path=0.0, train_size=[384, 384], overlap_size=[32, 32],  # save_memory=True,
                 save_memory_decoder=True, encoder='UFPNet',
                 pretrain=False, use_amp=False,  # stem_method='stem_multi',
                 state_dict_pth_encoder='/home/ubuntu/106-48t/personal_data/mxt/exp_results/ckpt/UFPNet/train_on_GoPro/net_g_latest.pth',
                 state_dict_pth_classifier=None,
                 state_dict_pth_decoder=None,
                 combine_train=False, test_only=True, cal_num_decoders=False, decoder_idx=4, dx='', classifier_type='single', num_classes=5, exit_idx=[4,3,2,1,0],
                 exit_threshold=0.25) -> None:
        super().__init__()
        self.classifier_type = classifier_type
        self.exit_idx = exit_idx
        self.decoder_idx = decoder_idx
        self.train_size = train_size
        self.overlap_size = overlap_size
        print(self.train_size, self.overlap_size)
        self.kernel_size = self.train_size
        self.inp_channels = 3
        self.out_channels = 3
        self.up_scale = 1
        self.cal_num_decoders = cal_num_decoders
        self.num_decoders = 0
        self.overlap_size_up = (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = [self.train_size[0] * self.up_scale, self.train_size[1] * self.up_scale]
        self.num_subdenet = len(baseblock)
        self.channels = width
        self.test_only = test_only
        thr_scale = 1
        if isinstance(exit_threshold, list):
            self.exit_threshold = [x * thr_scale for x in exit_threshold]
        else:
            self.exit_threshold = [exit_threshold * thr_scale] * (self.num_subdenet-1)
        self.pretrain = pretrain
        self.use_amp = use_amp
        # self.num_subnet = num_subnet
        self.combine_train = combine_train
        channels = [width, width * 2, width * 4, width * 8]
        middle_blk_num = 1
        chan = 2 * channels[-1]
        # self.downs_mid = nn.Conv2d(channels[-1], chan, 2, 2)
        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock(chan) for _ in range(middle_blk_num)]
        #     )
        self.dx = dx
        if not self.pretrain or state_dict_pth_classifier is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.classifier_type == 'hard_simple':
                if dx == 'e4':
                    self.ape = nn.Sequential(
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
                elif dx == 'e3':
                    self.ape = nn.Sequential(
                        nn.Conv2d(channels[-1], chan, 2, 2),
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
            elif self.classifier_type == 'single':
                if dx == 'fd4_d4':
                    self.ape = Classifier(2 * channels[-1], 1)
                elif dx == 'fd4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3':
                    self.ape = Classifier(channels[-2], 1)
                elif dx == 'fd2':
                    self.ape = Classifier(channels[-3], 1)
                elif dx == 'fd1':
                    self.ape = Classifier(channels[0], 1)
                elif dx == '_d4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3_d4':
                    self.ape = Classifier(channels[-2] + channels[-1], 1)
                elif dx == 'fd2_d4':
                    self.ape = Classifier(channels[-3] + channels[-1], 1)
                elif dx == 'fd1_d4':
                    self.ape = Classifier(channels[0] + channels[-1], 1)
            else:
                if dx == 'fd4_d4':
                    self.ape = nn.ModuleList([Classifier(2 * channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd3':
                    self.ape = nn.ModuleList([Classifier(channels[-2], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd2':
                    self.ape = nn.ModuleList([Classifier(channels[-3], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd1':
                    self.ape = nn.ModuleList([Classifier(channels[0], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == '_d4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd3_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[-2] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd2_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[-3] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd1_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[0] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
        if state_dict_pth_classifier is not None:
            self.load_classifier(state_dict_pth_classifier)
        self.subdenets = nn.ModuleList()
        # self.encoder_feature_projs = nn.ModuleList()
        self.conv_features = nn.ParameterList()
        # print(channels)
        h, w = train_size
        # for i in range(len(channels)):
        #     self.encoder_feature_projs.append(MLP_UHD([h, w, channels[i]], 1))
        #     h, w = h//2, w//2
        # self.encoder_feature_projs.append(MLP_UHD([h, w, chan], 4))
        for i in range(len(channels)):
            self.conv_features.append(
                nn.Parameter(torch.ones((1, channels[i], 1, 1)),
                             requires_grad=True))
        self.conv_features.append(
            nn.Parameter(torch.ones((1, chan, 1, 1)),
                         requires_grad=True))


        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(decoder_layers))]

        channels.reverse()
        for i in range(self.num_subdenet):
            first_col = False
            self.subdenets.append(UniDecoder(channels=channels, layers=decoder_layers, kernel_size=kernel_size,
                                             first_col=first_col, dp_rates=dp_rate, save_memory=save_memory_decoder,
                                             baseblock=baseblock[i], train_size=train_size, sub_idx=(i+1)/self.num_subdenet))
        if state_dict_pth_decoder is not None:
            self.subdenets[0].load_pretain_model(state_dict_pth_decoder)

        # if encoder == 'UFPNet':
        # self.encoder = UFPNetLocal(width=width)
        if encoder == 'UFPNet':
            self.encoder = UFPNetLocal(width=width)
        # elif encoder == 'FFTformer':
        #     self.encoder = fftformerEncoder()
        #     self.encoderX = fftformerEncoderPro()
        else:
            self.encoder = NAFNetLocal(width=width)

        self.encoder.eval()

        self.outputs = nn.ModuleList()

        for i in range(self.num_subdenet):
            self.outputs.append(nn.Sequential(
                nn.Conv2d(width, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
                # LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
            ))
    def load_classifier(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)
        # print(checkpoint.keys())

        # state_dict = checkpoint["params"]

        self.ape.load_state_dict(checkpoint, strict=False)

        print('-----------load classifier from ' + state_dict_pth + '----------------')
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = self.kernel_size[0] // n_level, self.kernel_size[1] // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h * self.up_scale, w * self.up_scale)
            self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, img, batch=None, pix_max=1., post_aug=False, out_device=None):
        h, w = img.shape[-2:]
        img_inp = check_image_size(img, 16)
        with torch.no_grad():
            e0, e1, e2, e3, e4 = self.encoder(img_inp)

        features_t = [self.conv_features[0] * (e0),
                      self.conv_features[1] * (e1),
                      self.conv_features[2] * (e2),
                      self.conv_features[3] * (e3),
                      self.conv_features[4] * (e4)]
        features = []
        for i in range(len(features_t)):
            features.append(self.grids(features_t[i], i))
        all_batch = features[0].shape[0]
        # print(all_batch)
        # batch_num = all_batch // batch
        if batch is not None:
            batchs = range(0, all_batch, batch)

            out_parts = []
            # print(inp_img_.shape)
            for batch_z in batchs:
                if batch_z + batch <= all_batch:
                    features_x = [x[batch_z:batch_z + batch, ...] for x in features]
                    if len(features_x[0].shape) == 3:
                        features_x = [x.unsqueeze(0) for x in features_x]
                    out_parts.append(self.forward_feature(features_x))
                else:
                    features_x = [x[batch_z:, ...] for x in features]
                    if len(features_x[0].shape) == 3:
                        features_x = [x.unsqueeze(0) for x in features_x]
                    out_parts.append(self.forward_feature(features_x))
            out_parts = torch.cat(out_parts, dim=0)
        else:
            out_parts = self.forward_feature(features)
        # print(out_parts.shape)

        x = self.grids_inverse(out_parts, out_device=out_device, pix_max=pix_max)
        # print(x.shape)
        if self.cal_num_decoders:
            return x[:, :, :h, :w] + img, self.num_decoders
        else:
            return x[:, :, :h, :w] + img
    def forward_feature(self, features):

        if self.pretrain:
            return self._forward_pretrain(features)
        else:
            return self._forward(features)

    def _forward(self, features):
        x_tmp_out_list = []
        e0, e1, e2, e3, e4 = features
        # ics = self.ape(e4)
        b, c, h, w = e0.shape
        ics = []
        num_tmp = 0
        if self.classifier_type == 'hard_simple':
            if self.dx == 'e4':
                class_hard_simple = self.ape(e4)
            elif self.dx == 'e3':
                class_hard_simple = self.ape(e3)
            # class_hard_simple = self.ape(e4)
            class_hard_simple_idx = torch.argmax(class_hard_simple, dim=-1, keepdim=False)
            out_x_idx = torch.tensor([self.exit_idx[i] for i in class_hard_simple_idx], device=e0.device)

            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)
            # if self.cal_num_decoders:
            #     self.num_decoders += b

            exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            # if self.stem_method != 'stem_once':
            x_img_out = torch.zeros((b, self.out_channels, h, w), device=e0.device)
            # print(ics)
            ic = torch.zeros_like(class_hard_simple_idx)
            for i in range(self.num_subdenet):
                # print(ic, out_x_idx)
                # e3_pre = e3
                if len(pass_index) > 0:
                    exit_id = torch.where(ic == out_x_idx)[0]
                    # print(ic, out_x_idx, exit_id, pass_index)
                    ic = ic + 1
                    remain_id = torch.where(exit_index < 0.0)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                    # if len(intersection_id) > 0:
                    #     x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                        # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                    pass_index = torch.where(exit_index < 0.0)[0]
                    # print('pass_index: ', pass_index)
                    # print('intersection_id: ', intersection_id)
                    # print('exit_id: ', exit_id)
                    # if i == 0:
                    #     if len(exit_id) > 0:
                    #         x_img_out[exit_id, ...] = 0
                    # print(exit_id, pass_index)
                    if len(exit_id) > 0 and i >= 1:
                        x_img_out[exit_id, ...] += self.outputs[i-1](e0[exit_id, ...])
                    if len(pass_index) > 0:
                        e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[
                            i].level0(
                            e4[pass_index, ...], e2[pass_index, ...])
                        e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[
                            i].level1(
                            e3[pass_index, ...], e1[pass_index, ...])
                        e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[
                            i].level2(
                            e2[pass_index, ...], e0[pass_index, ...])
                        e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[
                            i].level3(
                            e1[pass_index, ...], None)
                        if self.cal_num_decoders:
                            self.num_decoders += len(pass_index)
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])

                    else:
                        break
            return x_img_out
        else:
            e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)
            if self.cal_num_decoders:
                self.num_decoders += b
            # if not self.test_only:
            #     e0, e1, e2, e3, e4 = features
            #     # print(e0.shape, e1.shape, e2.shape, e3.shape, e4.shape)
            #     for i in range(1, self.num_subdenet):  # range(1): #
            #         e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
            #     # d3_img = self.outputs[0](e0)
            #     d3_img = self.outputs[-1](e0)
            #     return d3_img
            # else:
            # print('Adaptive')
            # self.exit_threshold = 0.
            exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            # if self.stem_method != 'stem_once':
            x_img_out = torch.zeros((b, self.out_channels, h, w), device=e0.device)
            # print(ics)
            ic = torch.ones([e4.shape[0], 1], device=e4.device) * (-1.)
            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)  # getattr(self, f'subdenet0')(d4, d0, d1, d2, d3)
            # x_img_out += self.outputs[0](e0)
            # return self.outputs[0](e0)

            for i in range(1, self.num_subdenet):
                # print(pass_index)
                e3_pre = e3
                if len(pass_index) > 0:
                    if self.dx == 'fd4':
                        dc = e3[pass_index, ...]
                    elif self.dx == 'fd3':
                        dc = e2[pass_index, ...]
                    elif self.dx == 'fd2':
                        dc = e1[pass_index, ...]
                    elif self.dx == 'fd1':
                        dc = e0[pass_index, ...]
                    if '_d4' in self.dx:
                        e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                            e4[pass_index, ...], e2[pass_index, ...])
                    if self.dx == '_d4':
                        dc = e3[pass_index, ...]
                    elif self.dx == 'fd4_d4':
                        dc = torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1)
                    elif self.dx == 'fd3_d4':
                        dc = torch.cat([self.avgpool(e2[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    elif self.dx == 'fd2_d4':
                        dc = torch.cat([self.avgpool(e1[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    elif self.dx == 'fd1_d4':
                        dc = torch.cat([self.avgpool(e0[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    if self.classifier_type == 'single':
                        ic[pass_index, ...] = self.ape(dc)
                    else:
                        ic[pass_index, ...] = self.ape[i-1](dc)
                     # .squeeze(-1)
                    # e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                    #     e4[pass_index, ...], e2[pass_index, ...])
                    #
                    # ic[pass_index, ...] = self.ape(
                    #     torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1))  # .squeeze(-1)
                    # print('ic: ', ic)
                    # pass_index = torch.where(ic > self.exit_threshold)[0]
                    exit_id = torch.where(ic <= self.exit_threshold[i-1])[0]
                    remain_id = torch.where(exit_index < 0.0)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                    if len(intersection_id) > 0:
                        x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                        # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                    pass_index = torch.where(exit_index < 0.0)[0]
                    # print('pass_index: ', pass_index)
                    # print('intersection_id: ', intersection_id)
                    # print('exit_id: ', exit_id)
                    if len(pass_index) > 0:
                        if '_d4' not in self.dx:
                            e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                                e4[pass_index, ...], e2[pass_index, ...])
                        e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[i].level1(
                            e3[pass_index, ...], e1[pass_index, ...])
                        e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[i].level2(
                            e2[pass_index, ...], e0[pass_index, ...])
                        e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[i].level3(
                            e1[pass_index, ...], None)
                        if self.cal_num_decoders:
                            self.num_decoders += len(pass_index)
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])
                    else:
                        break
            # if self.stem_method == 'stem_once':
            #     x_img_out = self.output(e0) + img
            # print('yyyyyy: ', x_img_out.max(), x_img_out.min())
            return x_img_out

    def _forward_pretrain(self, features):

        e0, e1, e2, e3, e4 = features
        # print(e0.shape, e1.shape, e2.shape, e3.shape, e4.shape)
        # for i in range(self.num_subdenet): #range(1): #
        #     e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
        # # d3_img = self.outputs[0](e0)
        # d3_img = self.outputs[-1](e0)
        for i in range(self.decoder_idx): #range(1): #
            e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)

        d3_img = self.outputs[self.decoder_idx-1](e0)
        return d3_img
class PatchAdaRevIDSlideV2(nn.Module):
    def __init__(self, width=64, in_channels=3, out_channels=3,
                 decoder_layers=[1, 1, 1, 1], baseblock=['naf', 'naf', 'naf', 'naf'],
                 kernel_size=19, drop_path=0.0, train_size=[384, 384], overlap_size=[32, 32],  # save_memory=True,
                 save_memory_decoder=True, encoder='UFPNet',
                 pretrain=False, use_amp=False,  # stem_method='stem_multi',
                 state_dict_pth_encoder='/home/ubuntu/106-48t/personal_data/mxt/exp_results/ckpt/UFPNet/train_on_GoPro/net_g_latest.pth',
                 state_dict_pth_classifier=None,
                 state_dict_pth_decoder=None,
                 combine_train=False, test_only=True, cal_num_decoders=False, decoder_idx=4, dx='', classifier_type='single', num_classes=5, exit_idx=[4,3,2,1,0],
                 exit_threshold=0.25) -> None:
        super().__init__()
        self.classifier_type = classifier_type
        self.exit_idx = exit_idx
        self.decoder_idx = decoder_idx
        self.train_size = train_size
        self.overlap_size = overlap_size
        print(self.train_size, self.overlap_size)
        self.kernel_size = self.train_size
        self.inp_channels = 3
        self.out_channels = 3
        self.up_scale = 1
        self.cal_num_decoders = cal_num_decoders
        self.num_decoders = 0
        self.overlap_size_up = (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = [self.train_size[0] * self.up_scale, self.train_size[1] * self.up_scale]
        self.num_subdenet = len(baseblock)
        self.channels = width
        self.test_only = test_only
        thr_scale = 1
        if isinstance(exit_threshold, list):
            self.exit_threshold = [x * thr_scale for x in exit_threshold]
        else:
            self.exit_threshold = [exit_threshold * thr_scale] * (self.num_subdenet-1)
        self.pretrain = pretrain
        self.use_amp = use_amp
        # self.num_subnet = num_subnet
        self.combine_train = combine_train
        channels = [width, width * 2, width * 4, width * 8]
        middle_blk_num = 1
        chan = 2 * channels[-1]
        # self.downs_mid = nn.Conv2d(channels[-1], chan, 2, 2)
        # self.middle_blks = \
        #     nn.Sequential(
        #         *[NAFBlock(chan) for _ in range(middle_blk_num)]
        #     )
        self.dx = dx
        if not self.pretrain or state_dict_pth_classifier is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.classifier_type == 'hard_simple':
                if dx == 'e4':
                    self.ape = nn.Sequential(
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
                elif dx == 'e3':
                    self.ape = nn.Sequential(
                        nn.Conv2d(channels[-1], chan, 2, 2),
                        NAFBlock(chan),
                        Classifier(chan, num_classes=num_classes)
                    )
            elif self.classifier_type == 'single':
                if dx == 'fd4_d4':
                    self.ape = Classifier(2 * channels[-1], 1)
                elif dx == 'fd4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3':
                    self.ape = Classifier(channels[-2], 1)
                elif dx == 'fd2':
                    self.ape = Classifier(channels[-3], 1)
                elif dx == 'fd1':
                    self.ape = Classifier(channels[0], 1)
                elif dx == '_d4':
                    self.ape = Classifier(channels[-1], 1)
                elif dx == 'fd3_d4':
                    self.ape = Classifier(channels[-2] + channels[-1], 1)
                elif dx == 'fd2_d4':
                    self.ape = Classifier(channels[-3] + channels[-1], 1)
                elif dx == 'fd1_d4':
                    self.ape = Classifier(channels[0] + channels[-1], 1)
            else:
                if dx == 'fd4_d4':
                    self.ape = nn.ModuleList([Classifier(2 * channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd3':
                    self.ape = nn.ModuleList([Classifier(channels[-2], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd2':
                    self.ape = nn.ModuleList([Classifier(channels[-3], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd1':
                    self.ape = nn.ModuleList([Classifier(channels[0], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == '_d4':
                    self.ape = nn.ModuleList([Classifier(channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd3_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[-2] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd2_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[-3] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
                elif dx == 'fd1_d4':
                    self.ape = nn.ModuleList(
                        [Classifier(channels[0] + channels[-1], 1) for _ in range(self.num_subdenet - 1)])
        if state_dict_pth_classifier is not None:
            self.load_classifier(state_dict_pth_classifier)
        self.subdenets = nn.ModuleList()
        # self.encoder_feature_projs = nn.ModuleList()
        self.conv_features = nn.ParameterList()
        # print(channels)
        h, w = train_size
        # for i in range(len(channels)):
        #     self.encoder_feature_projs.append(MLP_UHD([h, w, channels[i]], 1))
        #     h, w = h//2, w//2
        # self.encoder_feature_projs.append(MLP_UHD([h, w, chan], 4))
        for i in range(len(channels)):
            self.conv_features.append(
                nn.Parameter(torch.ones((1, channels[i], 1, 1)),
                             requires_grad=True))
        self.conv_features.append(
            nn.Parameter(torch.ones((1, chan, 1, 1)),
                         requires_grad=True))


        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(decoder_layers))]

        channels.reverse()
        for i in range(self.num_subdenet):
            first_col = False
            self.subdenets.append(UniDecoder(channels=channels, layers=decoder_layers, kernel_size=kernel_size,
                                             first_col=first_col, dp_rates=dp_rate, save_memory=save_memory_decoder,
                                             baseblock=baseblock[i], train_size=train_size, sub_idx=(i+1)/self.num_subdenet))
        if state_dict_pth_decoder is not None:
            self.subdenets[0].load_pretain_model(state_dict_pth_decoder)

        # if encoder == 'UFPNet':
        # self.encoder = UFPNetLocal(width=width)
        if encoder == 'UFPNet':
            self.encoder = UFPNet(width=width)
        # elif encoder == 'FFTformer':
        #     self.encoder = fftformerEncoder()
        #     self.encoderX = fftformerEncoderPro()
        else:
            self.encoder = NAFNetLocal(width=width)

        self.encoder.eval()

        self.outputs = nn.ModuleList()

        for i in range(self.num_subdenet):
            self.outputs.append(nn.Sequential(
                nn.Conv2d(width, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
                # LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
            ))
    def load_classifier(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)
        # print(checkpoint.keys())

        # state_dict = checkpoint["params"]

        self.ape.load_state_dict(checkpoint, strict=False)

        print('-----------load classifier from ' + state_dict_pth + '----------------')
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = self.kernel_size[0] // n_level, self.kernel_size[1] // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h * self.up_scale, w * self.up_scale)
            self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, img, batch=None, pix_max=1., post_aug=False, out_device=None):
        h, w = img.shape[-2:]
        # img_inp = check_image_size(img, 16)
        img_inp = self.grids(img, 0)
        with torch.no_grad():
            e0, e1, e2, e3, e4 = self.encoder(img_inp)

        features_t = [self.conv_features[0] * (e0),
                      self.conv_features[1] * (e1),
                      self.conv_features[2] * (e2),
                      self.conv_features[3] * (e3),
                      self.conv_features[4] * (e4)]
        # features = []
        features = features_t
        # for i in range(len(features_t)):
        #     features.append(self.grids(features_t[i], i))

        all_batch = features[0].shape[0]
        # print(all_batch)
        # batch_num = all_batch // batch
        if batch is not None:
            batchs = range(0, all_batch, batch)

            out_parts = []
            # print(inp_img_.shape)
            for batch_z in batchs:
                if batch_z + batch <= all_batch:
                    features_x = [x[batch_z:batch_z + batch, ...] for x in features]
                    if len(features_x[0].shape) == 3:
                        features_x = [x.unsqueeze(0) for x in features_x]
                    out_parts.append(self.forward_feature(features_x))
                else:
                    features_x = [x[batch_z:, ...] for x in features]
                    if len(features_x[0].shape) == 3:
                        features_x = [x.unsqueeze(0) for x in features_x]
                    out_parts.append(self.forward_feature(features_x))
            out_parts = torch.cat(out_parts, dim=0)
        else:
            out_parts = self.forward_feature(features)
        # print(out_parts.shape)

        x = self.grids_inverse(out_parts, out_device=out_device, pix_max=pix_max)
        # print(x.shape)
        if self.cal_num_decoders:
            return x[:, :, :h, :w] + img, self.num_decoders
        else:
            return x[:, :, :h, :w] + img
    def forward_feature(self, features):

        if self.pretrain:
            return self._forward_pretrain(features)
        else:
            return self._forward(features)

    def _forward(self, features):
        x_tmp_out_list = []
        e0, e1, e2, e3, e4 = features
        # ics = self.ape(e4)
        b, c, h, w = e0.shape
        ics = []
        num_tmp = 0
        if self.classifier_type == 'hard_simple':
            if self.dx == 'e4':
                class_hard_simple = self.ape(e4)
            elif self.dx == 'e3':
                class_hard_simple = self.ape(e3)
            # class_hard_simple = self.ape(e4)
            class_hard_simple_idx = torch.argmax(class_hard_simple, dim=-1, keepdim=False)
            out_x_idx = torch.tensor([self.exit_idx[i] for i in class_hard_simple_idx], device=e0.device)

            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)
            # if self.cal_num_decoders:
            #     self.num_decoders += b

            exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            # if self.stem_method != 'stem_once':
            x_img_out = torch.zeros((b, self.out_channels, h, w), device=e0.device)
            # print(ics)
            ic = torch.zeros_like(class_hard_simple_idx)
            for i in range(self.num_subdenet):
                # print(ic, out_x_idx)
                # e3_pre = e3
                if len(pass_index) > 0:
                    exit_id = torch.where(ic == out_x_idx)[0]
                    # print(ic, out_x_idx, exit_id, pass_index)
                    ic = ic + 1
                    remain_id = torch.where(exit_index < 0.0)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                    # if len(intersection_id) > 0:
                    #     x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                        # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                    pass_index = torch.where(exit_index < 0.0)[0]
                    # print('pass_index: ', pass_index)
                    # print('intersection_id: ', intersection_id)
                    # print('exit_id: ', exit_id)
                    # if i == 0:
                    #     if len(exit_id) > 0:
                    #         x_img_out[exit_id, ...] = 0
                    # print(exit_id, pass_index)
                    if len(exit_id) > 0 and i >= 1:
                        x_img_out[exit_id, ...] += self.outputs[i-1](e0[exit_id, ...])
                    if len(pass_index) > 0:
                        e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[
                            i].level0(
                            e4[pass_index, ...], e2[pass_index, ...])
                        e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[
                            i].level1(
                            e3[pass_index, ...], e1[pass_index, ...])
                        e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[
                            i].level2(
                            e2[pass_index, ...], e0[pass_index, ...])
                        e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[
                            i].level3(
                            e1[pass_index, ...], None)
                        if self.cal_num_decoders:
                            self.num_decoders += len(pass_index)
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])

                    else:
                        break
            return x_img_out
        else:
            e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)
            if self.cal_num_decoders:
                self.num_decoders += b
            # if not self.test_only:
            #     e0, e1, e2, e3, e4 = features
            #     # print(e0.shape, e1.shape, e2.shape, e3.shape, e4.shape)
            #     for i in range(1, self.num_subdenet):  # range(1): #
            #         e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
            #     # d3_img = self.outputs[0](e0)
            #     d3_img = self.outputs[-1](e0)
            #     return d3_img
            # else:
            # print('Adaptive')
            # self.exit_threshold = 0.
            exit_index = torch.ones(e4.shape[0], device=e4.device) * (-1.)
            pass_index = torch.arange(0, e4.shape[0], device=e4.device)
            # if self.stem_method != 'stem_once':
            x_img_out = torch.zeros((b, self.out_channels, h, w), device=e0.device)
            # print(ics)
            ic = torch.ones([e4.shape[0], 1], device=e4.device) * (-1.)
            # e3, e2, e1, e0 = self.subdenets[0](e4, e3, e2, e1, e0)  # getattr(self, f'subdenet0')(d4, d0, d1, d2, d3)
            # x_img_out += self.outputs[0](e0)
            # return self.outputs[0](e0)

            for i in range(1, self.num_subdenet):
                # print(pass_index)
                e3_pre = e3
                if len(pass_index) > 0:
                    if self.dx == 'fd4':
                        dc = e3[pass_index, ...]
                    elif self.dx == 'fd3':
                        dc = e2[pass_index, ...]
                    elif self.dx == 'fd2':
                        dc = e1[pass_index, ...]
                    elif self.dx == 'fd1':
                        dc = e0[pass_index, ...]
                    if '_d4' in self.dx:
                        e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                            e4[pass_index, ...], e2[pass_index, ...])
                    if self.dx == '_d4':
                        dc = e3[pass_index, ...]
                    elif self.dx == 'fd4_d4':
                        dc = torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1)
                    elif self.dx == 'fd3_d4':
                        dc = torch.cat([self.avgpool(e2[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    elif self.dx == 'fd2_d4':
                        dc = torch.cat([self.avgpool(e1[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    elif self.dx == 'fd1_d4':
                        dc = torch.cat([self.avgpool(e0[pass_index, ...]), self.avgpool(e3[pass_index, ...])], 1)
                    if self.classifier_type == 'single':
                        ic[pass_index, ...] = self.ape(dc)
                    else:
                        ic[pass_index, ...] = self.ape[i-1](dc)
                     # .squeeze(-1)
                    # e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                    #     e4[pass_index, ...], e2[pass_index, ...])
                    #
                    # ic[pass_index, ...] = self.ape(
                    #     torch.cat([e3[pass_index, ...], e3_pre[pass_index, ...]], 1))  # .squeeze(-1)
                    # print('ic: ', ic)
                    # pass_index = torch.where(ic > self.exit_threshold)[0]
                    exit_id = torch.where(ic <= self.exit_threshold[i-1])[0]
                    remain_id = torch.where(exit_index < 0.0)[0]
                    intersection_id = []
                    for id in exit_id:
                        if id in remain_id: intersection_id.append(int(id))
                    exit_index[intersection_id] = i  # (i - (self.exit_interval - 1)) // self.exit_interval
                    if len(intersection_id) > 0:
                        x_img_out[intersection_id, ...] += self.outputs[i - 1](e0[intersection_id, ...])
                        # print('xxxxxx: ', x_img_out[intersection_id, ...].max(), x_img_out[intersection_id, ...].min())
                    pass_index = torch.where(exit_index < 0.0)[0]
                    # print('pass_index: ', pass_index)
                    # print('intersection_id: ', intersection_id)
                    # print('exit_id: ', exit_id)
                    if len(pass_index) > 0:
                        if '_d4' not in self.dx:
                            e3[pass_index, ...] = (self.subdenets[i].alpha0) * e3[pass_index, ...] + self.subdenets[i].level0(
                                e4[pass_index, ...], e2[pass_index, ...])
                        e2[pass_index, ...] = (self.subdenets[i].alpha1) * e2[pass_index, ...] + self.subdenets[i].level1(
                            e3[pass_index, ...], e1[pass_index, ...])
                        e1[pass_index, ...] = (self.subdenets[i].alpha2) * e1[pass_index, ...] + self.subdenets[i].level2(
                            e2[pass_index, ...], e0[pass_index, ...])
                        e0[pass_index, ...] = (self.subdenets[i].alpha3) * e0[pass_index, ...] + self.subdenets[i].level3(
                            e1[pass_index, ...], None)
                        if self.cal_num_decoders:
                            self.num_decoders += len(pass_index)
                        if i == self.num_subdenet - 1:
                            x_img_out[pass_index, ...] += self.outputs[i](e0[pass_index, ...])
                    else:
                        break
            # if self.stem_method == 'stem_once':
            #     x_img_out = self.output(e0) + img
            # print('yyyyyy: ', x_img_out.max(), x_img_out.min())
            return x_img_out

    def _forward_pretrain(self, features):

        e0, e1, e2, e3, e4 = features
        # print(e0.shape, e1.shape, e2.shape, e3.shape, e4.shape)
        # for i in range(self.num_subdenet): #range(1): #
        #     e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
        # # d3_img = self.outputs[0](e0)
        # d3_img = self.outputs[-1](e0)
        for i in range(self.decoder_idx): #range(1): #
            e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)

        d3_img = self.outputs[self.decoder_idx-1](e0)
        return d3_img
class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        train_size = (1, 3, 256, 256)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

class AdaRevIDLocal(Local_Base, AdaRevIDSlide):
    def __init__(self, *args, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        AdaRevIDSlide.__init__(self, *args, **kwargs)

        train_size = (1, 3, 256, 256)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

def get_parameter_number(net):
    total_num = sum(np.prod(p.size()) for p in net.parameters())
    trainable_num = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)
if __name__ == '__main__':
    net = AdaRevID(test_only=True, pretrain=True, save_memory_decoder=False, encoder='UFPNet', baseblock=['fcnaf', 'fcnaf', 'fcnaf', 'fcnaf']).cuda()

    # for n, p in net.named_parameters()
    #     print(n, p.shape)

    # inp = torch.randn((1, 3, 128, 128)).cuda()
    # print(inp.shape, inp.max())
    # out = net(inp)
    # print(len(out))
    # # print(torch.mean(torch.abs(out['img'][-1] - inp)))
    # print(torch.mean(torch.abs(out[-1] - inp)))
    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    print(macs, params)
    get_parameter_number(net)
    # print(params_2)

