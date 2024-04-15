"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False,
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

            # self.norm_layer = nn.BatchNorm2d(out_channel)
            # self.norm_layer = nn.InstanceNorm2d(out_channel)
            # self.relu = nn.PReLU()
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=5, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=5, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        return self.main(x) + x
##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [ResBlock(n_feat) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size=3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
##########################################################################
class RSNet(nn.Module):
    def __init__(self,in_c=3, out_c=3, n_feat=96, kernel_size=3, bias=False, num_cab=8):
        super(RSNet, self).__init__()
        self.shallow_feat = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                          ResBlock(n_feat))
        self.orb1 = ORB(n_feat, num_cab, CAB=ResBlock)
        self.orb2 = ORB(n_feat, num_cab, CAB=ResBlock)
        self.orb3 = ORB(n_feat, num_cab, CAB=ResBlock)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)
    def forward(self, img):
        x = self.shallow_feat(img)
        x = self.orb1(x)

        x = self.orb2(x)

        x = self.orb3(x)
        x = self.tail(x) + img
        return x
class RSNet_1(nn.Module):
    def __init__(self,in_c=3, out_c=3, n_feat=96, kernel_size=3, bias=False, num_cab=8):
        super(RSNet_1, self).__init__()

        self.shallow_feat = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        orb1 = [ResBlock(n_feat) for _ in range(num_cab)]
        self.orb1 = nn.Sequential(*orb1)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)
    def forward(self, img):
        x = self.shallow_feat(img)
        x = self.orb1(x)
        x = self.tail(x) + img
        return x
class RSNet_cat(nn.Module):
    def __init__(self,in_c=6, out_c=3, n_feat=96, kernel_size=3, bias=False, num_cab=8):
        super(RSNet_cat, self).__init__()
        # ResBlock = ResBlock_fft_bench
        self.shallow_feat = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        orb1 = [ResBlock(n_feat) for _ in range(num_cab)]
        self.orb1 = nn.Sequential(*orb1)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)
    def forward(self, img, img_):
        x = self.shallow_feat(torch.cat([img, img_], dim=1))
        x = self.orb1(x)
        x = self.tail(x) + img
        return x

class DeepDeblur(nn.Module):
    def __init__(self,in_c=3, out_c=3, n_feat=64, kernel_size=3, bias=False, num_cab=19):
        super(DeepDeblur, self).__init__()

        self.stage1 = RSNet_1(in_c, out_c, n_feat, kernel_size, bias, num_cab)
        self.stage2 = RSNet_cat(in_c*2, out_c, n_feat, kernel_size, bias, num_cab)
        self.stage3 = RSNet_cat(in_c*2, out_c, n_feat, kernel_size, bias, num_cab)
        self.up1 = UpSample(3, 0)
        self.up2 = UpSample(3, 0)

    def forward(self, img):
        x_pry = kornia.geometry.build_pyramid(img, 3)
        x1 = self.stage1(x_pry[2])
        x = self.up1(x1)
        x2 = self.stage2(x_pry[1], x)
        x = self.up2(x2)
        x = self.stage3(x_pry[0], x)
        return x, x2, x1

if __name__=='__main__':
    import torch

    net = DeepDeblur().cuda()

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    # print(params)
    # params = float(params[:-3])
    # macs = float(macs[:-4])

    print('FLOPs: ', macs)
    print('params: ', params)