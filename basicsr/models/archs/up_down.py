import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.models.losses import MSELoss

class DownShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DownShuffle, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels//4, kernel_size=kernel_size,
                                            stride=1, padding=(kernel_size-1)//2, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=(kernel_size-1)//2, bias=False),
                    nn.PixelShuffle(2)
                )

    def forward(self, x):
        return self.up(x)

class Up_ConvTranspose2d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1):
        return self.up(x1)


class UpShuffle_freq(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.up = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=(kernel_size-1)//2, bias=False),
                    nn.PixelShuffle(2)
                )
        self.freq_up = freup_Periodicpadding(in_channels, out_channels)
    def forward(self, x):
        return self.up(x) + self.freq_up(x)
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 2, 2)
                )
    def forward(self, x):
        return self.down(x)


class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H / 2), 0:int(W / 2)] = output[:, :, 0:int(H / 2), 0:int(W / 2)]
        crop[:, :, int(H / 2):H, 0:int(W / 2)] = output[:, :, int(H * 1.5):2 * H, 0:int(W / 2)]
        crop[:, :, 0:int(H / 2), int(W / 2):W] = output[:, :, 0:int(H / 2), int(W * 1.5):2 * W]
        crop[:, :, int(H / 2):H, int(W / 2):W] = output[:, :, int(H * 1.5):2 * H, int(W * 1.5):2 * W]
        crop = F.interpolate(crop, (2 * H, 2 * W))

        return self.post(crop)


class freup_Periodicpadding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(freup_Periodicpadding, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(in_channels, in_channels, 1, 1, 0))

        self.post = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        # N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)


class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        # self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        r = x.size(2)  # h
        c = x.size(3)  # w

        I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()
        I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()

        if r % 2 == 1:  # odd
            ir1, ir2 = r // 2 + 1, r // 2 + 1
        else:  # even
            ir1, ir2 = r // 2 + 1, r // 2
        if c % 2 == 1:  # odd
            ic1, ic2 = c // 2 + 1, c // 2 + 1
        else:  # even
            ic1, ic2 = c // 2 + 1, c // 2

        I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
        I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
        I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
        I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
            I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
            I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5

        I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
        I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
        I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
        I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
            I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
            I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5

        real = I_Mup * torch.cos(I_Pup)
        imag = I_Mup * torch.sin(I_Pup)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output
class DownLift(nn.Module):
    def __init__(self, in_channels, return_loss=False):
        super().__init__()
        self.sum = sum
        self.pu = PU(in_channels, in_channels, return_loss)
        # self.pu2 = PU(in_channels, in_channels, return_loss)

        self.return_loss = return_loss # nn.Conv2d(in_channels, out_channels, 1, 1)
        # else:
        #     self.out_proj = nn.Conv2d(in_channels*4, out_channels, 1, 1)

    def forward(self, x):

        # if self.sum:
        #     x = ss + sd + ds + dd
        # else:

        # x = self.out_proj(x)
        if self.return_loss:
            xo, xe = x[:, :, :, 1::2], x[:, :, :, 0::2]
            s, d, loss1 = self.pu(xo, xe)
            so, se = s[:, :, 1::2, :], s[:, :, 0::2, :]
            do, de = d[:, :, 1::2, :], d[:, :, 0::2, :]
            ss, sd, loss2 = self.pu(so, se)
            ds, dd, loss3 = self.pu(do, de)
            x = torch.cat([ss, sd, ds, dd], dim=0)
            return x, loss1 + loss2 + loss3
        else:
            xo, xe = x[:, :, :, 1::2], x[:, :, :, 0::2]
            s, d = self.pu(xo, xe)
            so, se = s[:, :, 1::2, :], s[:, :, 0::2, :]
            do, de = d[:, :, 1::2, :], d[:, :, 0::2, :]
            ss, sd = self.pu(so, se)
            ds, dd = self.pu(do, de)
            x = torch.cat([ss, sd, ds, dd], dim=0)
            return x

class UpLift(nn.Module):
    def __init__(self, in_channels, return_loss=False):
        super().__init__()

        # self.in_proj = nn.Conv2d(in_channels, out_channels*4, 1, 1)
        self.pu = UP(in_channels, in_channels)
        self.return_loss = return_loss
    def forward(self, x):
        if self.return_loss:
            ss, sd, ds, dd = torch.chunk(x, 4, dim=0)
            so, se, loss1 = self.pu(ss, sd)
            do, de, loss2 = self.pu(ds, dd)
            B, C, H, W = so.shape
            s = torch.zeros([B, C, H * 2, W], device=so.device)
            d = torch.zeros([B, C, H * 2, W], device=so.device)
            s[:, :, 0::2, :] = se
            s[:, :, 1::2, :] = so
            d[:, :, 0::2, :] = de
            d[:, :, 1::2, :] = do
            # s = shuffle_h(torch.cat([se, so], dim=-2))
            # d = shuffle_h(torch.cat([de, do], dim=-2))
            xo, xe, loss3 = self.pu(s, d)
            x = torch.zeros([B, C, H * 2, W * 2], device=so.device)
            x[:, :, :, 0::2] = xe
            x[:, :, :, 1::2] = xo
            # x = shuffle_h(torch.cat([xe, xo], dim=-1))
            return x, loss1 + loss2 + loss3
        else:
            ss, sd, ds, dd = torch.chunk(x, 4, dim=0)
            so, se = self.pu(ss, sd)
            do, de = self.pu(ds, dd)
            B, C, H, W = so.shape
            s = torch.zeros([B, C, H * 2, W], device=so.device)
            d = torch.zeros([B, C, H * 2, W], device=so.device)
            s[:, :, 0::2, :] = se
            s[:, :, 1::2, :] = so
            d[:, :, 0::2, :] = de
            d[:, :, 1::2, :] = do
            # s = shuffle_h(torch.cat([se, so], dim=-2))
            # d = shuffle_h(torch.cat([de, do], dim=-2))
            xo, xe = self.pu(s, d)
            x = torch.zeros([B, C, H * 2, W * 2], device=so.device)
            x[:, :, :, 0::2] = xe
            x[:, :, :, 1::2] = xo
            # x = shuffle_h(torch.cat([xe, xo], dim=-1))
            return x


class PU(nn.Module):
    def __init__(self, in_channels, out_channels, return_loss=False):
        super().__init__()
        self.p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, padding=2, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, groups=1),
            nn.Tanh()
        )
        self.u = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, padding=2, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, groups=1),
            nn.Tanh()
        )
        self.return_loss = return_loss

        self.loss = MSELoss()
        self.alpha1 = 0.01
        self.alpha2 = 0.1
    def forward(self, xo, xe):
        p = self.p(xe)
        d = xo - p
        s = xe + self.u(d)
        if self.return_loss:
            return s, d, self.alpha1 * self.loss(s, xo) + self.alpha2 * self.loss(p, xo)
        else:
            return s, d
class UP(nn.Module):
    def __init__(self, in_channels, out_channels, return_loss=False):
        super().__init__()
        self.p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, padding=2, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, groups=1),
            nn.Tanh()
        )
        self.u = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, padding=2, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, groups=1),
            nn.Tanh()
        )
        self.loss = MSELoss()
        self.alpha1 = 0.01
        self.alpha2 = 0.1
        self.return_loss = return_loss
    def forward(self, s, d):
        xe = s - self.u(d)
        p = self.p(xe)
        xo = d + p
        if self.return_loss:
            return xo, xe, self.alpha1 * self.loss(s, xo) + self.alpha2 * self.loss(p, xo)
        else:
            return xo, xe
def shuffle_h(x):
    x = rearrange(x, 'b c (k h) w -> b c k h w', k=2)
    x = torch.transpose(x, 2, 3).contiguous()
    return rearrange(x, 'b c h k w -> b c (h k) w')
def shuffle_w(x):
    x = rearrange(x, 'b c h (k w) -> b c h k w', k=2)
    x = torch.transpose(x, 3, 4).contiguous()
    return rearrange(x, 'b c h w k -> b c h (w k)')