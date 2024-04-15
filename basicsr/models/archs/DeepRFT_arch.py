import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import *



class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResFourier_complex, num_heads=1):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, num_heads=num_heads) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResFourier_complex, num_heads=1):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, num_heads=num_heads) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out
def get_P(phase, fnorm='backward'):
    return torch.fft.irfft2(torch.exp(1j*phase), norm=fnorm)
def get_iP(Phase, fnorm='backward'):
    rf_P = torch.fft.rfft2(Phase, norm=fnorm)
    x_ir_angle = torch.log(rf_P) / 1j
    return x_ir_angle.real, x_ir_angle.imag # torch.abs(en1x_ir_angle) #

class DeepRFT(nn.Module):
    def __init__(self, width=32, num_res=8, window_size=None, inference=False):
        super(DeepRFT, self).__init__()
        self.window_size = window_size
        self.inference = inference
        # ResBlockx = ResBlock
        # ResBlockx = ResFourier_complex_gelu
        # ResBlockx = ResFourier_complex
        # ResBlockx = ResFourier_dual3x3
        ResBlockx = ResFourier_LPF
        # ResBlockx = ResFourier_conv1x1_in_spatial
        # ResBlockx = ResFourier_conv3x3_in_spatial
        # ResBlockx = ResFourier_spatialattn_in_fourier
        # ResBlockx = ResFourier_real_conv3x3
        base_channel = width
        num_heads = [1, 2, 4]
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlockx, num_heads=num_heads[0]),
            EBlock(base_channel*2, num_res, ResBlock=ResBlockx, num_heads=num_heads[1]),
            EBlock(base_channel*4, num_res, ResBlock=ResBlockx, num_heads=num_heads[2]),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlockx, num_heads=num_heads[2]),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlockx, num_heads=num_heads[1]),
            DBlock(base_channel, num_res, ResBlock=ResBlockx, num_heads=num_heads[0])
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        # print(x.shape)
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        if not self.inference:
            z_ = self.ConvsOut[0](z)
            out_3 = z_+x_4
            if self.window_size is not None and (H != self.window_size or W != self.window_size):
                out_3 = window_reversex(out_3, self.window_size//4, H//4, W//4, batch_list)
            outputs.append(out_3)
        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        if not self.inference:
            z_ = self.ConvsOut[1](z)
            out_2 = z_ + x_2
            if self.window_size is not None and (H != self.window_size or W != self.window_size):
                out_2 = window_reversex(out_2, self.window_size // 2, H // 2, W // 2, batch_list)
            outputs.append(out_2)
        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            # outputs.append(z+x)
            out = z + x
            if self.window_size is not None and (H != self.window_size or W != self.window_size):
                out = window_reversex(out, self.window_size, H, W, batch_list)
            outputs.append(out)
            return outputs # [::-1]
        else:
            out = z + x
            if self.window_size is not None and (H != self.window_size or W != self.window_size):
                out = window_reversex(out, self.window_size, H, W, batch_list)
            return out

class LocalResFourier_complex(ResFourier_complex):
    def __init__(self, n_feat, kernel_size_ffc=3, base_size=None, kernel_size=None, fast_imp=False, train_size=None):
        super().__init__(n_feat, kernel_size_ffc)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp
        self.train_size = train_size

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

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
                parts.append(x[:, :, i:i + k1, j:j + k2])
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
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def _pad(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1 - h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (mod_pad_w // 2, mod_pad_w - mod_pad_w // 2, mod_pad_h // 2, mod_pad_h - mod_pad_h // 2)
        x = F.pad(x, pad, 'reflect')
        return x, pad

    def forward(self, x):
        # print(x.shape)
        print(self.kernel_size, self.base_size, self.train_size)
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        # b, c, h, w = x.shape

        # x_res =

        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            x_fft = self.grids(x)  # convert to local windows
            # print(x_fft.shape)
            x_fft = self.main_fft(x_fft) + self.main(x_fft)
            # print(x_fft.shape)
            x_fft = self.grids_inverse(x_fft)  # reverse

        out = x_fft + x #  + x_res
        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, ResFourier_complex):
            # print(m)
            # print(m.main.conv1.in_channels)
            attn = LocalResFourier_complex(n_feat=m.main.conv1.in_channels, kernel_size_ffc=m.main.conv1.kernel_size[0], base_size=base_size, fast_imp=False,
                                  train_size=train_size)
            setattr(model, n, attn)


'''
ref. 
@article{chu2021tsc,
  title={Improving Image Restoration by Revisiting Global Information Aggregation},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class DeepRFTLocalx(Local_Base, DeepRFT):
    def __init__(self, *args, train_size=(1, 3, 256, 256), base_size=None, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        DeepRFT.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class DeepRFTLocal(DeepRFT):
    def __init__(self, width=32, num_res=8, window_size=None, inference=True, kernel_size=(384, 384), fast_imp=False):
        super().__init__(width=width, num_res=num_res, window_size=window_size, inference=inference)
        # N, C, H, W = train_size
        # if base_size is None:
        #     # base_size = (int(H * 1.5), int(W * 1.5))
        #     # base_size = (320, 320)
        #     base_size = (256, 256)
        # self.base_size = base_size
        self.overlap_size = (32, 32)
        print(self.overlap_size)
        # print(kernel_size)
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp

        self.window_size = window_size
        self.inference = inference
        # ResBlockx = ResBlock
        # ResBlockx = ResFourier_complex_gelu
        # ResBlockx = ResFourier_complex
        ResBlockx = ResFourier_conv1x1_in_spatial
        base_channel = width

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlockx),
            EBlock(base_channel * 2, num_res, ResBlock=ResBlockx),
            EBlock(base_channel * 4, num_res, ResBlock=ResBlockx),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlockx),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlockx),
            DBlock(base_channel, num_res, ResBlock=ResBlockx)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)
    def forward_(self, x):
        # print(x.shape)
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)

        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        out = z + x

        return out

    def _pad(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1 - h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (mod_pad_w // 2, mod_pad_w - mod_pad_w // 2, mod_pad_h // 2, mod_pad_h - mod_pad_h // 2)
        x = F.pad(x, pad, 'reflect')
        return x, pad
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
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
                parts.append(x[:, :, i:i + k1, j:j + k2])
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
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
    def forward(self, x):
        # print(self.kernel_size, self.base_size, self.train_size)
        # if self.kernel_size is None and self.base_size:
        #     if isinstance(self.base_size, int):
        #         self.base_size = (self.base_size, self.base_size)
        #     self.kernel_size = list(self.base_size)

        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            x = self.grids(x)  # convert to local windows
            # print(x.shape)
            x = self.forward_(x)
            # print(x_fft.shape)
            x = self.grids_inverse(x)  # reverse

        return x
if __name__=='__main__':
    import torch
    TLPLN = DeepRFT() # .cuda()
    x = torch.randn(1,3,256,256)
    x = x # .cuda()
    y = TLPLN(x)
    print(x.shape, y)

