import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    if isinstance(padder_size, int):
        padder_size_h = padder_size
        padder_size_w = padder_size
    else:
        padder_size_h, padder_size_w = padder_size
    mod_pad_h = (padder_size_h - h % padder_size_h) % padder_size_h
    mod_pad_w = (padder_size_w - w % padder_size_w) % padder_size_w
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x

def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]
def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    # print(windows[:batch_list[0], ...].shape)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    if torch.is_complex(windows):
        res = torch.complex(torch.zeros([B, C, H, W]), torch.zeros([B, C, H, W]))
        res = res.to(windows.device)
    else:
        res = torch.zeros([B, C, H, W], dtype=windows.dtype, device=windows.device)

    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res

def window_partitionxy(x, window_size, start=[0, 0]):
    s_h, s_w = start
    assert 0 <= s_h < window_size and 0 <= s_w < window_size
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main, b_main = window_partitionx(x[:, :, s_h:, s_w:], window_size)
    # print(x_main.shape, b_main, x[:, :, s_h:, s_w:].shape)
    if s_h == 0 and s_w == 0:
        return x_main, b_main
    if s_h != 0 and s_w != 0:
        x_l = window_partitions(x[:, :, -h:, :window_size], window_size)
        b_l = x_l.shape[0] + b_main[-1]
        b_main.append(b_l)
        x_u = window_partitions(x[:, :, :window_size, -w:], window_size)
        b_u = x_u.shape[0] + b_l
        b_main.append(b_u)
        x_uu = x[:, :, :window_size, :window_size]
        b_uu = x_uu.shape[0] + b_u
        b_main.append(b_uu)
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_l, x_u, x_uu], dim=0), b_main

def window_reversexy(windows, window_size, H, W, batch_list, start=[0, 0]):
    s_h, s_w = start
    assert 0 <= s_h < window_size and 0 <= s_w < window_size

    if s_h == 0 and s_w == 0:
        x_main = window_reversex(windows, window_size, H, W, batch_list)
        return x_main
    else:
        h, w = window_size * (H // window_size), window_size * (W // window_size)
        # print(windows[:batch_list[-4], ...].shape, batch_list[:-3], H-s_h, W-s_w)
        x_main = window_reversex(windows[:batch_list[-4], ...], window_size, H-s_h, W-s_w, batch_list[:-3])
        B, C, _, _ = x_main.shape
        res = torch.zeros([B, C, H, W], device=windows.device)
        x_uu = window_reverses(windows[batch_list[-2]:, ...], window_size, window_size, window_size)
        res[:, :, :window_size, :window_size] = x_uu[:, :, :, :]
        x_l = window_reverses(windows[batch_list[-4]:batch_list[-3], ...], window_size, h, window_size)
        res[:, :, -h:, :window_size] = x_l
        x_u = window_reverses(windows[batch_list[-3]:batch_list[-2], ...], window_size, window_size, w)
        res[:, :, :window_size, -w:] = x_u[:, :, :, :]

        res[:, :, s_h:, s_w:] = x_main
        return res
class WindowPartition(nn.Module):
    def __init__(self, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
    def forward(self, x):
        if self.window_size is None:
            return x, []
        H, W = x.shape[-2:]
        if H > self.window_size and W > self.window_size:
            if not self.shift_size:
                x, batch_list = window_partitionx(x, self.window_size)
                return x, batch_list
            else:
                x, batch_list = window_partitionxy(x, self.window_size, [self.shift_size, self.shift_size])
                return x, batch_list
        else:
            return x, []

class WindowReverse(nn.Module):
    def __init__(self, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x, H, W, batch_list):
        # print(x.shape, batch_list)
        if self.window_size is None:
            return x
        if len(batch_list) > 0 and (H > self.window_size and W > self.window_size):
            if not self.shift_size:
                x = window_reversex(x, self.window_size, H, W, batch_list)
            else:
                x = window_reversexy(x, self.window_size, H, W, batch_list, [self.shift_size, self.shift_size])
        return x

class GridNet(nn.Module):
    def __init__(self, window_size=64, overlap_size=16):

        super(GridNet, self).__init__()

        self.train_size = window_size
        self.kernel_size = (window_size, window_size)
        self.overlap_size = (overlap_size, overlap_size)
        self.batch_size = None
        # self.nr = None

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])
        self.stride = stride
        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1
        # self.nr = num_row
        # self.nc = num_col

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
                parts.append(x[:, :, i:i + k1, j:j + k2].unsqueeze(0))
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        # if nr is None:
        #     nr = self.nr
        # if nc is None:
        #     nc = self.nc
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])
        nr = (h - self.overlap_size[0] - 1) // stride[0] + 1
        nc = (w - self.overlap_size[1] - 1) // stride[1] + 1
        self.h = h
        self.w = w
        self.ek1 = nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = nc * self.stride[1] + self.overlap_size[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, 1, self.overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, 1, self.overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, 1, self.overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, 1, self.overlap_size[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, 1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, 1, self.ek1, 1)

    def grids_inverse(self, outs, original_size=None):
        if original_size is None:
            original_size = self.original_size
        preds = torch.zeros(original_size).to(outs.device)
        b, c, h, w = original_size

        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :, :self.overlap_size[0], :] *= self.fuse_matrix_h2.to(outs.device)
            if i + k1 * 2 - self.ek1 < h:
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
                outs[cnt, :, :, -self.overlap_size[0]:, :] *= self.fuse_matrix_h1.to(outs.device)
            if i + k1 == h:
                outs[cnt, :, :, :self.ek1, :] *= self.fuse_matrix_eh2.to(outs.device)
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, :, -self.ek1:, :] *= self.fuse_matrix_eh1.to(outs.device)

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :, :self.overlap_size[1]] *= self.fuse_matrix_w2.to(outs.device)
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, :, -self.overlap_size[1]:] *= self.fuse_matrix_w1.to(outs.device)
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :, :self.ek2] *= self.fuse_matrix_ew2.to(outs.device)
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, :, -self.ek2:] *= self.fuse_matrix_ew1.to(outs.device)
            # print(preds[:, :, i:i + k1, j:j + k2].shape, outs[cnt, :, :, :, :].shape)
            preds[:, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :, :]
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, inp_img, grid=True, batch_size=None, original_size=None):

        if grid:
            self.batch_size = inp_img.shape[0]
            x = self.grids(inp_img)
            x = rearrange(x, 'n b c h w -> (n b) c h w')
        else:
            if batch_size is None:
                batch_size = self.batch_size
            x = rearrange(inp_img, '(n b) c h w -> n b c h w', b=batch_size)
            x = self.grids_inverse(x, original_size)

        return x


if __name__ == '__main__':
    import resource

    net = GridNet().cuda()


    # for n, p in net.named_parameters()
    #     print(n, p.shape)

    inp = torch.randn((4, 3, 256, 256)).cuda()
    print(inp.shape, inp.max())
    out1 = net(inp, True)
    out = net(out1, False)
    print(torch.mean(torch.abs(out - inp)))
