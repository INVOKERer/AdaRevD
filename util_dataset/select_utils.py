import numpy as np
import glob
import cv2
import os
import kornia
import tqdm
import torch
import spectral
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from natsort import natsorted
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class FocusDataset():
    def __init__(self, train_size=256, overlap_size=32, window_size=1,
                 diff_max=2, diff_max_method=2, cal_method='fft_hsi',
                 preprocess=False, img_write=True, band_select=20,
                 rgb_or_hsi='hsi', low_memory=False, select_list=[], out_range_list=[], pix_max=255.,
                 ext='.png', save_confuse=False
                 ):
        '''
        :param train_size:
        :param overlap_size:
        :param window_size:
        :param diff_max:
        :param cal_method: fft_hsi, fft_rgb, laplace, brenner
        '''
        self.train_size = train_size
        self.preprocess = preprocess
        self.img_write = img_write
        self.ext = ext
        self.save_confuse = save_confuse
        self.band_select = band_select
        self.rgb_or_hsi = rgb_or_hsi
        self.low_memory = low_memory
        self.select_list = select_list
        self.out_range_list = out_range_list
        self.pix_max = pix_max
        self.grid = True
        self.overlap_size = (overlap_size, overlap_size)
        self.window_size = window_size
        self.diff_max = diff_max
        self.diff_max_method = diff_max_method
        self.sel_start = self.select_list[0] - self.out_range_list[0]
        self.sel_end = self.select_list[-1] - self.out_range_list[0]
        # self.overlap_size = (16, 16)
        print(self.overlap_size)
        self.kernel_size = [train_size, train_size]
        # self.pix_max = pix_max
        ###########################
        self.h = None
        self.w = None
        self.cal_method = cal_method
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
        self.nr = num_row
        self.nc = num_col
        step_j = k2 if num_col == 1 else stride[1] # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0] # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
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
                if self.cal_method == 'fft_rgb':
                    parts.append(self.cal_best_fft_rgb(x[:, :, i:i + k1, j:j + k2].cuda()))
                elif self.cal_method == 'laplace_rgb':
                    parts.append(self.cal_best_laplace_rgb(x[:, :, i:i + k1, j:j + k2].cuda()))
                elif self.cal_method == 'brenner_rgb':
                    parts.append(self.cal_best_brenner_rgb(x[:, :, i:i + k1, j:j + k2].cuda()))
                elif self.cal_method == 'laplace_hsi':
                    parts.append(self.cal_best_laplace(x[:, :, i:i + k1, j:j + k2].cuda()))
                elif self.cal_method == 'brenner_hsi':
                    parts.append(self.cal_best_brenner(x[:, :, i:i + k1, j:j + k2].cuda()))
                elif self.cal_method == 'fft_hsi':
                    parts.append(self.cal_best_fft(x[:, :, i:i + k1, j:j + k2].cuda()))
                # parts.append(x[3, :, i:i + k1, j:j + k2].unsqueeze(0))
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts
    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)
    def grids_inverse(self, outs):
        b, c, h, w = self.original_size
        preds = torch.zeros([1, c, h, w]).to(outs.device)
        # b, c, h, w = self.original_size
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # print(outs.shape)
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size[0], :] *= self.fuse_matrix_h2.to(outs.device)
            if i + k1*2 - self.ek1 < h:
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
                outs[cnt, :,  -self.overlap_size[0]:, :] *= self.fuse_matrix_h1.to(outs.device)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] *= self.fuse_matrix_eh2.to(outs.device)
            if i + k1*2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] *= self.fuse_matrix_eh1.to(outs.device)

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size[1]] *= self.fuse_matrix_w2.to(outs.device)
            if j + k2*2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size[1]:] *= self.fuse_matrix_w1.to(outs.device)
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] *= self.fuse_matrix_ew2.to(outs.device)
            if j + k2*2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] *= self.fuse_matrix_ew1.to(outs.device)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds # / count_mt

    def forward(self, inp_img, get_idx=False):
        B, C, H, W = inp_img.shape
        if get_idx:
            parts, idx = self.cal_best_fft_rgb(inp_img.cuda(), True)
            return idx
        else:
            x = self.grids(inp_img)
            # print(x)
            x = self.grids_inverse(x)
            return x[:,:,:H,:W].contiguous()

    def cal_best_fft_rgb(self, inp, get_idx=False):
        # b = inp.shape[0]
        # inpx = inp.clone()
        # mean_x = torch.mean(inp[0])
        # for i in range(b):
        #     # inp[i, ...] = kornia.enhance.normalize_min_max(inp[i, ...], min_val=k_min, max_val=k_max.cpu().numpy())
        #     inp[i, ...] = inp[i, ...] / torch.mean(inp[i, ...]) * mean_x
        x = torch.fft.rfft2(inp)
        x.real = torch.relu(x.real)
        x.imag = torch.relu(x.imag)
        x = torch.fft.irfft2(x) - inp / 2.
        # x = torch.fft.fftshift(x, dim=[-2, -1])
        # x = kornia.geometry.center_crop(x, [2, 2])
        x = x[:, :, :self.window_size, :self.window_size] + x[:, :, -self.window_size:, :self.window_size] + \
            x[:, :, :self.window_size, -self.window_size:] + x[:, :, -self.window_size:, -self.window_size:]
        x, idx = torch.max(torch.mean(x, dim=[1, 2, 3], keepdim=False), dim=0)

        if get_idx:
            return idx
        else:
            return inp[idx, ...].unsqueeze(0)

    def cal_best_laplace_rgb(self, inp, get_idx=False):
        x = kornia.filters.laplacian(inp, kernel_size=5)
        x, idx = torch.max(torch.mean(x**2, dim=[1, 2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return inp[idx, ...].unsqueeze(0)
    def cal_best_brenner_rgb(self, inp, get_idx=False):
        # kernel = torch.zeros([5, 5])
        # kernel[2, 2] = -1
        # kernel[2, -1] = 1
        # x = kornia.filters.filter2d(inp, kernel.unsqueeze(0))
        x = torch.nn.functional.pad(inp, (0, 2, 0, 0))
        x = x[:, :, :, 2:] - inp
        x, idx = torch.max(torch.mean(x**2, dim=[1, 2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return inp[idx, ...].unsqueeze(0)

    def check_idx(self, idx):

        idx_new1 = []
        idx_new2 = []
        c_range = list(range(idx.shape[0]))
        center_band = len(c_range) // 2
        # t_band_start = len(c_range) // 3
        # t_band_end = len(c_range)*2 // 3
        # center_idx = torch.mean(idx[t_band_start:t_band_end])
        c_range1 = c_range[center_band:]
        c_k_forward = None
        for c in c_range1:
            k = idx[c]
            # print(inp_c.shape) # b 1 h w
            if c_k_forward is None:
                c_k_forward = k
                # inp_z = torch.index_select(inp_c, dim=0, index=k)
            if abs(k.item() - c_k_forward.item()) >= self.diff_max:
                k = c_k_forward
            else:
                c_k_forward = k
            idx_new1.append(k.item())
        c_range2 = c_range[:center_band]
        c_k_forward = None
        for c in c_range2[::-1]:
            k = idx[c]
            # print(inp_c.shape) # b 1 h w
            if c_k_forward is None:
                c_k_forward = k
                # inp_z = torch.index_select(inp_c, dim=0, index=k)
            if abs(k.item() - c_k_forward.item()) >= self.diff_max:
                k = c_k_forward
            else:
                c_k_forward = k
            idx_new2.append(k.item())
        idx_new2 = idx_new2[::-1]
        idx_new = idx_new2 + idx_new1
        return idx_new
    def fuse_channel(self, inp, idx):
        results = []
        # c_k_forward = None
        c_range = list(range(idx.shape[0]))
        idx = self.check_idx(idx)
        # center_band = len(c_range) // 2
        # c_range = c_range[center_band:] + c_range[:center_band]
        for c in c_range:
            k = idx[c]
            inp_c = torch.index_select(inp, dim=1, index=torch.tensor(c, device=inp.device))
            # print(inp_c.shape) # b 1 h w
            # if c_k_forward is None:
            #     c_k_forward = k
            #     # inp_z = torch.index_select(inp_c, dim=0, index=k)
            # if abs(k - c_k_forward) >= self.diff_max:
            #     k = c_k_forward
            inp_z = torch.index_select(inp_c, dim=0, index=torch.tensor(k, device=inp.device))

            # print(inp_z.shape)
            results.append(inp_z)  # inp[idx, i, :, :].unsqueeze(0))
        # results = results[center_band + 1:] + results[:center_band + 1]
        result = torch.cat(results, dim=1)
        return result
    def cal_best_fft(self, inp, get_idx=False):
        x = torch.fft.rfft2(inp)
        x.real = torch.relu(x.real)
        x.imag = torch.relu(x.imag)
        x = torch.fft.irfft2(x) - inp / 2.
        # x = torch.fft.fftshift(x, dim=[-2, -1])
        # x = kornia.geometry.center_crop(x, [2, 2])
        x = x[:, :, :self.window_size, :self.window_size] + x[:, :, -self.window_size:, :self.window_size] + \
            x[:, :, :self.window_size, -self.window_size:] + x[:, :, -self.window_size:, -self.window_size:]
        x, idx = torch.max(torch.mean(x, dim=[2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return self.fuse_channel(inp, idx)
    def cal_best_laplace(self, inp, get_idx=False):
        x = kornia.filters.laplacian(inp, kernel_size=5)
        x, idx = torch.max(torch.mean(x**2, dim=[2, 3], keepdim=False), dim=0, keepdim=False)

        if get_idx:
            return idx
        else:
            return self.fuse_channel(inp, idx)
    def cal_best_brenner(self, inp, get_idx=False):
        # kernel = torch.zeros([5, 5])
        # kernel[2, 2] = -1
        # kernel[2, -1] = 1
        # x = kornia.filters.filter2d(inp, kernel.unsqueeze(0))
        x = torch.nn.functional.pad(inp, (0, 2, 0, 0))
        x = x[:, :, :, 2:] - inp
        x, idx = torch.max(torch.mean(x ** 2, dim=[2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return self.fuse_channel(inp, idx)

    @staticmethod
    def judge_white_frelu_torch(img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.
        # img = kornia.image_to_tensor(img, keepdim=False)
        c = img.shape[1]
        if c > 1:
            img = kornia.color.rgb_to_grayscale(img)
        # print(img.shape)
        img_min, img_max = img.min(), img.max()
        img = (img-img_min)/(img_max-img_min)
        # img = kornia.enhance.normalize_min_max(img, 0., 1.)
        x = torch.fft.rfft2(img)
        # y_blur_f = np.fft.fftshift(y) 0.017

        x.real = torch.relu(x.real)
        x.imag = torch.relu(x.imag)
        x = torch.fft.irfft2(x)
        resx = x - img / 2.
        resx = resx.view(resx.shape[0], -1)
        resx = torch.quantile(resx, torch.tensor([0.05, 0.95], dtype=resx.dtype, device=resx.device), dim=1, keepdim=False)
        # print(resx.shape)
        res1, res2 = resx.chunk(2, dim=0)
        # print(res1.shape)
        res = res2 - res1 # .squeeze(1)
        # print(res)
        # _, idx = torch.where(res > thr)
        # print(idx)
        # print(res.item())
        return res.item() # > thr

    @staticmethod
    def sort_key(x):
        x = os.path.basename(x)
        x, ext = os.path.splitext(x)
        x = x.split('_')[-1]
        # print(x)
        return float(x)

    def get_rois(self, ):
        return
    def get_blank_img(self, ):
        return
    def get_hr_patch(self, img_blur_list, j, i, x_blank):

        x_blur_list = []
        for idx, c in enumerate(img_blur_list):
            # print(c)
            if self.rgb_or_hsi == 'rgb':
                x_bx = cv2.imread(c)[j:j + self.patch_size, i:i + self.patch_size, :]
            else:
                x_bx = spectral.open_image(c)

                if idx < self.sel_start or idx > self.sel_end:
                    x_bx = np.array(x_bx[j:j + self.patch_size, i:i + self.patch_size, :])
                else:
                    x_bx = np.array(x_bx[j:j + self.patch_size, i:i + self.patch_size, self.band_select])
            if self.preprocess:
                # print(x_blank.shape, x_bx.shape)
                x_b = x_bx / x_blank[j:j + self.patch_size, i:i + self.patch_size, :]  # [512:-512, 512:-512, :]
            else:
                x_b = x_bx / self.pix_max
            x_b = kornia.image_to_tensor(x_b, keepdim=False)
            # print(x_b.shape)
            x_blur_list.append(x_b.cuda())  #
            # hr_patch = torch.cat(x_blur_list, dim=0)
            # hr_patch = torch.clamp(hr_patch, 0., 1.)
        # continue
        hr_patch = torch.cat(x_blur_list, dim=0)
        hr_patch = torch.clamp(hr_patch, 0., 1.)
        return hr_patch
    def get_dataset(self, out_dir, img_blur_root, rois):
        rois = self.get_rois()

        out_dir_sharp = os.path.join(out_dir, 'sharp')
        out_dir_blur = os.path.join(out_dir, 'blur')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_sharp, exist_ok=True)
        os.makedirs(out_dir_blur, exist_ok=True)
        if self.save_confuse:
            out_dir_confuse = os.path.join(os.path.dirname(out_dir), 'confuse')
            out_dir_confuse_sharp = os.path.join(out_dir_confuse, 'sharp')
            out_dir_confuse_blur = os.path.join(out_dir_confuse, 'blur')
            os.makedirs(out_dir_confuse, exist_ok=True)
            os.makedirs(out_dir_confuse_sharp, exist_ok=True)
            os.makedirs(out_dir_confuse_blur, exist_ok=True)
        for roi in tqdm.tqdm(rois):
            # print(roi)
            roi_i = os.path.dirname(os.path.dirname(os.path.dirname(roi)))
            slide_name = os.path.basename(roi_i)  # .split('/')[-2]  # [-3:] + roi_i
            slide_type = os.path.basename(os.path.dirname(roi_i))
            print(slide_name)
            # img_blur_list = []
            # img_blur_list_all = []
            # for sel in out_range_list:
            img_blur_list = glob.glob(os.path.join(roi, 'digi', '*' + self.ext))
            img_blur_list = natsorted(img_blur_list, key=self.sort_key)
            # if sel in select_list:
            #     img_blur_list.append(os.path.join(roi, roi_i + '_' + str(sel) + '.hdr'))
            if self.preprocess:
                x_blank = self.get_blank_img()
                # print(x_blank.type)
            # img_blur_list = glob.glob(img_root+'/' + roi_i[:-2] +'_*')
            # print(roi_i[:-2], len(img_blur_list))
            x_blur_list = []
            # print(img_blur_list)

            # for i, c in tqdm.tqdm(enumerate(img_blur_list)):
            if not self.low_memory:
                for idx, c in enumerate(img_blur_list):
                    # print(c)
                    if self.rgb_or_hsi == 'rgb':
                        x_bx = cv2.imread(c)
                    else:
                        x_bx = spectral.open_image(c)

                        if idx < self.sel_start or idx > self.sel_end:
                            x_bx = np.array(x_bx[:, :, :])
                        else:
                            x_bx = np.array(x_bx[:, :, self.band_select])
                    # x_bx = x_bx[512:-512, 512:-512, :]
                    # print(x_blank.shape)
                    # x_blank = x_blank
                    # print(x_blank.shape, x_bx.shape)
                    if self.preprocess:
                        # print(x_blank.shape, x_bx.shape)
                        x_b = x_bx / x_blank  # [512:-512, 512:-512, :]
                    else:
                        x_b = x_bx / self.pix_max
                    x_b = kornia.image_to_tensor(x_b, keepdim=False)
                    x_blur_list.append(x_b.cuda())  #
                # continue
                x = torch.cat(x_blur_list, dim=0)
                x = torch.clamp(x, 0., 1.)
            num_patch = 0
            if not self.low_memory:
                _, _, h, w = x.shape
            else:
                x_tmp = cv2.imread(img_blur_list[0])
                h, w, _ = x_tmp.shape
            w1 = list(np.arange(0, w - self.patch_size, self.patch_size - self.overlap, dtype=np.int32))
            h1 = list(np.arange(0, h - self.patch_size, self.patch_size - self.overlap, dtype=np.int32))
            w1.append(w - self.patch_size)
            h1.append(h - self.patch_size)

            for i in w1:
                for j in h1:
                    num_patch += 1
                    if not self.low_memory:
                        hr_patch = x[:, :, j:j + self.patch_size, i:i + self.patch_size]
                    else:
                        x_blur_list = []
                        for idx, c in enumerate(img_blur_list):
                            # print(c)
                            if self.rgb_or_hsi == 'rgb':
                                x_bx = cv2.imread(c)[j:j + self.patch_size, i:i + self.patch_size, :]
                            else:
                                x_bx = spectral.open_image(c)

                                if idx < sel_start or idx > sel_end:
                                    x_bx = np.array(x_bx[j:j + self.patch_size, i:i + self.patch_size, :])
                                else:
                                    x_bx = np.array(x_bx[j:j + self.patch_size, i:i + self.patch_size, self.band_select])
                            if self.preprocess:
                                # print(x_blank.shape, x_bx.shape)
                                x_b = x_bx / x_blank[j:j + self.patch_size, i:i + self.patch_size, :]  # [512:-512, 512:-512, :]
                            else:
                                x_b = x_bx / self.pix_max
                            x_b = kornia.image_to_tensor(x_b, keepdim=False)
                            # print(x_b.shape)
                            x_blur_list.append(x_b.cuda())  #
                            # hr_patch = torch.cat(x_blur_list, dim=0)
                            # hr_patch = torch.clamp(hr_patch, 0., 1.)
                        # continue
                        hr_patch = torch.cat(x_blur_list, dim=0)
                        hr_patch = torch.clamp(hr_patch, 0., 1.)
                    b = hr_patch.shape[0]
                    # print(b)
                    sel_patch = hr_patch[sel_start:sel_end, ...]
                    # print(hr_patch.shape, i, j)
                    torch.cuda.synchronize()
                    start = time.time()
                    idx_fft = self.cal_best_fft_rgb(sel_patch, True) + sel_start

                    torch.cuda.synchronize()
                    end1 = time.time()
                    idx_brenner = self.cal_best_brenner_rgb(sel_patch, True) + sel_start

                    # _, idx_laplace = gridx.cal_best_laplace_rgb(hr_patch, True)
                    torch.cuda.synchronize()
                    end2 = time.time()
                    idx_laplace = self.cal_best_laplace_rgb(sel_patch, True) + sel_start

                    torch.cuda.synchronize()
                    # print(idx_fft)
                    if idx_fft.item() == idx_brenner.item() or idx_fft.item() == idx_laplace.item():
                        idx_sharp = idx_fft.item()
                    else:
                        idx_sharp = 0
                    center = len(self.out_range_list) // 2
                    if self.img_write:
                        if (idx_sharp <= sel_start + 1 or idx_sharp >= sel_end - 1):
                            print(idx_fft.item(), idx_laplace.item(), idx_brenner.item())
                            if self.save_confuse:
                                k_name = os.path.basename(img_blur_root)

                                sharp_patch = hr_patch[idx_fft.item(), ...].unsqueeze(0)
                                sharp_patch = kornia.tensor_to_image(sharp_patch) * 255.
                                fft_name = os.path.basename(img_blur_list[idx_fft.item()])
                                out_way = os.path.join(out_dir_confuse_sharp,
                                                       slide_type + '_' + slide_name + 'x' + str(num_patch) + '_' + fft_name)
                                # cv2.imwrite(out_way, np.uint8(sharp_patch))
                                for i_image in range(b):
                                    if i_image == idx_fft.item():
                                        continue
                                    blur_patch = kornia.tensor_to_image(hr_patch[i_image, ...])
                                    # blur_patch = kornia.tensor_to_image(blur_patch)
                                    blur_patch = blur_patch * 255.
                                    out_way = os.path.join(out_dir_confuse_blur,
                                                           slide_type + '_' + slide_name + 'x' + str(num_patch) + '_' + str(
                                                               i_image) + '.png')
                                cv2.imwrite(out_way, np.uint8(blur_patch))
                        else:
                            sharp_name = os.path.basename(img_blur_list[idx_sharp])

                            sharp_patch = hr_patch[idx_sharp, ...].unsqueeze(0)
                            sharp_patch = kornia.tensor_to_image(sharp_patch) * 255.
                            out_way = os.path.join(out_dir_sharp,
                                                   slide_type + '_' + slide_name + '_' + str(i) + '-' + str(
                                                       j) + '_' + sharp_name)
                            cv2.imwrite(out_way, np.uint8(sharp_patch))
                            for i_image in range(b):
                                # print(len(img_blur_list))
                                if i_image == idx_sharp:
                                    continue
                                blur_name = os.path.basename(img_blur_list[i_image])
                                blur_patch = kornia.tensor_to_image(hr_patch[i_image, ...])
                                # blur_patch = kornia.tensor_to_image(blur_patch)
                                blur_patch = blur_patch * 255.
                                out_way = os.path.join(out_dir_blur,
                                                       slide_type + '_' + slide_name + '_' + str(i) + '-' + str(
                                                           j) + '_' + blur_name)
                                cv2.imwrite(out_way, np.uint8(blur_patch))
        return

class AutoFocusDataset(FocusDataset):
    def __init__(self, *args, **kwargs):
        FocusDataset.__init__(self, *args, **kwargs)
    def get_rois(self, ):
        return
    def get_blank_img(self, ):
        return
    def get_dataset(self, out_dir, img_blur_root, rois):
        return
