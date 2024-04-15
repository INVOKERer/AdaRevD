from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (single_rgb_paths_from_folder,
                                    paired_paths_from_lmdb, single_focus_rgb_paths_from_subfolder,
                                    paired_paths_from_meta_info_file, single_rgb_paths_from_subfolder)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation, paired_center_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
import spectral
import glob
import random
import numpy as np
import kornia
import torch
import os

class Dataset_Classify(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.range = opt['range'] if 'range' in opt else None
        self.flag = opt['flag'] if 'flag' in opt else 'color'
        self.lq_folder = opt['dataroot_lq']
        self.gt_class = {}
        gt_class_name = opt['gt_class']
        self.gt_class_name = gt_class_name
        # print(gt_class_name)
        for i in range(len(gt_class_name)):
            if isinstance(gt_class_name[i], list):
                for j in range(len(gt_class_name[i])):
                    self.gt_class[gt_class_name[i][j]] = i
            else:
                self.gt_class[gt_class_name[i]] = i
        print(self.gt_class)
        self.gt_cal = opt['gt_cal'] if 'gt_cal' in opt else False
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        pths = [self.lq_folder]
        self.paths = single_rgb_paths_from_subfolder(pths, ['lq'], self.filename_tmpl, self.range, self.gt_class_name)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print(self.paths)
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        class_name = lq_path.split('/')[-2]

        gt_cls = torch.tensor([self.gt_class[class_name]])
        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_lq_bytes, flag=self.flag, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        if len(img_lq.shape) == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)
        # augmentation for training
        if self.opt['phase'] == 'train':

            # flip, rotation augmentations
            if self.geometric_augs:
                img_lq = random_augmentation(img_lq)[0]

        # print(img_lq.shape)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)
        # print(img_lq.shape)
        flag_aug = random.randint(0, 4)
        if flag_aug == 0:
            factor = random.randint(80, 101) / 100.
            img_lq = img_lq * factor

            # kornia.enhance.adjust_contrast(img_lq, factor, clip_output=True)

        elif flag_aug == 1:
            factor = random.randint(100, 121) / 100.
            img_lq = img_lq * factor
            # img_lq = kornia.enhance.adjust_brightness(img_lq, factor, clip_output=True)

        elif flag_aug == 2:
            factor = random.randint(80, 121) / 100.
            img_lq = kornia.enhance.adjust_gamma(img_lq, factor)

        img_lq = torch.clamp(img_lq, 0., 1.)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        # print({
        #     'gt': gt_cls,
        #     'lq_path': lq_path
        # })
        # print(img_gt.shape, img_lq.shape)
        return {
            'lq': img_lq,
            'gt': gt_cls,
            'lq_path': lq_path,
            'gt_path': lq_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_Multi_Class_Paired_Focus(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.range = opt['range'] if 'range' in opt else None
        self.flag = opt['flag'] if 'flag' in opt else 'color'
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.gt_cal = opt['gt_cal'] if 'gt_cal' in opt else False
        self.scalex = opt['scalex'] if 'scalex' in opt else 1.
        class_name = opt['img_class']
        self.aug_pre = opt['aug_pre'] if 'aug_pre' in opt else [0, 1, 2]
        self.class_name = class_name
        # print(self.opt)
        # print(self.scalex, self.gt_cal)
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            if self.gt_cal:
                pths = [self.lq_folder, self.gt_folder]
            else:
                pths = [self.lq_folder]
            self.paths = single_focus_rgb_paths_from_subfolder(pths, ['lq'], self.filename_tmpl, self.range, self.class_name)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print(self.paths)
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        file_name = os.path.basename(lq_path)
        file_name_ = file_name.split('_')
        file_name_x = file_name_[0]
        for x in file_name_[1:-1]:
            file_name_x = file_name_x + '_' + x
        file_name_b, ext = os.path.splitext(file_name)
        class_name = lq_path.split('/')[-3]
        gt_path_ = os.path.join(self.gt_folder, class_name, 'sharp', file_name_x + '_*'+ext)

        # try:
        gt_path = glob.glob(gt_path_)[0]
        file_name_s, ext = os.path.splitext(gt_path)
        # except:
        #     print(file_name_x, gt_path_, glob.glob(gt_path_))

        gt_distance = file_name_s.split('_')[-1]
        blur_distance = file_name_b.split('_')[-1]
        # print(self.opt)
        # print(self.scalex, self.gt_cal)
        if isinstance(self.scalex, dict):
            distance = (float(gt_distance) - float(blur_distance)) * self.scalex[class_name]
        else:
            distance = (float(gt_distance) - float(blur_distance)) * self.scalex
        # print(file_name_s.split('_'), file_name_b.split('_'), distance)
        img_gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_gt_bytes, flag=self.flag, float32=True)
        except:
            print(gt_path, file_name_)
            raise Exception("gt path {} not working".format(gt_path))

        # lq_path_list = glob.glob(os.path.join(self.lq_folder, file_name_[0] + '_*_' + file_name_[-1]))
        # lq_path = random.choice(lq_path_list)
        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_lq_bytes, flag=self.flag, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)
        if len(img_lq.shape) == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=False,
                                    float32=True)
        distance = torch.tensor([distance])
        # flag_aug = random.randint(0, 4)
        # flag_add = random.randint(0, 1)
        # if flag_add == 0:
        #     add_factor = random.randint(-20, 21) / 100.
        #     img_lq = img_lq + add_factor
        #     img_gt = img_gt + add_factor
        if isinstance(self.aug_pre, dict):
            flag_aug = random.choice(self.aug_pre[class_name])
            if flag_aug == 'lower':
                flag_add = random.randint(0, 1)
                if flag_add == 0:
                    add_factor = random.randint(-20, 0) / 100.
                    img_lq = img_lq + add_factor
                    img_gt = img_gt + add_factor
                factor = random.randint(50, 100) / 100.
                img_lq = img_lq * factor
                img_gt = img_gt * factor
                # kornia.enhance.adjust_contrast(img_lq, factor, clip_output=True)
            elif flag_aug == 'higher':
                flag_add = random.randint(0, 1)
                if flag_add == 0:
                    add_factor = random.randint(0, 20) / 100.
                    img_lq = img_lq + add_factor
                    img_gt = img_gt + add_factor
                factor = random.randint(100, 150) / 100.
                img_lq = img_lq * factor
                img_gt = img_gt * factor
                # img_lq = kornia.enhance.adjust_brightness(img_lq, factor, clip_output=True)
            elif flag_aug == 'gamma':
                factor = random.randint(80, 120) / 100.
                img_lq = kornia.enhance.adjust_gamma(img_lq, factor)
                img_gt = kornia.enhance.adjust_gamma(img_gt, factor)
            # elif flag_aug == 'adjust_brightness':
            #     factor = random.randint(0, 21) / 100.
            #     img_lq = kornia.enhance.adjust_brightness(img_lq, factor, clip_output=True)
            #     img_gt = kornia.enhance.adjust_brightness(img_gt, factor, clip_output=True)
        else:
            flag_aug = random.randint(0, 5)
            if flag_aug == 1:
                factor = random.randint(50, 100) / 100.
                img_lq = img_lq * factor
                img_gt = img_gt * factor
                # kornia.enhance.adjust_contrast(img_lq, factor, clip_output=True)
            elif flag_aug == 2:
                factor = random.randint(100, 150) / 100.
                img_lq = img_lq * factor
                img_gt = img_gt * factor
                # img_lq = kornia.enhance.adjust_brightness(img_lq, factor, clip_output=True)
            elif flag_aug == 3:
                factor = random.randint(80, 120) / 100.
                img_lq = kornia.enhance.adjust_gamma(img_lq, factor)
                img_gt = kornia.enhance.adjust_gamma(img_gt, factor)

        img_lq = torch.clamp(img_lq, 0., 1.)
        img_gt = torch.clamp(img_gt, 0., 1.)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print(img_gt.shape, img_lq.shape)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'distance': distance
        }

    def __len__(self):
        return len(self.paths)

class Dataset_Multi_Paired_Focus(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.range = opt['range'] if 'range' in opt else None
        self.flag = opt['flag'] if 'flag' in opt else 'color'
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.gt_cal = opt['gt_cal'] if 'gt_cal' in opt else False
        self.scalex = opt['scalex'] if 'scalex' in opt else 1.
        # print(self.opt)
        # print(self.scalex, self.gt_cal)
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            if self.gt_cal:
                pths = [self.lq_folder, self.gt_folder]
            else:
                pths = [self.lq_folder]
            self.paths = single_rgb_paths_from_folder(pths, ['lq'], self.filename_tmpl, self.range)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print(self.paths)
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        file_name = os.path.basename(lq_path)
        file_name_ = file_name.split('_')
        file_name_x = file_name_[0]
        for x in file_name_[1:-1]:
            file_name_x = file_name_x + '_' + x
        file_name_b, ext = os.path.splitext(file_name)

        gt_path_ = os.path.join(self.gt_folder, file_name_x + '_*'+ext)

        # try:
        gt_path = glob.glob(gt_path_)[0]
        file_name_s, ext = os.path.splitext(gt_path)
        # except:
        #     print(file_name_x, gt_path_, glob.glob(gt_path_))

        gt_distance = file_name_s.split('_')[-1]
        blur_distance = file_name_b.split('_')[-1]
        # print(self.opt)
        # print(self.scalex, self.gt_cal)
        distance = (float(gt_distance) - float(blur_distance)) * self.scalex
        # print(distance)
        img_gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_gt_bytes, flag=self.flag, float32=True)
        except:
            print(gt_path, file_name_)
            raise Exception("gt path {} not working".format(gt_path))

        # lq_path_list = glob.glob(os.path.join(self.lq_folder, file_name_[0] + '_*_' + file_name_[-1]))
        # lq_path = random.choice(lq_path_list)
        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_lq_bytes, flag=self.flag, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)
        if len(img_lq.shape) == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=False,
                                    float32=True)
        distance = torch.tensor([distance])
        flag_aug = random.randint(0, 4)
        if flag_aug == 0:
            factor = random.randint(80, 101) / 100.
            img_lq = img_lq * factor
            img_gt = img_gt * factor
            # kornia.enhance.adjust_contrast(img_lq, factor, clip_output=True)

        elif flag_aug == 1:
            factor = random.randint(100, 121) / 100.
            img_lq = img_lq * factor
            img_gt = img_gt * factor
            # img_lq = kornia.enhance.adjust_brightness(img_lq, factor, clip_output=True)

        elif flag_aug == 2:
            factor = random.randint(80, 121) / 100.
            img_lq = kornia.enhance.adjust_gamma(img_lq, factor)
            img_gt = kornia.enhance.adjust_gamma(img_gt, factor)
        img_lq = torch.clamp(img_lq, 0., 1.)
        img_gt = torch.clamp(img_gt, 0., 1.)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print(img_gt.shape, img_lq.shape)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'distance': distance
        }

    def __len__(self):
        return len(self.paths)

class Dataset_Multi_Paired_TuiSaoFocus(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.range = opt['range'] if 'range' in opt else None
        self.flag = opt['flag'] if 'flag' in opt else 'color'
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.gt_cal = opt['gt_cal'] if 'gt_cal' in opt else False
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            if self.gt_cal:
                pths = [self.lq_folder, self.gt_folder]
            else:
                pths = [self.lq_folder]
            self.paths = single_rgb_paths_from_folder(pths, ['lq'], self.filename_tmpl, self.range)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print(self.paths)
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        file_name = os.path.basename(lq_path)
        file_name_ = file_name.split('-')
        file_name_x = file_name_[0]
        for x in file_name_[1:-1]:
            file_name_x = file_name_x + '-' + x
        file_name_b, ext = os.path.splitext(file_name)

        gt_path_ = os.path.join(self.gt_folder, file_name_x + '-*'+ext)

        # try:
        gt_path = glob.glob(gt_path_)[0]
        file_name_s, ext = os.path.splitext(gt_path)
        # except:
        #     print(file_name_x, gt_path_, glob.glob(gt_path_))

        gt_distance = file_name_s.split('-')[-1]
        blur_distance = file_name_b.split('-')[-1]
        # print(gt_path, lq_path)
        # print(gt_distance, blur_distance)
        distance = float(gt_distance) - float(blur_distance)
        img_gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_gt_bytes, flag=self.flag, float32=True)
        except:
            print(gt_path, file_name_)
            raise Exception("gt path {} not working".format(gt_path))

        # lq_path_list = glob.glob(os.path.join(self.lq_folder, file_name_[0] + '_*_' + file_name_[-1]))
        # lq_path = random.choice(lq_path_list)
        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_lq_bytes, flag=self.flag, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)
        if len(img_lq.shape) == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=False,
                                    float32=True)
        distance = torch.tensor([distance])
        flag_aug = random.randint(0, 4)
        if flag_aug == 0:
            factor = random.randint(80, 101) / 100.
            img_lq = img_lq * factor
            img_gt = img_gt * factor
            # kornia.enhance.adjust_contrast(img_lq, factor, clip_output=True)

        elif flag_aug == 1:
            factor = random.randint(100, 121) / 100.
            img_lq = img_lq * factor
            img_gt = img_gt * factor
            # img_lq = kornia.enhance.adjust_brightness(img_lq, factor, clip_output=True)

        elif flag_aug == 2:
            factor = random.randint(80, 121) / 100.
            img_lq = kornia.enhance.adjust_gamma(img_lq, factor)
            img_gt = kornia.enhance.adjust_gamma(img_gt, factor)
        img_lq = torch.clamp(img_lq, 0., 1.)
        img_gt = torch.clamp(img_gt, 0., 1.)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print(img_gt.shape, img_lq.shape)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'distance': distance
        }

    def __len__(self):
        return len(self.paths)
class Dataset_Multi_Paired_Defocus(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.range = opt['range'] if 'range' in opt else None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.flag = opt['flag'] if 'flag' in opt else 'color'
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = single_rgb_paths_from_folder([self.lq_folder], ['lq'], self.filename_tmpl, self.range)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print(self.paths)
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        file_name = os.path.basename(lq_path)
        file_name_ = file_name.split('_')
        file_name_x = file_name_[0]
        for x in file_name_[1:-1]:
            file_name_x = file_name_x + '_' + x
        file_name_b, ext = os.path.splitext(file_name)
        ext_gt = '.png'
        gt_path_ = os.path.join(self.gt_folder, file_name_x+ext_gt)

        try:
            gt_path = glob.glob(gt_path_)[0]
        # file_name_s, ext_gt = os.path.splitext(gt_path)
        except:
            print(file_name_x, gt_path_, glob.glob(gt_path_))

        # gt_distance = file_name_s.split('_')[-1]
        # blur_distance = file_name_b.split('_')[-1]
        # distance = float(gt_distance) - float(blur_distance)
        img_gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_gt_bytes, flag=self.flag, float32=True)
        except:
            print(gt_path, file_name_)
            raise Exception("gt path {} not working".format(gt_path))

        # lq_path_list = glob.glob(os.path.join(self.lq_folder, file_name_[0] + '_*_' + file_name_[-1]))
        # lq_path = random.choice(lq_path_list)
        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_lq_bytes, flag=self.flag, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)
        if len(img_lq.shape) == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        if self.opt['phase'] == 'val':
            gt_size = self.opt['crop_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_center_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=False,
                                    float32=True)
        # distance = torch.tensor([distance])
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print(img_gt.shape, distance.shape)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

