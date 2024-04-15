import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.sv_blur import BatchBlur_SV, BatchBlur_SV2, BatchBlur_SV3
from torch import nn
from basicsr.utils.dist_util import get_dist_info
import math
import kornia
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        self.gt_for_train = self.opt['train'].get('gt_for_train', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        self.loss_clamp = train_opt.get('loss_clamp', None)
        # total_iter = train_opt.get('total_iter', 0)
        # ema_warmup_iters = 4000 # 2000
        # ema_keep_iters = total_iter + 1 - ema_warmup_iters # 46000
        # w_warm = np.linspace(0, 1., ema_warmup_iters)
        # w_keep = np.ones(ema_keep_iters)
        # # w_end = np.linspace(self.ema_decay, 1e-3, total_iter-ema_warmup_iters-ema_keep_iters)
        # self.ema_decays_weight = np.concatenate([w_warm, w_keep], axis=0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        # print(data)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        # print(data)
        # val_opt = self.opt['val']
        center_crop = self.opt['val'].get('center_crop', None)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if center_crop:
            self.gt = kornia.geometry.center_crop(self.gt, [center_crop, center_crop])
            self.lq = kornia.geometry.center_crop(self.lq, [center_crop, center_crop])
        # if 'distance' in data:
        #     self.distance = data['distance'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.gt_for_train:
            preds = self.net_g(self.lq, self.gt)
        else:
            preds = self.net_g(self.lq)
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        l_reblur = 0.
        l_deblur = 0.
        l_cls = 0.
        l_local = 0.
        if isinstance(preds, list):
            preds_img = preds[-1]
        elif isinstance(preds, dict):
            if 'loss_reblur' in preds.keys():
                # print(preds['loss'])
                l_reblur += preds['loss_reblur']
                loss_dict['l_reblur'] = l_reblur
                # l_reblur *= preds['reblur_loss_weight']
            if 'loss_deblur' in preds.keys():
                # print(preds['loss'])
                l_deblur += preds['loss_deblur']
                loss_dict['l_deblur'] = l_deblur
                # l_deblur *= preds['deblur_loss_weight']
            if 'loss_cls' in preds.keys():
                # print(preds['loss'])
                l_cls += preds['l_cls']
                loss_dict['l_cls'] = l_cls
            if 'loss_local' in preds.keys():
                # print(preds['loss'])
                l_local += preds['loss_local']
                loss_dict['l_local'] = l_local
            preds = preds['img']

            if isinstance(preds, list):
                preds_img = preds[-1]
            else:
                preds_img = preds
        else:
            preds_img = preds
        self.output = preds_img

        # if isinstance(preds, dict):
        #     if 'loss' in preds.keys():
        #         print(preds['loss'])
        #         l_pix += preds['loss']
        # if isinstance(preds, dict):
        #     l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        # else:
        l_pix += self.cri_pix(preds, self.gt) # + 0. * sum(p.sum() for p in self.net_g.parameters())
        # print(preds.shape)

        if self.loss_clamp is not None:
            l_pix = torch.clamp(l_pix, self.loss_clamp[0], self.loss_clamp[1])
        loss_dict['l_pix'] = l_pix
        # print(current_iter, loss_dict)
        # if 'l_reblur' in loss_dict.keys():
        l_total = l_pix + l_reblur + l_deblur + l_local + l_cls
        loss_dict['l_total'] = l_total
        # print(l_pix)
        # print(l_total, l_pix, l_reblur, l_deblur)
        # print(l_reblur, l_pix)
        l_total.backward()
        # else:
        #     l_pix.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            ema_decay = self.ema_decay # * self.ema_decays_weight[current_iter]
            # print(ema_)
            self.model_ema(decay=ema_decay)
        if current_iter == 3:
            # self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 6:
            self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
    def _get_gpu_memory(self):
        logger = get_root_logger()

        mem = torch.cuda.max_memory_allocated(0)
        mem_mb = int(mem) // (1024 * 1024)
        # print(mem_mb)
        logger.info(f'Model gpu memory: {mem_mb} MB')
    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img,  rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        max_idx = self.opt['val'].get('max_num', 4000)
        for idx, val_data in enumerate(dataloader):
            if idx >= max_idx:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class ImageReblurModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageReblurModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        self.gt_for_train = self.opt['train'].get('gt_for_train', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        self.loss_clamp = train_opt.get('loss_clamp', None)
        # total_iter = train_opt.get('total_iter', 0)
        # ema_warmup_iters = 4000 # 2000
        # ema_keep_iters = total_iter + 1 - ema_warmup_iters # 46000
        # w_warm = np.linspace(0, 1., ema_warmup_iters)
        # w_keep = np.ones(ema_keep_iters)
        # # w_end = np.linspace(self.ema_decay, 1e-3, total_iter-ema_warmup_iters-ema_keep_iters)
        # self.ema_decays_weight = np.concatenate([w_warm, w_keep], axis=0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        # print(data)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        # print(data)
        # val_opt = self.opt['val']
        center_crop = self.opt['val'].get('center_crop', None)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if center_crop:
            self.gt = kornia.geometry.center_crop(self.gt, [center_crop, center_crop])
            self.lq = kornia.geometry.center_crop(self.lq, [center_crop, center_crop])
        # if 'distance' in data:
        #     self.distance = data['distance'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.gt_for_train:
            preds = self.net_g(self.lq, self.gt)
        else:
            preds = self.net_g(self.lq)
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        l_reblur = 0.
        l_deblur = 0.
        l_cls = 0.
        l_local = 0.
        if isinstance(preds, list):
            preds_img = preds[-1]
        elif isinstance(preds, dict):
            if 'loss_reblur' in preds.keys():
                # print(preds['loss'])
                l_reblur += preds['loss_reblur']
                loss_dict['l_reblur'] = l_reblur
                # l_reblur *= preds['reblur_loss_weight']
            if 'loss_deblur' in preds.keys():
                # print(preds['loss'])
                l_deblur += preds['loss_deblur']
                loss_dict['l_deblur'] = l_deblur
                # l_deblur *= preds['deblur_loss_weight']
            if 'loss_cls' in preds.keys():
                # print(preds['loss'])
                l_cls += preds['l_cls']
                loss_dict['l_cls'] = l_cls
            if 'loss_local' in preds.keys():
                # print(preds['loss'])
                l_local += preds['loss_local']
                loss_dict['l_local'] = l_local
            preds = preds['img']

            if isinstance(preds, list):
                preds_img = preds[-1]
            else:
                preds_img = preds
        else:
            preds_img = preds
        self.output = preds_img

        # if isinstance(preds, dict):
        #     if 'loss' in preds.keys():
        #         print(preds['loss'])
        #         l_pix += preds['loss']
        # if isinstance(preds, dict):
        #     l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        # else:
        l_pix += self.cri_pix(preds, self.lq)  # + 0. * sum(p.sum() for p in self.net_g.parameters())
        # print(preds.shape)

        if self.loss_clamp is not None:
            l_pix = torch.clamp(l_pix, self.loss_clamp[0], self.loss_clamp[1])
        loss_dict['l_pix'] = l_pix
        # print(current_iter, loss_dict)
        # if 'l_reblur' in loss_dict.keys():
        l_total = l_pix + l_reblur + l_deblur + l_local + l_cls
        loss_dict['l_total'] = l_total
        # print(l_total, l_pix, l_reblur, l_deblur)
        # print(l_reblur, l_pix)
        # print(l_pix)
        l_total.backward()
        # else:
        #     l_pix.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            ema_decay = self.ema_decay  # * self.ema_decays_weight[current_iter]
            # print(ema_)
            self.model_ema(decay=ema_decay)
        if current_iter == 3:
            # self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 6:
            self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)

    def _get_gpu_memory(self):
        logger = get_root_logger()

        mem = torch.cuda.max_memory_allocated(0)
        mem_mb = int(mem) // (1024 * 1024)
        # print(mem_mb)
        logger.info(f'Model gpu memory: {mem_mb} MB')

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img, self.gt)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, self.gt)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        max_idx = self.opt['val'].get('max_num', 4000)
        for idx, val_data in enumerate(dataloader):
            if idx >= max_idx:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
class ImageClassifyHardModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageClassifyHardModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        self.loss_clamp = train_opt.get('loss_clamp', None)
        # total_iter = train_opt.get('total_iter', 0)
        # ema_warmup_iters = 4000 # 2000
        # ema_keep_iters = total_iter + 1 - ema_warmup_iters # 46000
        # w_warm = np.linspace(0, 1., ema_warmup_iters)
        # w_keep = np.ones(ema_keep_iters)
        # # w_end = np.linspace(self.ema_decay, 1e-3, total_iter-ema_warmup_iters-ema_keep_iters)
        # self.ema_decays_weight = np.concatenate([w_warm, w_keep], axis=0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        # print(data)
        # for k in data.keys():
        #     print(k)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'psnr_cls' in data:
            self.psnr_cls = data['psnr_cls'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        # print(data)
        # val_opt = self.opt['val']
        center_crop = self.opt['val'].get('center_crop', None)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'psnr_cls' in data:
            self.psnr_cls = data['psnr_cls'].to(self.device)
        if center_crop:
            self.gt = kornia.geometry.center_crop(self.gt, [center_crop, center_crop])
            self.lq = kornia.geometry.center_crop(self.lq, [center_crop, center_crop])
        # if 'distance' in data:
        #     self.distance = data['distance'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if isinstance(preds, list):
            preds_img = preds[0]
        elif isinstance(preds, dict):
            preds_img = preds['psnr_cls']
            if isinstance(preds_img, list):
                preds_img = preds_img[0]
        else:
            preds_img = preds
        self.output = preds_img
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.

        # if isinstance(preds, dict):
        #     l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        # else:
        l_pix += self.cri_pix(preds, self.psnr_cls)  # + 0. * sum(p.sum() for p in self.net_g.parameters())
        # print(preds.shape)
        # print(preds, self.psnr_cls)
        # print(current_iter, l_pix)
        if self.loss_clamp is not None:
            l_pix = torch.clamp(l_pix, self.loss_clamp[0], self.loss_clamp[1])
        loss_dict['l_pix'] = l_pix

        l_pix.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            ema_decay = self.ema_decay  # * self.ema_decays_weight[current_iter]
            # print(ema_)
            self.model_ema(decay=ema_decay)
        if current_iter == 3:
            # self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 6:
            self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)

    def _get_gpu_memory(self):
        logger = get_root_logger()

        mem = torch.cuda.max_memory_allocated(0)
        mem_mb = int(mem) // (1024 * 1024)
        # print(mem_mb)
        logger.info(f'Model gpu memory: {mem_mb} MB')

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        # _, _, h, w = self.output.size()
        # self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[0]
            elif isinstance(pred, dict):
                pred = pred['psnr_cls']
                if isinstance(pred, list):
                    pred = pred[0]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[0]
            elif isinstance(pred, dict):
                pred = pred['psnr_cls']
                if isinstance(pred, list):
                    pred = pred[0]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        max_idx = self.opt['val'].get('max_num', 4000)
        for idx, val_data in enumerate(dataloader):
            if idx >= max_idx:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            class_pred = visuals['result']
            class_pred = torch.argmax(class_pred, dim=-1, keepdim=False)
            gt = visuals['psnr_cls'].squeeze(-1)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])

                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    if class_pred == gt:
                        self.metric_results[name] += 1


            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'psnr_cls'):
            out_dict['psnr_cls'] = self.psnr_cls.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
class ImageClassifyModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageClassifyModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        self.loss_clamp = train_opt.get('loss_clamp', None)
        # total_iter = train_opt.get('total_iter', 0)
        # ema_warmup_iters = 4000 # 2000
        # ema_keep_iters = total_iter + 1 - ema_warmup_iters # 46000
        # w_warm = np.linspace(0, 1., ema_warmup_iters)
        # w_keep = np.ones(ema_keep_iters)
        # # w_end = np.linspace(self.ema_decay, 1e-3, total_iter-ema_warmup_iters-ema_keep_iters)
        # self.ema_decays_weight = np.concatenate([w_warm, w_keep], axis=0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        # print(data)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        # print(data)
        # val_opt = self.opt['val']
        center_crop = self.opt['val'].get('center_crop', None)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # if center_crop:
        #     self.gt = kornia.geometry.center_crop(self.gt, [center_crop, center_crop])
        #     self.lq = kornia.geometry.center_crop(self.lq, [center_crop, center_crop])
        # if 'distance' in data:
        #     self.distance = data['distance'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if isinstance(preds, list):
            preds_img = preds[-1]
        elif isinstance(preds, dict):
            preds_img = preds['img']
            if isinstance(preds_img, list):
                preds_img = preds_img[-1]
        else:
            preds_img = preds
        self.output = preds_img
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.

        # if isinstance(preds, dict):
        #     l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        # else:
        l_pix += self.cri_pix(preds, self.gt)  # + 0. * sum(p.sum() for p in self.net_g.parameters())
        # print(preds.shape)
        # print(current_iter, l_pix)
        if self.loss_clamp is not None:
            l_pix = torch.clamp(l_pix, self.loss_clamp[0], self.loss_clamp[1])
        loss_dict['l_pix'] = l_pix

        l_pix.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            ema_decay = self.ema_decay  # * self.ema_decays_weight[current_iter]
            # print(ema_)
            self.model_ema(decay=ema_decay)
        if current_iter == 3:
            # self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 6:
            self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)

    def _get_gpu_memory(self):
        logger = get_root_logger()

        mem = torch.cuda.max_memory_allocated(0)
        mem_mb = int(mem) // (1024 * 1024)
        # print(mem_mb)
        logger.info(f'Model gpu memory: {mem_mb} MB')

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        # _, _, h, w = self.output.size()
        # self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        max_idx = self.opt['val'].get('max_num', 4000)
        # total_num = len(dataloader)
        for idx, val_data in enumerate(dataloader):
            if idx >= max_idx:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            class_pred = visuals['result']
            class_pred = torch.argmax(class_pred, dim=-1, keepdim=False)
            if 'gt' in visuals:
                gt = visuals['gt'].squeeze(-1)
                # gt = torch.nn.functional.one_hot(gt, num_classes=class_pred.shape[-1])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        # metric_type = opt_.pop('type')
                        if class_pred == gt:
                            self.metric_results[name] += 1
                else:
                    for name, opt_ in opt_metric.items():
                        # metric_type = opt_.pop('type')
                        if class_pred == gt:
                            self.metric_results[name] += 1

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
class PANSRCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(PANSRCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        # print(train_opt)
        self.ema_decay = train_opt.get('ema_decay', 0)

        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        # print(data)
        # for k in data.keys():
        #     print(k)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'lr' in data:
            self.lr = data['lr'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        # for k in data.keys():
        #     print(k)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'lr' in data:
            self.lr = data['lr'].to(self.device)
        # if 'distance' in data:
        #     self.distance = data['distance'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if isinstance(preds, list):
            preds_img = preds[-1]
        elif isinstance(preds, dict):
            preds_img = preds['img']
            if isinstance(preds_img, list):
                preds_img = preds_img[-1]
        else:
            preds_img = preds
        self.output = preds_img
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.

        # if isinstance(preds, dict):
        #     l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        # else:
        l_pix += self.cri_pix(preds, self.gt, self.lr)

        # print(current_iter, l_pix)
        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
class FocusCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(FocusCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        # print(data.keys())
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'distance' in data:
            self.distance = data['distance'].to(self.device)
        else:
            raise AttributeError
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'distance' in data:
            self.distance = data['distance'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq, self.gt)
        # if isinstance(preds, list):
        #     preds_img = preds[-1]
        # elif isinstance(preds, dict):
        #     preds_img = preds['z_distance']
        #     if isinstance(preds_img, list):
        #         preds_img = preds_img[-1]
        # else:
        #     preds_img = preds
        l_reblur = 0.
        loss_dict = OrderedDict()
        if isinstance(preds, list):
            preds_img = preds[-1]
        elif isinstance(preds, dict):
            if 'loss_reblur' in preds.keys():
                # print(preds['loss'])
                l_reblur += preds['loss_reblur']
                loss_dict['l_reblur'] = l_reblur
            preds = preds['z_distance']
            if isinstance(preds, list):
                preds_img = preds[-1]
            else:
                preds_img = preds
        else:
            preds_img = preds
        self.output = preds_img

        # pixel loss
        l_pix = 0.
        l_total = 0.
        # if isinstance(preds, dict):
        #     l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        # else:

        l_pix += self.cri_pix(preds, self.distance)

        # print(current_iter, l_pix)
        loss_dict['l_pix'] = l_pix
        l_total = l_pix + l_reblur
        loss_dict['l_total'] = l_total
        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        # _, _, h, w = self.output.size()
        # self.output = self.output # [:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['z_distance']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['z_distance']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        max_idx = self.opt['val'].get('max_num', 4000)
        print('start val, max_idx: ', max_idx)
        for idx, val_data in enumerate(dataloader):
            # print(idx, max_idx)
            if idx >= max_idx:
                print('break in max_idx: ', max_idx)
                break
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            distance_pred = visuals['result']
            if 'distance' in visuals:
                distance = visuals['distance'].squeeze()
                del self.distance
            # print('pred: ', distance_pred)
            # print('gt: ', distance)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        # metric_type = opt_.pop('type')
                        self.metric_results[name] += np.mean(np.abs(distance.cpu().numpy() - distance_pred.cpu().numpy()))
                else:
                    for name, opt_ in opt_metric.items():
                        # metric_type = opt_.pop('type')
                        self.metric_results[name] += np.mean(np.abs(distance.cpu().numpy() - distance_pred.cpu().numpy()))

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'distance'):
            out_dict['distance'] = self.distance.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
class HSICleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(HSICleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if isinstance(preds, list):
            preds_img = preds[-1]
        elif isinstance(preds, dict):
            preds_img = preds['img']
            if isinstance(preds_img, list):
                preds_img = preds_img[-1]
        else:
            preds_img = preds
        self.output = preds_img
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.

        if isinstance(preds, dict):
            l_pix += self.cri_pix(preds['img'], self.gt) + preds['loss'] * 1.
        else:
            l_pix += self.cri_pix(preds, self.gt)
        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        # if isinstance(self.output, list):
        #     self.output = self.output[-1]
        # elif isinstance(self.output, dict):
        #     self.output = self.output['img']
        #     if isinstance(self.output, list):
        #         self.output = self.output[-1]
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=False)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=False)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)

        if load_path is not None:
            # self.load_network(self.net_g, load_path,
            #                   self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        kernel_size = self.opt['degrade_model'].get('kernel_size', 19)
        divisor_ = self.opt['degrade_model'].get('divisor_', True)
        self.device = torch.device('cuda')
        # self.degradation_model = DegradationModel(kernel_size=kernel_size, divisor_=divisor_)
        # self.degradation_model = nn.DataParallel(self.degradation_model)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('reblur_opt'):
            reblur_type = train_opt['reblur_opt'].pop('type')
            cri_reblur_cls = getattr(loss_module, reblur_type)
            self.cri_reblur = cri_reblur_cls(**train_opt['reblur_opt']).to(
                self.device)
        else:

            self.cri_reblur = None

        if train_opt.get('pretrain_reblur_opt'):
            reblur_type = train_opt['pretrain_reblur_opt'].pop('type')
            cri_pretrain_reblur_cls = getattr(loss_module, reblur_type)
            self.cri_pretrain_reblur = cri_pretrain_reblur_cls(**train_opt['pretrain_reblur_opt']).to(
                self.device)
        else:
            self.cri_pretrain_reblur = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_pretrain_reblur is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
    def feed_data(self, data, is_val=False):
        center_crop = self.opt['val'].get('center_crop', None)
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if center_crop:
            self.gt = kornia.geometry.center_crop(self.gt, [center_crop, center_crop])
            self.lq = kornia.geometry.center_crop(self.lq, [center_crop, center_crop])

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter): #  , tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        if self.cri_pretrain_reblur:
            preds = self.net_g(self.lq, self.gt)
        else:
            preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            l_pix += self.cri_pix(self.output, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # print(self.cri_reblur)
        if self.cri_reblur:
            l_reblur = 0.
            # fake_blur = self.degradation_model(self.output, preds[0])
            fake_blur = preds[0]
            l_reblur += self.cri_reblur(fake_blur.to(self.lq.device), self.lq)

            # print('l reblur ... ', l_reblur)
            l_total += l_reblur * 0.01
            loss_dict['l_reblur'] = l_reblur
        if self.cri_pretrain_reblur:
            l_reblur = 0.
            # fake_blur = self.degradation_model(self.gt[0], preds[0][0])
            fake_blur = preds[0]
            # print(fake_blur.max())
            l_reblur += self.cri_pretrain_reblur(fake_blur.to(self.lq.device), self.lq[0])

            # print('l reblur ... ', l_reblur)
            l_total += l_reblur * 0.01
            loss_dict['l_pretrain_reblur'] = l_reblur

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            #
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # print('output: ', self.output.max())
        # print('loss: ', l_total)
        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
        # print(l_total)
        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            ema_decay = self.ema_decay # * self.ema_decays_weight[current_iter]
            # print(ema_)
            self.model_ema(decay=ema_decay)

        if current_iter == 3:
            # self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 6:
            self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)

    def _get_gpu_memory(self):
        logger = get_root_logger()

        mem = torch.cuda.max_memory_allocated(0)
        mem_mb = int(mem) // (1024 * 1024)
        # print(mem_mb)
        logger.info(f'Model gpu memory: {mem_mb} MB')

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.cri_pretrain_reblur:
                    pred = self.net_g_ema(img, self.gt)
                else:
                    pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
                kernel = pred[0]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.kernel = kernel
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.cri_pretrain_reblur:
                    pred = self.net_g(img, self.gt)
                else:
                    pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
                kernel = pred[0]
            elif isinstance(pred, dict):
                pred = pred['img']
                if isinstance(pred, list):
                    pred = pred[-1]
            self.output = pred
            self.kernel = kernel
            self.net_g.train()
    # def test(self):
    #     self.net_g.eval()
    #     with torch.no_grad():
    #         n = len(self.lq)  # tensor[1, 3, H, W]
    #         outs = []
    #         m = self.opt['val'].get('max_minibatch', n)
    #         i = 0
    #         while i < n:
    #             j = i + m
    #             if j >= n:
    #                 j = n
    #             pred = self.net_g(self.lq[i:j])
    #             if isinstance(pred, list):
    #                 pred = pred[-1]
    #             outs.append(pred.detach().cpu())
    #             i = j
    #
    #         self.output = torch.cat(outs, dim=0)  # tensor[1, 3, H, W]
    #     self.net_g.train()
    #
    # def test_crop9(self):
    #     self.net_g.eval()
    #
    #     with torch.no_grad():
    #         N, C, H, W = self.lq.shape
    #         h, w = math.ceil(H / 3), math.ceil(W / 3)
    #         rf = 30
    #         imTL = self.net_g(self.lq[:, :, 0:h + rf, 0:w + rf])[:, :, 0:h, 0:w]
    #         imML = self.net_g(self.lq[:, :, h - rf:2 * h + rf, 0:w + rf])[:, :, rf:(rf + h), 0:w]
    #         imBL = self.net_g(self.lq[:, :, 2 * h - rf:, 0:w + rf])[:, :, rf:, 0:w]
    #         imTM = self.net_g(self.lq[:, :, 0:h + rf, w - rf:2 * w + rf])[:, :, 0:h, rf:(rf + w)]
    #         imMM = self.net_g(self.lq[:, :, h - rf:2 * h + rf, w - rf:2 * w + rf])[:, :, rf:(rf + h), rf:(rf + w)]
    #         imBM = self.net_g(self.lq[:, :, 2 * h - rf:, w - rf:2 * w + rf])[:, :, rf:, rf:(rf + w)]
    #         imTR = self.net_g(self.lq[:, :, 0:h + rf, 2 * w - rf:])[:, :, 0:h, rf:]
    #         imMR = self.net_g(self.lq[:, :, h - rf:2 * h + rf, 2 * w - rf:])[:, :, rf:(rf + h), rf:]
    #         imBR = self.net_g(self.lq[:, :, 2 * h - rf:, 2 * w - rf:])[:, :, rf:, rf:]
    #
    #         imT = torch.cat((imTL, imTM, imTR), 3)
    #         imM = torch.cat((imML, imMM, imMR), 3)
    #         imB = torch.cat((imBL, imBM, imBR), 3)
    #         output_cat = torch.cat((imT, imM, imB), 2)
    #
    #         self.output = output_cat
    #     self.net_g.train()
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()

            if self.cri_pretrain_reblur:
                sr_img = tensor2img([visuals['result']], rgb2bgr=False)
                # reblur_img = tensor2img([self.degradation_model(self.gt, visuals['kernel'])], rgb2bgr=False)
                # blur_img = tensor2img([visuals['lq']], rgb2bgr=False)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=False)
                    del self.gt
            else:
                sr_img = tensor2img([visuals['result']], rgb2bgr=False)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=False)
                    del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric
    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
    # def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     if with_metrics:
    #         self.metric_results = {
    #             metric: 0
    #             for metric in self.opt['val']['metrics'].keys()
    #         }
    #
    #     rank, world_size = get_dist_info()
    #     if rank == 0:
    #         pbar = tqdm(total=len(dataloader), unit='image')
    #
    #     cnt = 0
    #
    #     for idx, val_data in enumerate(dataloader):
    #         if idx % world_size != rank:
    #             continue
    #
    #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #
    #         self.feed_data(val_data, is_val=True)
    #         if self.opt['val'].get('grids', False):
    #             self.grids()
    #
    #         # self.test_crop9()
    #         self.test()
    #
    #         if self.opt['val'].get('grids', False):
    #             self.grids_inverse()
    #
    #         visuals = self.get_current_visuals()
    #         sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    #         if 'gt' in visuals:
    #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
    #             del self.gt
    #
    #         # tentative for out of GPU memory
    #         del self.lq
    #         del self.output
    #         torch.cuda.empty_cache()
    #
    #         if save_img:
    #             if sr_img.shape[2] == 6:
    #                 L_img = sr_img[:, :, :3]
    #                 R_img = sr_img[:, :, 3:]
    #
    #                 # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
    #                 visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
    #
    #                 imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
    #                 imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
    #             else:
    #                 if self.opt['is_train']:
    #
    #                     save_img_path = osp.join(self.opt['path']['visualization'],
    #                                              img_name,
    #                                              f'{img_name}_{current_iter}.png')
    #
    #                     save_gt_img_path = osp.join(self.opt['path']['visualization'],
    #                                                 img_name,
    #                                                 f'{img_name}_{current_iter}_gt.png')
    #                 else:
    #                     save_img_path = osp.join(
    #                         self.opt['path']['visualization'], dataset_name,
    #                         f'{img_name}.png')
    #                     save_gt_img_path = osp.join(
    #                         self.opt['path']['visualization'], dataset_name,
    #                         f'{img_name}_gt.png')
    #
    #                 imwrite(sr_img, save_img_path)
    #                 # imwrite(gt_img, save_gt_img_path)
    #
    #         if with_metrics:
    #             # calculate metrics
    #             opt_metric = deepcopy(self.opt['val']['metrics'])
    #             if use_image:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(sr_img, gt_img, **opt_)
    #             else:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
    #
    #         cnt += 1
    #         if rank == 0:
    #             for _ in range(world_size):
    #                 pbar.update(1)
    #                 pbar.set_description(f'Test {img_name}')
    #     if rank == 0:
    #         pbar.close()
    #
    #     # current_metric = 0.
    #     collected_metrics = OrderedDict()
    #     if with_metrics:
    #         for metric in self.metric_results.keys():
    #             collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
    #         collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)
    #
    #         self.collected_metrics = collected_metrics
    #
    #     keys = []
    #     metrics = []
    #     for name, value in self.collected_metrics.items():
    #         keys.append(name)
    #         metrics.append(value)
    #     metrics = torch.stack(metrics, 0)
    #     # torch.distributed.reduce(metrics, dst=0)
    #     if self.opt['rank'] == 0:
    #         metrics_dict = {}
    #         cnt = 0
    #         for key, metric in zip(keys, metrics):
    #             if key == 'cnt':
    #                 cnt = float(metric)
    #                 continue
    #             metrics_dict[key] = float(metric)
    #
    #         for key in metrics_dict:
    #             metrics_dict[key] /= cnt
    #
    #         self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
    #                                            tb_logger, metrics_dict)
    #     return 0.
    #
    # def nondist_validation(self, *args, **kwargs):
    #     logger = get_root_logger()
    #     logger.warning('nondist_validation is not implemented. Run dist_validation.')
    #     self.dist_validation(*args, **kwargs)

    # def _log_validation_metric_values(self, current_iter, dataset_name,
    #                                   tb_logger, metric_dict):
    #     log_str = f'Validation {dataset_name}, \t'
    #     for metric, value in metric_dict.items():
    #         log_str += f'\t # {metric}: {value:.4f}'
    #     logger = get_root_logger()
    #     logger.info(log_str)
    #
    #     log_dict = OrderedDict()
    #     # for name, value in loss_dict.items():
    #     for metric, value in metric_dict.items():
    #         log_dict[f'm_{metric}'] = value
    #
    #     self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'kernel'):
            out_dict['kernel'] = self.kernel.detach().cpu()
        return out_dict
    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
    # def save(self, epoch, current_iter):
        # self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_training_state(epoch, current_iter)


class DegradationModel(nn.Module):
    def __init__(self, kernel_size=19, divisor_=False):
        super(DegradationModel, self).__init__()
        self.blur_layer = BatchBlur_SV(l=kernel_size, padmode='reflection')# , divisor_=divisor_) # replication

    def forward(self, image, kernel):
        return self.blur_layer(image, kernel)
# if isinstance(preds, list):
#     self.output = preds[-1]
#
#     loss_dict = OrderedDict()
#     # pixel loss
#     l_pix = 0.
#     # for pred in preds:
#     l_pix += self.cri_pix(preds, self.gt)
#
#     loss_dict['l_pix'] = l_pix
#     l_pix.backward()
#     if self.opt['train']['use_grad_clip']:
#         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
#     self.optimizer_g.step()
#
#     self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     if self.ema_decay > 0:
#         self.model_ema(decay=self.ema_decay)
# elif isinstance(preds, dict):
#     preds_img = preds['img']
#     if isinstance(preds_img, list):
#         self.output = preds_img[-1]
#     else:
#         self.output = preds_img
#     preds_loss = preds['loss']
#     loss_dict = OrderedDict()
#     # pixel loss
#     l_pix = 0.
#     # for pred in preds:
#     alpha = 1. #
#     l_pix += self.cri_pix(preds_img, self.gt) + preds_loss * alpha
#
#     loss_dict['l_pix'] = l_pix
#     l_pix.backward()
#     if self.opt['train']['use_grad_clip']:
#         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
#     self.optimizer_g.step()
#
#     self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     if self.ema_decay > 0:
#         self.model_ema(decay=self.ema_decay)
# else:
#     self.output = preds
#     loss_dict = OrderedDict()
#     # pixel loss
#     l_pix = 0.
#     l_pix += self.cri_pix(preds, self.gt)
#
#     loss_dict['l_pix'] = l_pix
#
#     l_pix.backward()
#     if self.opt['train']['use_grad_clip']:
#         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
#     self.optimizer_g.step()
#     self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     if self.ema_decay > 0:
#         self.model_ema(decay=self.ema_decay)