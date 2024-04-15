# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
from basicsr.models.archs.encoder1 import ConvEncoderLoss
from basicsr.models.archs.discriminator import MultiscaleDiscriminator
from basicsr.models.archs.loss import GANLoss, VGGLoss

class ImageRestorationModel2(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel2, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        # self.net_g.load_state_dict(torch.load(opt['pretrained_network_g'])['params'], strict=False)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # self.net_loss = ConvEncoderLoss()
        # self.net_loss.load_state_dict(torch.load("/workspace/lidasong/HINet/latest_net_E.pth"), strict=False)
        # self.net_loss = self.model43_to_device(self.net_loss, True)
        
        self.net_d = MultiscaleDiscriminator()
        # self.net_d.load_state_dict(torch.load('/workspace/lidasong/SPADE/checkpoints/generator37/latest_net_D.pth'))
        self.net_d = self.model_to_device(self.net_d)
        
        self.criterionGAN = GANLoss('hinge')
        self.criterionVGG = VGGLoss(0)
        self.criterionFeat = torch.nn.L1Loss()
        self.L1 = torch.nn.L1Loss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
    def model43_to_device(self, net, find_unused_parameters):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        # elif optim_type == 'SGD':
        #     self.optimizer_g = torch.optim.SGD(optim_params,
        #                                        **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        d_optim_params = []
        d_optim_params_lowlr = []
        for k, v in self.net_d.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    d_optim_params_lowlr.append(v)
                else:
                    d_optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        self.optimizer_d = torch.optim.Adam([{'params': d_optim_params}, {'params': d_optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_d)
        
        
        # print(self.optimizer_g)
        # exit(0)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t


    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
    def compute_generator_loss(self):
        G_losses = OrderedDict() #{}
        # fake_image, KLD_loss = self.generate_fake(input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
        pred, fake_image = self.net_g(self.lq, self.gt)
        pred_fake, pred_real = self.discriminate(
            self.gt, fake_image, self.lq)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        num_D = len(pred_fake)
        GAN_Feat_loss = torch.FloatTensor(1).cuda().fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * 10 / num_D
        G_losses['GAN_Feat'] = GAN_Feat_loss
        G_losses['VGG'] = self.criterionVGG(fake_image, self.lq)*30
        
        l_pix = 0.
        # for pred in preds:
        l_pix += 10 *self.L1(pred, self.gt)
            # print('l pix ... ', l_pix)
        G_losses['l_pix'] = l_pix
        
        # blur_list1 = self.net_loss(self.gt)
        # blur_list2 = self.net_loss(preds[-1])
        # l_blur = 0.
        #for kk in range(len(blur_list1)):
        #    l_blur_ele = self.L1(blur_list1[kk], blur_list2[kk])
        #    l_blur += l_blur_ele
            
        # G_losses['l_blur'] = l_blur
        return pred, fake_image, G_losses
                                                                                             
        
        
    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.net_d(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
    def compute_discriminator_loss(self):
        D_losses = OrderedDict() # {}
        with torch.no_grad():
            _, fake_image = self.net_g(self.lq, self.gt)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(
            self.gt, fake_image, self.lq)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        return D_losses


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds, inverse_preds, G_losses = self.compute_generator_loss()
        #if not isinstance(preds, list):
        #     preds = [preds]
        self.output = preds
        self.output1 = inverse_preds
        l_total = (G_losses['GAN'] + G_losses['GAN_Feat'] + G_losses['VGG'])+G_losses['l_pix']
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()
        # use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        # if use_grad_clip:
        #     print("use grad clip", use_grad_clip)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        # exit(0)
        self.optimizer_g.step()
        
        self.optimizer_d.zero_grad()
        D_losses = self.compute_discriminator_loss()
        D_loss = sum(D_losses.values()).mean()
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01)
        self.optimizer_d.step()

        # exit(0)
        # whole_loss = {}
        # whole_loss.update(G_losses)
        # whole_loss.update(D_losses)
        # print(G_losses)
        # self.log_dict = self.reduce_loss_dict(G_losses)
        self.log_g_dict = G_losses
        # exit(0)
        self.log_d_dict = D_losses
        self.log_dict = [self.log_g_dict, self.log_d_dict]
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                # print(self.lq.shape())
                pred = self.net_g(self.lq[i:j, :, :, :], self.lq[i:j, :, :, :])
                if isinstance(pred, list):
                    pred = pred[0]
                # print('pred .. size', pred.size())
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()
    def get_latest_images(self):
        return [self.lq[0], self.output[0], self.output1[0]]

    def single_image_inference(self, img, save_path):
        self.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if self.opt['val'].get('grids', False):
            self.grids()

        self.test()

        if self.opt['val'].get('grids', False):
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
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
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # if img_name[-1] != '9':
            #     continue

            # print('val_data .. ', val_data['lq'].size(), val_data['gt'].size())
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

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

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
            # if cnt == 300:
            #     break
        pbar.close()

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
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
