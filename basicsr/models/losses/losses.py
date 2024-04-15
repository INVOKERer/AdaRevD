import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
import kornia
from basicsr.models.losses.loss_util import weighted_loss
# from basicsr.metrics.psnr_ssim import calculate_psnr
# from kornia.constants import pi
_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')
@weighted_loss
def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target, reduction='mean', beta=1.47/2.)
# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]])
# hed_from_rgb = linalg.inv(rgb_from_hed)
hed_from_rgb = torch.linalg.inv(rgb_from_hed)


def separate_stains(rgb, conv_matrix):
    # rgb = _prepare_colorarray(rgb, force_copy=True, channel_axis=-1)
    # rgb = rgb.astype(np.float32)

    # rgb = torch.maximum(rgb, torch.tensor(1e-6))  # avoiding log artifacts
    rgb = torch.clamp(rgb, min=1e-6)
    log_adjust = torch.log(torch.tensor(1e-6))  # used to compensate the sum above
    rgb = rearrange(rgb, 'b c h w -> b h w c')
    stains = (torch.log(rgb) / log_adjust) @ conv_matrix
    stains = rearrange(stains, 'b h w c -> b c h w')
    # stains = torch.maximum(stains, torch.tensor(0.))
    stains = torch.clamp(stains, min=0.)
    return stains

def combine_stains(stains, conv_matrix):

    # stains = stains.astype(np.float32)

    # log_adjust here is used to compensate the sum within separate_stains().
    log_adjust = -torch.log(torch.tensor(1e-6))
    stains = rearrange(stains, 'b c h w -> b h w c')
    log_rgb = -(stains * log_adjust) @ conv_matrix
    rgb = torch.exp(log_rgb)
    rgb = rearrange(rgb, 'b h w c -> b c h w')
    return torch.clamp(rgb, min=0., max=1.)

class rgb2hed(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mat = hed_from_rgb

    def forward(self, x):
        return separate_stains(x, self.mat.to(x.device))
class hed2rgb(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mat = rgb_from_hed

    def forward(self, x):
        return combine_stains(x, self.mat.to(x.device))
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += l1_loss(
                predi, target, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * l1_loss(
                pred, target, weight, reduction=self.reduction)

class ClassifyLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ClassifyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred.shape, gt.shape)

        return self.loss_weight * self.loss(pred, gt.squeeze(-1))
class ClassifyLossTrain(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ClassifyLossTrain, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        # self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred.shape, gt.shape)

        return self.loss_weight * self.loss(pred, gt)
class FocusDistanceL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocusDistanceL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, distance_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += l1_loss(
                predi, distance_gt, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * l1_loss(
                pred, distance_gt, weight, reduction=self.reduction)
class FocusDistanceSmoothL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocusDistanceSmoothL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, distance_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += smooth_l1_loss(
                predi, distance_gt, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * smooth_l1_loss(
                pred, distance_gt, weight, reduction=self.reduction)
class L1LossPry(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossPry, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = 0.
        target_pry = kornia.geometry.build_pyramid(target, len(pred))
        loss += l1_loss(
            pred[-1], target, weight, reduction=self.reduction)
        pred.pop(-1)
        for i, predi in enumerate(pred):
            loss += l1_loss(
            predi, target_pry[i], weight, reduction=self.reduction) * 0.33

        return self.loss_weight * loss

class ApeLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ApeLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()

    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 0
        for sr_i, ic_i in zip(sr, ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            ic_i_gt = 1 - torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.squeeze(-1), ic_i_gt)
            # print(self.diff_loss(sr_i, hr), self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # loss = 0.1 * loss + self.diff_loss(sr[-1], hr)
        # print('loss', loss)
        return loss
class ApeV2Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ApeV2Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()

    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 10.0 * torch.log10(1. ** 2 / ((sr[0] - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
        loss = loss + self.diff_loss(sr[0], hr)
        for sr_i, ic_i in zip(sr[1:], ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            ic_i_gt = 1 - torch.tanh(now_psnr - pre_psnr)
            # ic_i_gt = torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.squeeze(-1), ic_i_gt)
            # print('ics: ', ic_i.squeeze(-1), 'ics_gt: ', ic_i_gt, 'psnr: ', pre_psnr)
            # print('divide: ', self.diff_loss(sr_i, hr) / self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # loss = 0.1 * loss + self.diff_loss(sr[-1], hr)
        # print('loss', loss)
        return loss
class AdaptiveFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptiveFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()
        self.gamma = 10.

    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 10.0 * torch.log10(1. ** 2 / ((sr[0] - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
        # loss = loss + self.diff_loss(sr[0], hr)
        for sr_i, ic_i in zip(sr[1:-1], ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            ic_i_gt = torch.tanh((now_psnr - pre_psnr) * self.gamma)
            # ic_i_gt = torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.mse_loss(ic_i.squeeze(-1), ic_i_gt) # * 10.
            # print('ics: ', ic_i.squeeze(-1), 'ics_gt: ', ic_i_gt, 'psnr: ', pre_psnr)
            # print('divide: ', self.diff_loss(sr_i, hr) / self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # print('loss', loss, self.diff_loss(sr[-1], hr))
        loss = loss + self.diff_loss(sr[-1], hr) #  * 0.1
        # print('loss', loss)
        return loss
class AdaptiveLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptiveLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        # self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.L1Loss() # nn.MSELoss()
        self.gamma = 1. # 5. # 10.
    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 10.0 * torch.log10(1. ** 2 / ((sr[0] - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)

        for sr_i, ic_i in zip(sr[1:], ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            # ic_i_gt = torch.tanh((now_psnr - pre_psnr) * self.gamma)
            ic_i_gt = (now_psnr - pre_psnr) * self.gamma
            # ic_i_gt = torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.mse_loss(ic_i.squeeze(-1), ic_i_gt) # * 10.
            # print('ics: ', ic_i.squeeze(-1), 'ics_gt: ', ic_i_gt)
            # print('divide: ', self.diff_loss(sr_i, hr) / self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # loss = 0.1 * loss + self.diff_loss(sr[-1], hr)
        # print('loss', loss)
        return self.loss_weight * loss / len(ic)
class ApeLossOld(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ApeLossOld, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        target_pry = kornia.geometry.build_pyramid(target, 5)
        pred_images_dict = pred[0]
        dict_list = pred_images_dict.keys()
        ics_dict = pred[1]
        for i in dict_list:
            k = int(i.split('_')[-1])
            pre_psnr = 0.
            sr, ic = pred_images_dict[i], ics_dict[i]
            hr = target_pry[k]
            for sr_i, ic_i in zip(sr, ic):
                now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
                # kornia.metrics.psnr(sr_i, hr, 1.)
                # print(now_psnr)
                ic_i_gt = 1 - torch.tanh(now_psnr - pre_psnr)
                # print(ic_i_gt.squeeze().shape, ic_i.shape)
                # print(now_psnr, ic_i_gt, self.mse_loss(ic_i, ic_i_gt))
                pre_psnr = now_psnr
                # print(ic_i.shape, ic_i_gt, sr_i.shape, now_psnr)
                loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i, ic_i_gt.squeeze())
        loss = 0.1 * loss + self.diff_loss(pred[-1], target)
        # print(loss)
        return loss
class ShiftLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ShiftLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = torch.fft.rfft2(predi) - torch.fft.rfft2(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * loss_freq
            return loss
        else:
            p_w = torch.fft.rfft(pred, dim=-1)
            t_w = torch.fft.rfft(target, dim=-1)
            # diff = (p_w * torch.abs(t_w)) / (t_w * torch.abs(p_w))
            diff = torch.angle(t_w) - torch.angle(p_w)

            loss = torch.mean(torch.abs(diff))
            # print(loss)
            return self.loss_weight * loss * 0.05
class FreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = torch.fft.rfft2(predi) - torch.fft.rfft2(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * (loss_freq * 0.01 + self.l1_loss(predi, target))
            return loss / len(pred)
        else:
            diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
            loss = torch.mean(torch.abs(diff))
            # print(loss)
            return self.loss_weight * (loss * 0.01 + self.l1_loss(pred, target))
def FReLU(img):
    x = torch.fft.rfft2(img)
    x_real = torch.relu(x.real)
    x_imag = torch.relu(x.imag)
    x = torch.complex(x_real, x_imag)
    x = torch.fft.irfft2(x) - img / 2.
    return x
class FRLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = FReLU(predi) - FReLU(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * loss_freq * 4. + self.l1_loss(predi, target)
            return loss
        else:
            diff = FReLU(pred) - FReLU(target)
            loss = torch.mean(torch.abs(diff))
            # print(loss, self.l1_loss(pred, target))
            return self.loss_weight * loss * 4. + self.l1_loss(pred, target)
class SelfFRLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SelfFRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.window_size = 6
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target, lr):
        if isinstance(pred, list):
            loss = 0.
            target_ = [lr, target, target]
            for i, predi in enumerate(pred):
                x = torch.fft.rfft2(predi)
                x.real = torch.relu(x.real)
                x.imag = torch.relu(x.imag)
                x = torch.fft.irfft2(x) - predi / 2.
                x = torch.fft.fftshift(x, dim=[-2, -1])
                x_center = kornia.geometry.center_crop(x, [self.window_size, self.window_size])
                loss_freq = -(torch.log(torch.mean(torch.abs(x_center))+1e-5))
                y = torch.fft.rfft2(target_[i])
                y.real = torch.relu(y.real)
                y.imag = torch.relu(y.imag)
                y = torch.fft.irfft2(y) - target_[i] / 2.
                y = torch.fft.fftshift(y, dim=[-2, -1])
                # loss_freq = -(torch.log(torch.mean(torch.abs(x-y)) + 1e-5)) + loss_freq_c
                loss_freq = torch.clamp(loss_freq, -1., 1.)
                L_reg = torch.abs(torch.mean(x)-torch.mean(y))
                L1 = self.l1_loss(predi, target_[i])
                # print(loss_freq * 1., L_reg * 1000., L1) # * 0.01
                loss += self.loss_weight * loss_freq*0.1 + L_reg * 10. # + L1*0.05
            return loss
        else:
            x = torch.fft.rfft2(pred)
            x.real = torch.relu(x.real)
            x.imag = torch.relu(x.imag)
            x = torch.fft.irfft2(x) - pred / 2.
            x = torch.fft.fftshift(x, dim=[-2, -1])
            x_center = kornia.geometry.center_crop(x, [self.window_size, self.window_size])
            loss_freq = -torch.mean(torch.abs(x_center))
            y = torch.fft.rfft2(target)
            y.real = torch.relu(y.real)
            y.imag = torch.relu(y.imag)
            y = torch.fft.irfft2(y) - target / 2.
            y = torch.fft.fftshift(y, dim=[-2, -1])
            L_reg = torch.abs(torch.mean(x - y))
            L1 = self.l1_loss(pred, target)
            loss = self.loss_weight * loss_freq * 0.01 + L_reg * 1000. + L1*5
            # print(loss)
            return loss

class CharFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)
class StainLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(StainLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
        self.rgb2hed = rgb2hed()

    def forward(self, pred, target):
        hed = self.rgb2hed(pred)
        Hem, Eos, DAB = torch.chunk(hed, 3, dim=1)
        hedx = self.rgb2hed(target)
        Hemx, Eosx, DABx = torch.chunk(hedx, 3, dim=1)
        HE = torch.cat([Hem, Eos], dim=1)
        HEx = torch.cat([Hemx, Eosx], dim=1)
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target) + self.l1_loss(HE, HEx)
class PhasefreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PhasefreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        pred_angle = torch.angle(pred_f)
        tar_f = torch.fft.rfft2(target)
        phase_diff = torch.abs(torch.angle(tar_f) - pred_angle)
        loss = torch.mean(phase_diff)
        return self.loss_weight * loss * 0.01 # + self.l1_loss(pred, target)
class STNPhaseOnlyLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(STNPhaseOnlyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        pred_angle = torch.angle(pred_f)
        tar_f = torch.fft.rfft2(target)
        tar_angle = torch.angle(tar_f)
        phase_diff = torch.abs(torch.mean(torch.fft.irfft2(torch.exp(1j*tar_angle))) -
                               torch.mean(torch.fft.irfft2(torch.exp(1j*pred_angle))))
        loss = torch.mean(phase_diff)
        return self.loss_weight * loss * 0.01 # + self.l1_loss(pred, target)
class Phase2freqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Phase2freqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        tar_f = torch.fft.rfft2(target)
        pred_angle = torch.angle(pred_f)
        tar_angle = torch.angle(tar_f)
        # pred_mag = torch.abs(pred_f)
        p_f = torch.exp(1j * pred_angle) # pred_mag * torch.exp(1j * pred_angle)
        t_f = torch.exp(1j * tar_angle) # pred_mag * torch.exp(1j * tar_angle)
        pred_if = torch.fft.irfft2(p_f)
        target_if = torch.fft.irfft2(t_f)
        # var = torch.var(pred_if, dim=1)
        phase_diff = torch.abs(p_f - t_f)
        # diff = torch.abs(tar_f-pred_f)
        loss = torch.mean(phase_diff)
        return (self.l1_loss(pred_if, target_if) + loss * 0.05) * self.loss_weight# self.loss_weight * loss * 0.01 +
class Phase3freqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Phase3freqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss2 = Phase2freqLoss(loss_weight, reduction)
    def forward(self, pred, target):
        pred1, pred2 = pred

        return torch.mean(torch.abs(pred1)) + self.freq_loss2(pred2, target)# self.loss_weight * loss * 0.01 +
class PANSRFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PANSRFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
    def forward(self, pred, target, lr):
        loss = 0.
        # for predi in pred:
        loss += self.freq_loss(pred[0], lr)
        loss += self.freq_loss(pred[1], target)
        loss += self.freq_loss(pred[2], target)
        return loss
class PANSR_ShiftFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PANSR_ShiftFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
        self.shift_loss = ShiftLoss(loss_weight, reduction)
    def forward(self, pred, target, lr):
        loss = 0.
        # for predi in pred:
        loss += self.freq_loss(pred[0], lr) + self.shift_loss(pred[0], lr)
        loss += self.freq_loss(pred[1], target) + self.shift_loss(pred[1], target)
        loss += self.freq_loss(pred[2], target) + self.shift_loss(pred[2], target)
        return loss
class MultiFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MultiFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
    def forward(self, pred, target):
        loss = 0.
        for predi in pred:
            loss += self.freq_loss(predi, target)
        return loss
class MultiScaleFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.freq_loss = FreqLoss(loss_weight, reduction)

    def forward(self, pred, target):
        tar = target
        loss = 0.
        if isinstance(pred, list):
            for predi in pred[::-1]:
                loss += self.freq_loss(predi, tar)
                tar = F.interpolate(tar, scale_factor=0.5)
        else:
            loss += self.freq_loss(pred, tar)
        return loss
class FocalfreqsinLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocalfreqsinLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        tar_f = torch.fft.rfft2(target)
        diff = pred_f - tar_f
        phase_diff = torch.abs(torch.angle(tar_f) - torch.angle(pred_f))
        # phase_diff = phase_diff / (2 * 3.1415926536)
        phase_diff = torch.sin(phase_diff / 2.)
        loss = torch.mean(torch.abs(diff) * 0.01 + torch.pow(phase_diff, 2) * 0.1)
        return self.loss_weight * loss + self.l1_loss(pred, target)
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for pre in pred:
                loss += self.forward_single(pre, target)
            return loss / len(pred)
        else:
            return self.forward_single(pred, target)

    def forward_single(self, pred, target):

        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
