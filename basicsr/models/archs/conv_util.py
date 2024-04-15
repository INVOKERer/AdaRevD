'''
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
'''
import torch
import kornia
import cv2

import torch.fft as fft

def deconv_m(input_image, kernel, lambda_value=1e-5, max_iter=2):
    """
    使用PyTorch实现deconv函数中的去卷积功能
    :param input_image: 带噪声的输入图像 Tensor, size: num_channels x height x width
    :param kernel: 用于卷积的卷积核 Tensor, size: num_channels x kernel_size x kernel_size
    :param lambda_value: Tikhonov正则化参数
    :param max_iter: 迭代次数上限
    :return: 恢复后的图像 Tensor, size: num_channels x height x width
    """

    # 计算傅里叶变换
    input_image_fft = fft.fftn(input_image, dim=(-2, -1))
    kernel_fft = fft.fftn(kernel, dim=(-2, -1))

    # 使用Tikhonov正则化方法修复图像
    kernel_fft_conj = torch.conj(kernel_fft)
    kernel_fft_sqnorm = kernel_fft.pow(2).type(torch.float) # torch.sum(kernel_fft * kernel_fft_conj, dim=1, keepdim=True)

    kernel_fft_sqnorm[kernel_fft_sqnorm < lambda_value] = lambda_value
    r = (input_image_fft * kernel_fft_conj) / kernel_fft_sqnorm
    restored_image_fft = r #  torch.sum(r * kernel_fft, dim=1)

    # 迭代多次，以获得更好的恢复结果
    for i in range(max_iter):
        kernel_fft = fft.fftn(kernel, dim=(-2, -1))
        kernel_fft_sqnorm = kernel_fft.pow(2).type(torch.float) # torch.sum(kernel_fft * torch.conj(kernel_fft), dim=1, keepdim=True)
        kernel_fft_sqnorm[kernel_fft_sqnorm < lambda_value] = lambda_value
        r = (restored_image_fft * kernel_fft_conj) / kernel_fft_sqnorm
        restored_image_fft = r # torch.sum(r * kernel_fft, dim=1)

    # 计算傅里叶反变换得到恢复后的图像
    restored_image = fft.ifftn(restored_image_fft, dim=(-2, -1)).real

    return restored_image

# --------------------------------
# --------------------------------
def get_uperleft_denominator(img, kernel, img_clean=None, cir_shift=True):
    if cir_shift:
        kernel = torch.fft.fftshift(kernel, dim=[-1, -2])
    if img_clean is None:
        img_clean = kornia.filters.median_blur(img, (3,3))
    ker_f = convert_psf2otf(kernel, torch.zeros_like(img)) # discrete fourier transform of kernel

    numerator = torch.fft.fft2(img)
    # h, w = numerator.shape[-2:]
    # psd = torch.sum(numerator.pow(2), dim=(2, 3), keepdim=True) / (h*w)
    # snr = image_snr(img, img_clean)
    # noise_psd = psd / snr
    noise_psd = wiener_filter_para(img, img_clean)
    denominator = inv_fft_kernel_est(ker_f, noise_psd)#
    # img1 = img.cuda()
     #, 3, onesided=False)
    deblur = deconv(denominator, numerator)
    return deblur
def image_snr(input_blur, img_clean):
    noise_image = img_clean - input_blur
    noise_std = torch.std(noise_image, dim=[2, 3], keepdim=True)
    signal_mean = torch.mean(img_clean, dim=[2, 3], keepdim=True)
    return torch.log10(signal_mean / (noise_std+1e-5))
# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur, img_clean):
    if img_clean is None:
        img_clean = kornia.filters.median_blur(_input_blur, (3,3))
    diff = img_clean - _input_blur

    num = (diff.shape[2]*diff.shape[2])
    mean_n = torch.sum(diff, (2,3), keepdim=True)/num
    # print(diff.shape, mean_n.shape)
    var_n = torch.sum((diff - mean_n) ** 2, (2,3), keepdim=True)/(num-1)
    mean_input = torch.sum(_input_blur, (2, 3), keepdim=True)/num
    var_s2 = (torch.sum((_input_blur-mean_input) ** 2, (2, 3), keepdim=True)/(num-1))**(0.5)
    NSR = var_n / (var_s2 + 1e-5) * 8.0 / 3.0 / 10.0
    # print(NSR.shape)
    # NSR = NSR.view(-1,1,1,1)
    return NSR

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    # print('ker_f: ', ker_f.abs().max(), ker_f.abs().min())
    # print('NSR: ', NSR)
    inv_denominator = ker_f.pow(2) + NSR # + 1e-5
    # print('inv_denominator: ', inv_denominator.min(), inv_denominator.max())
    # pseudo inverse kernel in flourier domain.
    # inv_ker_f = torch.zeros_like(ker_f)
    # print(inv_denominator)
    ker_f = torch.conj(ker_f) / inv_denominator
    # ker_f.real /= inv_denominator
    # ker_f.imag /= -inv_denominator
    return ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    # deblur_f = torch.zeros_like(inv_ker_f).cuda()
    # deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
    #                         - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    # deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
    #                         + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur_f = inv_ker_f * fft_input_blur
    # print('inv_ker_f: ', inv_ker_f.abs().max(), inv_ker_f.abs().min())
    deblur = torch.fft.ifft2(deblur_f) # , 3, onesided=False)
    deblur = torch.abs(deblur)
    # print('deblur: ', deblur.max(), deblur.min())
    return deblur # torch.clamp(deblur, 0., 1.)

# --------------------------------
# --------------------------------
def convert_psf2otf(ker, psf):
    # psf = torch.zeros(size).cuda()
    # circularly shift
    # centre = ker.shape[2]//2 + 1
    C_k = psf.shape[0] // ker.shape[0]
    # print(psf.shape, ker.shape)
        # for b in range(ker.shape[0]):
        #     psf[b*C_k:(b+1)*C_k, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
        #     psf[b*C_k:(b+1)*C_k, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
        #     psf[b*C_k:(b+1)*C_k, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
        #     psf[b*C_k:(b+1)*C_k, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # else:
    if C_k > 1:
        for b in range(ker.shape[0]):
            psf[b * C_k:(b + 1) * C_k, ...] = ker[b, ...] # .unsqueeze(0)
    else:
        psf = ker
    # compute the otf
    otf = torch.fft.fft2(psf) # , 3, onesided=False)
    return otf


def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]

if __name__=='__main__':
    import numpy as np
    x = cv2.imread('/home/ubuntu/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/0.png')
    x_tensor = kornia.image_to_tensor(x / 255., keepdim=False)
    # y = cv2.imread('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/100.png')
    # y_tensor = kornia.image_to_tensor(y / 255., keepdim=False)
    # xy = torch.cat([x_tensor, y_tensor], dim=0)
    # h, w = xy.shape[-2:]
    # xy = xy.view(-1, 1, h, w)
    # z = xy[:3, :, :, :]
    # z = z.view(1,3,h,w)
    # img_ = kornia.tensor_to_image(z)
    # cv2.imwrite('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/0_z.png', np.uint8(img_ * 255.))
    kernel = kornia.filters.get_motion_kernel2d(31, angle=21)
    print(kernel.shape)
    H, W = x_tensor.shape[-2:]
    h, w = kernel.shape[-2:]
    pad_1 = (H-h)//2
    pad_2 = (W-w)//2
    pad_1_ = H - (H - h) // 2 - h
    pad_2_ = W - (W - w) // 2 - w
    pad = torch.nn.ZeroPad2d((pad_1, pad_1_, pad_2, pad_2_))
    x_blur = kornia.filters.filter2d(x_tensor, kernel)
    kernel_x = kernel.unsqueeze(0)
    # print(kernel_x.shape, pad_1, pad_1_, pad_2)
    # print(kernel_x)
    kernel_x = pad(kernel_x)
    print(kernel_x.shape)
    out_weiner = get_uperleft_denominator(x_blur, kernel_x, cir_shift=False)
    # out_weiner = kornia.enhance.normalize_min_max(out_weiner, 0., 1.)
    # out_weiner = torch.clamp(out_weiner, 0., 1.)
    print(out_weiner.max(), out_weiner.min())
    print(torch.mean(out_weiner - x_blur))
    print('mean_x', torch.mean(out_weiner - x_tensor))
    img_ = kornia.tensor_to_image(out_weiner)
    cv2.imwrite('/home/ubuntu/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/0_deconv.png', np.uint8(img_*255.))