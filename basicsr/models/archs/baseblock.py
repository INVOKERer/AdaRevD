import kornia.filters
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import * # LayerNorm2d, window_reversex, window_partitionx, FFT_ReLU, Attention_win
from einops.layers.torch import Rearrange, Reduce
# from basicsr.models.archs.ops_dcnv3.modules.dcnv3 import DCNv3, DCNv3Xt, DCNv3_pytorch
from basicsr.models.archs.ops_dcnv3.dcnv3_torch import DCNv3_pytorch, DCNv3_pytorch_aaai, DCNv3_pytorch_dual, DCNv3_pytorch_calayer
from basicsr.models.archs.attn_util import *
from basicsr.models.archs.dct_util import *
from basicsr.models.archs.utils_deblur import *
from basicsr.models.archs.local_arch import *



def MLP_cubic(h, w, c, deep):
    return nn.Sequential(*[nn.Sequential(
        nn.Linear(w, w),
        nn.PReLU(),
        Rearrange('b c h w -> b c w h'),
        nn.Linear(h, h),
        nn.PReLU(),
        Rearrange('b c w h -> b w h c'),
        nn.Linear(c, c),
        nn.PReLU(),
        Rearrange('b w h c -> b c h w'),
        nn.Linear(h, h),
        nn.PReLU(),
    ) for _ in range(deep)])
class MLP_UHD(nn.Module):
    def __init__(self,size_pry=[256, 256, 32], deep=1):
        super().__init__()
        h, w, c = size_pry
        self.path_real = MLP_cubic(h, w, c, deep)

        self.path_imag = MLP_cubic(h, w, c, deep)

    def forward(self, x):

        a_f = torch.fft.fft2(x)
        a_f_real = a_f.real
        a_f_imag = a_f.imag

        # print(a_f.shape)
        coeff_a = torch.real(torch.fft.ifft2(torch.complex(self.path_real(a_f_real), self.path_imag(a_f_imag))))

        return coeff_a + x
class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output
class ResBlock1(nn.Module):
    def __init__(self, ch):
        super(ResBlock1, self).__init__()
        self.body = nn.Sequential(
                        # nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output
class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(kernel_size*kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
def deblurfeaturefilter_fft(image, kernel, NSR=None):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    # otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-7)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur

def featurefilter_fft(image, kernel, NSR=None):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    # otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    # otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-7)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur

def featurefilter_deconv_fft(image, kernel, NSR=None):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    # otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    if NSR is None:
        otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-5)
    else:
        otf = torch.conj(otf) / (torch.abs(otf) ** 2 + NSR)
    # otf = torch.conj(otf) / (torch.abs(otf) + 1e-5)
    # otf_real = torch.clamp(otf.real, -100., 100.)
    # otf_imag = torch.clamp(otf.imag, -100., 100.)
    # otf = torch.complex(otf_real, otf_imag)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        # self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # identity_input = x  # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        return low_part
class kernel_deblur(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64]):
        super(kernel_deblur, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )

        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)
    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x = featurefilter_deconv_fft(x, kernel, NSR=NSR)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output

class kernel_deblur_feature(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64]):
        super(kernel_deblur_feature, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(1)
        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch * 2, 1, 1), requires_grad=True)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)
    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x_deblur = featurefilter_deconv_fft(x, kernel, NSR=NSR)
        # att = torch.cat([x, input], dim=1)
        att = torch.cat([x, kernel_feature], dim=1)
        att = self.conv_2(att * self.gamma)
        x = x * att
        output = x + input
        output = self.conv_attn(torch.cat([output, x_deblur], dim=1))

        return output
class kernel_attn_deblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attn_deblur, self).__init__()
        self.kernel_size = kernel_size
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU()
        # )
        self.conv_attn = nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                        nn.Softmax(1)
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*3, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*3, 1, 1), requires_grad=True)
    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # ker_attn = 1
        ker_attn = self.conv_attn(x)
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel_key = ker_attn @ kernel_attn

        kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(input - b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x = featurefilter_deconv_fft(x, kernel_key, NSR=NSR)
        # x = featurefilter_fft(x, kernel_key, NSR=None)
        # k_y = self.conv_kernel(kernel)
        # att = torch.cat([x, input, k_y], dim=1)
        att = torch.cat([x, input, kernel_feature], dim=1)
        att = self.conv_2(att*self.gamma)
        x = x * att
        output = x + input

        return output
class kernel_reblur(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_reblur, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )

        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*2, 1, 1), requires_grad=True)
    def forward(self, input, kernel):
        x = self.conv_1(input)
        # ker_attn = 1
        # ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        # ker_attn = torch.softmax(ker_attn, dim=-1)
        # kernel_code_uncertain = ker_attn @ kernel_code_uncertain.reshape(kernel_code.shape[0], kernel_code_uncertain.shape[1], -1)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = featurefilter_fft(x, kernel, NSR=None)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att*self.gamma)
        x = x * att
        output = x + input

        return output

class kernel_attn_reblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attn_reblur, self).__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        # self.conv_blur = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     # nn.GELU(),
        #     # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     # nn.GELU()
        # )
        self.kernel_up = nn.Upsample(size=kernel_size)
        self.conv_attn = nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                        # nn.Softmax(1)
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*2, 1, 1), requires_grad=True)
    def forward(self, input, kernel):
        x = self.conv_1(input)
        # ker_attn = 1
        ker_attn = self.conv_attn(x)
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel_key = ker_attn @ kernel_attn
        in_ker_size = int(np.sqrt(kernel_key.shape[-1]))
        if in_ker_size != self.kernel_size:
            kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], in_ker_size,
                                            in_ker_size)
            kernel_key = self.kernel_up(kernel_key)
        else:
            kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # result_dir = '/home/ubuntu/90t/personal_data/mxt/MXT/RevIR/Motion_Deblurring/results_kernel_feature_reblur'
        # kernelx = kernel_key.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        # for k in range(kernelx.shape[-1]):
        #     out_way = os.path.join(result_dir, 'x_kernel_' + str(k) + '.png')
        #     sns_plot = plt.figure()
        #     # print(kernelx[:, :, k].max())
        #     ker = kernelx[:, :, k]
        #     sns.heatmap(ker, cmap='Reds', linewidths=0., vmin=ker.min(), vmax=ker.max(),
        #                 xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds_r .invert_yaxis()
        #     sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
        #     plt.close()
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = featurefilter_fft(x, kernel_key, NSR=None)
        # k_y = self.conv_kernel(kernel)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att*self.gamma)
        x = x * att # + self.conv_blur(x)
        output = x + input

        return output
class kernel_attn_reblur_feature(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attn_reblur_feature, self).__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU()
        # )
        self.conv_attn = nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                        # nn.Softmax(1)
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*3, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*3, 1, 1), requires_grad=True)
    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # ker_attn = 1
        ker_attn = self.conv_attn(x)
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel_key = ker_attn @ kernel_attn

        kernel_key = kernel_key.view(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = featurefilter_fft(x, kernel_key, NSR=None)
        # k_y = self.conv_kernel(kernel)
        att = torch.cat([x, input, kernel_feature], dim=1)
        att = self.conv_2(att*self.gamma)
        x = x * att
        output = x + input

        return output

class kernel_reblur_feature(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_reblur_feature, self).__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU()
        # )
        self.conv_attn = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=1, stride=1, padding=0),
                        # nn.Softmax(1)
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*2, 1, 1), requires_grad=True)
    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # ker_attn = 1
        # ker_attn = self.conv_attn(x)
        # ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        # kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        # ker_attn = torch.softmax(ker_attn, dim=-1)
        # kernel_key = ker_attn @ kernel_attn
        #
        # kernel_key = kernel_key.view(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x_reblur = featurefilter_fft(x, kernel, NSR=None)
        # k_y = self.conv_kernel(kernel)
        att = torch.cat([x, kernel_feature], dim=1)
        att = self.conv_2(att*self.gamma)
        x = x * att
        output = x + input
        output = self.conv_attn(torch.cat([output, x_reblur], dim=1))
        return output
class Spatial_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Spatial_Conv, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.dwconv = nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, groups=out_ch, stride=1, padding=1)
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = deblurfeaturefilter_fft(x, kernel, NSR=None)
        x = self.dwconv(x)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
class kernel_deblur2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_deblur2, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_deconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x_deconv = deblurfeaturefilter_fft(x, kernel, NSR=None)
        x_deconv = self.conv_deconv(x_deconv)
        att = torch.cat([x, x_deconv], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
class feature_attention(nn.Module):
    def __init__(self, feat_ch, in_ch, out_ch):
        super(feature_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        nn.Conv2d(feat_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output
class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock_kernel_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = feature_attention(feature_ch, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y, kernel)
        return y

class NAFBlock_deblur_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur_feature(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp
        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel, kf)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel, kf)
        return y
class NAFBlock_reblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_reblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel = xk
        
        # inp = self.kernel_deblur(inp, kernel)
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        #
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y, kernel)
        return y
class NAFBlock_reblur_attn(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_reblur = kernel_attn_reblur(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel = xk
        x = inp
        # x = self.kernel_reblur(x, kernel)
        x = self.norm1(x)

        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_reblur(y, kernel)
        return y
class NAFBlock_reblur_attn_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_attn_reblur_feature(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel, kernel_feature = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel, kernel_feature)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y, kernel, kernel_feature)
        return y
class NAFBlock_reblur_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_reblur_feature(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel, kernel_feature = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel, kernel_feature)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y, kernel, kernel_feature)
        return y
class NAFBlock_deblur_attn(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_attn_deblur(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp
        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel, kf)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma

        return y
class NAFBlock_reblur_maxidx(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        self.kernel_size = kernel_size
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y_xk = torch.nn.functional.interpolate(y, size=[kernel.shape[-2], kernel.shape[-1]])
        b, c, h, w = y_xk.shape
        _, max_idx = torch.max(y_xk.view(b, c, -1), dim=-1)
        ks2 = kernel.shape[1]
        kernel = kernel.view(b, 1, ks2, -1)
        kernel = kernel.repeat(1, c, 1, 1)
        # print(max_idx.shape)
        max_idx = max_idx.unsqueeze(-1).repeat(1, 1, ks2).unsqueeze(-1)

        # print(max_idx.shape, max_idx)
        # print(max_idx.shape, kernel.shape)
        kernel = torch.gather(kernel, dim=-1, index=max_idx)
        # print(kernel.shape)
        kernel = kernel.view(b, c, self.kernel_size, self.kernel_size)
        y = self.kernel_deblur(y, kernel)
        return y

##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self, x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        # print(x.shape)
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class NAFBlock_reblur_softmax(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        self.kernel_size = kernel_size
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.yk_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.kernel_deblur = kernel_reblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        sq = int(np.sqrt(kernel.shape[1]))
        y_xk = torch.nn.functional.interpolate(y, size=[sq, sq])
        y_xk = self.yk_conv(y_xk)
        # print(kernel.shape)
        ker_attn = rearrange(y_xk, 'b c h w -> b c (h w)')
        # ker_attn = torch.sigmoid(ker_attn)
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel = rearrange(kernel, 'b n k1 k2 -> b n (k1 k2)')
        # print(kernel.shape, ker_attn.shape)
        kernel = ker_attn @ kernel
        kernel = torch.softmax(kernel, dim=-1)
        kernel = rearrange(kernel, 'b n (k1 k2) -> b n k1 k2', k1=self.kernel_size, k2=self.kernel_size)
        y = self.kernel_deblur(y, kernel)
        return y
class NAFBlock_conv(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = Spatial_Conv(in_ch=c, out_ch=c)

    def forward(self, inp):
        # inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y)
        return y
class FNAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FNAFBlock_kernel_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = feature_attention(feature_ch, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FCNAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FC1NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FCNAFBlock_kernel_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = feature_attention(feature_ch, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
def generate_k(model, code, n_row=1):
    # eval???
    model.eval()

    # unconditional model
    # for a random Gaussian vector, its l2norm is always close to 1.
    # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

    u = code  # [B, 19*19]
    samples, _ = model.inverse(u)

    samples = model.post_process(samples)

    return samples
def generate_k_forward(model, code, n_row=1):
    # eval???
    # model.eval()

    # unconditional model
    # for a random Gaussian vector, its l2norm is always close to 1.
    # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

    u = code  # [B, 19*19]
    # samples, _ = model(u)
    samples, _ = model.inverse(u)
    samples = model.post_process(samples)

    return samples
class SimpleGate(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlockold(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., attn_type='SCA'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.attn_type = attn_type
        # Simplified Channel Attention
        # if attn_type == 'SCA':
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # AvgPool2d(base_size=[], fast_imp=fast_imp, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # elif attn_type == 'grid_dct_SE':
        #     self.sca = WDCT_SE(dw_channel // 2, num_heads=num_heads, bias=True, window_size=window_size)

        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        if self.attn_type == 'SCA':
            x = x * self.sca(x)
        elif self.attn_type == 'grid_dct_SE':
            x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class DCNv3Block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., attn_type='SCA'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.dcn_conv2 = DCNv3(channels=dw_channel // 2, kernel_size=3, pad=1, stride=1, group=num_heads)
        self.dcn_conv2 = DCNv3_pytorch(channels=dw_channel//2, kernel_size=3, pad=1, stride=1, group=num_heads,
                                        offset_scale=1.0,
                                        act_layer='GELU',
                                        norm_layer='LN',
                                        center_feature_scale=True,
                                        remove_center=False)
        self.conv2 = nn.Conv2d(in_channels=dw_channel//2, out_channels=dw_channel//2, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel//2,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.attn_type = attn_type
        # Simplified Channel Attention
        if attn_type == 'SCA':
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        elif attn_type == 'grid_dct_SE':
            self.sca = WDCT_SE(dw_channel // 2, num_heads=num_heads, bias=True, window_size=window_size)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        # x1 = x1.permute(0, 2, 3, 1)
        x1 = self.dcn_conv2(x1)
        # x1 = x1.permute(0, 3, 1, 2)
        x2 = self.conv2(x2)
        x = x1*x2
        if self.attn_type == 'SCA':
            x = x * self.sca(x)
        elif self.attn_type == 'grid_dct_SE':
            x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class FDCNv3Block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.dcn_conv2 = DCNv3_pytorch(channels=dw_channel // 2, kernel_size=3, pad=1, stride=1, group=num_heads,
                                       offset_scale=1.0,
                                       act_layer='GELU',
                                       norm_layer='LN',
                                       center_feature_scale=True,
                                       remove_center=False)
        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=3, padding=1,
                               stride=1,
                               groups=dw_channel // 2,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()

        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x1, x2 = x.chunk(2, dim=1)
        # x1 = x1.permute(0, 2, 3, 1)
        x1 = self.dcn_conv2(x1)
        # x1 = x1.permute(0, 3, 1, 2)
        x2 = self.conv2(x2)
        x = x1 * x2

        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.avg = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))
        # x = x * self.sca(self.avg(x))
        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FC1NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x

class FC1NAFBlockDFFN(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.ffn = DFFN(dim=c, ffn_expansion_factor=FFN_Expand, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.ffn(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FC1NAFBlockFFN(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.ffn = FFN(dim=c, ffn_expansion_factor=FFN_Expand, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.ffn(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):

        super(FFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        # self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # x_patch_fft = torch.fft.rfft2(x_patch.float())
        # x_patch_fft = x_patch_fft * self.fft
        # x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        # x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #               patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class FC1XNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1, train_size=[256, 256]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv_phase_tig(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU, train_size=train_size) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block1(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.sc = nn.Conv2d(features * 2, features * 2, 1, 1, 0)

        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = AvgPool2d(base_size=80)

        self.out = nn.Conv2d(features, features, 1, 1, 0)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)
        attention_vectors = self.sc(attention_vectors)
        # attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        out = fea_high + fea_low
        out = self.out(out)
        return out
class FC1SENAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sea = SFconv(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x_spatial = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x_fourier = self.fft_block1(x_)
        # x_all = x_spatial + x_fourier
        x = self.sea(x_spatial, x_fourier)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FC2NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        # self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x))) + self.fft_block2(x)


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FOCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))
        x = self.dropout2(x)
        x = y + x * self.gamma + self.fft_block1(inp)
        return x
class FLC1NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1, kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_large_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU, kernel_size=kernel_size) # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta + self.fft_block1(x_)

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))
        x = self.dropout2(x)
        x = y + x * self.gamma
        return x
class FCANAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_conv_multiact(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_conv_multiact(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FUNAFBlock(nn.Module):
    def __init__(self, c, scale_up=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv_with_scale_up(in_dim=c, dim=c//scale_up, dw=DW_Expand, bias=True, act_method=nn.GELU, scale_up=scale_up) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv_with_scale_up(in_dim=c, dim=c//scale_up, dw=DW_Expand, bias=True, act_method=nn.GELU, scale_up=scale_up)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FDCANAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv_dycalayer(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        # y, ca = self.fft_block1(x_)
        # x = x * (self.sca(x)+ca)
        y = self.fft_block1(x_)
        x = x * self.sca(x)
        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + y
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FDCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv2(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv2(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0: multihead_fft_bench_complex_conv fft_bench_complex_conv multihead_fft_bench_real_complex_dconv
        self.fft_block1 = multihead_fft_bench_real_complex_resconv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0: multihead_fft_bench_complex_conv fft_bench_complex_conv multihead_fft_bench_real_complex_dconv
        self.fft_block1 = multihead_fft_bench_real_complex_3x3conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FDepthCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_depthconv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FPNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = Fourier_phase_DConvblock(c, dw_channel, kernel_size=sub_idx, bias=True) # , act_method=nn.GELU
        self.fft_block2 = Fourier_phase_DConvblock(c, dw_channel, kernel_size=sub_idx, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FPTNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = Fourier_phaseT_DConvblock(c, dw_channel, kernel_size=sub_idx, bias=True) # , act_method=nn.GELU
        self.fft_block2 = Fourier_phaseT_DConvblock(c, dw_channel, kernel_size=sub_idx, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FPTv2NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = Fourier_phaseTv2_DConvblock(c, dw_channel, num_heads=num_heads, kernel_size=sub_idx, bias=True) # , act_method=nn.GELU
        self.fft_block2 = Fourier_phaseTv2_DConvblock(c, dw_channel, num_heads=num_heads, kernel_size=sub_idx, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class Fourier_phase_DConvblock(nn.Module):
    def __init__(self, dim, hdim, kernel_size, bias=True, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        if kernel_size > 1:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=dim, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])
            self.conv_mag = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=dim, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])
        else:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])
            self.conv_mag = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])

        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        # self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        # y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)

        y = self.conv_mag(y_mag) * torch.exp(-1j * y_phase)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class Fourier_phaseT_DConvblock(nn.Module):
    def __init__(self, dim, hdim, kernel_size, bias=True, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        if kernel_size > 1:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=dim, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])

        else:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])

        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        # self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)

        # y_phase = y_phase / self.pi
        y_mag, y_phase = self.conv_phase(torch.cat([y_mag, y_phase], dim=0)).chunk(2, dim=0)

        y = y_mag * torch.exp(-1j * y_phase)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class Fourier_phaseTv2_DConvblock(nn.Module):
    def __init__(self, dim, hdim, kernel_size, num_heads, bias=True, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.proj_in = nn.Sequential(*[nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                                       nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=3,
                                                 padding=1, stride=1, groups=dim, bias=bias),
                                       ])
        self.proj_out = nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        if kernel_size > 1:
            self.conv_phase = nn.Sequential(*[
                nn.ReflectionPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=num_heads,
                          bias=bias),
                nn.GELU(),
                nn.ReflectionPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=num_heads,
                          bias=bias),
            ])

        else:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=1, padding=0, stride=1,
                          groups=num_heads, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=1, padding=0, stride=1,
                          groups=num_heads, bias=bias),
            ])

        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        # self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.proj_in(x)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)

        # y_mag, y_phase = self.conv_phase(torch.cat([y_mag, y_phase], dim=0)).chunk(2, dim=0)
        y_phase = self.conv_phase(y_phase) + y_phase
        y = y_mag * torch.exp(-1j * y_phase)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.proj_out(y)
class DCTNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        if sub_idx == 1:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
            self.fft_block2 = nn.Sequential(DCT2x(),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
        else:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.ZeroPad2d((0, sub_idx-1, 0, sub_idx-1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.GELU(),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
            self.fft_block2 = nn.Sequential(DCT2x(),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))


        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class DctFuseBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        if sub_idx == 1:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
        else:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.GELU(),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)

        return x * self.beta
class Fusext_dct(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32, sub_idx=1):
        super(Fusext_dct, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = DctFuseBlock(dim * 2, sub_idx=sub_idx)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.conv_spatial(x_in) + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output
class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class DCTFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):

        super(DCTFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.dct_mix = nn.Parameter(torch.ones((1, hidden_features * 2, 1, 1, self.patch_size, self.patch_size)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)
        self.dct = DCT2(window_size=self.patch_size)

        self.idct = IDCT2(window_size=self.patch_size)
    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_dct = self.dct(x_patch)
        x_patch_dct = x_patch_dct * self.dct_mix

        x_patch = self.idct(x_patch_dct)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, cs='channel_mlp'):
        super(AttnBlock, self).__init__()

        self.dct = DCT2x()
        self.idct = IDCT2x()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.normz = LayerNorm2d(dim)
        self.attn = AttentionX(dim, num_heads=num_heads, bias=True, cs=cs)
        self.conv_spatial = DFFN(dim, ffn_expansion_factor, bias)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.fft_block1 = fft_bench_complex_mlp(dim, ffn_expansion_factor, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(dim, ffn_expansion_factor, window_size=None, bias=True, act_method=nn.GELU)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, x_in):  # left_down - right_up?

        x = self.norm1(self.dct(x_in))

        x = self.idct(self.attn(x)) + self.fft_block1(self.normz(x_in))
        x = x * self.gamma + x_in

        y = self.norm2(x)
        y = self.conv_spatial(y) + self.fft_block2(y)
        x = y * self.beta + x
        return x
class Fusext(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', dim_head=32):
        super(Fusext, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.norm1 = LayerNorm2d(dim * 2)
        self.norm2 = LayerNorm2d(dim)
        self.attn = AttentionX(dim * 2, num_heads=dim//dim_head, bias=True, cs='channel', out_dim=dim)
        self.conv_spatial = DFFN(dim, ffn_expansion_factor, bias)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.norm1(x_in)
        x = self.attn(x)*self.gamma + self.conv2(x_cat)

        x = self.conv_spatial(self.norm2(x))*self.beta + x

        return x
class Fusextv2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32):
        super(Fusextv2, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = DFFN(dim * 2, ffn_expansion_factor, bias)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.norm1(x_in)
        x = self.conv_spatial(x)*self.beta + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output
class FuseBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU) # , act_method=nn.GELU

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)

        return x * self.beta
class Fusextv3(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32):
        super(Fusextv3, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = FuseBlock(dim * 2)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.conv_spatial(x_in) + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output
class FuseXBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, fft_bench=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0: fft_bench_complex_conv
        if fft_bench:
            self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_bench = fft_bench
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.fft_bench:
            if self.sin:
                x = x + torch.sin(self.fft_block1(x_))
            else:
                x = x + self.fft_block1(x_)
        x = self.dropout1(x)

        return x * self.beta
class FusNext(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32, fft_bench=True):
        super(FusNext, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = FuseXBlock(dim * 2, fft_bench=fft_bench)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        # print(left_down.shape, right_up.shape)
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.conv_spatial(x_in) + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output
class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm2d(dim * 2)

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class Ffftformer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=True, att=True):
        super(Ffftformer, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm2d(dim)
            self.attn = FSAS(dim, bias)
            self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

            self.fft_block1 = fft_bench_complex_mlp(dim, ffn_expansion_factor, window_size=None, bias=True,
                                                    act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(dim, ffn_expansion_factor, window_size=None, bias=True,
                                                act_method=nn.GELU)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            z = self.norm1(x)
            x = x + self.gamma * (self.attn(z) + self.fft_block1(z))
        y = self.norm2(x)
        x = x + self.beta * (self.ffn(y)+self.fft_block2(y))

        return x
class AttentionX(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim // num_heads
        self.out_dim = out_dim // num_heads
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 2+out_dim, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 2+out_dim, dim * 2+out_dim, kernel_size=3,
                                stride=1, padding=1, groups=dim * 2+out_dim, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(out_dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (head c) (h h1) (w w1) -> (b h1 w1) head c (h w)', head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (head c) (h1 h) (w1 w) -> (b h1 w1) head c (h w)', head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv.split([self.out_dim, self.dim, self.dim], dim=2)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)

        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv.split([self.out_dim, self.dim, self.dim], dim=2)
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, left, right):
        batch_size = left.shape[0]
        n_feats = left.shape[1]

        inp_feats = torch.cat([left, right], dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V
class fft_bench_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_complex_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y

class fft_bench_complex_conv_phase_tig(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, train_size=[256, 256]):
        super(fft_bench_complex_conv_phase_tig, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)
        self.phase_para = nn.Parameter(torch.zeros(1, dim, train_size[0], train_size[1]))
        torch.nn.init.kaiming_uniform_(self.phase_para, a=math.sqrt(2))
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # phase_proj = torch.exp(1j*self.phase_para)
        phase_proj = kornia.geometry.subpix.spatial_softmax2d(self.phase_para)
        phase_proj = torch.fft.rfft2(phase_proj, norm=self.norm)
        # phase_proj = phase_proj / (torch.abs(phase_proj)+1e-6)
        # print(y.shape, phase_proj.shape)
        y = y * phase_proj
        # print(torch.mean(torch.abs(self.phase_para)))
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y

class fft_bench_complex_large_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, kernel_size=19):
        super(fft_bench_complex_large_conv, self).__init__()
        self.act_fft = act_method()
        # self.act_spatial = nn.GELU()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.bias = bias
        self.norm = norm
        # init_kers = torch.zeros([1, hid_dim, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
        # torch.nn.init.kaiming_uniform_(self.kers, a=math.sqrt(5))
        self.kernel_size = kernel_size
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.ker_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hid_dim, out_channels=num_heads*kernel_size**2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.BatchNorm2d(num_heads*kernel_size**2),
            # nn.Softmax(1)
        )
        self.num_heads = num_heads
        # nn.init.kaiming_normal(self.kers)
        # self.min = inf
        # self.max = -inf
    def forward(self, x):

        y = self.complex_conv1(x)
        batch_size = y.shape[0]
        ker_x = self.ker_gen(y).view(batch_size, self.num_heads, self.kernel_size, self.kernel_size)
        ker_x = kornia.geometry.subpix.spatial_softmax2d(ker_x)
        y = self.pad(y)
        # print(y.shape)
        H, W = y.shape[-2:]
        y = rearrange(y, 'b (nh c1) h w -> b nh c1 h w', nh=self.num_heads)

        kers = convert_psf2otf(ker_x, (batch_size, self.num_heads, H, W))
        kers = kers.unsqueeze(2)
        y = torch.fft.rfft2(y, norm=self.norm)
        # y = y * kers
        y = y * torch.conj(kers) / (torch.abs(kers)+1e-5)
        y = rearrange(y, 'b nh c1 h w -> b (nh c1) h w')
        y = torch.cat([y.real, y.imag], dim=1)
        # print(self.kers.shape, (1, y.shape[1]//2, H, W))


        # kers = torch.cat([kers.real, kers.imag], dim=1)
        # # y1, y2 = y.chunk(2, dim=1)
        # # ker1, ker2 = kers.chunk(2, dim=1)
        # # kers_mag = torch.abs(kers)
        # y = y * kers
        y = self.act_fft(y)

        y_real, y_imag = y.chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # ker_x = self.ker_norm(self.kers)
        # ker_x = kornia.geometry.subpix.spatial_softmax2d(ker_x)

        # y2 = y * ker2  # / (kers_mag ** 2 + 1e-3)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # y = torch.concat([y, y1, y2], dim=1)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # print(y.shape)
        y = y[:, :, self.kernel_size//2-1:-self.kernel_size//2, self.kernel_size//2-1:-self.kernel_size//2].contiguous()
        # print(y.shape, self.kernel_size//2)
        y = self.complex_conv2(y)

        # y, y1, y2 = y.chunk(3, dim=1)
        # y1 = torch.softmax(y1, dim=1)
        # y = y * torch.sum(y1 * y2, dim=1, keepdim=True)
        # y = self.act_spatial(y)
        return y
class fft_bench_complex_large_conv1(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, kernel_size=19):
        super(fft_bench_complex_large_conv1, self).__init__()
        self.act_fft = act_method()
        # self.act_spatial = nn.GELU()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.bias = bias
        self.norm = norm
        init_kers = torch.zeros([1, dim, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        self.kers = nn.Parameter(init_kers, requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.kers, a=math.sqrt(5))
        self.ker_norm = LayerNorm2d(dim)
        # nn.init.kaiming_normal(self.kers)
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        # print(self.kers.shape, (1, y.shape[1]//2, H, W))


        y = self.act_fft(y)
        y = self.complex_conv2(y)
        y_real, y_imag = y.chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        ker_x = self.ker_norm(self.kers)
        ker_x = kornia.geometry.subpix.spatial_softmax2d(ker_x)
        kers = convert_psf2otf(ker_x, (1, y.shape[1], H, W))
        # kers = torch.cat([kers.real, kers.imag], dim=1)
        # y1, y2 = y.chunk(2, dim=1)
        # ker1, ker2 = kers.chunk(2, dim=1)
        # kers_mag = torch.abs(kers)
        y1 = y * kers
        y2 = y * torch.conj(kers)  # / (kers_mag ** 2 + 1e-3)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = torch.concat([y, y1, y2], dim=1)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y, y1, y2 = y.chunk(3, dim=1)
        y = y + y1 * y2
        # y = self.act_spatial(y)
        return self.project_out(y)

class fft_bench_conv_multiact(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_conv_multiact, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y_real, y_imag = y.chunk(2, dim=1)
        y_real1, y_real2 = y_real.chunk(2, dim=1)
        y_real1 = self.act_fft(y_real1)
        y_real2 = -self.act_fft(-y_real2)
        y_imag1, y_imag2, y_imag3, y_imag4 = y_imag.chunk(4, dim=1)
        y_imag1 = self.act_fft(y_imag1)
        y_imag2 = -self.act_fft(-y_imag2)
        y_imag3 = -self.act_fft(-y_imag3)
        y_imag4 = self.act_fft(y_imag4)

        y_real = torch.cat([y_real1, y_real2], dim=1)
        y_imag = torch.cat([y_imag1, y_imag2, y_imag3, y_imag4], dim=1)
        y = torch.cat([y_real, y_imag], dim=1)
        # y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_deform_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_deform_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_deform_conv = DCNv3_pytorch(hid_dim * 2, kernel_size=1, group=num_heads,
                                                 center_feature_scale=False, remove_center=False)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.complex_deform_conv(y)
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_deform_conv2(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_deform_conv2, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_deform_conv = DCNv3_pytorch(hid_dim * 2, kernel_size=1, group=num_heads,
                                                 center_feature_scale=False, remove_center=False)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))

        y = self.act_fft(y)
        y = self.complex_deform_conv(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, spatial_size=[128, 128]):
        super(fft_bench_mlp, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        h, w = spatial_size
        self.spatial_size = spatial_size
        # print(dim, hid_dim)
        self.complex_mlp = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias),
            act_method(),
            nn.Linear(w//2+1, w//2+1),
            act_method(),
            Rearrange('b c h w -> b c w h'),
            nn.Linear(h, h),
            act_method(),
            Rearrange('b c w h -> b c h w'),
            nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        )])

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if H != self.spatial_size[0] or W != self.spatial_size[1]:
            x = F.interpolate(x, size=self.spatial_size,   mode='bilinear', align_corners=True)
        y = torch.fft.rfft2(x, norm=self.norm)
        y_real, y_imag = self.complex_mlp(torch.cat([y.real, y.imag], dim=1)).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=self.spatial_size, norm=self.norm)
        if H != self.spatial_size[0] or W != self.spatial_size[1]:
            y = F.interpolate(y, size=[H, W],   mode='bilinear', align_corners=True)

        return y

class fft_bench_complex_conv_with_scale_up(nn.Module):
    def __init__(self, in_dim, dim, dw=1, norm='backward', act_method=nn.ReLU, scale_up=2, bias=False):
        super(fft_bench_complex_conv_with_scale_up, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        if scale_up > 1:
            self.conv_scale_up = nn.Sequential(nn.Conv2d(in_dim, dim*(scale_up**2), 1, bias=False),
                                    nn.PixelShuffle(scale_up)
                                    )
            self.conv_scale_down = nn.Sequential(nn.PixelUnshuffle(scale_up),
                                                 nn.Conv2d(dim * (scale_up ** 2), in_dim, 1, bias=False),
                                                 )
        else:
            self.conv_scale_up = nn.Identity()
            self.conv_scale_down = nn.Identity()
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):

        x = self.conv_scale_up(x)
        _, _, H, W = x.shape
        # print(x.shape)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.conv_scale_down(y)
        return y
class fft_bench_complex_conv_dycalayer(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_complex_conv_dycalayer, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)
        self.ca = DCNv3_pytorch_calayer(dim, kernel_size=3) # kernel_size=3 5
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)

        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        ca = self.ca(torch.abs(y))
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y * ca
class fft_bench_complex_conv2(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_complex_conv2, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_dconv = nn.Conv2d(hid_dim * 2, hid_dim * 2, groups=num_heads*2, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.complex_dconv(y)
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fft_bench_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_complex_conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim, hid_dim, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv1(x)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=0))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.conv2(y)
class multihead_fft_bench_real_complex_3x3conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_3x3conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        # self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fft_bench_real_complex_resconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_resconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fft_bench_real_complex_noresconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_noresconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) # + x_fft
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fft_bench_real_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        # self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        # self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=0)
        # x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        # x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y)
        y_real, y_imag = y.chunk(2, dim=0)
        # y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fft_bench_real_complex_dconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_dconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=hid_dim, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=hid_dim, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fft_bench_resconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_resconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim*2, hid_dim*2, groups=hid_dim, kernel_size=1, padding=0, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, hid_dim*2, groups=hid_dim, kernel_size=1, padding=0, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_fft = self.conv1(x_fft)
        # x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft
        # y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(y)).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class multihead_fourier_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fourier_complex_conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = Complex_Conv(dim, hid_dim, kernel_size=1, bias=bias)
        # self.norm1 = nn.Identity() # LayerNorm2d(hid_dim)
        self.complex_conv1 = Complex_Conv(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = Complex_Conv(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = Complex_Conv(hid_dim, dim, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=0)
        x_fft = self.conv1(x_fft)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft

        y_real, y_imag = self.conv2(self.act_fft(y)).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fourier_complex_conv1x1(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fourier_complex_conv1x1, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = Complex_Conv(dim, hid_dim, kernel_size=1, bias=bias)

        self.conv2 = Complex_Conv(hid_dim, dim, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=0)
        x_fft = self.conv1(x_fft)
        y_real, y_imag = self.conv2(self.act_fft(x_fft)).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_depthconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_complex_depthconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_dconv = nn.Conv2d(hid_dim * 2, hid_dim * 2, groups=hid_dim * 2, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.complex_dconv(y)
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class Complex_Conv(nn.Module):
    def __init__(self, dim, dim_out, groups=1, kernel_size=1, padding=0, bias=True):
        super(Complex_Conv, self).__init__()
        # padding = kernel_size // 2 - 1
        self.complex_conv = nn.Conv2d(dim, dim_out * 2, kernel_size=kernel_size, bias=bias, padding=padding, groups=groups)

    def forward(self, x):
        a, b = self.complex_conv(x).chunk(2, dim=1)
        a_real, a_imag = a.chunk(2, dim=0)
        b_imag, b_real = b.chunk(2, dim=0)

        return torch.cat([a_real - b_real, b_imag + a_imag], dim=0)
