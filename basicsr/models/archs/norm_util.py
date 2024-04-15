from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
# from basicsr.models.archs.attn_util import save_feature
# from basicsr.models.archs.dct_util import DCT2x
##########################################################################
## Layer Norm

def d4_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def d3_to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
def d5_to_3d(x):
    # x = x.permute(0, 2, 1, 3, 4)
    return rearrange(x, 'b c s h w -> b (s h w) c')


def d3_to_5d(x, s, h, w):
    x = rearrange(x, 'b (s h w) c -> b c s h w', s=s, h=h, w=w)
    # x = x.permute(0, 2, 1, 3, 4)
    return x
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight
        if self.mu_sigma:
            return x, mu, sigma
        else:
            return x
class WithBias_LayerNorm2x(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm2x, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight1 = nn.Parameter(torch.ones(normalized_shape))
        self.weight2 = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(normalized_shape))
            self.bias2 = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma+1e-5)
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5)
        x1 = x * self.weight1 + self.bias1
        x2 = x * self.weight2 + self.bias2
        return torch.cat([x1, x2], dim=-1)
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm(dim, bias, mu_sigma)
        self.out_dir = out_dir
        # if self.out_dir:
        #     self.dct = DCT2x()
    def forward(self, x):
        # if self.out_dir:
        #     save_feature(self.out_dir+'/beforeLN', x, min_max='log')
        h, w = x.shape[-2:]
        x = d4_to_3d(x)
        
        if self.mu_sigma:
            x, mu, sigma = self.body(x)
            return d3_to_4d(x, h, w), d3_to_4d(mu, h, w), d3_to_4d(sigma, h, w)
        else:
            x = self.body(x)
            # if self.out_dir:
            #     x = d3_to_4d(x, h, w)
            #     save_feature(self.out_dir + '/afterLN', x, min_max='log')
            #     return x
            # else:
            return d3_to_4d(x, h, w)


class LayerNorm2x(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm2x, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm2x(dim, bias, mu_sigma)
        self.out_dir = out_dir
        # if self.out_dir:
        #     self.dct = DCT2x()

    def forward(self, x):
        # if self.out_dir:
        #     save_feature(self.out_dir+'/beforeLN', x, min_max='log')
        h, w = x.shape[-2:]
        x = d4_to_3d(x)
        x = self.body(x)
        return d3_to_4d(x, h, w)

class nnLayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(nnLayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.norm(x).permute(0, 2, 1).contiguous()
        return x.view(b, c, h, w)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class WLayerNorm2d(nn.Module):

    def __init__(self, C, num_waves, k, eps=1e-6):
        super(WLayerNorm2d, self).__init__()
        # self.register_parameter('weight', nn.Parameter(torch.ones(1, channels, 1, k)))
        # self.register_parameter('bias', nn.Parameter(torch.zeros(1, channels, 1, k)))S
        self.weight = nn.Parameter(torch.ones(1, 1, num_waves, k))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_waves, k))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)

        x = (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias
        return x
class LayerNorm3d(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False):
        super(LayerNorm3d, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm(dim, bias, mu_sigma)

    def forward(self, x):
        s, h, w = x.shape[-3:]
        x = d5_to_3d(x)

        if self.mu_sigma:
            x, mu, sigma = self.body(x)
            return d3_to_5d(x, s, h, w), d3_to_5d(mu, s, h, w), d3_to_5d(sigma, s, h, w)
        else:
            x = self.body(x)
            return d3_to_5d(x, s, h, w)