import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.arch_util import fft_bench_complex_mlp
# from basicsr.models.archs.dct_util import *
# from basicsr.models.archs.ops_dcnv3.dcnv3_torch import DCNv3_pytorch, DCNv3_pytorch_aaai, DCNv3_pytorch_dual
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


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
class DefDCTFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):

        super(DefDCTFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        num_heads = hidden_features // 32
        self.dcn_conv2 = DCNv3_pytorch(channels=hidden_features, kernel_size=3, pad=1, stride=1, group=num_heads,
                                       offset_scale=1.0,
                                       act_layer='GELU',
                                       norm_layer='LN',
                                       center_feature_scale=True,
                                       remove_center=False)
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
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x2 = self.dcn_conv2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class FuseAS(nn.Module):
    def __init__(self, dim, bias):
        super(FuseAS, self).__init__()

        self.to_hiddenx = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.to_hiddeny = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=bias)
        self.to_hidden_dw_q = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x, y):
        hiddenx = self.to_hiddenx(x)
        q = self.to_hidden_dw_q(self.to_hiddeny(y))
        k, v = self.to_hidden_dw(hiddenx).chunk(2, dim=1)

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
class FuseV2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(FuseV2, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.normy = LayerNorm(dim, LayerNorm_type)
        self.attn = FuseAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, y, x):

        x = x + self.attn(self.norm1(x), self.normy(y))

        x = x + self.ffn(self.norm2(x))

        return x
class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

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


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class Fuse(nn.Module):
    def __init__(self, n_feat, ffn_expansion_factor=2.66):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2, ffn_expansion_factor=ffn_expansion_factor, bias=True, att=False)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        # enc, dnc = x_
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d

        return output
class FuseV3(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(FuseV3, self).__init__()
        self.n_feat = dim
        self.norm = LayerNorm(dim*2, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

        self.conv = nn.Conv2d(dim*2, dim*2, 1, 1, 0)
        self.conv_fft = fft_bench_complex_mlp(dim, bias=bias)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
    def forward(self, left_down, right_up):
        x = torch.cat([left_down, right_up], dim=1)
        x1, x2 = self.norm(self.conv(x)).chunk(2, dim=1)
        x = torch.cat([self.ffn(x1), self.conv_fft(x2)], dim=1) + x
        x = self.conv2(x)
        e, d = x.chunk(2, dim=1)
        output = e + d

        return output
class FuseV4x(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(FuseV4x, self).__init__()

        self.conv_right_up = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
                        nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                        nn.GELU(),
                        nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
                        )
        self.conv_left_down = nn.Sequential(
                        fft_bench_complex_mlp(dim, bias=bias)
                        )
        self.conv_att = nn.Sequential(
                        nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0),
                        LayerNorm(dim, LayerNorm_type),
                        DFFN(dim, ffn_expansion_factor, bias),
                        # nn.Sigmoid()
                        )

    def forward(self, left_down, right_up):
        x_right_up = self.conv_right_up(right_up) + right_up
        x_left_down = self.conv_left_down(left_down) + left_down
        att = torch.cat([x_right_up, x_left_down], dim=1)
        att = self.conv_att(att)
        x_right_up = x_right_up * torch.sin(att)
        output = x_right_up + x_left_down

        return output
class FuseV4(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(FuseV4, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv_a = nn.Sequential(
                        nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1, padding=0),
                        nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim),
                        nn.GELU(),
                        nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1, padding=0)
                        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(dim * 2, dim*2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1, padding=0)
        )
        self.conv_spatial = nn.Sequential(
            # nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0),
            DFFN(dim*2, ffn_expansion_factor/2., bias, dim_out=dim*2)
        )
        self.norm = LayerNorm(dim*2, LayerNorm_type)

    def forward(self, left_down, right_up):
        x_in = self.conv1(torch.cat([left_down, right_up], dim=1))
        x = self.norm(x_in)
        x_fft = torch.fft.rfft2(x)
        # x_mag = torch.abs(x_fft)
        # x_phase = torch.angle(x_fft)
        # x_mag = self.conv_mag(x_mag)
        # x_phase = self.conv_phase(x_phase)
        # x_fourier = x_mag * torch.exp(-1j*x_phase)
        x_real = x_fft.real
        x_imag = x_fft.imag
        x_real = self.conv_a(x_real)
        x_imag = self.conv_b(x_imag)
        x_fourier = torch.complex(x_real, x_imag)
        x = self.conv_spatial(x) + torch.fft.irfft2(x_fourier) + x_in
        e, d = self.conv2(x).chunk(2, dim=1)
        # e, d = x
        output = e + d

        return output
class FuseV5(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(FuseV5, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)

        self.conv_spatial = DCTFFN(dim*2, ffn_expansion_factor, bias, dim_out=dim*2)

        self.norm = LayerNorm(dim*2, LayerNorm_type)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_in = self.conv1(torch.cat([left_down, right_up], dim=1))
        x = self.norm(x_in)
        x = self.conv_spatial(x) + x_in
        e, d = self.conv2(x).chunk(2, dim=1)
        # e, d = x
        output = e + d

        return output
class FuseV6(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', group_chan=32):
        super(FuseV6, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        num_heads = dim // group_chan
        self.conv_spatial = DFFN(dim*2, ffn_expansion_factor, bias, dim_out=dim*2)
        self.dcn_conv = DCNv3_pytorch_dual(channels=dim, channels_ref=dim, kernel_size=3, pad=1, stride=1, group=num_heads,
                                       offset_scale=1.0,
                                       act_layer='GELU',
                                       norm_layer='LN',
                                       center_feature_scale=True,
                                       remove_center=False)
        self.norm = LayerNorm(dim*2, LayerNorm_type)

    def forward(self, left_down, right_up):  # left_down - right_up?
        res = left_down - right_up
        left_down = self.dcn_conv(left_down, res)
        x_in = self.conv1(torch.cat([left_down, right_up], dim=1))
        x = self.norm(x_in)

        x = self.conv_spatial(x) + x_in
        e, d = self.conv2(x).chunk(2, dim=1)
        # e, d = x
        output = e + d

        return output
class FuseV8(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(FuseV8, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)

        self.conv_spatial = DFFN(dim*2, ffn_expansion_factor, bias, dim_out=dim*2)
        self.fft_conv = fft_bench_complex_mlp(dim * 2, bias=True)
        self.norm = LayerNorm(dim*2, LayerNorm_type)
        self.fuse = SKFF(dim*2, height=2, reduction=4, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
    def forward(self, left_down, right_up):  # left_down - right_up?
        x_in = self.conv1(torch.cat([left_down, right_up], dim=1))
        x = self.norm(x_in)
        x = self.fuse(self.conv_spatial(x), self.fft_conv(x))
        x = x * self.gamma + x_in
        e, d = self.conv2(x).chunk(2, dim=1)
        # e, d = x
        output = e + d

        return output
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
class PromptEmbBlock(nn.Module):
    def __init__(self, dim, prompt_dim=128, prompt_len=5, prompt_size=[96, 96], num_heads=1, ffn_expansion_factor=2.66):
        super(PromptEmbBlock, self).__init__()
        bias=True
        LayerNorm_type = 'WithBias'
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size[0], prompt_size[1]))
        self.linear_layer = nn.Linear(dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, groups=prompt_dim, stride=1, padding=1, bias=False)
        self.noise_level = TransformerBlock(dim=dim+prompt_dim, bias=True)
        self.reduce_noise_level = nn.Conv2d(dim+prompt_dim,dim,kernel_size=1,bias=bias)
    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
                 self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        # prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        x = torch.cat([x, prompt], 1)
        x = self.noise_level(x)
        x = self.reduce_noise_level(x)

        return x
class PromptGenBlock(nn.Module):
    def __init__(self, dim, prompt_dim=128, prompt_len=5, prompt_size=[96, 96], num_heads=1, ffn_expansion_factor=2.66):
        super(PromptGenBlock, self).__init__()
        bias=True
        LayerNorm_type = 'WithBias'
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size[0], prompt_size[1]))
        self.linear_layer = nn.Linear(dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, groups=prompt_dim, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
                 self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=True))

    def forward(self, x):
        return self.body(x)
class UpsampleX(nn.Module):
    def __init__(self, n_feat, n_feat_o):
        super(UpsampleX, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat_o, 3, stride=1, padding=1, bias=True))

    def forward(self, x):
        return self.body(x)
class fftformerDecoder(nn.Module):
    def __init__(self,
                 dim=[32, 64, 128, 256],
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=True,
                 out_channels=3
                 ):
        super(fftformerDecoder, self).__init__()

        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[3]), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[3])])

        self.up4_3 = UpsampleX(int(dim[3]), int(dim[2]))
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[2]), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up3_2 = UpsampleX(int(dim[2]), int(dim[1]))
        # self.reduce_chan_level2 = nn.Conv2d(int(dim[2]), int(dim[1]), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[1]), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.up2_1 = UpsampleX(int(dim[1]), int(dim[0]))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[0]), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim[0]), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])
        self.fuse3 = Fuse(dim[2])
        self.fuse2 = Fuse(dim[1])
        self.fuse1 = Fuse(dim[0])
        self.output = nn.Conv2d(int(dim[0]), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, out_enc_level1, out_enc_level2, out_enc_level3, out_enc_level4):
        out_dec_level4 = self.decoder_level4(out_enc_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)

        inp_dec_level3 = self.fuse3(inp_dec_level3, out_enc_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1
class fftformerEncoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(fftformerEncoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])


    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)


        return [out_enc_level1, out_enc_level2, out_enc_level3]
class fftformerEncoderPro(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 1],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(fftformerEncoderPro, self).__init__()

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[0])])

        self.down4_5 = Downsample(int(dim * 2 ** 3))
        self.encoder_level5 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 4), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

    def forward(self, out_enc_level3):

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4)

        inp_enc_level5 = self.down4_5(out_enc_level4)
        out_enc_level5 = self.encoder_level5(inp_enc_level5)
        return [out_enc_level4, out_enc_level5]
##########################################################################
##---------- FFTformer -----------------------
class fftformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(fftformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
