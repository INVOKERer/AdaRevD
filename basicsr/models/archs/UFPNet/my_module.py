
import numpy as np
import torch.nn as nn
import torch

# input channel: 3


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class kernel_extra_Encoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Encoding_Block, self).__init__()
        self.Conv_head = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.Conv_head(input)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        skip = self.act(output)
        output = self.downsample(skip)

        return output, skip


class kernel_extra_conv_mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_mid, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                        )

    def forward(self, input):
        output = self.body(input)
        return output


class kernel_extra_Decoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Decoding_Block, self).__init__()
        self.Conv_t = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.Conv_head = nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.act = nn.ReLU()

    def forward(self, input, skip):
        output = self.Conv_t(input, output_size=[skip.shape[0], skip.shape[1], skip.shape[2], skip.shape[3]])
        output = torch.cat([output, skip], dim=1)
        output = self.Conv_head(output)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        output = self.act(output)

        return output


class kernel_extra_conv_tail(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_tail, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel_mean = self.mean(input)
        # kernel_feature = kernel_mean

        kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = kernel_mean.reshape(kernel_mean.shape[0], self.kernel_size, self.kernel_size, kernel_mean.shape[2] * kernel_mean.shape[3]).permute(0, 3, 1, 2)
        # kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)

        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mean


class kernel_extra_conv_tail_mean_var(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_tail_mean_var, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                        )
        self.var = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel_mean = self.mean(input)
        kernel_var = self.var(input)
        # kernel_feature = kernel_mean

        kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = kernel_mean.reshape(kernel_mean.shape[0], self.kernel_size, self.kernel_size, kernel_mean.shape[2] * kernel_mean.shape[3]).permute(0, 3, 1, 2)
        # kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)

        kernel_var = kernel_var.reshape(kernel_var.shape[0], self.kernel_size, self.kernel_size, kernel_var.shape[2] * kernel_var.shape[3]).permute(0, 3, 1, 2)
        kernel_var = kernel_var.mean(dim=[2, 3], keepdim=True)
        kernel_var = kernel_var.repeat(1, 1, self.kernel_size, self.kernel_size)

        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mean, kernel_var


class kernel_extra(nn.Module):
    def __init__(self, kernel_size):
        super(kernel_extra, self).__init__()
        self.kernel_size = kernel_size
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 128)
        self.Conv_mid = kernel_extra_conv_mid(128, 256)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(256, 128)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(128, 64)
        self.Conv_tail = kernel_extra_conv_tail(64, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        kernel = self.Conv_tail(output)

        return kernel


class code_extra_mean_var(nn.Module):
    def __init__(self, kernel_size):
        super(code_extra_mean_var, self).__init__()
        self.kernel_size = kernel_size
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 128)
        self.Conv_mid = kernel_extra_conv_mid(128, 256)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(256, 128)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(128, 64)
        self.Conv_tail = kernel_extra_conv_tail_mean_var(64, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        code, var = self.Conv_tail(output)

        return code, var


if __name__ == '__main__':
    input1 = torch.rand(2, 3, 128, 128).cuda()
    net = code_extra_mean_var(19).cuda()
    code, var = net(input1)
    # print(net)
    print(var.size())
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))








