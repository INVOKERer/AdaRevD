
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import kornia
import torch
matplotlib.use('PS')


# spatially variant blur
class BatchBlur_SV(nn.Module):
    def __init__(self, l=19, padmode='reflection'):
        super(BatchBlur_SV, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))


    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            pad = pad.view(C * B, 1, H_p, W_p)
            pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]
            kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
            out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
            out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)

            return out
class BatchBlur_SV_test(nn.Module):
    def __init__(self, l=19, padmode='reflection'):
        super(BatchBlur_SV_test, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))


    def forward(self, input):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        pad = pad.view(C * B, 1, H_p, W_p)
        pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]

        # print(pad.shape, kernel.shape)
        out_unf = pad.sum(-1).unsqueeze(1)
        # out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
        out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)
        # print(out.shape)
        return out
class BatchBlur_SV2(nn.Module):
    def __init__(self, l=19, padmode='reflection', divisor_=True):
        super(BatchBlur_SV2, self).__init__()
        self.l = l
        pad_1 = l // 2
        pad_2 = l // 2 if l % 2 == 1 else l // 2 - 1
        self.pad_1 = pad_1
        self.pad_2 = pad_2
        if padmode == 'reflection':
            self.pad = nn.ReflectionPad2d((pad_1, pad_2, pad_1, pad_2))
        elif padmode == 'zero':
            self.pad = nn.ZeroPad2d((pad_1, pad_2, pad_1, pad_2))
        elif padmode == 'replication':
            self.pad = nn.ReplicationPad2d((pad_1, pad_2, pad_1, pad_2))
        # pad_size = l // 2
        # if self.mode != 'reflect':
            # self.pad_size = pad_size
        self.fold_params = dict(kernel_size=1, dilation=1, padding=0, stride=1)
        self.unfold_params = dict(kernel_size=l, dilation=1, padding=0, stride=1)
        output_size = [1, 1, 128, 128]
        self.output_size = output_size
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.unfold_params)
        self.divisor_ = divisor_
        if divisor_:
            input_size = [1, 1, 128+pad_1+pad_2, 128+pad_1+pad_2]
            self.input_ones = torch.ones(input_size)
            # print(self.unfold(self.input_ones).shape)
            self.divisor = self.fold(self.unfold(self.input_ones))
        # print('divisor: ', self.divisor.sum(1).shape)
    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        if self.divisor_:
            if pad.shape[-2:] != self.divisor.shape[-2:]:
                h, w = pad.shape[-2:]
                self.input_ones = torch.ones([1, 1, h, w])
                self.input_ones = self.input_ones.to(pad.device)
                self.fold = nn.Fold(output_size=[H, W], **self.fold_params)
                # print(self.unfold((self.input_ones).shape))
                self.divisor = self.fold(self.unfold(self.input_ones)).sum(1)
            print('divisor: ', self.divisor.mean())
            if self.divisor.device != pad.device:
                self.divisor = self.divisor.to(pad.device)
        elif self.output_size[-2:] != input.shape[-2:]:
            self.output_size = [H, W]
            self.fold = nn.Fold(output_size=[H, W], **self.fold_params)
        H_p, W_p = pad.size()[-2:]
        pad = pad.view(B*C, 1, H_p, W_p)
        pad = self.unfold(pad) # B*C K^2 HW
        K2, HW = pad.shape[-2:]
        # print('pad1: ', pad.shape)
        pad = pad.view(B, C, K2, HW)
        # print('pad2: ', pad.shape)
        # print('kernel: ', kernel.shape)
        kernel = kernel.view(B, K2, H * W).unsqueeze(1)

        pad = pad * kernel # .transpose(-1, -2)
        out_unf = pad.sum(-2).view(B*C, HW).unsqueeze(1)
        if self.divisor_:
            out = self.fold(out_unf) / self.divisor.contiguous()
            # print(torch.sum(out - self.fold(out_unf / 361.)))
        else:
            out = self.fold(out_unf)
        # out = out[:, :, self.pad_1:-self.pad_2, self.pad_1:-self.pad_2].contiguous()
        # print(out.shape)
        return out.view(B, C, H, W)
        # kernel = kernel.flatten(2).unsqueeze(0)
        # if len(kernel.size()) == 2:
        #     input_CBHW = pad.view((C * B, 1, H_p, W_p))
        #     kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
        #     return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        # else:
        #     pad = pad.view(C * B, 1, H_p, W_p)
        #     pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]
        #     kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
        #     kernel = kernel.squeeze(1).transpose(-1, -2)
        #     # print(pad.shape, kernel.shape)
        #     out_unf = (pad * kernel.contiguous()).sum(-1).unsqueeze(1)
        #     # out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
        #     out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)
        #     # print(out.shape)
        #     return out
class BatchBlur_SV3(nn.Module):
    def __init__(self, l=19, padmode='reflection', divisor_=True):
        super(BatchBlur_SV3, self).__init__()
        self.l = l
        pad_1 = l // 2
        pad_2 = l // 2 if l % 2 == 1 else l // 2 - 1
        self.pad_1 = pad_1
        self.pad_2 = pad_2
        if padmode == 'reflection':
            self.pad = nn.ReflectionPad2d((pad_1, pad_2, pad_1, pad_2))
        elif padmode == 'zero':
            self.pad = nn.ZeroPad2d((pad_1, pad_2, pad_1, pad_2))
        elif padmode == 'replication':
            self.pad = nn.ReplicationPad2d((pad_1, pad_2, pad_1, pad_2))
        # pad_size = l // 2
        # if self.mode != 'reflect':
            # self.pad_size = pad_size
        self.fold_params = dict(kernel_size=1, dilation=1, padding=0, stride=1)
        self.unfold_params = dict(kernel_size=l, dilation=1, padding=0, stride=1)
        output_size = [1, 1, 128, 128]
        self.output_size = output_size
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.unfold_params)
        self.divisor_ = divisor_
        if divisor_:
            self.divisor = float(l**2)
        # print('divisor: ', self.divisor.sum(1).shape)
    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        if self.output_size[-2:] != input.shape[-2:]:
            self.fold = nn.Fold(output_size=[H, W], **self.fold_params)
        H_p, W_p = pad.size()[-2:]
        pad = pad.view(B*C, 1, H_p, W_p)
        pad = self.unfold(pad) # B*C K^2 HW
        K2, HW = pad.shape[-2:]
        # print('pad1: ', pad.shape)
        pad = pad.view(B, C, K2, HW)
        # print('pad2: ', pad.shape)
        # print('kernel: ', kernel.shape)
        # kernel = kernel.view(B, K2, H * W).unsqueeze(1)
        kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
        kernel = kernel.permute(1, 0, 3, 2)
        pad = pad * kernel # .transpose(-1, -2)
        out_unf = pad.sum(-2).view(B*C, HW).unsqueeze(1)
        if self.divisor_:
            out = self.fold(out_unf) / self.divisor
            # print(torch.sum(out - self.fold(out_unf / 361.)))
        else:
            out = self.fold(out_unf)
        # out = out[:, :, self.pad_1:-self.pad_2, self.pad_1:-self.pad_2].contiguous()
        # print(out.shape)
        return out.view(B, C, H, W)
        # kernel = kernel.flatten(2).unsqueeze(0)
        # if len(kernel.size()) == 2:
        #     input_CBHW = pad.view((C * B, 1, H_p, W_p))
        #     kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
        #     return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        # else:
        #     pad = pad.view(C * B, 1, H_p, W_p)
        #     pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]
        #     kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
        #     kernel = kernel.squeeze(1).transpose(-1, -2)
        #     # print(pad.shape, kernel.shape)
        #     out_unf = (pad * kernel.contiguous()).sum(-1).unsqueeze(1)
        #     # out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
        #     out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)
        #     # print(out.shape)
        #     return out

class BatchBlur_nopad(nn.Module):
    def __init__(self, l=19, divisor_=False):
        super(BatchBlur_nopad, self).__init__()
        self.l = l
        pad_1 = l // 2
        pad_2 = l // 2 if l % 2 == 1 else l // 2 - 1
        self.pad_1 = pad_1
        self.pad_2 = pad_2
        # print(pad_1, pad_2)
        # if padmode == 'reflection':
        #     self.pad = nn.ReflectionPad2d((pad_1, pad_2, pad_1, pad_2))
        # elif padmode == 'zero':
        #     self.pad = nn.ZeroPad2d((pad_1, pad_2, pad_1, pad_2))
        # elif padmode == 'replication':
        #     self.pad = nn.ReplicationPad2d((pad_1, pad_2, pad_1, pad_2))
        # pad_size = l // 2
        # if self.mode != 'reflect':
            # self.pad_size = pad_size
        self.fold_params = dict(kernel_size=1, dilation=1, padding=0, stride=1)
        self.unfold_params = dict(kernel_size=l, dilation=1, padding=0, stride=1)
        output_size = [1, 1, 128, 128]
        self.output_size = output_size
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.unfold_params)
        self.divisor_ = divisor_
        if divisor_:
            self.divisor = float(l**2)
        # print('divisor: ', self.divisor.sum(1).shape)
    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        # pad = self.pad(input)
        H_o, W_o = H-self.pad_1-self.pad_2, W-self.pad_1-self.pad_2
        pad = input
        if self.output_size[-2:] != input.shape[-2:]:
            self.fold = nn.Fold(output_size=[H_o, W_o], **self.fold_params)
        H_p, W_p = pad.size()[-2:]
        pad = pad.view(B*C, 1, H_p, W_p)
        pad = self.unfold(pad) # B*C K^2 HW
        K2, HW = pad.shape[-2:]
        # print('pad1: ', pad.shape)
        pad = pad.view(B, C, K2, HW)
        kernelx = kernel[:, :, self.pad_1:-self.pad_2, self.pad_1:-self.pad_2].contiguous()
        # print('pad2: ', pad.shape)
        # print('kernel: ', kernelx.shape)
        kernelx = kernelx.view(B, K2, HW).unsqueeze(1)
        # kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
        # kernel = kernel.permute(1, 0, 3, 2)
        pad = pad * kernelx # .transpose(-1, -2)
        out_unf = pad.sum(-2).view(B*C, HW).unsqueeze(1)
        if self.divisor_:
            out = self.fold(out_unf) / self.divisor
            # print(torch.sum(out - self.fold(out_unf / 361.)))
        else:
            out = self.fold(out_unf)
        # out = out[:, :, self.pad_1:-self.pad_2, self.pad_1:-self.pad_2].contiguous()
        # print(out.shape)
        return out.view(B, C, H_o, W_o)
        # kernel = kernel.flatten(2).unsqueeze(0)
        # if len(kernel.size()) == 2:
        #     input_CBHW = pad.view((C * B, 1, H_p, W_p))
        #     kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
        #     return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        # else:
        #     pad = pad.view(C * B, 1, H_p, W_p)
        #     pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]
        #     kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
        #     kernel = kernel.squeeze(1).transpose(-1, -2)
        #     # print(pad.shape, kernel.shape)
        #     out_unf = (pad * kernel.contiguous()).sum(-1).unsqueeze(1)
        #     # out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)
        #     out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)
        #     # print(out.shape)
        #     return out
if __name__=='__main__':
    import torch
    import kornia
    import cv2
    import os

    x = torch.ones(1, 3, 128, 128).cuda()
    y = x # [:, :, 9:-9, 9:-9]
    k = torch.randn(1, 19*19, 128, 128).cuda()
    k = torch.softmax(k, dim=1)
    # model1 = BatchBlur_SV2_test(divisor_=True).cuda()stop
    model1 = BatchBlur_SV2(divisor_=False).cuda()
    model2 = BatchBlur_SV2(divisor_=True).cuda()
    # print(torch.mean(torch.abs(model1(x, k / 19. ** 2) - model2(x, k))))
    # print(torch.mean(torch.abs(model1(x, k) - model2(x, k))))
    print(torch.mean(torch.abs(y - model1(x, k))))
