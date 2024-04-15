import torch
import numpy as np
import cv2 as cv
from torch import nn
from tqdm import tqdm
import kornia
import utils
from collections import OrderedDict
# from torchvision.models import resnet18
from basicsr.models.archs.dctformer_iccv_arch import DCTformer_iccv as Net
from natsort import natsorted
from glob import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

def get_effective_receptive_field(input, output, log=True, grad=512):
    """
    draw effective receptive field
    :param input: image (1, C, H, W)
    :param output: feature e.g. (1, C', H', W')
    :return: numpy array (0~255)
    """

    init_grad = torch.zeros_like(output)

    # according to the feature shape
    init_grad[:, :, init_grad.shape[2] // 2, init_grad.shape[3] // 2] += grad

    grad = torch.autograd.grad(output, input, init_grad)

    rf = torch.transpose(torch.squeeze(grad[0]), 0, 2).numpy()
    # rf[rf < 0] = 0
    rf = np.abs(rf)
    if log:
        rf = np.log(rf+1)
    rf /= np.ptp(rf)

    return rf


if __name__ == '__main__':

    # resnet = resnet18(pretrained=True)
    heads = [1, 2, 4, 8]
    window_size_dct_enc = [256, 128, 64]
    window_size_dct_mid = 32

    model_type = 'DCTformer-S'  # DCTformer-B Restormer
    attn_type = 'nodct_global_channel' # 'nodct_global_channel'
    if attn_type == 'channel_mlp':
        weight_name = 'ICCV_Abalation_DCTformer_DCT_LN_local_channel_mlp_4gpu_freqloss'
    elif attn_type == 'channel':
        weight_name = 'ICCV_Abalation_DCTformer_DCT_LN_local_channel_4gpu_freqloss'
    elif attn_type == 'nodct_global_channel':
        weight_name = 'ICCV_Abalation_DCTformer_nodct_global_channel_4gpu_freqloss'
    if model_type == 'DCTformer-S':
        # DCTformer-S  36.26 38.24
        dim = 32
        enc_blk_nums = [1, 2, 3]
        middle_blk_num = 7
        dec_blk_nums = [3, 2, 2]
    elif model_type == 'DCTformer-B':
        # DCTformer-B
        dim = 36
        enc_blk_nums = [1, 2, 6]
        middle_blk_num = 9
        dec_blk_nums = [6, 2, 2]
    else:
        # Restormer
        dim = 48
        enc_blk_nums = [2, 3, 3]
        middle_blk_num = 4
        dec_blk_nums = [3, 3, 4]
    cs_e = [attn_type, attn_type]
    cs_m = [attn_type, attn_type]
    cs_d = [attn_type, attn_type]
    test_layer = Net(dim=dim, enc_blk_nums=enc_blk_nums, middle_blk_num=middle_blk_num, dec_blk_nums=dec_blk_nums,
                         cs_e=cs_e, cs_m=cs_m, cs_d=cs_d,
                         window_size_dct_enc=window_size_dct_enc, window_size_dct_mid=window_size_dct_mid,
                         bias=True).cuda()
    # test_layer = Net().cuda() # nn.Sequential(*list(resnet.children())[:-3])
    weights_root = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/experiments'
    pth_w = 'models/net_g_80000.pth'
    weights = os.path.join(weights_root, weight_name, pth_w)
    img_end = 100
    crop_size = 64
    checkpoint = torch.load(weights)
    inp_dir = '../../../../Datasets/Deblur/GoPro/test/blur'
    out_dir = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/ICCV2023/figs/ERF64'
    os.makedirs(out_dir, exist_ok=True)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    # model_restoration.load_state_dict(checkpoint['params'])
    try:
        test_layer.load_state_dict(checkpoint['params'])
        state_dict = checkpoint["params"]
        # for k, v in state_dict.items():
        #     print(k)
    except:
        try:
            test_layer.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # print(k)
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            test_layer.load_state_dict(new_state_dict)
    print("===>Testing using weights: ", weights)
    erf = np.zeros((crop_size, crop_size))
    # test_layer.train()
    # with torch.no_grad():
    for file_ in tqdm(files[:img_end]):  # [:1]
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_)) / 255.

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()
        # print(input_.shape, input_.dtype)
        input_tensor = kornia.geometry.center_crop(input_, [crop_size, crop_size])
        img_gray = cv.cvtColor(kornia.tensor_to_image(input_tensor.cpu()), cv.COLOR_BGR2GRAY)
        # print(input_tensor.shape)
        input_tensor.requires_grad = True
        # inputx = input.clone().detach().requires_grad_(True) # torch.tensor(input, requires_grad=True)
        feature = test_layer(input_tensor)
        feature = feature - input_tensor
        # 特征图的channel维度算均值且去掉batch维度，得到二维张量
        feature_map = feature.mean(dim=[0,1], keepdim=False).squeeze()
        # 对二维张量中心点（标量）进行backward
        # init_grad = feature_map[(feature_map.shape[0] // 2 - 1)][(feature_map.shape[1] // 2 - 1)]
        # init_grad = torch.zeros_like(feature)
        #
        # # accrding to the feature shape
        # init_grad[:, :, init_grad.shape[2] // 2, init_grad.shape[3] // 2] += 512
        # init_grad.backward(retain_graph=True)
        feature_map[(feature_map.shape[0] // 2 - 1)][(feature_map.shape[1] // 2 - 1)].backward(retain_graph=True)
        # 对输入层的梯度求绝对值
        grad = torch.abs(input_tensor.grad)
        # 梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
        grad = grad.mean(dim=1, keepdim=False).squeeze()

        # 累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
        rf = grad.cpu().numpy()
        # rf[(rf.shape[0] // 2 - 1), (rf.shape[1] // 2 - 1)] = np.log(rf[(rf.shape[0] // 2 - 1), (rf.shape[1] // 2 - 1)] + 1)
        # rf = np.log(rf + 1)
        rf = np.clip(rf, 0., 0.25)
        rf /= np.ptp(rf)
        # rf = (img_gray + rf) / 2.
        # print(rf.max())
        # rf = get_effective_receptive_field(input_tensor.cpu(), feature.cpu(), log=True, grad=512)
        erf += rf # np.mean(rf, axis=-1)
        # print(erf.max())
    erf /= img_end
    print(erf.max())
    # erf[(erf.shape[0] // 2 - 1), (erf.shape[1] // 2 - 1)] -= 0.7
    # cv.imshow('receptive field', erf)
    # cv.waitKey(0)
    # cv.imwrite(os.path.join(out_dir, 'test.png'), erf * 255)
    sns_plot = plt.figure()
    sns.heatmap(erf, cmap='Blues_r', linewidths=0.0, vmin=0, vmax=0.25,
                xticklabels=False, yticklabels=False, cbar=True) # RdBu_r Reds_r .invert_yaxis()
    # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
    out_way = os.path.join(out_dir, attn_type + '_heatmap_br_nolog' + '.png')
    sns_plot.savefig(out_way, dpi=700)