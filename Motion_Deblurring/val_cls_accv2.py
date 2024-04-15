
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
import time
from natsort import natsorted
from glob import glob

from basicsr.models.archs.AdaRevID_arch import AdaRevID as Net

from skimage import img_as_ubyte
from collections import OrderedDict
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim
# from skimage.metrics import structural_similarity as ssim_loss
import cv2
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='/home/ubuntu/Data1/mxt/GoPro/test/input_crops_384', type=str, help='Directory of validation images') # blur sharp_phase
parser.add_argument('--tar_dir', default='/home/ubuntu/Data1/mxt/GoPro/test/target_crops_384', type=str, help='Directory of gt images') # sharp sharp_phase
parser.add_argument('--result_dir', default='./results/RealBlur_AdaRevID-S_2fcnaf_norev/RealBlur_J', type=str, help='Directory for results')

parser.add_argument('--weights',
                    default='/home/ubuntu/90t/personal_data/mxt/MXT/ckpts/RevD-B/net_g_GoPro.pth',
                    type=str, help='Path to weights')
parser.add_argument('--multi_focus', default=False, type=bool, help='save')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset')
# ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--save_result', default=False, type=bool, help='save')
parser.add_argument('--get_psnr', default=True, type=bool, help='psnr')
parser.add_argument('--get_ssim', default=True, type=bool, help='ssim')
parser.add_argument('--gpus', default='0', type=str, help='gpu')
args = parser.parse_args()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
####### Load yaml #######

# psnr_classes = [20., 25., 30., 35., 40.]
step_ = 'avg6'
psnr_classes = [22.0162, 24.0619, 26.1443, 28.5015, 31.7516]
# psnr_classes = [15., 25., 35.] # [21., 27., 33., 39.] # [19., 23., 27., 31., 35., 39.] # [18., 21., 24., 27., 30., 33., 36., 39.]
# psnr_classes = [18., 21., 24., 27., 30., 33., 36., 39.] # [15., 25., 35.] # [21., 27., 33., 39.] # [19., 23., 27., 31., 35., 39.] # [18., 21., 24., 27., 30., 33., 36., 39.]
numclass = len(psnr_classes)+1
cls_len = [0]*numclass
yaml_file = 'Options/AdaRevIDB-Adaptive-test.yml'
confusin_matrix = np.zeros((numclass+1, numclass+1))
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

print(args.weights)
if args.weights is not None:
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################
    print(x['network_g'])
    model_restoration = Net(**x['network_g'])
    checkpoint = torch.load(args.weights)
    # model_restoration.load_state_dict(checkpoint['params'])
    try:
        model_restoration.load_state_dict(checkpoint['params'], strict=False)
        state_dict = checkpoint["params"]
        # for k, v in state_dict.items():
        #     print(k)
    except:
        try:
            model_restoration.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # print(k)
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_restoration.load_state_dict(new_state_dict)
else:
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    try:
        model_restoration = Net(**x['network_g'])
    except:
        print('default model')
        model_restoration = Net()
    print('inference for time')
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

psnr_val_rgb = []
ssim_val_rgb = []
ssim = 0
psnr = 0
factor = 8
all_time = 0.
dataset = args.dataset
if args.save_result:
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

# inp_dir = os.path.join(args.input_dir, dataset, 'test', 'blur')
# tar_dir = os.path.join(args.tar_dir, dataset, 'test', 'sharp')
inp_dir = args.input_dir
tar_dir = args.tar_dir
# print(inp_dir)
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
total_right = 0
with torch.no_grad():
    for file_ in tqdm(files): # [:1]
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # blur = utils.load_img(file_)
        img_blur = np.float32(utils.load_img(file_))/255.
        img = torch.from_numpy(img_blur).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        # h,w = input_.shape[2], input_.shape[3]
        # H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        # padh = H-h if h%factor!=0 else 0
        # padw = W-w if w%factor!=0 else 0
        # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        torch.cuda.synchronize()
        start = time.time()
        cls_id = model_restoration(input_)
        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start

        # Unpad images to original dimensions

        # restored = restored[:,:,:h,:w]

        # if args.get_psnr or args.get_ssim:
        gt_file_way = os.path.join(tar_dir, os.path.basename(file_))
        if os.path.exists(gt_file_way):
            rgb_gt = cv2.imread(gt_file_way)
        else:
            rgb_gt = cv2.imread(gt_file_way[:-3] + 'png')
        rgb_gt = np.float32(cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)) / 255.
        psnr = 10.0 * np.log10(1. ** 2 / np.mean((rgb_gt - img_blur) ** 2) + 1e-8)
        psnr_cls = 0
        for p in range(len(psnr_classes)):
            if psnr > psnr_classes[p]:  # >= or >
                psnr_cls = p + 1
        gt_cls = psnr_cls

        cls_id = cls_id.cpu().detach().numpy()[0]
        # print(gt_cls, cls_id, cls_id == gt_cls)
        if cls_id == gt_cls:
            total_right += 1
        confusin_matrix[gt_cls, -1] += 1
        confusin_matrix[-1, cls_id] += 1
        confusin_matrix[gt_cls, cls_id] += 1
print('average_time: ', all_time / len(files))

acc = total_right / len(files)
print("acc: %f" % acc)

print(confusin_matrix)
result_dir = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/rebuttal'
pd.DataFrame(confusin_matrix).to_excel(os.path.join(result_dir, step_+'_confuse_matrix_val.xls'), index=True, header=False)

