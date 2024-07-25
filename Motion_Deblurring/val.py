
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

from basicsr.models.archs.AdaRevID_arch import AdaRevIDSlide as Net
from skimage import img_as_ubyte
from collections import OrderedDict
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim
# from skimage.metrics import structural_similarity as ssim_loss
import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='/data/mxt_data/GoPro/test/blur', type=str, help='Directory of validation images') # blur sharp_phase
parser.add_argument('--tar_dir', default='/data/mxt_data/GoPro/test/sharp', type=str, help='Directory of gt images') # sharp sharp_phase
# parser.add_argument('--input_dir', default='/home/ubuntu/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/tmp/405_cut/blur', type=str, help='Directory of validation images') # blur sharp_phase
# parser.add_argument('--tar_dir', default='/home/ubuntu/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/tmp/405_cut/sharp', type=str, help='Directory of gt images') # sharp sharp_phase
# parser.add_argument('--result_dir', default='./results/RealBlur_AdaRevID-S_2fcnaf_norev/RealBlur_J', type=str, help='Directory for results')
parser.add_argument('--result_dir', default='/home/ubuntu/90t/personal_data/mxt/MXT/exp_results/RevD-xxx/GoPro', type=str, help='Directory for results')
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
# yaml_file = 'Options/DeepRFT-width32-4gpu.yml'
# yaml_file = 'Options/AdaRevID-GoPro-rebuttal-test.yml'
# yaml_file = 'Options/NAFNet-width64.yml'
# yaml_file = 'Options/AdaRevIDB-Adaptive-test.yml'
yaml_file = 'Options/AdaRevID-B-GoPro-test.yml'
# yaml_file = 'Options/AdaRevID-RealBlur-R-test.yml'
# yaml_file = 'Options/AdaRevID-RealBlur-J-test.yml'
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
    # print(checkpoint)
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
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
with torch.no_grad():
    for file_ in tqdm(files): # [:1]
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        # h,w = input_.shape[2], input_.shape[3]
        # H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        # padh = H-h if h%factor!=0 else 0
        # padw = W-w if w%factor!=0 else 0
        # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        torch.cuda.synchronize()
        start = time.time()
        restored, num_decoders = model_restoration(input_)
        # restored = model_restoration(input_)
        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start

        if isinstance(restored, list):
            restored = restored[-1]
        elif isinstance(restored, dict):
            restored = restored['img']
            if isinstance(restored, list):
                restored = restored[-1]
        # Unpad images to original dimensions

        # restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_img = img_as_ubyte(restored)
        if args.get_psnr or args.get_ssim:
            gt_file_way = os.path.join(tar_dir, os.path.basename(file_))
            if os.path.exists(gt_file_way):
                rgb_gt = cv2.imread(gt_file_way)
            else:
                rgb_gt = cv2.imread(gt_file_way[:-3] + 'png')
            rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)

            if args.get_psnr:
                PSNR = psnr_loss(restored_img, rgb_gt)
                psnr_val_rgb.append(PSNR)
            if args.get_ssim:
                SSIM = calculate_ssim(restored_img, rgb_gt, crop_border=0)
                ssim_val_rgb.append(SSIM)
            # print('PSNR: ', sum(psnr_val_rgb) / len(psnr_val_rgb))
        if args.save_result:
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
print('average_time: ', all_time / len(files))
if args.get_psnr:
    psnr = sum(psnr_val_rgb) / len(files)
    print("PSNR: %f" % psnr)
if args.get_ssim:
    ssim = sum(ssim_val_rgb) / len(files)
    print("SSIM: %f" % ssim)
print('num_decoders: ', num_decoders)
