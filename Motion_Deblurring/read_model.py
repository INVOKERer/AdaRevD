
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

from basicsr.models.archs.msdi2_arch import prior_upsampling

from skimage import img_as_ubyte
from collections import OrderedDict
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim

import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--weights',
                    default='../experiments/GoPro-MSDINet2e/models/net_g_100000.pth',
                    type=str, help='Path to weights')
args = parser.parse_args()


checkpoint = torch.load(args.weights)
state_dict = checkpoint["params"]
# for k, v in state_dict.items():
#     print(k)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # print(k)
    if 'prior_upsampling' in k:
        name = k.replace('prior_upsampling.', '')  # remove `module.`
        print(k)
        new_state_dict[name] = v
prior_upsampling = prior_upsampling()
prior_upsampling.load_state_dict(new_state_dict, strict=True)
# print(checkpoint)

