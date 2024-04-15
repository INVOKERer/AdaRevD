import os
import argparse
import torch.nn as nn
import torch
# from torch.utils.data import DataLoader
# from basicsr.models.archs.WNAFNet_arch import WNAFNet as Net
# from basicsr.models.archs.restormer_arch import RestormerLocal as Net
# from basicsr.models.archs.stripformer_arch import Stripformer as Net
# from DeepRFT_MIMO import DeepRFT as mynet
from basicsr.models.archs.maxim_arch import MAXIM as Net
from tqdm import tqdm

import time


parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# yaml_file = 'Options/Deblurring_DCTformer_test.yml'
yaml_file = 'Options/Deblurring_Restormer.yml'
# model_restoration = mynet(inference=True)
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
# model_restoration = Net(**x['network_g'])
model_restoration = Net()
# print number of model
# utils.load_checkpoint_compress_doconv(model_restoration, args.weights)
# print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# test_dataset = get_test_data(rgb_dir_test, img_options={})
num_imgs = 1111
img_size_h = 720
img_size_w = 1280
all_time = 0.
with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for ii in tqdm(range(num_imgs)):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = torch.randn((1, 3, img_size_h, img_size_w)).cuda()
        # print()
        # filenames = data_test[1]
        # _, _, Hx, Wx = input_.shape
        # filenames = data_test[1]
        # print(input_.shape)
        torch.cuda.synchronize()
        start = time.time()
        # input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_)
        # restored = torch.clamp(restored, 0, 1)
        # print(restored.shape)
        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start
print('average_time: ', all_time / num_imgs)
