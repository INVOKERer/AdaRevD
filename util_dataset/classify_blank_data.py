import numpy as np
import glob
import cv2
import os
import kornia
import tqdm
import random
from sklearn.model_selection import train_test_split


img_root = '/home/ubuntu/90t/personal_data/mxt/Dataset/company_defocus/image_prepro/blank'
out_root = '/home/ubuntu/qlli-3090-1/mxt/BlankClassify_HCSaveImages_Pre'
name = 'pre_blank'
os.makedirs(out_root, exist_ok=True)

images_paths = glob.glob(os.path.join(img_root, '*', '*'))
random.shuffle(images_paths)
train_list, test_list = train_test_split(images_paths, test_size=0.1, random_state=42)

for k in tqdm.tqdm(train_list[:3600]):

    sharp_outdir = os.path.join(out_root, 'train', name)
    os.makedirs(sharp_outdir, exist_ok=True)
    file_name = os.path.basename(k)
    dir_name = k.split('/')[-2]
    try:
        x_sharp = cv2.imread(k)
        x_s = cv2.resize(x_sharp, [256, 256])

        out_way = os.path.join(sharp_outdir, dir_name+'_'+file_name)
        cv2.imwrite(out_way, x_s)
    except:
        print(k)

for k in tqdm.tqdm(test_list[:500]):

    sharp_outdir = os.path.join(out_root, 'test', name)
    os.makedirs(sharp_outdir, exist_ok=True)
    file_name = os.path.basename(k)
    dir_name = k.split('/')[-2]
    try:
        x_sharp = cv2.imread(k)
        x_s = cv2.resize(x_sharp, [256, 256])
        # print(dir_name)
        out_way = os.path.join(sharp_outdir, dir_name+'_'+file_name)
        cv2.imwrite(out_way, x_s)
    except:
        print(k)

# for img_file in tqdm.tqdm(img_list[:10]):
#     x = cv2.imread(img_file)
#     # x = x[:256, :256, :]
#     file_name = os.path.basename(img_file)
#     y = np.fft.rfft2(x, axes=[0,1])
#     # mask = np.complex(mask, np.zeros_like(mask))
#     y.real[np.where(y.real < 0)] = 0
#     y.imag[np.where(y.imag < 0)] = 0
#     # y = np.log(abs(y)+1)
#
#     y_ift = np.fft.irfft2(y, axes=[0, 1])
#     out_way = os.path.join(out_dir, file_name)
#     # cv2.imwrite(out_way, mask*255)
#     cv2.imwrite(out_way, y_ift)
