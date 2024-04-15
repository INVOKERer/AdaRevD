import numpy as np
import tqdm

# import pickle
# import gzip
import cca_core

from CKA import linear_CKA, linear_CKA_torch, kernel_CKA
import glob
import os
import cv2
from natsort import natsorted
import torch
import kornia

compare_cka = 'sharp-blur' # 'cls' 'sharp-blur'
enc_dec = 'dec' # dec enc
torch_or_npy = 'torch'
mean = True
print(compare_cka, enc_dec, torch_or_npy)

out_dir = os.path.join('')
os.makedirs(out_dir, exist_ok=True)

enc_feature_root = os.path.join('')
feature_root = os.path.join('')

image_blur_root = os.path.join('')
image_sharp_root = os.path.join('')
if enc_dec == 'dec':
    decoders = ['sub_decoder_0', 'sub_decoder_1', 'sub_decoder_2', 'sub_decoder_3', 'sub_decoder_4', 'sub_decoder_5', 'sub_decoder_6', 'sub_decoder_7']
    features = ['e0', 'e1', 'e2', 'e3']
else:
    decoders = ['encoder']
    features = ['e0', 'e1', 'e2', 'e3', 'e4']

image_blur_list = natsorted(glob.glob(os.path.join(image_blur_root, '*.png')) + glob.glob(os.path.join(image_blur_root, '*.jpg')))
image_sharp_list = natsorted(glob.glob(os.path.join(image_sharp_root, '*.png')) + glob.glob(os.path.join(image_sharp_root, '*.jpg')))

# m = len(features)
# n = len(decoders)
matrix = np.zeros([len(features), len(decoders)])
crop_size = 256
end = 10
# e4_list = natsorted(glob.glob(os.path.join(enc_feature_root, 'encoder_e4_*.npy')))
cls_list = natsorted(glob.glob(os.path.join(enc_feature_root, 'encoder_classifyfeature_*.npy')))
features_list = natsorted(glob.glob(os.path.join(feature_root, decoders[0]+'_'+features[0]+'_*.npy')))
# print(features_list)
for n in tqdm.tqdm(range(len(features_list[:]))):
    if n > 1:
        if compare_cka == 'cls':
            # data_e4 = np.load(e4_list[n])
            data_cls = np.load(cls_list[n])
            if mean:
                input_ = np.mean(data_cls, axis=-1)
            else:
                input_ = data_cls
        else:
            image_blur = cv2.imread(image_blur_list[n], cv2.COLOR_BGR2RGB)
            image_sharp = cv2.imread(image_sharp_list[n], cv2.COLOR_BGR2RGB)
            img = np.float32(image_sharp) / 255. - np.float32(image_blur) / 255. # np.float32(image_blur) / 255. #
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0)
            input_ = kornia.geometry.center_crop(input_, [crop_size, crop_size])
            input_ = kornia.tensor_to_image(input_, keepdim=True)
            if mean:
                input_ = np.mean(input_, axis=-1)

        input = np.concatenate([input, input_], axis=0)
    else:
        if compare_cka == 'cls':
            # data_e4 = np.load(e4_list[n])
            data_cls = np.load(cls_list[n])
            if mean:
                data_cls = np.mean(data_cls, axis=-1)
            input = data_cls
        else:
            image_blur = cv2.imread(image_blur_list[n], cv2.COLOR_BGR2RGB)
            image_sharp = cv2.imread(image_sharp_list[n], cv2.COLOR_BGR2RGB)
            img = np.float32(image_sharp) / 255. - np.float32(image_blur) / 255. # np.float32(image_sharp) / 255. # - np.float32(image_blur) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input = img.unsqueeze(0)
            input = kornia.geometry.center_crop(input, [crop_size, crop_size])
            input = kornia.tensor_to_image(input, keepdim=True)
            if mean:
                input = np.mean(input, axis=-1)
acts2 = input
if mean:
    num_datapoints, h, w = acts2.shape
    f_acts2 = acts2.reshape((num_datapoints, h * w))
else:
    num_datapoints, h, w, channels = acts2.shape
    f_acts2 = acts2.reshape((num_datapoints, channels * h * w))
f_acts2 = torch.tensor(f_acts2)
f_acts2 = f_acts2.cuda()
for i in tqdm.tqdm(range(len(features))):
    for j in range(len(decoders)):
        # print(decoders[j], features[i])
        features_list = natsorted(glob.glob(os.path.join(feature_root, decoders[j]+'_'+features[i]+'_*.npy')))
        # print(len(features_list))

        for n in tqdm.tqdm(range(len(features_list[:]))):
            if n > 1:
                datax = np.load(features_list[n])
                if mean:
                    datax = np.mean(datax, axis=-1)
                data = np.concatenate([data, datax], axis=0)
                # image_blur = cv2.imread(image_blur_list[n], cv2.COLOR_BGR2RGB)
                # image_sharp = cv2.imread(image_sharp_list[n], cv2.COLOR_BGR2RGB)
                # img = np.float32(image_sharp) / 255. - np.float32(image_blur) / 255.
                # img = torch.from_numpy(img).permute(2, 0, 1)
                # input_ = img.unsqueeze(0)
                # input_ = kornia.geometry.center_crop(input_, [crop_size, crop_size])
                # input_ = kornia.tensor_to_image(input_, keepdim=True)
                # input = np.concatenate([input, input_], axis=0)
            else:
                data = np.load(features_list[n])
                if mean:
                    data = np.mean(data, axis=-1)
                # image_blur = cv2.imread(image_blur_list[n], cv2.COLOR_BGR2RGB)
                # image_sharp = cv2.imread(image_sharp_list[n], cv2.COLOR_BGR2RGB)
                # img = np.float32(image_sharp) / 255. - np.float32(image_blur) / 255.
                # img = torch.from_numpy(img).permute(2, 0, 1)
                # input = img.unsqueeze(0)
                # input = kornia.geometry.center_crop(input, [crop_size, crop_size])
                # input = kornia.tensor_to_image(input, keepdim=True)

        acts1 = data

        print(acts1.shape, acts2.shape)
        if mean:
            num_datapoints, h, w = acts1.shape
            # acts1 = np.resize(acts1, [num_datapoints, ])
            f_acts1 = acts1.reshape((num_datapoints, h * w))
        else:
            num_datapoints, h, w, channels = acts1.shape
            # acts1 = np.resize(acts1, [num_datapoints, ])
            f_acts1 = acts1.reshape((num_datapoints, channels * h * w))


        # print(acts1.shape)
        # avg_acts1 = np.mean(acts1, axis=(1, 2))
        # avg_acts2 = np.mean(acts2, axis=(1, 2))
        # print(avg_acts1.shape, avg_acts2.shape)
        if torch_or_npy == 'torch':
            f_acts1 = torch.tensor(f_acts1)

            K_CKA = linear_CKA_torch(f_acts1.cuda(), f_acts2)
            K_CKA = K_CKA.cpu().detach().numpy()
        else:
            K_CKA = linear_CKA(f_acts1, f_acts2)
        # L_CKA = linear_CKA(f_acts1, f_acts2)
        matrix[i, j] = K_CKA
        print(i, j, decoders[j], features[i], K_CKA)
print(matrix)
npy_out_way = os.path.join(out_dir, compare_cka+'_'+enc_dec+'_linear_CKA_'+torch_or_npy+'.npy')
np.save(npy_out_way, matrix)
# matrix = np.load(npy_out_way)
# img_out_way = os.path.join(out_dir, 'linear_CKA_img.png')
# cv2.imwrite(img_out_way, matrix * 255)
# matrix_flip = cv2.flip(matrix, 0)
# img_flip_out_way = os.path.join(out_dir, 'linear_CKA_img_flip.png')
# cv2.imwrite(img_flip_out_way, matrix_flip * 255)

# CKA
# print('Linear CKA, between X and Y: {}'.format(linear_CKA(avg_acts1, avg_acts2)))




# print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(avg_acts1, avg_acts2)))
# print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(avg_acts1, avg_acts1)))
#
# num_datapoints, h, w, channels = acts1.shape
# f_acts1 = acts1.reshape((num_datapoints*h*w, channels))
#
# num_datapoints, h, w, channels = acts2.shape
# f_acts2 = acts2.reshape((num_datapoints*h*w, channels))
# print(f_acts1.shape)
# # CCA
# f_results = cca_core.get_cca_similarity(f_acts1.T, f_acts2.T, epsilon=1e-10, verbose=False)
# print("Mean CCA similarity", np.mean(f_results["cca_coef1"]))

# f_results_avg = cca_core.get_cca_similarity(avg_acts1.T, avg_acts2.T, epsilon=1e-10, verbose=False)
# print("Avg Mean CCA similarity", np.mean(f_results_avg["cca_coef1"]))
