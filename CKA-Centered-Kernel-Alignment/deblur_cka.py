import numpy as np
# import pickle
# import gzip
import cca_core
from CKA import linear_CKA, kernel_CKA
import glob
import os
import cv2


path = 'ubuntu' # 'ubuntu' 'mxt'
out_dir = os.path.join('/home', path, '106-48t/personal_data/mxt/MXT/Deblur2022/results/CKA/complex_concat')
os.makedirs(out_dir, exist_ok=True)
f1_root = os.path.join('/home', path, '106-48t/personal_data/mxt/MXT/Deblur2022/results/ResFourier_complex/feature_x_concat')
f2_root = os.path.join('/home', path, '106-48t/personal_data/mxt/MXT/Deblur2022/results/ResFourier_concat/feature_x_concat')

f1_list = glob.glob(os.path.join(f1_root, '*.npy'))
f2_list = glob.glob(os.path.join(f2_root, '*.npy'))
f1_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
f2_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
m = len(f1_list)
n = len(f2_list)
matrix = np.zeros([m, n])
for i in range(m):
    for j in range(n):
        f1 = f1_list[i]
        f2 = f2_list[j]
        if i == j and f1_root == f2_root:
            CKA = 1.
            matrix[i, j] = CKA
            print('ij: ', i, j, CKA)
        elif i > j and matrix[j, i] != 0 and f1_root == f2_root:
            matrix[i, j] = matrix[j, i]
            print('reverse: ', i, j, matrix[i, j])
        else:
            acts1 = np.load(f1)
            acts2 = np.load(f2)
            num_datapoints, h, w, channels = acts1.shape
            # acts1 = np.resize(acts1, [num_datapoints, ])
            f_acts1 = acts1.reshape((num_datapoints, channels * h * w))

            num_datapoints, h, w, channels = acts2.shape
            f_acts2 = acts2.reshape((num_datapoints, channels * h * w))
            # print(acts1.shape)
            # avg_acts1 = np.mean(acts1, axis=(1, 2))
            # avg_acts2 = np.mean(acts2, axis=(1, 2))
            # print(avg_acts1.shape, avg_acts2.shape)
            K_CKA = kernel_CKA(f_acts1, f_acts2)
            # L_CKA = linear_CKA(f_acts1, f_acts2)
            matrix[i, j] = K_CKA
            print(i, j, K_CKA)
print(matrix)
npy_out_way = os.path.join(out_dir, 'CKA_npy.npy')
np.save(npy_out_way, matrix)
matrix = np.load(npy_out_way)
img_out_way = os.path.join(out_dir, 'CKA_img.png')
cv2.imwrite(img_out_way, matrix * 255)
matrix_flip = cv2.flip(matrix, 0)
img_flip_out_way = os.path.join(out_dir, 'CKA_img_flip.png')
cv2.imwrite(img_flip_out_way, matrix_flip * 255)

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