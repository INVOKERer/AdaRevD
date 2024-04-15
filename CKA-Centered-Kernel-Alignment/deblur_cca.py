import numpy as np
# import pickle
# import gzip
import cca_core
from CKA import linear_CKA, kernel_CKA
import glob
import os
import cv2


path = 'ubuntu' # 'ubuntu' 'mxt'
out_dir = os.path.join('/home', path, '106-48t/personal_data/mxt/MXT/Deblur2022/results/CCA/nofft_nofft')
os.makedirs(out_dir, exist_ok=True)
f1_root = os.path.join('/home', path, '106-48t/personal_data/mxt/MXT/Deblur2022/results/ResFourier_nofft/feature_x_concat')
f2_root = os.path.join('/home', path, '106-48t/personal_data/mxt/MXT/Deblur2022/results/ResFourier_nofft/feature_x_concat')

f1_list = glob.glob(os.path.join(f1_root, '*.npy'))
f2_list = glob.glob(os.path.join(f2_root, '*.npy'))
# print(f1_list)
f1_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
f2_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
# print(f1_list)
m = len(f1_list)
n = len(f2_list)
matrix = np.zeros([m, n])
for i in range(m):
    for j in range(n):

        f1 = f1_list[i]
        f2 = f2_list[j]
        if i == j and f1_root == f2_root:
            cca = 1.
            matrix[i, j] = cca
            print('ij: ', i, j, cca)
        elif i > j and matrix[j, i] != 0 and f1_root == f2_root:
            matrix[i, j] = matrix[j, i]
            print('reverse: ', i, j, matrix[i, j])
        else:
            acts1 = np.load(f1)
            acts2 = np.load(f2)
            # CCA
            num_datapoints, h, w, channels = acts1.shape
            f_acts1 = acts1.reshape((num_datapoints * h * w, channels))

            num_datapoints, h, w, channels = acts2.shape
            f_acts2 = acts2.reshape((num_datapoints * h * w, channels))

            f_results = cca_core.get_cca_similarity(f_acts1.T, f_acts2.T, epsilon=1e-10, verbose=False)
            cca = np.mean(f_results["cca_coef1"])
            matrix[i, j] = cca
            print(i, j, cca)
print(matrix)
npy_out_way = os.path.join(out_dir, 'CCA_npy.npy')
np.save(npy_out_way, matrix)
matrix = np.load(npy_out_way)
img_out_way = os.path.join(out_dir, 'CCA_img.png')
cv2.imwrite(img_out_way, matrix * 255)
matrix_flip = cv2.flip(matrix, 0)
img_flip_out_way = os.path.join(out_dir, 'CCA_img_flip.png')
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