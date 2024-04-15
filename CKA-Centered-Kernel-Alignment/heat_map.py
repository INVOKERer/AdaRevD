import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


# file_way = '/home/mxt/106-48t/personal_data/mxt/MXT/Deblur2022/results/CCA/nofft_nofft/CCA_npy.npy'
file_way = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/linear_ckav2_Large_bs64/decoder_cka_cls.npy'
out_root = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/linear_ckav2_Large_bs64'
file_name = os.path.basename(file_way)
filename, ext = os.path.splitext(file_name)
os.makedirs(out_root, exist_ok=True)
x = np.load(file_way)
print(x)
print(x.max(), x.min())
# x = 1-x
# x = cv2.flip(x, 0)
# h, w = x.shape
# x = x[h // 2 - 16: h // 2 + 16, w // 2 - 16: w // 2 + 16]
# x[np.where(x >= 1)] = 255
# cv2.imwrite('/home/mxt/106-48t/personal_data/mxt/MXT/Deblur2022/results/erf/DeepRFT_255.png', x)
# x = x / np.ptp(x)
sns_plot = plt.figure()
sns.heatmap(x, vmax=0.85, vmin=0.4, cmap='Reds_r', linewidths=0.5, square=True, xticklabels=False, yticklabels=False, cbar=False) # .invert_yaxis()
# blur pattern v2_linear: 1. 0.5
# cls v2_linear: 0.85 0.4
# blur pattern v2: 1. 0.
# cls v2: 0.9 0.1
# blur pattern v1: 1. 0.5
# cls v1: 0.8 0.3

# print(y)
# plt.ylabel('features', fontsize=12, color='k') #y轴label的文本和字体大小
# plt.xlabel('decoders', fontsize=12, color='k') #x轴label的文本和字体大小

# out_way = os.path.join(out_root, filename+'-heatmap' + '.png')
# sns_plot.savefig(out_way, dpi=100)
# plt.close()

# plt.show()