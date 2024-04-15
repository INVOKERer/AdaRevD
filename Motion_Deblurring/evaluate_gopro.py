
import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from tqdm import tqdm
import concurrent.futures
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim
def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test):

  return psnr_loss(image_true, image_test)


def compute_ssim(tar_img, prd_img):

    ssim = calculate_ssim(tar_img, prd_img, crop_border=0)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0
    # try:
    #     prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)
    # except:
    #     print('x: ' + prd.split('/')[-1])
    #     cr1 = np.ones_like(prd_img)
    # print(prd_img.shape, tar_img.shape, cr1.shape, cr1.max(), cr1.min())
    PSNR = compute_psnr(tar_img, prd_img)
    SSIM = compute_ssim(tar_img, prd_img)
    return (PSNR,SSIM)

datasets = ['GoPro', 'HIDE']
# datasets = ['RealBlur_R']
# datasets = ['RealBlur_J']
for dataset in datasets:

    # file_path = os.path.join('/home/ubuntu/90t/personal_data/mxt/MXT/exp_results/RevD-L_RealBlur', dataset)
    
    file_path = os.path.join('/home/ubuntu/106-48t/personal_data/mxt/exp_results/images_deblur/NAFNet64', dataset)

    print(file_path)
    gt_path = os.path.join('/home/ubuntu/106-48t/personal_data/mxt/Datasets/Deblur', dataset, 'test/sharp')

    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    print(len(path_list), len(gt_list))
    assert len(path_list) != 0, "Predicted files not found"
    assert len(gt_list) != 0, "Target files not found"

    psnr, ssim = [], []
    # img_files =[(i, j) for i,j in zip(gt_list,path_list)]
    # try:
    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    # for filename in img_files:
    #     # print(filename)
    #     PSNR_SSIM = proc(filename)
    #     psnr.append(PSNR_SSIM[0])
    #     ssim.append(PSNR_SSIM[1])
    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    #     for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
    #         # print(img_files)
    for i1, i2 in zip(gt_list, path_list):
        PSNR_SSIM = proc([i1, i2])
        psnr.append(PSNR_SSIM[0])
        ssim.append(PSNR_SSIM[1])
    # except:
    #     print('x')
    #     psnr.append(0.)
    #     ssim.append(0.)

    avg_psnr = sum(psnr)/len(psnr)
    avg_ssim = sum(ssim)/len(ssim)

    print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
