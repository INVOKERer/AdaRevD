[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adarevd-adaptive-patch-exiting-reversible-1/image-deblurring-on-gopro)](https://paperswithcode.com/sota/image-deblurring-on-gopro?p=adarevd-adaptive-patch-exiting-reversible-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adarevd-adaptive-patch-exiting-reversible-1/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=adarevd-adaptive-patch-exiting-reversible-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adarevd-adaptive-patch-exiting-reversible-1/deblurring-on-realblur-j-1)](https://paperswithcode.com/sota/deblurring-on-realblur-j-1?p=adarevd-adaptive-patch-exiting-reversible-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adarevd-adaptive-patch-exiting-reversible-1/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=adarevd-adaptive-patch-exiting-reversible-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adarevd-adaptive-patch-exiting-reversible-1/deblurring-on-realblur-r)](https://paperswithcode.com/sota/deblurring-on-realblur-r?p=adarevd-adaptive-patch-exiting-reversible-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adarevd-adaptive-patch-exiting-reversible-1/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=adarevd-adaptive-patch-exiting-reversible-1)

# [AdaRevD: Adaptive Patch Exiting Reversible Decoder Pushes the Limit of Image Deblurring (CVPR2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Mao_AdaRevD_Adaptive_Patch_Exiting_Reversible_Decoder_Pushes_the_Limit_of_CVPR_2024_paper.html)
[Xintian Mao](https://scholar.google.es/citations?user=eM5Ogs8AAAAJ&hl=en), Qingli Li and [Yan Wang](https://scholar.google.com/citations?user=5a1Cmk0AAAAJ&hl=en)





## Quick Run

## Training
1. Download GoPro training and testing data
2. To train the main body of AdaRevD, download the pretrained model from [NAFNet](https://github.com/megvii-research/NAFNet) or [UFPNet](https://github.com/Fangzhenxuan/UFPDeblur) [百度网盘](https://pan.baidu.com/s/1ibUqnlcl_F21CiBukEQHBA)(提取码: xpjh), modify state_dict_pth_encoder in GoPro-AdaRevIDB-pretrain-4gpu.yml and run
 ```
cd AdaRevD
./train_4gpu.sh Motion_Deblurring/Options/GoPro-AdaRevIDB-pretrain-4gpu.yml
```
3. To train the classifier of AdaRevD, modify the pretrain_network_g in GoPro-AdaRevIDB-classify-4gpu.yml and run
 ```
./train_4gpu.sh Motion_Deblurring/Options/GoPro-AdaRevIDB-classify-4gpu.yml
```

## Evaluation
To test the pre-trained models [Google Drive](https://drive.google.com/drive/folders/1r4SKc6jRBK5gr1Kl4iORHBjd-5TRKTSY?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1AttePVDB1IcfLPMscg1-Bg)(提取码:dfce) on your own images (turn the 'pretrain' in yml from false to true for RevD), run 
```
python Motion_Deblurring/val.py 
```

## Results
Results on GoPro, HIDE, Realblur test sets:
[Google Drive](https://drive.google.com/drive/folders/11mYTAqCln1dg7TmeQw3dAffpZ2uecUpe?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1xoEgsisdnMnjbEHcCXOa2Q)(提取码:27ex)

## Citation
If you use , please consider citing:
```
@inproceedings{xintm2024AdaRevD, 
    title = {AdaRevD: Adaptive Patch Exiting Reversible Decoder Pushes the Limit of Image Deblurring},
    author = {Xintian Mao, Qingli Li and Yan Wang}, 
    booktitle = {Proc. CVPR}, 
    year = {2024}
    }
```
## Contact
If you have any question, please contact mxt_invoker1997@163.com

## Our Related Works
- Deep Residual Fourier Transformation for Single Image Deblurring, arXiv 2021. [Paper](https://arxiv.org/abs/2111.11745v1) | [Code](https://github.com/INVOKERer/DeepRFT)
- Intriguing Findings of Frequency Selection for Image Deblurring, AAAI 2023. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/DeepMed-Lab-ECNU/DeepRFT-AAAI2023)
- LoFormer: Local Frequency Transformer for Image Deblurring, ACM MM 2024. [Paper](https://arxiv.org/abs/2407.16993) | [Code](https://github.com/INVOKERer/LoFormer)

## Reference Code:
- https://github.com/Fangzhenxuan/UFPDeblur
- https://github.com/megvii-research/NAFNet
- https://github.com/megvii-research/RevCol
- https://github.com/littlepure2333/APE
- https://github.com/INVOKERer/DeepRFT/tree/AAAI2023

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 
