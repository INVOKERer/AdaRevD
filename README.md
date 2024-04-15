

# AdaRevD: Adaptive Patch Exiting Reversible Decoder Pushes the Limit of Image Deblurring (CVPR)
[Xintian Mao](https://scholar.google.es/citations?user=eM5Ogs8AAAAJ&hl=en), Qingli Li and [Yan Wang](https://scholar.google.com/citations?user=5a1Cmk0AAAAJ&hl=en)



**Paper**: (https://github.com/INVOKERer/AdaRevD/blob/master/AdaRevD.pdf)


## Quick Run

## Training
1. Download GoPro training and testing data
2. To train the main body of AdaRevD, run
 ```
cd AdaRevD
./train_4gpu.sh Motion_Deblurring/Options/GoPro-AdaRevIDB-pretrain-4gpu.yml
```
3. To train the classifier of AdaRevD, modify the pretrain_network_g in GoPro-AdaRevIDB-classify-4gpu.yml and run
 ```
./train_4gpu.sh Motion_Deblurring/Options/GoPro-AdaRevIDB-classify-4gpu.yml
```

## Evaluation
To test the pre-trained models [百度网盘](https://pan.baidu.com/s/1AttePVDB1IcfLPMscg1-Bg)(提取码:dfce) on your own images, run 
```
python Motion_Deblurring/val.py 
```

## Results
Results on GoPro, HIDE, Realblur test sets:
[百度网盘](https://pan.baidu.com/s/1xoEgsisdnMnjbEHcCXOa2Q)(提取码:27ex)

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
- Deep Residual Fourier Transformation for Single Image Deblurring, arXiv 2021. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/INVOKERer/DeepRFT)
- Intriguing Findings of Frequency Selection for Image Deblurring, AAAI 2023. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/DeepMed-Lab-ECNU/DeepRFT-AAAI2023)

## Reference Code:
- https://github.com/Fangzhenxuan/UFPDeblur
- https://github.com/megvii-research/NAFNet
- https://github.com/megvii-research/RevCol
- https://github.com/littlepure2333/APE
- https://github.com/INVOKERer/DeepRFT/tree/AAAI2023

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 
