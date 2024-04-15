# Installation

This repository is built in PyTorch 1.12.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/INVOKERer/AdaRevD.git
cd AdaRevD
```

2. Make conda environment
```
conda create -n pytorch112 python=3.8
conda activate pytorch112
```

3. Install dependencies
```
conda install pytorch=1.12.0 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm ptflops
pip install seaborn spectral einops kornia timm gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext --record install_files.txt
```
