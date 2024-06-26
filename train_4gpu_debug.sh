#!/usr/bin/env bash

CONFIG=$1

CUDA_VISIBLE_DEVICES=4,5,6,7 TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=4 --master_port=9994 basicsr/train.py -opt $CONFIG --launcher pytorch
