#!/usr/bin/env bash

CONFIG=$1

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=9993 basicsr/train.py -opt $CONFIG --launcher pytorch
