#!/usr/bin/env bash

CONFIG=$1

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9995 basicsr/train.py -opt $CONFIG --launcher pytorch
