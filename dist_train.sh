#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1
CONFIFG=$2


$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIFG --launcher pytorch --world_size=$GPUS ${@:3}
