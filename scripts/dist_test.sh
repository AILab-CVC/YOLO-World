#!/usr/bin/env bash

SIZE=$1
CHECKPOINT=$2
GPUS=$3
OMP=${OMP:-4}

OMP_NUM_THREADS=$OMP torchrun --nproc_per_node=$GPUS \
$(dirname "$0")/test_lvis.py $SIZE $CHECKPOINT
