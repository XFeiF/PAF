#!/usr/bin/env bash
set -e
cd ..
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/src"

python "${SRC_DIR}"/main.py train\
  --desc='test' \
  --cuda=0 \
  --dataset='ImageNet100'\
  --model='res50' \
  --action='base' \
  --epoch=10 \
  --batch_size=64
