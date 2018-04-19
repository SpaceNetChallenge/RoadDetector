#!/usr/bin/env bash
set -e

rm -rf /results/*
rm -rf /wdata/*
pushd /opt/app/src
python create_spacenet_masks.py $1 $2 $3 $4 --training
CUDA_VISIBLE_DEVICES=0 python train_eval.py resnet34_512_02_02.json --training
popd
