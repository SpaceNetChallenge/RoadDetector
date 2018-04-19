#!/usr/bin/env bash
set -e

rm -rf /results/*
rm -rf /wdata/*
pushd /opt/app/src
python create_spacenet_masks.py $1 $2 $3 $4 --training
./run_training.sh
popd
