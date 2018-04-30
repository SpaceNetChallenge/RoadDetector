#!/bin/bash
set -e
bash prep.sh
python run_test.py $@
rm -rf /wdata/*
