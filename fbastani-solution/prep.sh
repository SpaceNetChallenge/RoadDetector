#!/bin/bash
set -e
apt-get update
#apt-get dist-upgrade -y
apt-get install -y python2.7 python-gdal python-pip libpython2.7-dev python-tk build-essential golang libgdal-dev libspatialindex-dev git
pip install --upgrade pip
pip install georasters numpy scikit-image scipy cython rtree
#pip install tensorflow-gpu
mkdir -p /wdata/spacenet2017/favyen/graphs/ /wdata/spacenet2017/favyen/truth/
