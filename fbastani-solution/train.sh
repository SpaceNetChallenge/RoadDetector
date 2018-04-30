#!/bin/bash
set -e
rm -rf models/*
bash prep.sh

# install go packages
export GOPATH=/go
mkdir -p /go
go get github.com/ajstarks/svgo
go get github.com/dhconnelly/rtreego
go get github.com/qedus/osmpbf

# create graphs, training masks
go run 1_convertgraphs.go $@
go run 2_truth_tiles.go

# run training
python do_the_training.py $@

rm -rf /wdata/*
