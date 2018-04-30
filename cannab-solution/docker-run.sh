#!/usr/bin/env bash

nvidia-docker run -v /local_data/SpaceNet_Roads_Dataset:/data:ro -v /local_data/SpaceNet_Roads_Dataset/results/cannab:/wdata --rm -ti --ipc=host cannab
