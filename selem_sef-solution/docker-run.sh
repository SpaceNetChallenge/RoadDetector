#!/usr/bin/env bash

nvidia-docker run -v /local_data/SpaceNet_Roads_Dataset:/data -v /local_data/SpaceNet_Roads_Dataset/results/selim_sef:/wdata --rm -ti --ipc=host selim_sef
