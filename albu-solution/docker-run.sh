#!/usr/bin/env bash

nvidia-docker run -v /mnt/disk2/roads/data:/data:ro -v /mnt/disk2/roads/wdata:/wdata --rm -ti --ipc=host albu
