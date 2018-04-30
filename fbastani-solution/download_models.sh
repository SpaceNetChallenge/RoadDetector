#!/usr/bin/env bash
mkdir models
aws s3 sync s3://spacenet-dataset/SpaceNet_Roads_Competition/Pretrained_Models/05-fbastani/ models/