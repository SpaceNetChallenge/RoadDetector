# PyTorch Pretrained Dual Path Networks (DPN)

This repository includes a PyTorch implementation of DualPathNetworks (https://arxiv.org/abs/1707.01629) that works with cypw's pretrained weights. 

The code is based upon cypw's original MXNet implementation (https://github.com/cypw/DPNs) with oyam's PyTorch implementation (https://github.com/oyam/pytorch-DPNs) as a reference.

If anyone would like to host a direct link of PyTorch pth files I am happy to do the conversion and upload somewhere. I do not have the resources to host myself.

All testing of these models and all validation was done with torch (0.2.0.post1) and mxnet (0.11.0) pip packages installed. 

## Usage

Download and untar trained weights files from https://github.com/cypw/DPNs#trained-models into a './pretrained' folder where this code is located. The pretrained weights can then be used in two ways:

1. They can be converted to PyTorch pth files by using the convert_from_mxnet.py script from the command line and then used as a normal PyTorch checkpoint.
2. They can be used via the model creation functions with pretrained=True if executing in an environment with MXNet available and weights in the './pretrained' folder.

### Conversion Script

    python convert_from_mxnet.py ./pretrained/ --model dpn107
    
### Pretrained 

    python validate.py /imagenet/validation/ --pretrained --model dpn92 --multi-gpu --img-size 320

Ensure you are executing the above with the appropriate MXNet model weights untarred into the './pretrained' folder.

## TODO

* Add conversion support for 5k models and test (need 5K Imagenet)
* Add/test training code from PyTorch imagenet ref impl if any interest

## Results

The following tables contain the validation results (from included validation code) on ImageNet-1K. The DPN models are using the converted weights from the pretrained MXNet models. Also included are results from Torchvision ResNet, DenseNet as well as an InceptionV4 and InceptionResnetV2 port (by Cadene, https://github.com/Cadene/pretrained-models.pytorch) for reference. 

All DPN runs at image size above 224x224 are using the mean-max pooling scheme (https://github.com/cypw/DPNs#mean-max-pooling) described by cypw.

Note that results are sensitive to image crop, scaling interpolation, and even the image library used. All image operations for these models are performed with PIL. Bicubic interpolation is used for all but the ResNet models where bilinear produced better results. Results for InceptionV4 and InceptionResnetV2 where better at 100% crop, all other networks being evaluated at their native training resolution use 87.5% crop.

Models with a '*' are using weights that were trained on ImageNet-5k and fine-tuned on ImageNet-1k. The MXNet weights files for these have an '-extra' suffix in their name.

### Results @224x224

|Model   | Prec@1 (Err)   | Prec@5 (Err)   | #Params   | Crop  |
|---|---|---|---|---|
| DenseNet121 | 74.752 (25.248)  | 92.152 (7.848)  | 7.98  | 87.5%  |
| ResNet50 | 76.130 (23.870) | 92.862 (7.138) |	25.56 | 87.5% |
| DenseNet169 |	75.912 (24.088) | 93.024 (6.976) | 14.15 | 87.5% |
| DualPathNet68 | 76.346 (23.654) | 93.008 (6.992) | 12.61 | 87.5% |
| ResNet101 | 77.374 (22.626) | 93.546 (6.454) | 44.55 | 87.5% |
| DenseNet201 | 77.290 (22.710) | 93.478 (6.522) | 20.01 | 87.5% |
| DenseNet161 | 77.348 (22.652) | 93.646 (6.354) | 28.68 | 87.5% |
| DualPathNet68b* | 77.528 (22.472) | 93.846 (6.154) | 12.61 | 87.5% |
| ResNet152 | 78.312 (21.688) | 94.046 (5.954) | 60.19 | 87.5% |
| DualPathNet92 | 79.128 (20.872) | 94.448 (5.552) | 37.67 | 87.5% |
| DualPathNet98 | 79.666 (20.334) | 94.646 (5.354) | 61.57 | 87.5% |
| DualPathNet131 | 79.806 (20.194) | 94.706 (5.294) | 79.25 | 87.5% |
| DualPathNet92* | 80.034 (19.966) | 94.868 (5.132) | 37.67 | 87.5% |
| DualPathNet107 | 80.172 (19.828) | 94.938 (5.062) | 86.92 | 87.5% |

### Results @299x299

|Model   | Prec@1 (Err)   | Prec@5 (Err)   | #Params   | Crop  |
|---|---|---|---|---|
| InceptionV3 | 77.436 (22.564) | 93.476 (6.524) | 27.16 | 87.5% |
| DualPathNet68 | 78.006 (21.994) | 94.158 (5.842) | 12.61 | 100% |
| DualPathNet68b* | 78.582 (21.418) | 94.470 (5.530) | 12.61 | 100% |
| InceptionV4 | 80.138 (19.862) | 95.010 (4.99) | 42.68 | 100% |
| DualPathNet92* | 80.408 (19.592) | 95.190 (4.810) | 37.67 | 100% |
| DualPathNet92 | 80.480 (19.520) | 95.192 (4.808) | 37.67 | 100% |
| InceptionResnetV2 | 80.492 (19.508) | 95.270 (4.730) | 55.85 | 100% |
| DualPathNet98 | 81.062 (18.938) | 95.404 (4.596) | 61.57 | 100% |
| DualPathNet131 | 81.208 (18.792) | 95.630 (4.370) | 79.25 | 100% |
| DualPathNet107* | 81.432 (18.568) | 95.706 (4.294) | 86.92 | 100% |

### Results @320x320

|Model   | Prec@1 (Err)   | Prec@5 (Err)   | #Params   | Crop  |
|---|---|---|---|---|
| DualPathNet68 | 78.450 (21.550) | 94.358 (5.642) | 12.61 | 100% |
| DualPathNet68b* | 78.764 (21.236) | 94.726 (5.274) | 12.61 | 100% |
| DualPathNet92* | 80.824 (19.176) | 95.570 (4.430) | 37.67 | 100% |
| DualPathNet92 | 80.960 (19.040) | 95.500 (4.500) | 37.67 | 100% |
| DualPathNet98 | 81.276 (18.724) | 95.666 (4.334) | 61.57 | 100% |
| DualPathNet131 | 81.458 (18.542) | 95.786 (4.214) | 79.25 | 100% |
| DualPathNet107* | 81.800 (18.200) | 95.910 (4.090) | 86.92 | 100% |

