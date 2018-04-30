# 02-Cannab 
SpaceNet Road Detector - Solution Description

Overview

Congrats on winning this marathon match. As part of your final submission and in order to receive payment for this marathon match, please complete the following document.

1.** Introduction**
Tell us a bit about yourself, and why you have decided to participate in the contest.

●● Name: Victor Durnov

●● Handle: cannab

●● Placement you achieved in the MM: 3rd

●● About you: I'm independent Software Developer/Data Scientist interested in hard algorithmic challenges and machine learning

●● Why you participated in the MM: I prefer to learn on practice and challenges like this one is a very good opportunity to learn something new

2.** Solution Development**
How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

●● Neural Networks are most suitable tool for such kind of tasks. I've treated the problem as image segmentation task to find road masks (center ground truth line with 2-3 meters buffer).So, the main tool was UNet-like and LinkNet-like Encoder-Decoder Neural Networks with different pretrained encoders (Transfer Learning, models from https://keras.io/applications ). (UNet paper: https://arxiv.org/pdf/1505.04597.pdf LinkNet: https://arxiv.org/pdf/1707.03718.pdf )

●● I've tried a lot of Neural Network's architectures and came up with big final ensemble with different weights for each city. Tried different Tile sizes, different channels and input preprocessing

●● I've separated data to 4 folds and trained only for 2 of them due to time and resources limitation.

●● Post-processing for predicted masks to build final roads network graph also was challenging. I've skeletonized masks, built graph, cleaned it, removed small parts and crosses (even if it increased score) and tried to fix broken connections

3.** Final Approach**
Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

●● Finally, I've used 12 Neural Network models:

--UNet-like Neural Network with pretrained VGG16 as encoder. Input takes all 8 channels from MUL-PanSharpen images. Trained on 512*512 random crops from full 1300*1300 tile. Road line width for masks was 22 pixels. City Id one-hot encoded used as additional input for this model and fusioned with main model in last convolution layer. This allowed to pretrain model using all 4 cities and then fine-tune it for each city separately

--UNet-like Neural Network with pretrained VGG16 as encoder. All 8 channels from MUL images + 1 channels from PAN images concatenated and used as input. Trained on 512*512 random crops from tile resized to 650*650 for each city separately. Road line width for masks was 12 pixels
--Same model as previous, but trained on 320*320 crops from tile resized to 325*325. Road line width for masks was 4 pixels
--LinkNet-like Neural Network with pretrained VGG16 as encoder. All 8 channels from MUL-PanSharpen images + 1 channels from PAN images concatenated and used as input. Trained on 512*512 random crops from full 1300*1300 tile for each city separately. Road line width for masks was 22 pixels
--Same model as previous, but trained on 512*512 random crops from tile resized to 650*650. MUL images used instead of MUL-PanSharpen. Road line width for masks was 9 pixels
--Same model as previous, but trained on 512*512 random crops from tile resized to 520*520. Road line width for masks was 9 pixels
--Unet-like Neural Network with pretrained InceptionResNetV2 as encoder. All 8 channels from MUL images + 1 channels from PAN images concatenated and used as input. Trained on 512*512 random crops from tile resized to 650*650 for each city separately. Road line width for masks was 9 pixels
--Same model as previous, but trained on 512*512 random crops from tile resized to 520*520. Road line width for masks was 6 pixels
--Same model as previous, but trained on 320*320 crops from tile resized to 325*325. Road line width for masks was 4 pixels
--Unet-like Neural Network with pretrained InceptionV3 as encoder. All 8 channels from MUL images + 1 channels from PAN images concatenated and used as input. Trained on 512*512 random crops from tile resized to 520*520 for each city separately. Road line width for masks was 9 pixels
--LinkNet-like Neural Network with pretrained ResNet50 as encoder. All 8 channels from MUL images + 1 channels from PAN images concatenated and used as input. Trained on 512*512 random crops from tile resized to 650*650 for each city separately. Road line width for masks was 9 pixels
--Unet-like Neural Network with pretrained ResNet50 as encoder. All 8 channels from MUL images + 1 channels from PAN images concatenated and used as input. Trained on 320*320 crops from tile resized to 325*325. Road line width for masks was 4 pixels

●● Then best weights in ensemble found per city using out-of-fold predictions for 50% of train data (2 of 4 folds) using provided Visualizer.

●● Also some parameters for post-processing where manually found per city using out-of-fold predictions and Visualizer: where to use erosion or dilation, distance to connect broken roads, min road length to remove noise.

4.** Open Source Resources, Frameworks and Libraries**
Please specify the name of the open source resource along with a URL to where it's housed and it's license type:

●● Anaconda as base Python 3 environment, www.anaconda.com
●● Tensorflow, www.tensorflow.org Apache License
●● Keras, https://keras.io The MIT License
●● OpenCV, https://opencv.org BSD License
●● Shapely, https://github.com/Toblerity/Shapely

5.** Potential Algorithm Improvements**
Please specify any potential improvements that can be made to the algorithm:

●● Improve data quality: there are some bugs when road masks on missing image data, also when tiles cut along the road, etc.
●● Train Neural Networks longer/harder, find better architectures

6.** Algorithm Limitations**
Please specify any potential limitations with the algorithm:

●● Was not able to build one common model for all cities. So, need to train/tune separate models for each new city.

7.** Deployment Guide**
Please provide the exact steps required to build and deploy the code:

    Dockerized version prepared as requested. For clean installation python 3 required with libraries (all in anaconda3 default installation): numpy, sklearn + install OpenCV, Tensorflow, Keras, shapely
8.** Final Verification**
Please provide instructions that explain how to train the algorithm and have it execute against sample data:

train.sh and test.sh scripts meet required specification.

