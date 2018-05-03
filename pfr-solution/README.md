SpaceNet 3 – Road Detection Marathon Match - Solution Description

Overview


1.** Introduction**
Tell us a bit about yourself, and why you have decided to participate in the contest.


Handle: pfr

2.** Solution Development**
How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

 The overall approach was to create a dense bitmap predictor that would be fed into a vectorizer.
 My initial focus was on developing a good convolutional network for dense prediction that would be able to utilize as much context as possible and could be trained efficiently. The dense prediction was obtained by simply reshaping the last layer of a classifier into small blocks of 2D data, which worked well since roads are unidimensional and therefore even at junctions, each block can usually be well approximated by a point from a low-dimensional manifold.
 I also experimented with a deconvolution backend with U-Net-style connections, which I suspected may not perform as well but might be useful for ensembling. It was much faster to train, but performed poorly because it exhibited too much reliance on the local characteristics of the source image.
 For vectorization, my plan was to develop a neural network-based vectorizer, but I didn't have enough time left for that, in light of the fact that the most straightforward solutions were not feasible due to stringent constraints on inference speed. So I just refined the simple threshold-based vectorizer which I was using initially, which was surprisingly effective.
 I only used RGB data, partly for faster development, and partly because I noticed issues in multispectral data which I didn't think would get fixed on short notice – the offending tiles were eventually removed, but the announcement was made only after the contest had ended.

3.** Final Approach**
Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

** General approach:** The model uses two stages: an ensemble of 9 neural networks to create a dense 2D prediction, followed by a vectorizer.
** Dense prediction:** Dense prediction is done with a stride of 4, which corresponds to an output spatial resolution of 1.2 m. Each pixel is labeled with one of 3 classes depending on distance to the closest road pixel: the distance thresholds are 6 and 12 source-image pixels. The network architecture is based on DPN-92 [Chen 2017], initialized with pre-trained weights published by their authors. Two changes are made: first, the layers after the last subsampling are removed and replaced with a thinner but deeper variant (block depth 12 to 16 instead of 3). Next, average pooling is removed and the 192 output channels of the linear layer are reshaped into an 8x8 3-channel block, so that dense prediction of a whole image can be done by executing the network once. Since predictions near the edges have less context available, the output is cropped by 24 pixels from all edges.
** Training:** A random 352x352 crop of each tile is extracted, and a random flip or 90°-multiple rotation (or in 40% of samples, an arbitrary rotation) is applied. Training is done over 9 or 10 epochs by the Adam optimizer with a learning rate of 3.75e-4, a batch size of 12 per GPU, and a weight decay factor of 1e-7. The learning rate is divided by 5 every epoch after the first 6 epochs. Note: 6 of the models are only trained on two thirds of the dataset.
** External clipping:** The last 3 models are not trained on pixels that are outside the region where image data is available or within 10 pixels of the tile boundary, and their output is forced to zero outside the image data region. They are also subject to additional training augmentation by intersecting the image data with a randomly-offset axis-aligned quadrant.
** Prediction:** The source image is split into 4 overlapping corner patches. The model is then evaluated and the output is obtained by using the prediction for the patch closest to each pixel. This is done for all models, then averaged by arithmetic mean and reduced to a single channel by taking the probability of the closest-to-road class and adding 15% of the probability of the second-closest-to-road class.
** Vectorization:** The first step is upsampling by a factor of 2 to get a 0.6 m spatial resolution, and thresholding to obtain the set of pixels having probability at least 28.8% of being a road pixel. The image is then padded slightly, denoised with a 5x5 closing kernel and skeletonized with the scikit-image implementation. This skeleton is converted into a vector network, which is then smoothed by the Shapely library. Finally, short isolated or dead-end segments, which can occur due to skeletonization, are removed.

4.** Open Source Resources, Frameworks and Libraries**
Please specify the name of the open source resource along with a URL to where it's housed and it's license type:

 PyTorch, http://pytorch.org/, 3-clause BSD
 PyTorch Pretrained Dual Path Networks, https://github.com/rwightman/pytorch-dpn-pretrained, Apache 2.0
Paper: Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng. "Dual Path Networks" ( NIPS17). [Chen 2017]

 Shapely, https://github.com/Toblerity/Shapely, 3-clause BSD
 Scikit-image, http://scikit-image.org/, 3-clause BSD
 GDAL, http://www.gdal.org/, MIT License
 Standard Python scientific ecosystem (Python 3, Numpy, Scipy, Pandas, Pillow...)

5.** Potential Algorithm Improvements**
Please specify any potential improvements that can be made to the algorithm:

 Use more context to avoid tile boundary effects, which wasn't allowed in this contest.
 Use multi-spectral data.
 Use a more sophisticated vectorizer, such as graph-based regularization instead of thresholding, or an end-to-end neural network vectorizer.
 Experiment with deconvolution layers instead of the final linear layer.

6.** Algorithm Limitations**
Please specify any potential limitations with the algorithm:

 The contest is designed so that the test tiles are from the same region as the train tiles, therefore the quality of generalization to new regions has not been measured. It is likely to be limited by the fact that the training data is only composed of a single viewpoint for each region.
 Applying the algorithm to a new AOI requires filling in 6 calibration coefficients (see param/adjust_rgb_v1.csv) derived from estimation of the black and white points of the 16-bit image. Automatic determination of the coefficients would be possible, but naive approaches may underperform in areas such as the Paris AOI that have a large proportion of forest tiles. Therefore for this contest manual adjustment of the 4 regions was chosen.

7.** Deployment Guide**
Please provide the exact steps required to build and deploy the code:

Prerequisites:
-
-
- The host machine should support AVX2, 4 GPUs for training or 1 GPU for testing, and nvidia-docker 2 must be installed.
- The SpaceNet 3 dataset should be available in folder $DATA_PATH.
- Some temporary data will be placed in folder $WDATA_PATH.
- The code package should be available in folder $PACKAGE_PATH.

Run the command docker build -t spacenet3_pfr $PACKAGE_PATH
Launch the docker container with
nvidia-docker run -v $DATA_PATH:/data:ro -v $WDATA_PATH:/wdata --ipc=host -it spacenet3_pfr

8.** Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

The pre-trained models can be executed directly from the docker container: ./test.sh /path_to_first_test_region_folder ... /path_to_last_test_region_folder name_of_output_file
Training can be run by calling
./train.sh /path_to_first_train_region_folder ... /path_to_last_train_region_folder

The trained models are stored in trained_models/*.pth
Calling sh again after training will use the newly created models.

