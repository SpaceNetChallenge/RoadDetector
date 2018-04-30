**Marathon Match - Solution Description**

**Overview**

Congrats on winning this marathon match. As part of your final submission and in order to receive payment for this marathon match, please complete the following document.

**1.** **Introduction**

Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Selim Seferbekov
- Handle: selim\_sef

**2.** **Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- I solved the task in two stages: 1. semantic segmentation of road centerlines 2. vectorization for the binary masks to get final road graph.  To produce binary segmentation masks I used encoder-decoder architectures with skip connections similar to U-Net   [[Olaf et al, 2015]](https://arxiv.org/abs/1505.04597) and Linknet [[Chaurasia et al]](https://arxiv.org/abs/1707.03718). To produce  road graphs I used skeletonization + graph generation with sknw library and some basic postprocessing.
- Data Type: I decided to use MUL-Pansharpen images instead of RGB hoping that neural networks will find indices like road REA/BAI . Which in the end caused a lot of problems during training/testing due CPU/IO bottleneck. After the competition I think that it was better to use original data i.e. full size PAN and small MUL images with late fusion.
- One model for each city or a shared model? I decided to use shared model and added one hot city encoding as additional channels.
- Transfer learning or training from scratch: I used encoders pretrained on ImageNet and just initialized with He initialization additional input channels. Using pretrained encoders allows network to converge faster and produce better results even if it had less input channels originally.
- Originaly I added a topology loss term [[A. Mosinska et al]](https://arxiv.org/abs/1712.02190) which visually improved masks significantly but to due bugs in graph generation I could not get any improvement on the leaderboard and decided not to use it.

 **3.** **Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- For semantic segmentation I used different variation of Unet and Linknet architectures with InceptionV3 and Resnet50 encoders. I trained neworks with RmsProp optimizer and loss=bce+(1–soft dice). Using both crossentropy and soft dice in the loss is crucial to achieve good results in binary semantic segementation and to get better results with ensembling.
- Mask posprocessing: Guassian smoothing and binary dilation. That helped to fill some small gaps in the masks. I also padded masks with reflection in order to produce better graph near the borders which gave +15k on the leaderboard.
- Graph generation: I produced sekeletons and then simply used sknw library to get road graph. After than I simplified graphs to have less lines.
- For validation I used the same 20% holdout set for all models.
- I used contrast normalization (CLAHE) to preprocess images which gave a bit higher score than using original image with simple normalization.
- The final solution has ensemble of 6 models to produce binary masks. The masks produced by these models are averaged and after that vectorized to obtain the final graph.

 **4.** **Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- Docker, [https://www.docker.com](https://www.docker.com/) (Apache License 2.0)
- Tensorflow, [https://www.tensorflow.org](https://www.tensorflow.org/)/ (Apache License 2.0)
- Nvidia-docker, [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker), ( BSD 3-clause)
- Python 3, [https://www.python.org/](https://www.python.org/), ( PSFL (Python Software Foundation License))
- Scikti-image, [http://scikit-image.org/](http://scikit-image.org/), ( BSD 3-clause)
- Scikit-learn, [http://scikit-learn.org/stable/](http://scikit-learn.org/stable/), (BSD 3-clause)
- Numpy, [http://www.numpy.org/](http://www.numpy.org/), (BSD)
- Scipy, [https://www.scipy.org/](https://www.scipy.org/), (BSD)
- Tqdm, [https://github.com/noamraph/tqdm](https://github.com/noamraph/tqdm), ( The MIT License)
- Keras, [https://keras.io/](https://keras.io/), ( The MIT License)
- Anaconda, [https://www.continuum.io/Anaconda-Overview](https://www.continuum.io/Anaconda-Overview),( New BSD License)
- OpenCV, [https://opencv.org/](https://opencv.org/) (BSD)
- SKNW [https://github.com/yxdragon/sknw](https://github.com/yxdragon/sknw) (BSD 3-clause)
- Simplification [https://github.com/urschrei/simplification](https://github.com/urschrei/simplification) (MIT)

 **5.** **Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

- Use more masks of different width for different roads to train  networks
- Somehow incorporate loss term that punishes topology violations.
- For binary segmetation it is usually better to use simple architeture like Unet with VGG16 BN encoder. As vgg16bn is extremely slow I decided not to use it in this challenge.
- Add posproccessing to connect gaps in the graph
- Instead of using PAN sharpened images it could be benefitial to use original MUL and PAN images and fuse them with neural network.

 **6.** **Algorithm Limitations**

Please specify any potential limitations with the algorithm:

- The current approach doesn&#39;t handle bridges and multilevel intersections properly.

 **7.** **Deployment Guide**

Please provide the exact steps required to build and deploy the code:

1. In this contest, a Dockerized version of the solution was required, which should run out of the box

 **8.** **Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

1. The algorithm can be executed by the instructions provided for the contest.

