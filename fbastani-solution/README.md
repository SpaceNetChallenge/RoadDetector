Marathon Match - Solution Description

Overview

Congrats on winning this marathon match. As part of your final submission and in order to receive payment for this marathon match, please complete the following document.

**1.** Introduction

Tell us a bit about yourself, and why you have decided to participate in the contest.

- .Name: Favyen Bastani
- .Handle: fbastani

**2.** Solution Development

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- .I first tried to adapt an approach that I had developed in the past ( [https://arxiv.org/abs/1802.03680](https://arxiv.org/abs/1802.03680)); however, I found that it struggled in areas where roads were close together, such as parking lots.
- .I then decided to go with a segmentation-based road extraction method, similar to DeepRoadMapper ( [http://openaccess.thecvf.com/content\_iccv\_2017/html/Mattyus\_DeepRoadMapper\_Extracting\_Road\_ICCV\_2017\_paper.html](http://openaccess.thecvf.com/content_iccv_2017/html/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.html)). Segmentation yields more accurate geometry than the above approach. After extracting an initial road graph, DeepRoadMapper includes a second step to refine the topology of the graph by adding missing connections, but I didn&#39;t get a chance to implement that.
- .After that, I gradually improved performance by playing around with the CNN configuration, training process, and other parameters.

**3.** Final Approach

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

- .First, use a CNN to segment the satellite imagery, classifying each pixel as either &quot;road&quot; or &quot;non-road&quot;
  - .Input: random 256x256 crops of the 1300x1300 images (9 channels, 8 from MUL-PanSharpen and 1 from PAN), with random 0/90/180/270 degrees rotation
    - ■.I also tried 512x512 crops and using the full 1300x1300 images, but couldn&#39;t get it to train as reliably
  - .Targets: 128x128 image (half the input resolution) where we draw lines along the road segments at a fixed width
  - .Model: I used a U-Net like architecture with 128 channels at the top layer, and 512 at the lowest resolution layer
    - ■.I consistently found that, even with so many channels, the model would not overfit to the training data
    - ■.I tried many CNN configurations, but it is hard to say why this one worked (slightly) better than the others that I tried
  - .Learning rate: I used 1e-3 learning rate with ADAM, and dropped the learning rate after some number of training rounds
    - ■.I found that dropping the learning rate like this gave significantly higher performance in the end
  - .Loss function: softmax cross-entropy
    - ■.I also tried L2, it had about the same performance
  - .Validation set: I used 80 of the 2780 images as a validation set, and saved the model that had the best loss over several random 256x256 crops from these 80 images
  - .Inference: I used the full 1300x1300 resolution at inference time
  - .Training data: I originally trained a single model across all cities, but later found that training separate models per-city gave slightly higher performance
  - .Ensemble: for each city, I trained four models, and averaged the segmentation outputs
    - ■.This gave slightly higher performance than using just one model
    - ■.I tried using AdaBoost for the ensemble, but it kept giving lower performance than even the individual models
- .Second, we need to extract a road network graph from the segmentation output
  - .I applied a Gaussian blur with sigma=1pixel over the output
    - ■.My intuition was that the CNN would be less confident on large roads because the position of the centerline is more ambiguous on these roads; blurring could correct for this
    - ■.It could also correct for small artifacts in the output
    - ■.But the performance difference over not blurring was actually negligible
  - .Then, applied threshold to convert the output to a binary mask
    - ■.I determined the threshold by optimizing on the validation set
  - .Apply morphological thinning (similar to skimage.morphology.thin) so that we get single-pixel-width lines
  - .At this point, we basically have a graph, where every set pixel is a vertex, and there are edges between adjacent set pixels
  - .But I used Douglas-Peucker to reduce the number of vertices without significantly altering the road network
  - .Padding optimization: before thinning, pad the image with several copies of its border on all sides
    - ■.This makes it more likely that roads would be connected to the border after thinning, since thinning can unset pixels at the border
    - ■.This is actually very important because APLS will always put a vertex at these dead-end roads; if we miss most of the border vertices then performance goes down a lot
- .Finally, I included several additional post-processing steps
  - .The thinning can result in artifacts that need to be pruned
    - ■.Remove small connected components
    - ■.Remove short segments that have a dead-end (i.e., at least one endpoint of the edge has only one incident edge)
  - .If a road is close to the border, then connect it to the border
  - .If a road is close to another road, then connect it; I implemented this by trying to extend dead-end segments by a certain amount and see if they intersect another segment
    - ■.Overall I think this might have actually decreased performance slightly, but the resulting map always looks better since it doesn&#39;t have small holes
**4.** Open Source Resources, Frameworks and Libraries

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- .Golang standard library: [https://github.com/golang/go](https://github.com/golang/go), BSD-style license
- .Python standard library: [https://www.python.org/](https://www.python.org/), PSF license
- .TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/), Apache 2.0
- .NumPy: [http://www.numpy.org/](http://www.numpy.org/), BSD
- .SciPy: [https://www.scipy.org/](https://www.scipy.org/), BSD-new
- .scikit-image: [http://scikit-image.org/](http://scikit-image.org/), BSD
- .rtree: [https://pypi.python.org/pypi/Rtree/](https://pypi.python.org/pypi/Rtree/), LGPL
- .georasters: [https://github.com/ozak/georasters](https://github.com/ozak/georasters), GPL
- .libspatialindex: [https://libspatialindex.github.io/](https://libspatialindex.github.io/), LGPL
- .libgdal: [http://www.gdal.org/](http://www.gdal.org/), X/MIT

**5.** Potential Algorithm Improvements

Please specify any potential improvements that can be made to the algorithm:

- .Implementing DeepRoadMapper&#39;s system to add missing connections could improve the APLS metric.
- .The road matching threshold in APLS is very strict; a post-processing step to use another CNN to align vertices along the road with the road in the satellite imagery could improve performance.
- .I think there is also lots of room to improve the CNN, such as using residual networks or other more modern networks, and trying different targets.
- .Adding features from OpenStreetMap could help; in particular, I considered adding a third segmentation class for buildings, with ground truth labels based on OSM.

**6.** Algorithm Limitations

Please specify any potential limitations with the algorithm:

- .If the segmentation output is noisy, then the graph extraction process will yield a very noisy graph; this may have decent APLS performance, but it would be useless for routing.
- .If a large segment of road is occluded by trees, then it won&#39;t be detected.
- .I did not implement anything to deal with overpasses/underpasses, because there were very few in the dataset.

**7.** Deployment Guide

Please provide the exact steps required to build and deploy the code:

1. If using Docker:
  1. cd /path/to/code
  2. nvidia-docker build -t spacenet .
  3. nvidia-docker run --name spacenet -v /mnt/cmt/data:/data:ro -it spacenet
  4. bash prep.sh
2. Otherwise:
  1. See prep.sh for the software that you&#39;ll need to install; you may need to adapt the steps if you are not using Ubuntu 16.04

**8.** Final Verification

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

1. Training
  1. bash train.sh /path/to/training/data
  2. train.sh first runs 1\_convertgraphs.go to convert summaryData CSV files into a simpler file format
  3. Then, 2\_truth\_tiles.go produces target images from the graphs
  4. Finally, do\_the\_training.py trains four models per city across four GPUs; you can also use run\_train.py to have more control
2. Testing
  1. bash test.sh /path/to/testing/data out.csv
  2. This simply calls run\_test.py
  3. It outputs a CSV file in same format as the summaryData ones

