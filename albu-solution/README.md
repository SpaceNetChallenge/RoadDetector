**Marathon Match - Solution Description**

**Overview**

1.	Introduction
Tell us a bit about yourself, and why you have decided to participate in the contest.

    Name: Alexander Buslaev
    Handle: albu


**2.** **Solution Development**

How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- **●●** Start from generating dataset because semantic segmentation works on pairs (image, mask). For that, I used a tool from [https://github.com/CosmiQ/apls](https://github.com/CosmiQ/apls) repository.
- **●●** I decided to use only RGB channels because I think for such hard task and so little data lesser amount of features (channels) should be better.
- **●●** To generate masks I used your tool and just drew all roads with width 2m. Also tried 1m and 3m but 2m was better.
- **●●** Scaled RGB channels to the same range in this way: find mean min/max for all tiles per city in train dataset and scaled images to this range.
- **●●** After it, I tried some architectures based on resnet34/resnet50/inceptionv3 encoders and unet-like decoder.
- **●●** The hardest part was to transform probability map to a graph. I found sknw package in github and ran it on binarized probability maps. Most annoying problems were – gaps due to not perfect segmentation (mostly on crossroads), multigraphs, noise, and uncertainty on borders. I will describe solutions for them in next section.

**3.** **Final Approach**

Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

Neural network part:

- Split data to 4 folds randomly but the same number of each city tiles in every fold
- Use resnet34 as encoder and unet-like decoder (conv-relu-upsample-conv-relu) with skip connection from every layer of network. Loss function: 0.8\*binary\_cross\_entropy + 0.2\*(1 – dice\_coeff). Optimizer – Adam with default params.
- Train on image crops 512\*512 with batch size 11 for 30 epoch (8 times more images in one epoch)
- Train 20 epochs with lr 1e-4
- Train 5 epochs with lr 2e-5
- Train 5 epochs with lr 4e-6
- Predict on full image with padding 22 on borders (1344\*1344).
- Merge folds by mean

Image to graph part:

- Take probability map and threshold it, remove some noise - small objects and small holes (lesser than 300 pix area)
- On binary image run skeletonization algorithm to generate thin representation.
- On skeleton image run sknw – transform skeleton to a graph.
- Now it is multi-graph with vertices on crossroads and we need to transform it to graph with straight edges. For each edge I ran approximation algorithm from opencv, so now each edge is represented as a sequence of straight segments, we just need to add them to a graph.
- In addition, there was uncertainty on borders: if tile cut by a road – we have part of road on this tile, but the center of the road could be on another tile. If we would have relative tile positions during a test – we could also use adjacent tiles. However, for this competition, it was not clear if we could use such information in a testing phase, so I just cut border a little to distribute error a bit more randomly.
- Another hint was that skeletonization does not work properly on borders – it generates lines, which go not in the middle of a region as I wish. To fix it I replicated border and skeletonize image only after it. After skeletonization, I restored image size. This error also could be solved by providing adjacent tiles.
- Small hints just to fix obvious errors in a graph - fr all terminal vertices:

\* remove it if it lies on edge lesser then 10 pixels length

\* connect with other if distance to it lesser then 20 pixels

\* connect with other if they lie on almost on one line and distance lesser then 200pix

**4.** **Open Source Resources, Frameworks and Libraries**

Please specify the name of the open source resource along with a URL to where it&#39;s housed and it&#39;s license type:

- tqdm ( [https://pypi.python.org/pypi/tqdm](https://pypi.python.org/pypi/tqdm)), MPLv2, MIT
- numpy ( [https://pypi.python.org/pypi/numpy](https://pypi.python.org/pypi/numpy)), BSD
- pencv-python ( [https://pypi.python.org/pypi/opencv-python](https://pypi.python.org/pypi/opencv-python)), MIT
- matplotlib ( [https://pypi.python.org/pypi/matplotlib](https://pypi.python.org/pypi/matplotlib)), BSD
- scipy ( [https://pypi.python.org/pypi/scipy](https://pypi.python.org/pypi/scipy)), BSD
- scikit-image ( [https://pypi.python.org/pypi/scikit-image](https://pypi.python.org/pypi/scikit-image)), Modified BSD
- scikit-learn ( [https://pypi.python.org/pypi/scikit-learn](https://pypi.python.org/pypi/scikit-learn)), BSD
- tensorboardX ( [https://pypi.python.org/pypi/tensorboardX](https://pypi.python.org/pypi/tensorboardX)), MIT
- pytorch ( [http://pytorch.org/](http://pytorch.org/)), BSD
- torchvision ( [https://pypi.python.org/pypi/torchvision](https://pypi.python.org/pypi/torchvision)), BSD
- GDAL ( [https://anaconda.org/conda-forge/gdal](https://anaconda.org/conda-forge/gdal)), MIT
- Sknw ( [https://github.com/yxdragon/sknw](https://github.com/yxdragon/sknw)) , BSD
- APLS ( [https://github.com/CosmiQ/apls](https://github.com/CosmiQ/apls)) and it&#39;s requirements
- Numba ( [https://pypi.python.org/pypi/numba](https://pypi.python.org/pypi/numba)) BSD
- Pandas ( [https://pypi.python.org/pypi/pandas](https://pypi.python.org/pypi/pandas)), BSD

**5.** **Potential Algorithm Improvements**

Please specify any potential improvements that can be made to the algorithm:

-
  -
    - Use tile adjacency information
    - Find more data
    - Maybe somehow add OSM information
    - Maybe try other network architectures (wideresnet38?)
    - Stack more networks (on different hyperparameters, scales, crops)

**6.** **Algorithm Limitations**

Please specify any potential limitations with the algorithm:

   **●●** It should not generalize to new kinds of data (big difference in weather conditions, zoom, etc); it is limitation for all machine learning algorithms.

**7.** **Deployment Guide**

Please provide the exact steps required to build and deploy the code:

   **1.** Please use steps from Dockerfile. If you use clean system – you also need to install nvidia driver, cuda 8, cudnn 6.

**8.** **Final Verification**

Please provide instructions that explain how to train the algorithm and have it execute against sample data:

   **1.** It&#39;s mostly described in &quot;final verification&quot; document

