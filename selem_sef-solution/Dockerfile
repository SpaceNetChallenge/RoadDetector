FROM nvidia/cuda:8.0-cudnn6-devel

MAINTAINER Selim Seferbekov <selim.sef@gmail.com>

ARG TENSORFLOW_VERSION=1.4.1
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=2.1.3

RUN apt-get update && \
    apt-get install -y curl build-essential libpng12-dev libffi-dev \
      	libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libopencv-dev \
		libprotobuf-dev \
		libsnappy-dev \
		protobuf-compiler \
		git \
		 && \
    apt-get clean && \
    rm -rf /var/tmp /tmp /var/lib/apt/lists/*

RUN curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
    bash /installer.sh -b -f && \
    rm /installer.sh

ENV PATH "$PATH:/root/anaconda3/bin"

RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp36-cp36m-linux_x86_64.whl

RUN pip install --no-cache-dir --no-dependencies keras==${KERAS_VERSION}
RUN conda install tqdm
RUN conda install -c conda-forge opencv
RUN pip install git+https://github.com/yxdragon/sknw
RUN pip install pygeoif
RUN pip install shapely
RUN pip install simplification

WORKDIR /work

COPY . /work/


RUN chmod 777 train.sh
RUN chmod 777 test.sh

