# run only on GPU instance and with --ipc=host option
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update --fix-missing && apt-get install -y wget libglib2.0 libsm-dev libxrender-dev libjpeg-dev vim tmux libopenblas-dev libxext-dev
ENV PATH "/miniconda/bin:$PATH"
ENV VERSION 4.3.30
RUN wget https://repo.continuum.io/miniconda/Miniconda3-${VERSION}-Linux-x86_64.sh
RUN chmod +x Miniconda3-${VERSION}-Linux-x86_64.sh
RUN ./Miniconda3-${VERSION}-Linux-x86_64.sh -b -f -p /miniconda
RUN mkdir -p /root/.torch/models
RUN wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -P /root/.torch/models
RUN wget http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
RUN conda install -y GDAL
RUN conda install -y shapely
RUN conda install -y conda=4.4.7
RUN conda install -c conda-forge -y osmnx
RUN pip install torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV LD_LIBRARY_PATH /miniconda/lib:${LD_LIBRARY_PATH}
RUN apt install -y libgl1-mesa-glx
ADD weights /results/weights/

ADD ["train.sh", "test.sh", "train_single.sh", "/"]
ADD src /opt/app/src/
