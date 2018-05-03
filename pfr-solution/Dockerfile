# Requirements:
# - host machine with AVX2, 4 GPUs for training or 1 GPU for inference, and nvidia-docker
# Important: nvidia-docker must be run with the option --ipc=host

FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install -y awscli bzip2
RUN apt-get install -y python3-pip
RUN apt-get install -y iotop psmisc
RUN apt-get install -y python3-gdal
RUN pip3 install pip --upgrade
RUN pip3 install boto3
RUN pip3 install requests
RUN pip3 install numpy==1.13.3
RUN pip3 install pandas==0.19.2
RUN pip3 install scipy==1.0.0
RUN pip3 install mxnet==0.11.0
RUN pip3 install tqdm
RUN pip3 install Shapely==1.6.3
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision==0.2.0
RUN pip3 install scikit-image==0.13.1
RUN pip3 uninstall pillow -y
RUN apt-get install -y libjpeg-turbo8-dev zlib1g-dev
RUN CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd
RUN pip3 install gdown
RUN apt-get install -y wget unzip

# Only needed for train.sh: download generic pretrained models
WORKDIR /workdir/pretrained
RUN gdown 'https://drive.google.com/uc?id=0B_uPUDq5vVcAdkxOZExBSXI0dlU' && test -e dpn92-extra_2017_08_28.tar.gz
RUN tar xvzf dpn92-extra_2017_08_28.tar.gz

# Only needed for test.sh: download trained models
WORKDIR /workdir/trained_models
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model01.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model02.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model03.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model04.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model05.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model06.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model07.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model08.pth
RUN wget -nv https://s3.amazonaws.com/hello-tc/spacenet3-code/trained_models/model09.pth

# Install the code
WORKDIR /workdir
COPY *.sh /workdir/
COPY code /workdir/code
COPY model /workdir/model
COPY param /workdir/param
RUN chmod +x *.sh

WORKDIR /wdata
WORKDIR /workdir
