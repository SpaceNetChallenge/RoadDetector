#!/usr/bin/env bash

python3 preprocess_clahe.py --wdata_dir /wdata --dirs_to_process "$@"

python3 train.py \
 --gpu="0" \
 --data_dirs "$@" \
 --network=inception-swish \
 --models_dir trained_models \
 --loss_function=bce_dice\
 --preprocessing_function=tf \
 --freeze_till_layer=input_1 \
 --learning_rate=0.001 \
 --schedule="0.0005:2,0.0001:15,0.00005:20,0.00003:25,0.00001:30" \
 --optimizer=rmsprop \
 --fold="0" \
 --clahe \
 --ohe_city \
 --wdata_dir /wdata \
 --crop_size=384 \
 --crops_per_image=2 \
 --batch_size=4 \
 --epochs=30 \
 --steps_per_epoch=2000 &


# linknet inception with CLAHE
python3 train.py \
 --gpu="1" \
 --data_dirs "$@" \
 --network=linknet_inception \
 --models_dir trained_models \
 --loss_function=bce_dice\
 --preprocessing_function=tf \
 --freeze_till_layer=input_1 \
 --learning_rate=0.001 \
 --schedule="0.0005:2,0.0001:15,0.00005:20,0.00003:25,0.00001:30" \
 --optimizer=rmsprop \
 --fold="0" \
 --clahe \
 --ohe_city \
 --wdata_dir /wdata \
 --crop_size=384 \
 --crops_per_image=2 \
 --batch_size=4 \
 --epochs=30 \
 --steps_per_epoch=2000 &


# linknet inception without CLAHE
python3 train.py \
 --gpu="2" \
 --data_dirs "$@" \
 --network=linknet_inception \
 --models_dir trained_models \
 --loss_function=bce_dice\
 --preprocessing_function=tf \
 --freeze_till_layer=input_1 \
 --learning_rate=0.001 \
 --schedule="0.0005:2,0.0001:15,0.00005:20,0.00003:25,0.00001:30" \
 --optimizer=rmsprop \
 --fold="0" \
 --ohe_city \
 --wdata_dir /wdata \
 --crop_size=384 \
 --crops_per_image=2 \
 --batch_size=4 \
 --epochs=30 \
 --steps_per_epoch=2000 &


# unet inception without clahe with stretch and mean preprocessing
python3 train.py \
 --gpu="3" \
 --data_dirs "$@" \
 --network=inception-unet \
 --models_dir trained_models \
 --loss_function=bce_dice\
 --preprocessing_function=tf \
 --freeze_till_layer=input_1 \
 --learning_rate=0.001 \
 --schedule="0.0005:2,0.0001:15,0.00005:20,0.00003:25,0.00001:30" \
 --optimizer=rmsprop \
 --fold="0" \
 --stretch_and_mean \
 --wdata_dir /wdata \
 --crop_size=384 \
 --crops_per_image=2 \
 --batch_size=4 \
 --epochs=30 \
 --steps_per_epoch=2000 &
 
wait


# linknet resnet
python3 train.py \
 --gpu="0" \
 --data_dirs "$@" \
 --network=linknet_resnet50 \
 --models_dir trained_models \
 --loss_function=bce_dice\
 --preprocessing_function=caffe \
 --freeze_till_layer=input_1 \
 --learning_rate=0.001 \
 --schedule="0.0005:2,0.0001:15,0.00005:20,0.00003:25,0.00001:30" \
 --optimizer=rmsprop \
 --fold="0" \
 --clahe \
 --ohe_city \
 --wdata_dir /wdata \
 --crop_size=384 \
 --crops_per_image=2 \
 --batch_size=4 \
 --epochs=30 \
 --steps_per_epoch=2000 &


# linknet inception with transposed convolutions
python3 train.py \
 --gpu="1" \
 --data_dirs "$@" \
 --network=linknet_inception_lite \
 --models_dir trained_models \
 --loss_function=bce_dice\
 --preprocessing_function=tf \
 --freeze_till_layer=input_1 \
 --learning_rate=0.001 \
 --schedule="0.0005:2,0.0001:15,0.00005:20,0.00003:25,0.00001:30" \
 --optimizer=rmsprop \
 --fold="0" \
 --clahe \
 --ohe_city \
 --wdata_dir /wdata \
 --crop_size=384 \
 --crops_per_image=2 \
 --batch_size=4 \
 --epochs=30 \
 --steps_per_epoch=2000 &
 
wait


cp trained_models/000_all_linknet_inception.h5 trained_models/000_paris_linknet_inception.h5
cp trained_models/000_all_linknet_inception.h5 trained_models/000_vegas_linknet_inception.h5
cp trained_models/000_all_linknet_inception.h5 trained_models/000_shanghai_linknet_inception.h5
cp trained_models/000_all_linknet_inception.h5 trained_models/000_khartoum_linknet_inception.h5

cp trained_models/100_all_linknet_inception.h5 trained_models/100_paris_linknet_inception.h5
cp trained_models/100_all_linknet_inception.h5 trained_models/100_vegas_linknet_inception.h5
cp trained_models/100_all_linknet_inception.h5 trained_models/100_shanghai_linknet_inception.h5
cp trained_models/100_all_linknet_inception.h5 trained_models/100_khartoum_linknet_inception.h5

cp trained_models/010_all_inception-unet.h5 trained_models/010_paris_inception-unet.h5
cp trained_models/010_all_inception-unet.h5 trained_models/010_vegas_inception-unet.h5
cp trained_models/010_all_inception-unet.h5 trained_models/010_shanghai_inception-unet.h5
cp trained_models/010_all_inception-unet.h5 trained_models/010_khartoum_inception-unet.h5

cp trained_models/100_all_inception-swish.h5 trained_models/100_paris_inception-swish.h5
cp trained_models/100_all_inception-swish.h5 trained_models/100_vegas_inception-swish.h5
cp trained_models/100_all_inception-swish.h5 trained_models/100_shanghai_inception-swish.h5
cp trained_models/100_all_inception-swish.h5 trained_models/100_khartoum_inception-swish.h5

cp trained_models/100_all_linknet_inception_lite.h5 trained_models/100_paris_linknet_inception_lite.h5
cp trained_models/100_all_linknet_inception_lite.h5 trained_models/100_vegas_linknet_inception_lite.h5
cp trained_models/100_all_linknet_inception_lite.h5 trained_models/100_shanghai_linknet_inception_lite.h5
cp trained_models/100_all_linknet_inception_lite.h5 trained_models/100_khartoum_linknet_inception_lite.h5

cp trained_models/101_all_linknet_resnet50.h5 trained_models/101_paris_linknet_resnet50.h5
cp trained_models/101_all_linknet_resnet50.h5 trained_models/101_vegas_linknet_resnet50.h5
cp trained_models/101_all_linknet_resnet50.h5 trained_models/101_shanghai_linknet_resnet50.h5
cp trained_models/101_all_linknet_resnet50.h5 trained_models/101_khartoum_linknet_resnet50.h5
