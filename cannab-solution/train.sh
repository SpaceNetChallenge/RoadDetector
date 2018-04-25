echo "Creating masks..."
nohup python create_masks.py masks_small 650 12 "$@" &
nohup python create_masks.py masks_small_9 650 9 "$@" &
nohup python create_masks.py masks_520 520 6 "$@" &
nohup python create_masks.py masks_520_9 520 9 "$@" &
nohup python create_masks.py masks_smallest 325 4 "$@" &
python create_masks.py masks_22 1300 22 "$@"
wait
echo "Masks created"

CUDA_VISIBLE_DEVICES="0" python train_resnet_unet_smallest.py "$@" &> /wdata/resnet_unet_smallest.out &
CUDA_VISIBLE_DEVICES="1" python train_inception_unet_smallest.py "$@" &> /wdata/inception_unet_smallest.out &
CUDA_VISIBLE_DEVICES="2" python train_inception3_unet_520.py "$@" &> /wdata/inception3_unet_520.out &
CUDA_VISIBLE_DEVICES="3" python train_vgg.py "$@" | tee /wdata/vgg_pretrain.out
echo "Waiting all GPUs to complete..."
wait

CUDA_VISIBLE_DEVICES="0" python train_linknet_520.py "$@" &> /wdata/linknet_520.out &
CUDA_VISIBLE_DEVICES="1" python train_inc_v2_unet_520.py "$@" &> /wdata/inc_v2_unet_520.out &
CUDA_VISIBLE_DEVICES="2" python tune_vgg_city.py "$@" &> tee /wdata/vgg_tune.out &
CUDA_VISIBLE_DEVICES="3" python train_linknet_city_big.py "$@" | tee /wdata/linknet_big.out
echo "Waiting all GPUs to complete..."
wait

CUDA_VISIBLE_DEVICES="0" python train_vgg_unet_smallest.py "$@" &> /wdata/gg_unet_smallest.out &
CUDA_VISIBLE_DEVICES="1" python train_vgg2_city_small.py "$@" &> /wdata/vgg2_city_small.out &
CUDA_VISIBLE_DEVICES="2" python train_linknet_city_small.py "$@" &> /wdata/linknet_city_small.out &
CUDA_VISIBLE_DEVICES="3" python train_inception_city_small.py "$@" | tee /wdata/inception_city_small.out
echo "Waiting all GPUs to complete..."
wait

CUDA_VISIBLE_DEVICES="0" python train_resnet_linknet_city_small.py "$@" | tee /wdata/resnet_linknet_city_small.out
echo "All NNs trained!"