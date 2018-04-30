CUDA_VISIBLE_DEVICES="0" python train_inception_unet_smallest_fixed.py "$@" &> /wdata/inception_unet_smallest_fixed.out &
CUDA_VISIBLE_DEVICES="1" python tune_vgg_city_fixed.py "$@" | tee /wdata/linknet_big_fixed.out
echo "Waiting all GPUs to complete..."
wait

echo "All NNs trained!"