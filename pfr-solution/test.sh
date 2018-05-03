#!/bin/bash
set -eu

output_name="${@:$#}"
directories=${@:1:($#-1)}

COMPONENT_MODELS="model01 model02 model03 model04 model05 model06 model07 model08 model09"

bash code/setup_directories.sh $directories
echo "Output path: '$output_name.txt'"
time bash code/provision.sh
for model in $COMPONENT_MODELS; do
    time python3 code/predict.py model/$model.yaml
done
time python3 code/vectorize.py "$COMPONENT_MODELS" test "$output_name".txt
echo "Successfully generated '$output_name.txt'."
bash code/cleanup_directories.sh
