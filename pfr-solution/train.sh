#!/bin/bash
set -eu

bash code/setup_directories.sh "$@"
if [ -e trained_models ]; then
    echo "Removing all existing trained models."
    rm -f trained_models/*.pth
fi
time bash code/provision.sh
for yaml in model/*.yaml; do
    time python3 code/train.py "$yaml"
done
echo "Training complete."
bash code/cleanup_directories.sh
