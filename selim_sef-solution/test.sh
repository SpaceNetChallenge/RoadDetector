#!/usr/bin/env bash
all_args=( $@ )
arg_len=${#all_args[@]}
out_file=${all_args[$arg_len-1]}
city_dirs=${all_args[@]:0:$arg_len-1}

echo "Output file: $out_file"
echo "City dirs $out_file are: $city_dirs"

python3 preprocess_clahe.py --wdata_dir /wdata --dirs_to_process $city_dirs

python3 predict_all.py --gpu "0" --wdata_dir /wdata --dirs_to_process $city_dirs

python3 generate_submission.py --output_file $out_file --dirs_to_process $city_dirs