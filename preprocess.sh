#!/bin/bash

data_path=data_defense/data/aplc/Wiki10/None_bins_100_neg_10
model_path=models/Wiki10/aplc/None

python ./utils/data_utils.py \
--pos_label 30 \
--top 5 \
--max_neg_samples 10 \
--max_seq_length 512 \
--data_path $data_path \
--model_path $model_path \
--model_type aplc \
--bin_size 100
