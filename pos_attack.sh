#!/bin/bash

data_path=data_defense/data/aplc/Wiki10/None_bins_100_neg_10
model_path=models/Wiki10/aplc/None
result_path=data_defense/results/aplc/Wiki10/None_bins_100_pos_10

echo "Preprocesing data"
python ./utils/data_utils.py \
--pos_label 30 \
--top 5 \
--max_neg_samples 10 \
--max_seq_length 512 \
--data_path $data_path \
--model_path $model_path \
--model_type aplc \
--bin_size 100

mkdir -p $result_path

for file in "$data_path"/*;do
  file=${file##*/}
  if [[ "$file" == *"pos"* ]];then
    echo "\nRunning adversarial attacks on $file"
    python -W ignore adv_xmtc.py \
    --data_dir $data_path \
    --mlm_path bert-base-cased \
    --tgt_type aplc \
    --tgt_path $model_path \
    --use_sim_mat 0 \
    --output_path "${result_path}/${file%.csv}.tsv" \
    --task_name Wiki10 \
    --pos_label 30 \
    --use_bpe 0 \
    --k 48 \
    --max_seq_length 512 \
    --threshold_pred_score 0 \
    --top 5 \
    --dev_name $file \
    --attack_type A_pos \
    --change_threshold 0.5 \
    --pos_samples 10
  fi
done
