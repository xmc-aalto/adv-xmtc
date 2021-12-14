#!/bin/bash

result_path=data_defense/results/aplc/Wiki10/None_bins_100_pos_10

python utils/results_all.py --result_path $result_path
python utils/plot_results.py --result_path $result_path
