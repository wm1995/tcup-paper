#!/bin/bash

MODELS=(tcup ncup fixed)
DATASETS=(t normal outlier gaussian_mix laplace lognormal)
REPEATS=400

# for model in "${MODELS[@]}"; do
# 	for dataset in "${DATASETS[@]}"; do
# 		RUN_NAME=tcup-fixed-${model}-${dataset}
# 		qsub -N "${RUN_NAME}" -J 1-${REPEATS} \
# 			-v MODEL=${model},DATASET=${dataset} \
# 			hpc-scripts/analyse_fixed.sh
# 	done
# done

for model in "${MODELS[@]}"; do
	for dataset in "${DATASETS[@]}"; do
		python scripts/combine_fixed.py --dataset $dataset --model $model
	done
done