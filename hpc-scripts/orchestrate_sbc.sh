#!/bin/bash

MODELS=(tcup ncup fixed tobs)
DATASETS=(t tobs fixed normal outlier5 outlier10 outlier20 gaussian_mix laplace lognormal)
REPEATS=400

for model in "${MODELS[@]}"; do
	for dataset in "${DATASETS[@]}"; do
		RUN_NAME=tcup-sbc-${model}-${dataset}
		qsub -N "${RUN_NAME}" -J 1-${REPEATS} \
			-v MODEL=${model},DATASET=${dataset} \
			hpc-scripts/sbc.sh
	done
done
