#!/bin/bash

MODELS=(tcup ncup fixed tobs)
DATASETS=(t tobs fixed normal outlier5 outlier10 outlier20 gaussian_mix laplace lognormal cauchy_obs laplace_obs mixed_obs)

for model in "${MODELS[@]}"; do
	for dataset in "${DATASETS[@]}"; do
		RUN_NAME=tcup-sbc-${model}-${dataset}
		qsub -N "${RUN_NAME}" -v MODEL=${model},DATASET=${dataset} \
			hpc-scripts/plot_sbc.sh
	done
done
