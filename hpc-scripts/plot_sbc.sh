#!/bin/bash
#PBS -lwalltime=01:30:00
#PBS -lselect=1:ncpus=1:mem=4gb
#PBS -o /rds/general/user/wjm119/home/tcup-paper/run2/hpc-logs/
#PBS -j oe

# NB need to set MODEL and DATASET as environment variables
# MODEL = ( tcup | ncup | fixed | tobs )
# DATASET = ( t | tobs | fixed | normal | outlier5 | outlier10 | \
#             outlier20 | gaussian_mix | laplace | lognormal )

PYTHON=python

echo "=> tcup sbc plot run:"
echo "=> \tmodel=${MODEL}"
echo "=> \tdataset=${DATASET}"
echo "=> \tseed=${PBS_ARRAY_INDEX}"
echo "=> \tstart_time=$(date)"

export PATH=${PATH}:${HOME}/.local/texlive/2024/bin/x86_64-linux

# Define directories
DATA_DIR=data/sbc/${DATASET}/
RESULTS_DIR=results/sbc/${MODEL}/${DATASET}/
PLOTS_DIR=plots/sbc/${MODEL}/${DATASET}/

# Load virtual environment
module load anaconda3/personal
source activate tcup-paper-run2

# Set up directories (if not already set up)
mkdir -p ${DATA_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${PLOTS_DIR}

# Copy data and results to instance
echo "=> Copying files at $(date)"
rsync --recursive ${PBS_O_WORKDIR}/${DATA_DIR} ${DATA_DIR}
rsync --recursive ${PBS_O_WORKDIR}/${RESULTS_DIR} ${RESULTS_DIR}


# Set plotting flags
if [[ "$DATASET" == "t" ]]; then
	DATA_FLAG=(--t-dist)
elif [[ "$DATASET" == "tobs" ]]; then
	DATA_FLAG=(--t-obs)
elif [[ "$DATASET" == "fixed" ]]; then
	DATA_FLAG=(--fixed-nu)
elif [[ "$DATASET" == "normal" ]]; then
	DATA_FLAG=(--normal)
elif [[ "$DATASET" == "outlier5" ]]; then
	DATA_FLAG=(--outlier 5)
elif [[ "$DATASET" == "outlier10" ]]; then
	DATA_FLAG=(--outlier 10)
elif [[ "$DATASET" == "outlier20" ]]; then
	DATA_FLAG=(--outlier 20)
elif [[ "$DATASET" == "gaussian_mix" ]]; then
	DATA_FLAG=(--gaussian-mix)
elif [[ "$DATASET" == "laplace" ]]; then
	DATA_FLAG=(--laplace)
elif [[ "$DATASET" == "lognormal" ]]; then
	DATA_FLAG=(--lognormal)
fi

if [[ "$MODEL" == "tcup" ]]; then
	MODEL_FLAG=(--tcup)
elif [[ "$MODEL" == "ncup" ]]; then
	MODEL_FLAG=(--ncup)
elif [[ "$MODEL" == "fixed" ]]; then
	MODEL_FLAG=(--fixed)
elif [[ "$MODEL" == "tobs" ]]; then
	MODEL_FLAG=(--tobs)
fi

echo "=> Generating plots at $(date)"
${PYTHON} $PBS_O_WORKDIR/scripts/plot_sbc.py "${MODEL_FLAG[@]}" "${DATA_FLAG[@]}"

# Copy files to final directory
echo "=> Copying files at $(date)"
rsync --recursive plots $PBS_O_WORKDIR

echo "=> Finished run at $(date)"
