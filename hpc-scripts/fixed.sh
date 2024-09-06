#!/bin/bash
#PBS -lwalltime=01:30:00
#PBS -lselect=1:ncpus=1:mem=4gb
#PBS -o /rds/general/user/wjm119/home/tcup-paper/run2/hpc-logs/
#PBS -j oe

# NB need to set MODEL and DATASET as environment variables
# MODEL = ( tcup | ncup | fixed )
# DATASET = ( t | outlier | gaussian_mix | laplace | lognormal )

PYTHON=python

echo "=> tcup fixed run:"
echo "=> \tmodel=${MODEL}"
echo "=> \tdataset=${DATASET}"
echo "=> \tseed=${PBS_ARRAY_INDEX}"
echo "=> \tstart_time=$(date)"

# Define directories
DATA_DIR=data/fixed/${DATASET}/
RESULTS_DIR=results/fixed/${MODEL}/${DATASET}/

# Load virtual environment
module load anaconda3/personal
source activate tcup-paper-run2

# Set up directories (if not already set up)
mkdir -p ${DATA_DIR}
mkdir -p ${RESULTS_DIR}
if [[ "$DATASET" == "outlier" ]]; then
	mkdir -p data/fixed/normal/
	mkdir -p results/fixed/${MODEL}/normal/
fi

# Generate dataset
if [[ "$DATASET" == "t" ]]; then
	DATA_FLAG=(--t-dist)
elif [[ "$DATASET" == "outlier" ]]; then
	DATA_FLAG=(--outlier)
elif [[ "$DATASET" == "gaussian_mix" ]]; then
	DATA_FLAG=(--gaussian-mix)
elif [[ "$DATASET" == "laplace" ]]; then
	DATA_FLAG=(--laplace)
elif [[ "$DATASET" == "lognormal" ]]; then
	DATA_FLAG=(--lognormal)
fi
echo "=> Generating dataset at $(date)"
${PYTHON} $PBS_O_WORKDIR/scripts/gen_dataset.py "${DATA_FLAG[@]}" --seed ${PBS_ARRAY_INDEX}

if [[ "$MODEL" == "tcup" ]]; then
	MODEL_FLAG=()
elif [[ "$MODEL" == "ncup" ]]; then
	MODEL_FLAG=(-n)
elif [[ "$MODEL" == "fixed" ]]; then
	MODEL_FLAG=(-f 3)
elif [[ "$MODEL" == "tobs" ]]; then
	MODEL_FLAG=(-o)
fi

if [[ "$DATASET" == "outlier" ]]; then
	echo "Running normal MCMC at $(date)"
	${PYTHON} $PBS_O_WORKDIR/scripts/fit_model.py "${MODEL_FLAG[@]}" data/fixed/normal/${PBS_ARRAY_INDEX}.json results/fixed/${MODEL}/normal/${PBS_ARRAY_INDEX}.nc
fi
echo "=> Running MCMC at $(date)"
${PYTHON} $PBS_O_WORKDIR/scripts/fit_model.py "${MODEL_FLAG[@]}" ${DATA_DIR}/${PBS_ARRAY_INDEX}.json ${RESULTS_DIR}/${PBS_ARRAY_INDEX}.nc

# Copy files to final directory
echo "=> Copying files at $(date)"
rsync --recursive data $PBS_O_WORKDIR
rsync --recursive results $PBS_O_WORKDIR

echo "=> Finished run at $(date)"
