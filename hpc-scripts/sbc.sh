#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=2gb
#PBS -o /rds/general/user/wjm119/home/tcup-paper/run2/hpc-logs/
#PBS -j oe

# NB need to set MODEL and DATASET as environment variables
# MODEL = ( tcup | ncup | fixed | tobs )
# DATASET = ( t | tobs | fixed | normal | outlier5 | outlier10 | \
#             outlier20 | gaussian_mix | laplace | lognormal )

PYTHON=python

echo "=> tcup sbc run:"
echo "=> \tmodel=${MODEL}"
echo "=> \tdataset=${DATASET}"
echo "=> \tseed=${PBS_ARRAY_INDEX}"
echo "=> \tstart_time=$(date)"

# Define directories
DATA_DIR=data/sbc/${DATASET}/
RESULTS_DIR=results/sbc/${MODEL}/${DATASET}/

# Load virtual environment
module load anaconda3/personal
source activate tcup-paper-run2

# Set up directories (if not already set up)
mkdir -p ${DATA_DIR}
mkdir -p ${RESULTS_DIR}

# Generate dataset
if [[ "$DATASET" == "t" ]]; then
	DATA_FLAG=(--t-dist)
elif [[ "$DATASET" == "tobs" ]]; then
	DATA_FLAG=(--t-obs)
elif [[ "$DATASET" == "fixed" ]]; then
	DATA_FLAG=(--fixed 3)
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
echo "=> Generating dataset at $(date)"
${PYTHON} $PBS_O_WORKDIR/scripts/gen_sbc_dataset.py "${DATA_FLAG[@]}" --seed ${PBS_ARRAY_INDEX}

if [[ "$MODEL" == "tcup" ]]; then
	MODEL_FLAG=()
elif [[ "$MODEL" == "ncup" ]]; then
	MODEL_FLAG=(-n)
elif [[ "$MODEL" == "fixed" ]]; then
	MODEL_FLAG=(-f 3)
elif [[ "$MODEL" == "tobs" ]]; then
	MODEL_FLAG=(-o)
fi

echo "=> Running MCMC at $(date)"
${PYTHON} $PBS_O_WORKDIR/scripts/fit_sbc.py "${MODEL_FLAG[@]}" ${DATA_DIR}/${PBS_ARRAY_INDEX}.json ${RESULTS_DIR}/${PBS_ARRAY_INDEX}.nc

# Copy files to final directory
echo "=> Copying files at $(date)"
rsync --recursive data $PBS_O_WORKDIR
rsync --recursive results $PBS_O_WORKDIR

echo "=> Finished run at $(date)"
