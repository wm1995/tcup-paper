#!/bin/bash
#PBS -lwalltime=00:10:00
#PBS -lselect=1:ncpus=4:mem=1gb
#PBS -N tcup-fixed-tcup-outlier
#PBS -o /rds/general/user/wjm119/home/tcup-paper/run1/hpc-logs/
#PBS -j oe
#PBS -J 1-3

PYTHON=$PBS_O_WORKDIR/.venv/bin/python
MODEL=tcup
DATASET=outlier

export JAX_CHECK_TRACER_LEAKS=1

date

# Load necessary modules
module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0

# Set up directories (if not already set up)
mkdir -p data/fixed/${DATASET}/
mkdir -p results/fixed/${MODEL}/${DATASET}/
if [[ "$DATASET" == "outlier" ]]; then
	mkdir -p data/fixed/normal/
	mkdir -p results/fixed/${MODEL}/normal/
fi

if [[ "$DATASET" == "t" ]]; then
	DATA_FLAG=(--t-dist)
elif [[ "$DATASET" == "outlier" ]]; then
	DATA_FLAG=(--outlier)
elif [[ "$DATASET" == "gaussian_mix" ]]; then
	DATA_FLAG=(--gaussian-mix)
elif [[ "$DATASET" == "laplace" ]]; then
	DATA_FLAG=(--laplace)
fi
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
	${PYTHON} $PBS_O_WORKDIR/scripts/fit_model.py "${MODEL_FLAG[@]}" data/fixed/normal/${PBS_ARRAY_INDEX}.json results/fixed/tcup/normal/${PBS_ARRAY_INDEX}.nc
fi
${PYTHON} $PBS_O_WORKDIR/scripts/fit_model.py "${MODEL_FLAG[@]}" data/fixed/${DATASET}/${PBS_ARRAY_INDEX}.json results/fixed/tcup/${DATASET}/${PBS_ARRAY_INDEX}.nc

# Copy files to final directory
# rsync --recursive data $PBS_O_WORKDIR
diff data/fixed/normal/${PBS_ARRAY_INDEX}.json $PBS_O_WORKDIR/data/fixed/normal/${PBS_ARRAY_INDEX}.json
rsync --recursive results $PBS_O_WORKDIR

date
