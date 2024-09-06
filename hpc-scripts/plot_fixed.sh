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

echo "=> tcup fixed plot run:"
echo "=> \tstart_time=$(date)"

export PATH=${PATH}:${HOME}/.local/texlive/2024/bin/x86_64-linux

# Define directories
DATA_DIR=data/fixed/
RESULTS_DIR=results/fixed/
PLOTS_DIR=plots/fixed/

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


echo "=> Generating plots at $(date)"
# plots/corner_t.pdf: results/t_tcup.nc results/t_fixed3.nc
${PYTHON} scripts/plot_corner.py \
    --dataset data/fixed/t/1.json \
	--mcmc-file results/fixed/tcup/t/1.nc \
	--mcmc-file results/fixed/fixed/t/1.nc \
	--var-names alpha beta sigma_68 nu \
	--range alpha 2.6 3.4 \
	--range beta_0 1.81 2.19 \
	--range sigma_68 0 0.5 \
	--range nu 0 25 \
	--output plots/fixed/corner_t.pdf

# plots/corner_outlier_ncup.pdf: results/normal_tcup.nc results/outlier_tcup.nc results/outlier_ncup.nc
${PYTHON} scripts/plot_corner.py \
    --dataset data/fixed/outlier/1.json \
	--mcmc-file results/fixed/tcup/outlier/1.nc \
	--mcmc-file results/fixed/ncup/outlier/1.nc \
	--mcmc-file results/fixed/ncup/normal/1.nc \
	--var-names alpha beta sigma_68 \
	--range alpha -2 9 \
	--range beta_0 0.8 2.9 \
	--range sigma_68 0 4 \
	--output plots/fixed/corner_outlier_ncup.pdf

# plots/corner_outlier_tcup.pdf: results/normal_tcup.nc results/outlier_tcup.nc
${PYTHON} scripts/plot_corner.py \
    --dataset data/fixed/outlier/1.json \
	--mcmc-file results/fixed/tcup/normal/1.nc \
	--mcmc-file results/fixed/tcup/outlier/1.nc \
	--var-names alpha beta sigma_68 outlier_frac \
	--range alpha 0.3 4.8 \
	--range beta_0 1.6 2.49 \
	--range sigma_68 0 1.9 \
    --range outlier_frac 0 0.19 \
	--output plots/fixed/corner_outlier_tcup.pdf

# plots/corner_gaussian_mix.pdf: results/gaussian_mix_tcup.nc results/gaussian_mix_ncup.nc
${PYTHON} scripts/plot_corner.py \
    --dataset data/fixed/gaussian_mix/1.json \
	--mcmc-file results/fixed/tcup/gaussian_mix/1.nc \
	--mcmc-file results/fixed/ncup/gaussian_mix/1.nc \
	--var-names alpha beta sigma_68 outlier_frac \
	--range alpha 1.1 3.9 \
	--range beta_0 2.6 3.9 \
	--range beta_1 -1.2 -0.51 \
	--range sigma_68 0 1.8 \
	--range outlier_frac 0 0.17 \
	--output plots/fixed/corner_gaussian_mix.pdf
    
${PYTHON} scripts/plot_regression.py \
	--dataset data/fixed/outlier/1.json \
	--tcup-file results/fixed/tcup/outlier/1.nc \
	--ncup-file results/fixed/ncup/outlier/1.nc \
	--xlim 0.8 8.2 \
	--output plots/fixed/regression_outlier.pdf
    
# plots/corner_t.pdf: results/t_tcup.nc results/t_fixed3.nc
${PYTHON} scripts/plot_corner.py \
	--mcmc-file results/real/tcup/kelly.nc \
	--mcmc-file results/real/linmix/kelly.nc \
	--var-names alpha beta sigma_68 outlier_frac \
	--range alpha 1.8 5.2 \
	--range beta_0 -0.2 3.5 \
	--range sigma_68 0 0.65 \
	--range outlier_frac 0 0.19 \
	--output plots/real/corner_kelly.pdf
    
${PYTHON} scripts/plot_regression.py \
    --dataset data/real/kelly.json \
	--tcup-file results/real/tcup/kelly.nc \
	--ncup-file results/real/linmix/kelly.nc \
	--xlim -2.6 0.8 \
	--ylim -0.2 3.4 \
	--output plots/real/regression_kelly.pdf
    
${PYTHON} scripts/plot_corner.py \
	--mcmc-file results/real/tcup/park_FWHM.nc \
	--var-names alpha beta sigma_68 nu \
	--range alpha 6.7 8.3 \
	--range beta_0 0.21 0.65 \
	--range beta_1 -0.9 1.9 \
	--range sigma_68 0 0.45 \
	--range nu 0 19 \
	--output plots/real/corner_park_fwhm.pdf

# Copy files to final directory
echo "=> Copying files at $(date)"
rsync --recursive plots $PBS_O_WORKDIR

echo "=> Finished run at $(date)"
