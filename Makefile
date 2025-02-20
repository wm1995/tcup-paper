# A recipe for recreating the analysis presented in this paper
# Requires GNU Make > 4.3

# Directory for virtual environment
VENV := venv

# List of models to be tested
MODELS := tcup # ncup fixed3 tobs

# SBC dataset parameters
NUM_SBC_DATASETS := 400
SBC_DATASETS := t tobs fixed normal outlier5 outlier10 outlier20 gaussian_mix laplace lognormal
SBC_PLOT_TYPES := alpha_scaled beta_scaled sigma_scaled  # also nu but only for tcup/t

# Fixed run parameters
NUM_FIXED_DATASETS := 400
FIXED_DATASETS = t # normal outlier gaussian_mix laplace lognormal

# Real datasets
REAL_DATASETS := kelly park_FWHM park_line_disp park_MAD

# Linmix datasets
REAL_LINMIX_DATASETS := kelly

################################################################################
# Makefile variables
################################################################################
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip

SBC_RANDOM_SEEDS := $(shell seq ${NUM_SBC_DATASETS})
SBC_DATASET_DEPENDENCIES := scripts/gen_sbc_dataset.py tcup-paper/data/sbc/dataset.py
SBC_DATA_DIRS := $(foreach dataset, ${SBC_DATASETS}, data/sbc/${dataset})
SBC_DATASETS_JSON := $(foreach dir, ${SBC_DATA_DIRS}, $(foreach seed, ${SBC_RANDOM_SEEDS}, ${dir}/${seed}.json))
SBC_MCMC_DIRS := $(foreach dataset, $(SBC_DATASETS), $(foreach model, ${MODELS}, results/sbc/${model}/${dataset}))
SBC_MCMC := $(foreach dir, ${SBC_MCMC_DIRS}, $(foreach seed, ${SBC_RANDOM_SEEDS}, ${dir}/${seed}.nc))
SBC_PLOT_DIRS := $(foreach dataset, $(SBC_DATASETS), $(foreach model, ${MODELS}, plots/sbc/${model}/${dataset}))
SBC_PLOTS := $(foreach dir, ${SBC_PLOT_DIRS}, $(foreach plot, ${SBC_PLOT_TYPES}, ${dir}/${plot}.pdf)) plots/sbc/tcup/t/nu.pdf

FIXED_RANDOM_SEEDS := $(shell seq ${NUM_FIXED_DATASETS})
FIXED_DATASET_DEPENDENCIES := scripts/gen_dataset.py tcup-paper/data/io.py
FIXED_DATA_DIRS := $(foreach dataset, ${FIXED_DATASETS}, data/fixed/${dataset})
FIXED_DATASETS_JSON := $(foreach dir, ${FIXED_DATA_DIRS}, $(foreach seed, ${FIXED_RANDOM_SEEDS}, ${dir}/${seed}.json))
FIXED_MCMC_DIRS := $(foreach dataset, $(FIXED_DATASETS), $(foreach model, ${MODELS}, results/fixed/${model}/${dataset}))
FIXED_MCMC := $(foreach dir, ${FIXED_MCMC_DIRS}, $(foreach seed, ${FIXED_RANDOM_SEEDS}, ${dir}/${seed}.nc))
# FIXED_PLOT_DIRS := $(foreach dataset, $(FIXED_DATASETS), $(foreach model, ${MODELS}, plots/fixed/${model}/${dataset}))
# FIXED_PLOT_TYPES := alpha_scaled beta_scaled.0 beta_scaled.1 sigma_scaled  # also nu but only for tcup/t
# FIXED_PLOTS := $(foreach dir, ${FIXED_PLOT_DIRS}, $(foreach plot, ${FIXED_PLOT_TYPES}, ${dir}/${plot}.pdf))

REAL_DATA_DIRS := data/real data/real/kelly data/real/park
REAL_DATASETS_JSON := $(foreach dataset, ${REAL_DATASETS}, data/real/${dataset}.json)
REAL_MCMC_DIRS := $(foreach model, ${MODELS}, results/real/${model})
REAL_MCMC := $(foreach dir, ${REAL_MCMC_DIRS}, $(foreach dataset, ${REAL_DATASETS}, ${dir}/${dataset}.nc))

REAL_LINMIX_DATASETS_JSON := $(foreach dataset, ${REAL_DATASETS}, data/real/${dataset}.json)
REAL_LINMIX_DIRS := results/real/linmix
REAL_LINMIX := $(foreach dir, ${REAL_LINMIX_DIRS}, $(foreach dataset, ${REAL_LINMIX_DATASETS}, ${dir}/${dataset}.json))
# SBC_PLOT_DIRS := $(foreach dataset, $(SBC_DATASETS), $(foreach model, ${MODELS}, plots/sbc/${model}/${dataset}))
# SBC_PLOTS := $(foreach dir, ${SBC_PLOT_DIRS}, $(foreach plot, ${SBC_PLOT_TYPES}, ${dir}/${plot}.pdf)) plots/sbc/tcup/t/nu.pdf

DATASETS := t normal outlier gaussian_mix laplace lognormal kelly
CORNER_DATASETS := t gaussian_mix laplace lognormal kelly

DATASETS_JSON := $(addsuffix .json, $(addprefix data/, ${DATASETS}))

MCMC :=  $(foreach dataset, $(DATASETS), $(foreach model, ${MODELS}, results/${dataset}_${model}.nc))
CORNER_PLOTS := $(foreach dataset, $(CORNER_DATASETS), plots/corner_${dataset}.pdf) plots/corner_tcup.pdf plots/corner_ncup.pdf

.PHONY = analysis datasets mcmc templates venv clean deep-clean sbc-datasets sbc-mcmc sbc-plots

################################################################################
# Set up Python virtual environment
################################################################################

venv: ${VENV}

${VENV}:
	python -m venv ${VENV}
	${PIP} install -r requirements.txt

################################################################################
# Generate simulation-based calibration datasets
################################################################################

sbc-datasets: ${SBC_DATASETS_JSON}

${SBC_DATA_DIRS}:
	-mkdir -p $@

data/sbc/t/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/t.py | data/sbc/t
	${PYTHON} scripts/gen_sbc_dataset.py --t-dist --seed $*

data/sbc/tobs/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/tobs.py | data/sbc/t
	${PYTHON} scripts/gen_sbc_dataset.py --t-obs --seed $*

data/sbc/fixed/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/t.py | data/sbc/fixed
	${PYTHON} scripts/gen_sbc_dataset.py --fixed 3 --seed $*

data/sbc/normal/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/normal.py | data/sbc/normal
	${PYTHON} scripts/gen_sbc_dataset.py --normal --seed $*

data/sbc/outlier5/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/outlier.py | data/sbc/outlier5
	${PYTHON} scripts/gen_sbc_dataset.py --outlier 5 --seed $*

data/sbc/outlier10/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/outlier.py | data/sbc/outlier10
	${PYTHON} scripts/gen_sbc_dataset.py --outlier 10 --seed $*

data/sbc/outlier20/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/outlier.py | data/sbc/outlier20
	${PYTHON} scripts/gen_sbc_dataset.py --outlier 20 --seed $*

data/sbc/gaussian_mix/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/gaussian_mix.py | data/sbc/gaussian_mix
	${PYTHON} scripts/gen_sbc_dataset.py --gaussian-mix --seed $*

data/sbc/laplace/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/laplace.py | data/sbc/laplace
	${PYTHON} scripts/gen_sbc_dataset.py --laplace --seed $*

data/sbc/lognormal/%.json: ${SBC_DATASET_DEPENDENCIES} tcup-paper/data/sbc/lognormal.py | data/sbc/lognormal
	${PYTHON} scripts/gen_sbc_dataset.py --lognormal --seed $*

################################################################################
# Fit MCMC models to SBC datasets
################################################################################

sbc-mcmc: ${SBC_MCMC}

${SBC_MCMC_DIRS}:
	-mkdir -p $@

results/sbc/tcup/%.nc: data/sbc/%.json ${SBC_MCMC_DEPENDENCIES} | ${SBC_MCMC_DIRS}
	-${PYTHON} scripts/fit_sbc.py $< $@

results/sbc/ncup/%.nc: data/sbc/%.json ${SBC_MCMC_DEPENDENCIES} | ${SBC_MCMC_DIRS}
	-${PYTHON} scripts/fit_sbc.py -n $< $@

results/sbc/fixed3/%.nc: data/sbc/%.json ${SBC_MCMC_DEPENDENCIES} | ${SBC_MCMC_DIRS}
	-${PYTHON} scripts/fit_sbc.py -f 3 $< $@

results/sbc/tobs/%.nc: data/sbc/%.json ${SBC_MCMC_DEPENDENCIES} | ${SBC_MCMC_DIRS}
	-${PYTHON} scripts/fit_sbc.py -o $< $@

################################################################################
# Generate SBC plots
################################################################################

sbc-plots: ${SBC_PLOTS}

${SBC_PLOT_DIRS}:
	-mkdir -p $@

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/t/${plot}.pdf) plots/sbc/tcup/t/nu.pdf &: $(wildcard data/sbc/tcup/t/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --t-dist

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/fixed/${plot}.pdf) &: $(wildcard data/sbc/tcup/fixed/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --fixed-nu

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/normal/${plot}.pdf) &: $(wildcard data/sbc/tcup/normal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --normal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/outlier5/${plot}.pdf) &: $(wildcard data/sbc/tcup/outlier5/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --outlier 5

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/outlier10/${plot}.pdf) &: $(wildcard data/sbc/tcup/outlier10/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --outlier 10

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/outlier20/${plot}.pdf) &: $(wildcard data/sbc/tcup/outlier20/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --outlier 20

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/cauchy_mix/${plot}.pdf) &: $(wildcard data/sbc/tcup/cauchy_mix/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --cauchy-mix

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/gaussian_mix/${plot}.pdf) &: $(wildcard data/sbc/tcup/gaussian_mix/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --gaussian-mix

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/laplace/${plot}.pdf) &: $(wildcard data/sbc/tcup/laplace/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --laplace

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/lognormal/${plot}.pdf) &: $(wildcard data/sbc/tcup/lognormal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --lognormal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tcup/tobs/${plot}.pdf) &: $(wildcard data/sbc/tcup/tobs/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tcup --t-obs

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/t/${plot}.pdf) plots/sbc/tobs/t/nu.pdf &: $(wildcard data/sbc/tcup/t/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --t-dist

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/fixed/${plot}.pdf) &: $(wildcard data/sbc/tobs/fixed/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --fixed-nu

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/normal/${plot}.pdf) &: $(wildcard data/sbc/tobs/normal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --normal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/outlier5/${plot}.pdf) &: $(wildcard data/sbc/tobs/outlier5/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --outlier 5

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/outlier10/${plot}.pdf) &: $(wildcard data/sbc/tobs/outlier10/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --outlier 10

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/outlier20/${plot}.pdf) &: $(wildcard data/sbc/tobs/outlier20/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --outlier 20

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/cauchy_mix/${plot}.pdf) &: $(wildcard data/sbc/tobs/cauchy_mix/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --cauchy-mix

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/gaussian_mix/${plot}.pdf) &: $(wildcard data/sbc/tobs/gaussian_mix/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --gaussian-mix

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/laplace/${plot}.pdf) &: $(wildcard data/sbc/tobs/laplace/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --laplace

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/lognormal/${plot}.pdf) &: $(wildcard data/sbc/tobs/lognormal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --lognormal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/tobs/tobs/${plot}.pdf) &: $(wildcard data/sbc/tobs/tobs/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --tobs --t-obs

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/t/${plot}.pdf) &: $(wildcard data/sbc/ncup/t/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --t-dist

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/fixed/${plot}.pdf) &: $(wildcard data/sbc/ncup/fixed/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --fixed-nu

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/normal/${plot}.pdf) &: $(wildcard data/sbc/ncup/normal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --normal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/outlier5/${plot}.pdf) &: $(wildcard data/sbc/ncup/outlier5/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --outlier 5

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/outlier10/${plot}.pdf) &: $(wildcard data/sbc/ncup/outlier10/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --outlier 10

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/outlier20/${plot}.pdf) &: $(wildcard data/sbc/ncup/outlier20/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --outlier 20

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/gaussian_mix/${plot}.pdf) &: $(wildcard data/sbc/ncup/gaussian_mix/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --gaussian-mix

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/laplace/${plot}.pdf) &: $(wildcard data/sbc/ncup/laplace/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --laplace

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/lognormal/${plot}.pdf) &: $(wildcard data/sbc/ncup/lognormal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --lognormal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/ncup/tobs/${plot}.pdf) &: $(wildcard data/sbc/ncup/tobs/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --ncup --t-obs

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/t/${plot}.pdf) &: $(wildcard data/sbc/fixed3/t/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --t-dist

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/fixed/${plot}.pdf) &: $(wildcard data/sbc/fixed3/fixed/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --fixed-nu

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/normal/${plot}.pdf) &: $(wildcard data/sbc/fixed3/normal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --normal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/outlier5/${plot}.pdf) &: $(wildcard data/sbc/fixed3/outlier5/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --outlier 5

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/outlier10/${plot}.pdf) &: $(wildcard data/sbc/fixed3/outlier10/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --outlier 10

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/outlier20/${plot}.pdf) &: $(wildcard data/sbc/fixed3/outlier20/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --outlier 20

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/gaussian_mix/${plot}.pdf) &: $(wildcard data/sbc/fixed3/gaussian_mix/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --gaussian-mix

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/laplace/${plot}.pdf) &: $(wildcard data/sbc/fixed3/laplace/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --laplace

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/lognormal/${plot}.pdf) &: $(wildcard data/sbc/fixed3/lognormal/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --lognormal

$(foreach plot, ${SBC_PLOT_TYPES}, plots/sbc/fixed3/tobs/${plot}.pdf) &: $(wildcard data/sbc/fixed3/tobs/*.nc) | ${SBC_PLOT_DIRS}
	${PYTHON} scripts/plot_sbc.py --fixed --t-obs

################################################################################
# Generate mock datasets
################################################################################

fixed-datasets: ${FIXED_DATASETS_JSON}
	echo ${FIXED_DATA_DIRS}

${FIXED_DATA_DIRS}:
	-mkdir -p $@

data/fixed/t/%.json: ${FIXED_DATASET_DEPENDENCIES} tcup-paper/data/fixed/t.py | data/fixed/t
	${PYTHON} scripts/gen_dataset.py --t-dist --seed $*

data/fixed/outlier/%.json data/fixed/normal/%.json &: ${FIXED_DATASET_DEPENDENCIES} tcup-paper/data/fixed/outlier.py | data/fixed/outlier data/fixed/normal
	${PYTHON} scripts/gen_dataset.py --outlier --seed $*

data/fixed/gaussian_mix/%.json: ${FIXED_DATASET_DEPENDENCIES} tcup-paper/data/fixed/gaussian_mix.py | data/fixed/gaussian_mix
	${PYTHON} scripts/gen_dataset.py --gaussian-mix --seed $*

data/fixed/laplace/%.json: ${FIXED_DATASET_DEPENDENCIES} tcup-paper/data/fixed/laplace.py | data/fixed/laplace
	${PYTHON} scripts/gen_dataset.py --laplace --seed $*

data/fixed/lognormal/%.json: ${FIXED_DATASET_DEPENDENCIES} tcup-paper/data/fixed/lognormal.py | data/fixed/lognormal
	${PYTHON} scripts/gen_dataset.py --lognormal --seed $*

################################################################################
# Prepare real datasets
################################################################################

real-datasets: ${REAL_DATASETS_JSON}

${REAL_DATA_DIRS}:
	-mkdir -p $@

data/real/kelly.json: scripts/preprocess_Kelly.py plots/
	-mkdir -p data/real/kelly/
	cd data/real/kelly/ && curl https://arxiv.org/e-print/0705.2774 | tar zx f10a.ps f10b.ps
	${PYTHON} scripts/preprocess_Kelly.py

data/real/park_FWHM.json data/real/park_line_disp.json data/real/park_MAD.json &:
	-mkdir -p data/real/park/
	${PYTHON} scripts/preprocess_Park.py

################################################################################
# Fit MCMC models to real datasets
################################################################################

real-mcmc: ${REAL_MCMC}

${REAL_MCMC_DIRS}:
	-mkdir -p $@

results/real/tcup/%.nc: data/real/%.json | ${REAL_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py $< $@

results/real/fixed3/%.nc: data/real/%.json | ${REAL_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py -f 3 $< $@

results/real/ncup/%.nc: data/real/%.json | ${REAL_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py -n $< $@

results/real/tobs/%.nc: data/real/%.json | ${REAL_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py -o $< $@

################################################################################
# Fit linmix to real datasets
################################################################################

real-linmix: linmix ${REAL_LINMIX}

linmix: linmix/Dockerfile linmix/run_mcmc.py
	docker build -t tcup-linmix linmix
	touch linmix

${REAL_LINMIX_DIRS}:
	-mkdir -p $@

results/real/linmix/%.json: data/real/%.json linmix | ${REAL_LINMIX_DIRS}
	docker run \
		--mount type=bind,source=$(shell pwd)/data,target=/mcmc/data \
		--mount type=bind,source=$(shell pwd)/results,target=/mcmc/results \
		--rm tcup-linmix --quiet -r 0 $< $@

################################################################################
# Fit MCMC models to datasets
################################################################################

fixed-mcmc: ${FIXED_MCMC}

${FIXED_MCMC_DIRS}:
	-mkdir -p $@

results/fixed/tcup/%.nc: data/fixed/%.json ${FIXED_MCMC_DEPENDENCIES} | ${FIXED_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py $< $@

results/fixed/fixed3/%.nc: data/fixed/%.json ${FIXED_MCMC_DEPENDENCIES} | ${FIXED_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py -f 3 $< $@

results/fixed/ncup/%.nc: data/fixed/%.json ${FIXED_MCMC_DEPENDENCIES} | ${FIXED_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py -n $< $@

results/fixed/tobs/%.nc: data/fixed/%.json ${FIXED_MCMC_DEPENDENCIES} | ${FIXED_MCMC_DIRS}
	-${PYTHON} scripts/fit_model.py -o $< $@

################################################################################
# Produce plots and analysis
################################################################################

analysis: graphics results/results.csv

graphics: plots/ ${CORNER_PLOTS}
	cp plots/dag.pdf graphics/
	cp plots/pdf_nu.pdf graphics/
	cp plots/cdf_outlier_frac.pdf graphics/
	cp plots/pdf_mixture_schechter.pdf graphics/
	cp plots/pdf_mixture_double_power.pdf graphics/

plots:
	mkdir plots

plots/dag.pdf: scripts/plot_dag.py
	${PYTHON} scripts/plot_dag.py

plots/corner_t.pdf: results/t_tcup.nc results/t_fixed3.nc
	${PYTHON} scripts/plot_corner.py \
	    --dataset data/t.json \
		--mcmc-file results/t_tcup.nc \
		--mcmc-file results/t_fixed3.nc \
		--var-names alpha beta sigma nu \
		--range alpha 2.7 3.6 \
		--range beta 1.82 2.1 \
		--range sigma 0 0.6 \
		--range nu 0 50 \
		--output plots/corner_t.pdf

plots/corner_outlier_ncup.pdf: results/normal_tcup.nc results/outlier_tcup.nc results/outlier_ncup.nc
	${PYTHON} scripts/plot_corner.py \
	    --dataset data/t.json \
		--mcmc-file results/outlier_tcup.nc \
		--mcmc-file results/outlier_ncup.nc \
		--mcmc-file results/normal_ncup.nc \
		--var-names alpha beta sigma \
		--range alpha 0 7 \
		--range beta 1 2.5 \
		--range sigma 0 2 \
		--output plots/corner_outlier_ncup.pdf

plots/corner_outlier_tcup.pdf: results/normal_tcup.nc results/outlier_tcup.nc
	${PYTHON} scripts/plot_corner.py \
	    --dataset data/t.json \
		--mcmc-file results/normal_tcup.nc \
		--mcmc-file results/outlier_tcup.nc \
		--var-names alpha beta sigma outlier_frac \
		--range alpha 0 4 \
		--range beta 1 2.5 \
		--range sigma 0 8 \
		--output plots/corner_outlier_tcup.pdf

plots/corner_gaussian_mix.pdf: results/gaussian_mix_tcup.nc results/gaussian_mix_ncup.nc
	${PYTHON} scripts/plot_corner.py \
	    --dataset data/gaussian_mix.json \
		--mcmc-file results/gaussian_mix_tcup.nc \
		--mcmc-file results/gaussian_mix_ncup.nc \
		--var-names alpha beta sigma outlier_frac \
		--range alpha 1 8 \
		--range beta_0 2.6 5 \
		--range beta_1 -2.8 -0.2 \
		--range sigma 0 3.5 \
		--range outlier_frac 0 0.12 \
		--output plots/corner_gaussian_mix.pdf

plots/corner_kelly.pdf: results/real/linmix/kelly.nc results/real/tcup/kelly.nc
	${PYTHON} scripts/plot_corner.py \
		--mcmc-file results/real/tcup/kelly.nc \
		--mcmc-file results/real/linmix/kelly.nc \
		--var-names alpha beta sigma_68 outlier_frac \
		--range alpha 1 5 \
		--range beta_0 -1 3.9 \
		--range sigma_68 0 0.7 \
		--range outlier_frac 0 0.15 \
		--output plots/corner_kelly.pdf

plots/real/regression_kelly.pdf: results/real/linmix/kelly.nc results/real/tcup/kelly.nc
    ${PYTHON} scripts/plot_regression.py \
        --dataset data/real/kelly.json \
        --tcup-file results/real/tcup/kelly.nc \
        --ncup-file results/real/linmix/kelly.nc \
        --xlim -2.6 0.8 --ylim -0.2 3.4 \
        --xlabel 'Eddington ratio, $\log L / L_{\text{Edd}}$' \
        --ylabel 'X-ray spectral index, $\Gamma$' \
        --output plots/real/regression_kelly.pdf

plots/corner_%.pdf: results/%.nc scripts/plot_corner.py
	${PYTHON} scripts/plot_corner.py

results/results.csv: ${MCMC} scripts/summarise_mcmc.py
	${PYTHON} scripts/summarise_mcmc.py

################################################################################
# Build LaTeX templates from data
################################################################################

templates: datasets.tex

datasets.tex: datasets templates/datasets.tex scripts/build_templates.py
	${PYTHON} scripts/build_templates.py

################################################################################
# Clean up
################################################################################

clean:
	@echo "No cleaning to do"

deep-clean: clean
	rm -r ${VENV}
