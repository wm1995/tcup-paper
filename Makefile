VENV := venv
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip

DATASETS := t normal outlier laplace lognormal kelly
MODELS := tcup ncup fixed3
CORNER_DATASETS := t gaussian_mix laplace lognormal kelly

DATASETS_JSON := $(addsuffix .json, $(addprefix data/, ${DATASETS}))

MCMC :=  $(foreach dataset, $(DATASETS), $(foreach model, ${MODELS}, results/${dataset}_${model}.nc))
CORNER_PLOTS := $(foreach dataset, $(CORNER_DATASETS), plots/corner_${dataset}.pdf) plots/corner_tcup.pdf plots/corner_ncup.pdf

.PHONY = analysis datasets mcmc templates venv clean deep-clean

################################################################################
# Set up Python virtual environment
################################################################################

venv: ${VENV}

${VENV}:
	python -m venv ${VENV}
	${PIP} install -r requirements.txt

################################################################################
# Generate mock datasets
################################################################################

datasets: data/ ${DATASETS_JSON}

data:
	mkdir data

data/t.json: tcup-paper/data/t.py
	${PYTHON} scripts/gen_dataset.py --t-dist

data/normal.json data/outlier.json &: tcup-paper/data/outlier.py
	${PYTHON} scripts/gen_dataset.py --outlier

data/gaussian_mix.json: tcup-paper/data/gaussian_mix.py
	${PYTHON} scripts/gen_dataset.py --gaussian-mix

data/laplace.json: tcup-paper/data/laplace.py
	${PYTHON} scripts/gen_dataset.py --laplace

data/lognormal.json: tcup-paper/data/lognormal.py
	${PYTHON} scripts/gen_dataset.py --lognormal

data/kelly.json: scripts/preprocess_Kelly.py
	-mkdir data/kelly/
	cd data/kelly/ && curl https://arxiv.org/e-print/0705.2774 | tar zx f10a.ps f10b.ps
	${PYTHON} scripts/preprocess_Kelly.py

################################################################################
# Fit MCMC models to datasets
################################################################################

mcmc: results/ ${MCMC}

results:
	mkdir results

results/%_tcup.nc: data/%.json
	-${PYTHON} scripts/fit_model.py $< $@

results/%_fixed3.nc: data/%.json
	-${PYTHON} scripts/fit_model.py -f 3 $< $@

results/%_ncup.nc: data/%.json
	-${PYTHON} scripts/fit_model.py -n $< $@

################################################################################
# Produce plots and analysis
################################################################################

analysis: graphics results/results.csv

graphics: plots/ ${CORNER_PLOTS}
	cp plots/dag.pdf graphics/
	cp plots/pdf_nu.pdf graphics/
	cp plots/cdf_outlier_frac.pdf graphics/

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
