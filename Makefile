VENV := venv
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip

DATASETS := t normal gaussian_mix laplace lognormal

DATASETS_JSON := $(addsuffix .json, $(addprefix data/, ${DATASETS}))

MCMC :=  $(foreach dataset, $(DATASETS), results/${dataset}.nc)

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

${DATASETS_JSON}: scripts/gen_data.py
	${PYTHON} scripts/gen_data.py

data/kelly.json:
	-mkdir data/kelly/
	curl https://arxiv.org/e-print/0705.2774 | tar zx f10a.ps f10b.ps
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
	-${PYTHON} scripts/fit_model.py -f 3 $@

results/%_ncup.nc: data/%.json
	-${PYTHON} scripts/fit_model.py -n $< $@

results/normal_tcup.nc results/outlier_tcup.nc &: data/normal.json scripts/fit_normal.py
	-${PYTHON} scripts/fit_normal.py $< results/normal_tcup.nc results/outlier_tcup.nc

results/normal_fixed3.nc results/outlier_fixed3.nc &: data/normal.json scripts/fit_normal.py
	-${PYTHON} scripts/fit_normal.py -f 3 $<  results/normal_fixed3.nc results/outlier_fixed3.nc

results/normal_ncup.nc results/outlier_ncup.nc &: data/normal.json scripts/fit_normal.py
	-${PYTHON} scripts/fit_normal.py -n $< results/normal_ncup.nc results/outlier_ncup.nc

################################################################################
# Produce plots and analysis
################################################################################

analysis: graphics results/results.csv

graphics: plots/ plots/*.pdf
	cp plots/dag.pdf graphics/
	cp plots/pdf_nu.pdf graphics/
	cp plots/cdf_outlier_frac.pdf graphics/

plots:
	mkdir plots

plots/dag.pdf: scripts/plot_dag.py
	${PYTHON} scripts/plot_dag.py

plots/corner_%.pdf: results/%.nc scripts/make_plots.py
	${PYTHON} scripts/make_plots.py

results/results.csv: ${MCMC}
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
