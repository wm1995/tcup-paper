VENV := venv
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip

PRIORS := invgamma invgamma2 \
		  cauchy cauchy_scaled cauchy_truncated \
		  F18 F18reparam \
		  nu2 nu2_principled nu2_heuristic nu2_scaled \
		  invnu

DATASETS := linear_1D0 linear_1D1 linear_2D0 linear_2D1 linear_3D0 linear_3D1

DATASETS_JSON := $(addsuffix .json, $(addprefix data/, ${DATASETS}))

MCMC :=  $(foreach dataset, $(DATASETS), $(foreach prior, $(PRIORS), results/${dataset}_${prior}.nc))

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

################################################################################
# Fit MCMC models to datasets
################################################################################

mcmc: results/ ${MCMC}

results:
	mkdir results

$(foreach prior, ${PRIORS}, results/%_${prior}.nc): data/%.json
	-$(foreach prior, ${PRIORS}, ${PYTHON} scripts/fit_model.py -p ${prior} $< results/$*_${prior}.nc; )
	-${PYTHON} scripts/fit_model.py -n $< results/$*_ncup.nc
	-${PYTHON} scripts/fit_model.py -f 2 $< results/$*_fixed2.nc

################################################################################
# Produce plots and analysis
################################################################################

analysis: graphics results/results.csv

graphics: plots/ plots/*.pdf
	cp plots/dag.pdf graphics/

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
