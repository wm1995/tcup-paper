VENV := venv/
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip

PRIORS := invgamma invgamma2 \
		  cauchy cauchy_scaled cauchy_truncated \
		  F18 F18reparam \
		  nu2 nu2_principled nu2_heuristic nu2_scaled \
		  invnu

DATASETS := linear_1D0 linear_1D1 linear_2D0 linear_2D1 linear_3D0 linear_3D1

MCMC :=  $(foreach dataset, $(DATASETS), $(foreach prior, $(PRIORS), results/${dataset}_${prior}.nc))

.PHONY = datasets mcmc templates venv clean deep-clean

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

datasets: data/ data/*.json

data:
	mkdir data

data/*.json: scripts/gen_data.py
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
# Produce plots
################################################################################

graphics: plots/ plots/*.pdf
	cp plots/* graphics/*

plots:
	mkdir plots

plots/*.pdf: results/* scripts/run_models.py
	${PYTHON} scripts/run_models.py

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
