VENV = venv/
PYTHON = ${VENV}/bin/python
PIP = ${VENV}/bin/pip

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

mcmc: results/ results/*

results:
	mkdir results

results/*: data/*.json scripts/run_models.py
	${PYTHON} scripts/run_models.py

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
