VENV := venv
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip

PRIORS := invgamma invgamma2 \
		  cauchy cauchy_scaled cauchy_truncated \
		  F18 F18reparam \
		  nu2 nu2_principled nu2_heuristic nu2_scaled \
		  invnu

DATASETS := linear_1D linear_2D linear_3D

SUFFIXES := 0 1

DATASETS_JSON := $(addsuffix .json, $(addprefix data/, ${DATASETS}))

MCMC :=  $(foreach suffix, $(SUFFIXES), $(foreach dataset, $(DATASETS), $(foreach prior, $(PRIORS), results/${dataset}${suffix}_${prior}.nc)))

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
	curl https://cdsarc.cds.unistra.fr/ftp/J/ApJS/176/355/table1.dat > data/2008ApJS..176..355K_table1.dat
	curl https://cdsarc.cds.unistra.fr/ftp/J/ApJ/657/116/table4.dat > data/2007ApJ...657..116K_table4.dat
	${PYTHON} scripts/preprocess_Kelly.py

################################################################################
# Fit MCMC models to datasets
################################################################################

mcmc: results/ ${MCMC}

results:
	mkdir results

$(foreach prior, ${PRIORS}, results/%_${prior}.nc): data/%.json
	-$(foreach prior, ${PRIORS}, ${PYTHON} scripts/fit_model.py -p ${prior} -e $< results/$*0_${prior}.nc; )
	-${PYTHON} scripts/fit_model.py -n -e $< results/$*0_ncup.nc
	-${PYTHON} scripts/fit_model.py -f 2 -e $< results/$*0_fixed2.nc
	-$(foreach prior, ${PRIORS}, ${PYTHON} scripts/fit_model.py -p ${prior} $< results/$*1_${prior}.nc; )
	-${PYTHON} scripts/fit_model.py -n $< results/$*1_ncup.nc
	-${PYTHON} scripts/fit_model.py -f 2 $< results/$*1_fixed2.nc

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
