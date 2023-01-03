VENV = venv/
PYTHON = ${VENV}/bin/python
PIP = ${VENV}/bin/pip

.PHONY = templates venv clean deep-clean

venv: ${VENV}

${VENV}:
	python -m venv ${VENV}
	${PIP} install -r requirements.txt

templates: datasets.tex

datasets.tex: templates/datasets.tex scripts/build_templates.py
	${PYTHON} scripts/build_templates.py

clean:
	@echo "No cleaning to do"

deep-clean: clean
	rm -r ${VENV}
