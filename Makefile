PY := .venv/bin/python

.PHONY: setup db check smoke run reconcile analyze paper test quick all

setup:            ## create .venv and install python deps
	python3 -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt
db:               ## start SQL Server 2022
	(docker compose version >/dev/null 2>&1 && docker compose up -d || docker-compose up -d)
check:            ## pre-flight: goldens deploy, models pulled
	$(PY) scripts/check_env.py
test:             ## offline unit tests
	$(PY) -m pytest -q
smoke:            ## 2 cases x 1 model x 1 rep
	$(PY) scripts/run_experiment.py --cases C01,C08 --models qwen2.5-coder:7b --reps 1
run:              ## full grid from config.yaml (resumable)
	$(PY) scripts/run_experiment.py
reconcile:        ## G4 fault-injection study
	$(PY) scripts/reconcile_experiment.py --rows 2000 --trials 30
analyze:          ## regenerate paper tables, macros, figures
	$(PY) scripts/analyze.py
paper:            ## build the PDF
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
quick:            ## end-to-end quick profile (1 model, 1 rep)
	CONFIG=config.quick.yaml ./run_all.sh
all:              ## everything, full profile
	./run_all.sh
