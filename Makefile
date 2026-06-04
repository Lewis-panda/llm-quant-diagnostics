.PHONY: env install test diag-1.5b diag-0.5b compare all clean

env:                ## create the micromamba environment
	micromamba env create -f environment.yml

install:            ## editable install of the package
	pip install -e .

test:               ## run unit tests (CPU-only, no model download)
	pytest -q

diag-1.5b:          ## run the diagnostic on Qwen2.5-1.5B
	python scripts/run_diagnostic.py --model Qwen/Qwen2.5-1.5B

diag-0.5b:          ## run the diagnostic on Qwen2.5-0.5B
	python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B

compare:            ## build the cross-model comparison
	python scripts/compare_models.py results/diagnostic_*.json

all: diag-1.5b diag-0.5b compare   ## full reproduction

clean:              ## remove generated artifacts (keeps source)
	rm -rf results/*.json results/*.md figures/* **/__pycache__ .pytest_cache
