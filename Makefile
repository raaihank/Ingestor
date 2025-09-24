PYTHON ?= python3

.PHONY: install dev format lint type test build setup-fasttext

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m ruff check --fix .

lint:
	$(PYTHON) -m ruff check .

type:
	mypy ingestor

test:
	pytest -q

build:
	$(PYTHON) -m build

setup-fasttext:
	$(PYTHON) scripts/setup_fasttext.py

