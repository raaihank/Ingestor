PYTHON ?= python3

.PHONY: install dev format lint type test build

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

