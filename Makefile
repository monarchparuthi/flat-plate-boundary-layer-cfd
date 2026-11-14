# Portable GNU Make shortcuts for Windows, macOS and Linux.
# Override with `make test PY=python` if you use a custom interpreter.

ifeq ($(OS),Windows_NT)
PYTHON ?= py -3
PY ?= .venv/Scripts/python.exe
PIP ?= .venv/Scripts/pip.exe
else
PYTHON ?= python3
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
endif

.PHONY: setup run test lint report clean

setup:
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PY) main.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

report:
	pandoc REPORT.md -o REPORT.pdf --pdf-engine=pdflatex

clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '__pycache__', 'tests/__pycache__']]"
