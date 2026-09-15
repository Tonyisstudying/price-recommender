"""Project automation for Phase 1."""

.PHONY: install mock preprocess features test eda

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

mock:
	$(PYTHON) -m src.generate_mock_data

preprocess: mock
	$(PYTHON) -m src.preprocessing

features: preprocess
	$(PYTHON) -m src.features

test:
	$(PYTHON) -m pytest tests -q

eda:
	$(PYTHON) -m jupyter notebook notebooks/01_eda.ipynb
