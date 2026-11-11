.PHONY: setup data train test all

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py --n 25000 --seed 42 --out data/raw/discharges.csv

train:
	PYTHONPATH=src python -m readmission.pipeline --data data/raw/discharges.csv --out reports

test:
	python -m pytest -q

all: data train test
