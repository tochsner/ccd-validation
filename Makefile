.PHONY: run

run:
	sh src/datasets/create_yule_10_dataset.sh

build:
	sh src/distribution/create_ccd.sh

subsample:
	python src/datasets/subsample_datasets.py