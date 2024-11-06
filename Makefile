.PHONY: run

run-simulations:
	# sh src/datasets/create_yule_10_dataset.sh
	sh src/datasets/create_yule_10_fixed_dataset.sh

subsample:
	python src/preprocessing/subsample_datasets.py

calculate-golden-probabilities:
	python src/preprocessing/calculate_golden_probabilities.py

validate-models:
	python src/model_validation/validate_models.py

map-validation:
	python src/map_validation/map_validation.py

run-benchmark:
	python src/model_validation/model_benchmark.py