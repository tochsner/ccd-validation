.PHONY: run

run-simulations:
	sh src/datasets/create_yule_10_dataset.sh

subsample:
	python src/preprocessing/subsample_datasets.py

calculate-golden-probabilities:
	python src/preprocessing/calculate_golden_probabilities.py

density-validation:
	python src/distribution_validation/density_validation.py

marginals-validation:
	python src/distribution_validation/marginals_validation.py

posterior-ratio-validation:
	python src/distribution_validation/posterior_ratio_validation.py

map-validation:
	python src/map_validation/map_validation.py

validation:
	make density-validation
	make marginals-validation
	make posterior-ratio-validation
	make map-validation
