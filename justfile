set dotenv-load

run-simulations:
	uv run sh src/datasets/create_yule_datasets.sh

run-beast:
	uv run sh src/datasets/run_yule_beast_runs.sh

subsample:
	python src/preprocessing/subsample_datasets.py

calculate-golden-probabilities:
	python src/preprocessing/calculate_golden_probabilities.py

true-tree-density-validation:
	python src/distribution_validation/true_tree_density_validation.py

marginals-validation:
	python src/distribution_validation/marginals_validation.py

posterior-ratio-validation:
	python src/distribution_validation/posterior_ratio_validation.py

map-validation:
	python src/map_validation/map_validation.py

likelihood-validation:
	python src/distribution_validation/data_likelihood_validation.py

goodness-of-fit-validation:
	python src/distribution_validation/goodness_of_fit_validation.py

num-parameters-analysis:
	python src/distribution_validation/num_parameters_analysis.py

coverage-validation:
	python src/matrix_validation/coverage_validation.py

validation:
	make true-tree-density-validation
	make marginals-validation
	make posterior-ratio-validation
	make map-validation
	make likelihood-validation

train:
	python src/ml/run.py

tune:
	python src/ml/hyperparam_tuning.py

ml-flow-ui:
	mlflow ui
