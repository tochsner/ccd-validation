set dotenv-load

run-simulations:
	uv run sh src/datasets/create_yule_datasets.sh

run-beast:
	uv run sh src/datasets/run_yule_beast_runs.sh

subsample-ess:
	mkdir -p data/subsampled-to-ess
	java -jar src/jars/SubsampleToESS.jar data/mcmc data/subsampled-to-ess

subsample-ess-hpc:
	module load stack/2024-06 gcc/12.2.0 openjdk/17.0.8.1_1
	mkdir -p data/subsampled-to-ess
	sbatch --time=24:00:00 --mem-per-cpu=4G --cpus-per-task=16 -o data/subsampled-to-ess/out_err/out -e data/subsampled-to-ess/out_err/err --wrap="java -jar src/jars/SubsampleToESS.jar data/mcmc data/subsampled-to-ess"

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
