mkdir -p data
mkdir -p data/processed

uv run python src/datasets/load_phylodata.py
java -Xmx16G -jar src/jars/ProcessTrees.jar phylodata data/mcmc data/processed
