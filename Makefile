.PHONY: run

run:
	rm -f data/lphy/*
	rm -f data/beast/*

	sh src/datasets/create_yule_10_dataset.sh