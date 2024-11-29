Required env variables:

- `BEAST` (similar to `/Applications/BEAST 2.7.6` on MacOS)
- `BEAST_PKG` (similar to `/Users/tobiaochsner/Library/Application Support/BEAST/2.7` on MacOS)

Pipeline:

1. MCMC runs should be stored in `data/mcmc_runs`. The BEAST xml and the true tree should be stored in `data/mcmc_config` directory.
2. Run `ThinOutMCMCRuns` to thin out the MCMC runs to ESS trees.

True Tree Density Validation:

1. Run `TrueTreeDensityValidation`.
2. Run `make true-tree-density-validation` to create the plots.

Posterior Ratio Validation:

1. Run `DistributionValidation`.
2. Run `make posterior-ratio-validation` to create the plots.

Marginals Validation:

1. Run `DistributionValidation`.
2. Run `make marginals-validation` to create the plots.

Map Validation:

1. Run `MapValidation`.
2. Run `make map-validation` to create the plots.

Data Likelihood Validation:

1. Run `DistributionValidation`.
2. Run `make likelihood-validation` to create the plots.