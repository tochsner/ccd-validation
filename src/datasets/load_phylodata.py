from pathlib import Path
import shutil
from phylodata import FileType, load_experiments, ExperimentToLoad


def load_phylodata_experiments():
    experiments = load_experiments(
        [
            ExperimentToLoad("tanoyo-2024-systematics-frn3", version=2),
            ExperimentToLoad("mathers-2023-hybridisation-vwt0", version=2),
            ExperimentToLoad("bler-2022-phylogenetic-k54f", version=2),
            ExperimentToLoad("johnson-2021-systematics-dbeu", version=1),
            ExperimentToLoad("pela-2021-subterranean-zego", version=3),
            ExperimentToLoad("near-2021-phylogeny-hiv3", version=2),
            ExperimentToLoad("arbour-2021-little-vfxn", version=3),
            ExperimentToLoad("chaves-2020-evolutionary-w9n2", version=3),
            ExperimentToLoad("brandrud-2019-phylogenomic-73ad", version=1),
            ExperimentToLoad("munro-2019-climate-6tvf", version=2),
            ExperimentToLoad("stervander-2019-origin-1z13", version=3),
            ExperimentToLoad("stange-2018-bayesian-q044", version=1),
            ExperimentToLoad("salariato-2017-climatic-5lpn", version=2),
            ExperimentToLoad("saladin-2017-fossils-lfr1", version=2),
            ExperimentToLoad("salariato-2016-diversification-059x", version=2),
            ExperimentToLoad("v-2016-new-g4qu", version=4),
            ExperimentToLoad("tornabene-2016-repeated-k72b", version=3),
            ExperimentToLoad("jr-2015-multilocus-qbd4", version=3),
            ExperimentToLoad("winger-2015-inferring-g4kz", version=1),
            ExperimentToLoad("morin-2015-geographic-orf9", version=1),
            ExperimentToLoad("colombo-2015-diversity-inb1", version=3),
        ],
        files_to_download=[FileType.POSTERIOR_TREES],
        directory=Path("data/phylodata"),
    )

    # copy them into the data folder structure to be organized in the same way as the yule datasets
    for experiment in experiments:
        trees_file = experiment.get_file_of_type(FileType.POSTERIOR_TREES).local_path

        mcmc_name = f"phylodata-phylodata-{experiment.experiment.human_readable_id.replace('-', '_')}.trees"
        mcmc_file = Path("data/mcmc") / mcmc_name

        if not mcmc_file.exists():
            shutil.copyfile(trees_file, mcmc_file)

    return experiments
