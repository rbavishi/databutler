import fire

from scripts.mining.kaggle import nb_utils
from scripts.mining.kaggle.execution.mpl_seaborn_miner import MplSeabornVizMiner


def run_notebook(owner_username: str, kernel_slug: str):
    executor = MplSeabornVizMiner
    print(executor.run_notebook(owner_username, kernel_slug, nb_utils.get_data_sources(owner_username, kernel_slug),
                                "./kaggle_trial"))


if __name__ == "__main__":
    fire.Fire({
        "run_notebook": run_notebook,
    })
