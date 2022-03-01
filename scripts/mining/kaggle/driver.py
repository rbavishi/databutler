import fire

from scripts.mining.kaggle.execution.base import BaseExecutor


def run_notebook(owner_username: str, kernel_slug: str):
    executor = BaseExecutor()
    print(executor.run_notebook(owner_username, kernel_slug, {
        "./kaggle_data/titanic": f"{executor.KAGGLE_INPUT_DIR}/titanic"
    }, "./kaggle_trial"))


if __name__ == "__main__":
    fire.Fire()
