import attrs

from scripts.mining.kaggle import nb_utils
from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner


@attrs.define(eq=False, repr=False)
class MplSeabornVizMiner(BaseExecutor):
    @classmethod
    @register_runner(name="mpl_seaborn_viz_miner")
    def mining_runner(cls, source: str, source_type: nb_utils.NotebookSourceType):
        return "Hello World"
