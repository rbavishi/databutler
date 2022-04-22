import os

from scripts.mining.kaggle.execution.base import BaseExecutor, register_runner
from scripts.mining.kaggle.notebooks.notebook import KaggleNotebookSourceType

from databutler.pat import astlib


class SimpleExecutor(BaseExecutor):
    """
    Simply runs the Kaggle notebook as is.

    Useful for debugging.
    """

    @classmethod
    @register_runner(name="simple_executor")
    def simple_runner(cls, source: str, source_type: KaggleNotebookSourceType, output_dir_path: str):
        if source_type == KaggleNotebookSourceType.IPYTHON_NOTEBOOK:
            file_name = f'{output_dir_path}/source.ipynb'
            # writing the file to file_name
            with open(file_name, 'w') as f:
                f.write(source)
            print(f'Source written to {file_name}')
            # executing file from terminal
            os.system(f'jupyter nbconvert --execute {file_name} --to html')

        elif source_type == KaggleNotebookSourceType.PYTHON_SOURCE_FILE:
            file_name = f'{output_dir_path}/source.py'
            # writing the file to file_name
            with open(file_name, 'w') as f:
                f.write(source)
            print(f'Source written to {file_name}')
            # executing file from terminal
            os.system(f'python3 {file_name}')

        else:
            raise NotImplementedError(f'Unknown source type {source_type}')
