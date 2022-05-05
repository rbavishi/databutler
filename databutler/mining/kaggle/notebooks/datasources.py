import glob
import os
from enum import Enum

import attrs

from databutler.mining.kaggle import utils
from databutler.mining.kaggle.notebooks import utils as nb_utils


class KaggleDataSourceType(Enum):
    """
    Type of the data-source associated with a notebook.

    Competition data is data such as that of titanic or house-prices.
    Datasets are those that are uploaded by individuals.
    Kernel outputs are similar to datasets, but are associated with a specific kernel version.
    """
    COMPETITION = 0
    DATASET = 1
    KERNEL_OUTPUT = 2
    UNKNOWN = 3


@attrs.define
class KaggleDataSource:
    """
    Holds information about a data-source.
    """
    #  URL relative to kaggle.com
    url: str
    #  Mount location inside /kaggle/input.
    mount_slug: str
    #  Competition, Dataset or Kernel data.
    src_type: KaggleDataSourceType
    #  Where it will be stored locally, if downloaded.
    local_storage_path: str

    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_storage_path)

    def download(self, force: bool = False, verbose: bool = False):
        if (not force) and self.is_downloaded():
            return

        quiet = (not verbose)
        api = nb_utils.get_kaggle_api()

        if self.src_type == KaggleDataSourceType.COMPETITION:
            competition_slug: str = self.url.split("/")[-1]
            path = self.local_storage_path

            if (not os.path.exists(path)) or force:
                os.makedirs(path, exist_ok=True)
                api.competition_download_files(competition=competition_slug, path=path, force=force, quiet=quiet)

                #  If it is a zip, unzip it.
                outfile = os.path.join(path, competition_slug + '.zip')
                if os.path.exists(outfile):
                    utils.unzip(outfile, path, remove_zip=True)

                #  Unzip any zip inside
                for p in glob.glob(os.path.join(path, "*.zip")):
                    utils.unzip(p, os.path.dirname(p), remove_zip=False)

        elif self.src_type == KaggleDataSourceType.DATASET:
            owner, dataset_slug = self.url.split('/')[1:]

            path = self.local_storage_path
            os.makedirs(path, exist_ok=True)

            api.dataset_download_files(dataset=f"{owner}/{dataset_slug}", path=path, force=force, quiet=quiet,
                                       unzip=True)

        elif self.src_type == KaggleDataSourceType.KERNEL_OUTPUT:
            owner, dataset_slug = self.url.split('/')[1:]

            path = self.local_storage_path
            os.makedirs(path, exist_ok=True)
            api.kernels_pull(kernel=f"{owner}/{dataset_slug}", path=path, quiet=quiet, metadata=False)
            api.kernels_output(kernel=f"{owner}/{dataset_slug}", path=path, force=force, quiet=quiet)

        else:
            raise NotImplementedError(f"Unrecognized data-source {self.url}")
