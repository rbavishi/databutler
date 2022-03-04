import os
from enum import Enum
from typing import List, Dict

import attrs

from databutler.utils import caching
from scripts.mining.kaggle.notebooks.datasources import KaggleDataSource, KaggleDataSourceType
from scripts.mining.kaggle.notebooks import utils as nb_utils


class KaggleNotebookSourceType(Enum):
    """
    Source type of a Kaggle notebook.
    """
    IPYTHON_NOTEBOOK = 0
    PYTHON_SOURCE_FILE = 1


@attrs.define(eq=False, repr=False, slots=False)
class KaggleNotebook:
    owner: str
    slug: str

    @caching.cached_property
    def _raw_data(self) -> Dict:
        """
        Raw data associated with a notebook as obtained from Kaggle directly.
        """
        return nb_utils.retrieve_notebook_data(self.owner, self.slug)

    @property
    def source_code(self) -> str:
        """
        Source code of the notebook.
        """
        return self._raw_data["kernelBlob"]["source"]

    @property
    def source_type(self) -> KaggleNotebookSourceType:
        """
        Type of source. Can be an IPython notebook or a regular python script.
        """
        source_type = self._raw_data["kernelRun"]["sourceType"]

        if source_type == "EDITOR_TYPE_NOTEBOOK":
            return KaggleNotebookSourceType.IPYTHON_NOTEBOOK
        elif source_type == "EDITOR_TYPE_SCRIPT":
            return KaggleNotebookSourceType.PYTHON_SOURCE_FILE
        else:
            raise ValueError(f"Could not recognize source type {source_type}")

    @property
    def docker_image_digest(self) -> str:
        """
        Digest of the docker image the notebook was run on.
        """
        return self._raw_data["kernelRun"]["runInfo"]["dockerImageDigest"]

    @property
    def docker_image_url(self) -> str:
        """
        GCR Url for the docker image the notebook was run on.
        """
        orig_url = self._raw_data["kernelRun"]["runInfo"].get("dockerHubUrl", "")
        if "@sha256:" in orig_url:
            return orig_url

        #  We'll need to construct it ourselves using the image digest.
        if self.is_gpu_accelerated():
            return f"gcr.io/kaggle-gpu-images/python@sha256:{self.docker_image_digest}"
        else:
            return f"gcr.io/kaggle-images/python@sha256:{self.docker_image_digest}"

    @caching.cached_property
    def data_sources(self) -> List[KaggleDataSource]:
        """
        All the data-sources used by the notebook.
        """
        nb_data = self._raw_data
        result: List[KaggleDataSource] = []
        ds_root = nb_utils.get_local_datasources_storage_path()

        for ds in nb_data["renderableDataSources"]:
            try:
                url = ds["dataSourceUrl"]
                mount_slug = ds["reference"]["mountSlug"]
                native_type = ds["reference"].get("sourceType", "")

                #  Determine the type of data-source using certain heuristics.
                if url.startswith("/c/") and len(url.split('/')) == 3:
                    #  If it starts with "/c/", it is bound to be a competition data-source.
                    src_type = KaggleDataSourceType.COMPETITION
                    local_storage_path = os.path.join(ds_root, "c", url.split("/")[-1])

                elif url.startswith("/") and len(url.split('/')) == 3 and "DATASET_VERSION" in native_type:
                    #  We use the JSON extracted from the notebook's webpage.
                    src_type = KaggleDataSourceType.DATASET
                    local_storage_path = os.path.join(ds_root, "d", *url.split("/")[1:])

                elif url.startswith("/") and len(url.split('/')) == 3 and "KERNEL_VERSION" in native_type:
                    #  We use the JSON extracted from the notebook's webpage.
                    src_type = KaggleDataSourceType.KERNEL_OUTPUT
                    local_storage_path = os.path.join(ds_root, "k", *url.split("/")[1:])

                else:
                    src_type = KaggleDataSourceType.UNKNOWN
                    local_storage_path = ""

                result.append(KaggleDataSource(
                    url=url,
                    mount_slug=mount_slug,
                    src_type=src_type,
                    local_storage_path=local_storage_path,
                ))

            except KeyError:
                pass

        return result

    def is_gpu_accelerated(self) -> bool:
        return self._raw_data["kernelRun"].get("isGpuEnabled", False)
