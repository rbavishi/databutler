import os
from enum import Enum
from typing import List, Dict, Optional, Set

import attrs

from databutler.pat import astlib
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

    @classmethod
    def from_raw_data(cls, owner: str, slug: str, raw_data: Dict) -> 'KaggleNotebook':
        nb = KaggleNotebook(owner, slug)
        nb._raw_data = raw_data
        return nb

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

    @caching.caching_method
    def get_astlib_ast(self) -> astlib.AstNode:
        """
        Parses the source code using the PAT astlib module.
        """
        if self.source_type == KaggleNotebookSourceType.IPYTHON_NOTEBOOK:
            #  TODO: Handle magics
            return astlib.parse(self.source_code, extension='.ipynb')
        elif self.source_type == KaggleNotebookSourceType.PYTHON_SOURCE_FILE:
            return astlib.parse(self.source_code)
        else:
            raise NotImplementedError(f"Could not recognize source of type {self.source_type}")

    @caching.caching_method
    def is_parseable(self) -> bool:
        """
        Checks if the source is parseable.
        """
        try:
            _ = self.get_astlib_ast()
            return True
        except:
            return False

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

        for ds in nb_data.get("renderableDataSources", []):
            try:
                url = ds["dataSourceUrl"]
                mount_slug = ds["reference"]["mountSlug"]
                native_type = ds["reference"].get("sourceType", "")


                #  Determine the type of data-source using certain heuristics.
                if (url.startswith("/c/") or url.startswith('/competitions/')) and len(url.split('/')) == 3:
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
        """
        Returns whether the notebook is a GPU notebook.
        """
        return self._raw_data["kernelRun"].get("isGpuEnabled", False)

    @caching.cached_property
    def associated_competition(self) -> Optional[str]:
        """
        Get the competition slug the notebook is associated with. Is None if no competition is found.
        """
        #  The notebook data by itself does not contain a field for competitions.
        #  Instead we are going to try to use the data-sources to determine if the notebook is associated with a
        #  competition. We use the heuristic that if the notebook is using a competition's data, it is associated
        #  with that competition.
        found_competitions: List[str] = []
        for ds in self.data_sources:
            if ds.src_type == KaggleDataSourceType.COMPETITION:
                if ds.url.startswith("/c/"):
                    found_competitions.append(ds.url[len("/c/"):])
                elif ds.url.startswith('/competition/'):
                    found_competitions.append(ds.url[len("/competition/"):])

        if len(found_competitions) != 1:
            #  If no competitions found, or more than one competition's data-sources are being used, deem it
            #  unassociated.
            return None

        return found_competitions[0]

    def is_pure_competition_notebook(self) -> bool:
        """
        Checks if the notebook is associated with a competition, and does not use any data-sources apart from the
        competition data-sources.
        """
        return self.associated_competition is not None and len(self.data_sources) == 1

    def was_execution_successful(self) -> bool:
        """
        Checks if the notebook ran successfully on Kaggle
        """
        return self._raw_data["kernelRun"].get("runInfo", {}).get("succeeded", False)

    @caching.cached_property
    def runtime(self) -> Optional[float]:
        """
        Execution time in seconds, if available. None otherwise.
        """
        return self._raw_data["kernelRun"].get("runInfo", {}).get("runTimeSeconds", False)

    @caching.cached_property
    def imported_packages(self) -> List[str]:
        """
        All the external packages (non-relative) imported in the source.

        NOTE: This is not guaranteed to be correct in all circumstances.
        """

        #  Parse the source as an AST.
        code_ast = self.get_astlib_ast()

        package_strs: Set[str] = set()
        for node in astlib.walk(code_ast):
            if isinstance(node, astlib.Import):
                for alias in node.names:
                    #  For every alias, get the top-level module name.
                    name = alias.evaluated_name.split(".")[0]
                    package_strs.add(name)

            elif isinstance(node, astlib.ImportFrom) and node.module is not None and len(node.relative) == 0:
                #  For non-relative imports, get the top-level module name.
                name = astlib.to_code(node.module).split(".")[0]
                package_strs.add(name)

        return sorted(package_strs)

