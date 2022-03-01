import collections
import os
from abc import ABC
from typing import Optional, Dict, Set

import attrs
import git

from databutler.utils import paths
from scripts.mining.kaggle import nb_utils
from scripts.mining.kaggle.execution.result import NotebookExecResult, NotebookExecStatus, NotebookExecErrorType
from scripts.mining.kaggle.docker_tools.client import DockerShellClient


@attrs.define(eq=False, repr=False)
class BaseExecutor():
    """
    Base class for managing the execution of a Kaggle notebook.
    """
    KAGGLE_INPUT_DIR: str = "/kaggle/input"
    KAGGLE_WORKING_DIR: str = "/kaggle/working"

    def _get_databutler_project_root(self) -> str:
        return os.path.join(os.path.dirname(__file__), *(os.pardir for _ in range(4)))

    def _get_docker_client(self) -> DockerShellClient:
        #  The timeout is not for commands that will be run, rather it is the timeout for setting up the client.
        return DockerShellClient(shell='/bin/bash', timeout=60)

    def _download_image_if_not_available(self, client: DockerShellClient, owner_username: str, kernel_slug: str):
        image = nb_utils.get_docker_image(owner_username, kernel_slug)

        if not client.image_exists(image):
            client.pull_image(image, verbose=True)

    def _get_databutler_src_paths(self) -> Set[str]:
        """
        Return the paths to include as source code of the project.
        Currently, we return all the files under git.
        """
        project_root_path = self._get_databutler_project_root()
        file_paths = set(git.Git(project_root_path).ls_files().split('\n'))
        worklist = collections.deque(file_paths)
        file_paths = set()
        while len(worklist) > 0:
            path = worklist.popleft()
            if path in file_paths:
                continue

            file_paths.add(path)
            parent = os.path.dirname(path)
            if parent not in file_paths:
                worklist.append(parent)

        return file_paths

    def run_notebook(self,
                     owner_username: str,
                     kernel_slug: str,
                     data_sources: Dict[str, str],
                     output_dir_path: str,
                     timeout: Optional[int] = None) -> NotebookExecResult:
        """

        Args:
            owner_username:
            kernel_slug:
            data_sources:
            output_dir_path:
            timeout:

        Returns:

        """

        client = self._get_docker_client()
        #  Make sure the image is available before making a container.
        self._download_image_if_not_available(client, owner_username, kernel_slug)

        #  Initialize a container.

        #  First set up the volume mappings for the datasources and output dir.
        #  We create a separate directory for the datasources, and then symlink all files to the destination.
        #  This way, if the notebook makes changes to the destination, it won't mess with the host filesystem.
        ds_mappings = {(k, v): f"/host_{idx}" for idx, (k, v) in enumerate(data_sources.items())}
        container_output_path = "/host_output"

        #  Make sure the output dir exists on the host filesystem.
        os.makedirs(output_dir_path, exist_ok=True)

        volumes = {
            **{
                os.path.abspath(k): {
                    "bind": vol,
                    "mode": "ro"
                }
                for (k, _), vol in ds_mappings.items()
            },
            os.path.abspath(output_dir_path): {
                "bind": container_output_path,
                "mode": "rw"
            }
        }

        image = nb_utils.get_docker_image(owner_username, kernel_slug)
        try:
            container_id = client.create_container(image, volumes=volumes)

        except Exception as e:
            print("FAILURE", e)
            return NotebookExecResult(
                status=NotebookExecStatus.ERROR,
                error_type=NotebookExecErrorType.CONTAINER_START_ERROR,
                msg=str(e)
            )

        #  Make the necessary directories in the container.
        client.exec(container_id, cmd=f"mkdir -p {self.KAGGLE_INPUT_DIR}")
        client.exec(container_id, cmd=f"mkdir -p {self.KAGGLE_WORKING_DIR}")

        #  Copy over the data-sources using symlinks
        for (src, dst), vol in ds_mappings.items():
            #  NOTE: `cp -as` only works on linux containers.
            client.exec(container_id, cmd=f"cp -as {vol}/ {dst}/")

        #  Copy over the source code for databutler.
        #  This is needed to run the script with some of our own code (e.g. when we want to slicing)
        client.cp_to_remote(container_id,
                            src_path=self._get_databutler_project_root(),
                            dst_path=self.KAGGLE_WORKING_DIR,
                            include=self._get_databutler_src_paths())

        #  Finally write the Kaggle notebook source to a file
        source = nb_utils.get_source(owner_username, kernel_slug)
        source_type = nb_utils.get_source_type(owner_username, kernel_slug)

        if source_type == nb_utils.NotebookSourceType.IPYTHON_NOTEBOOK:
            script_name = "kaggle_script.ipynb"
        elif source_type == nb_utils.NotebookSourceType.PYTHON_SOURCE_FILE:
            script_name = "kaggle_script.py"
        else:
            raise NotImplementedError(f"Cannot run source of type {source_type}")

        client.write_file(container_id, filepath=f"{self.KAGGLE_WORKING_DIR}/{script_name}", contents=source)

        print(client.exec(container_id, cmd=f"ls -alh /kaggle > {container_output_path}/trial.txt")['stdout'])
        print(client.exec(container_id, cmd=f"ls -alh /kaggle/input > {container_output_path}/trial.txt")['stdout'])
        print(client.exec(container_id, cmd=f"ls -alh /kaggle/working > {container_output_path}/trial.txt")['stdout'])

        if source_type == nb_utils.NotebookSourceType.IPYTHON_NOTEBOOK:
            print(client.exec(container_id,
                              cmd=(f"jupyter nbconvert --to html --execute "
                                   f"--output {container_output_path}/{script_name}  {script_name}"),
                              workdir=self.KAGGLE_WORKING_DIR))
        else:
            print(client.exec(container_id,
                              cmd=f"python {script_name} > {container_output_path}/{script_name}",
                              workdir=self.KAGGLE_WORKING_DIR))

