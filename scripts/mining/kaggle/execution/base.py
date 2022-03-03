import collections
import inspect
import os
import pickle
import textwrap
import traceback
from abc import ABC
from typing import Optional, Dict, Set, Callable, List, Any, Tuple

import git

from scripts.mining.kaggle import nb_utils
from scripts.mining.kaggle.docker_tools.client import DockerShellClient
from scripts.mining.kaggle.execution.result import KaggleExecResult, KaggleExecStatus

_RUNNER_METADATA_KEY = "__databutler_mining_runner"


def register_runner(_method: Callable = None, *, name: str = None):
    """
    Decorates a method to be a runner.
    
    Every runner must have a unique name. A runner must be a classmethod that has the following signature
    
    ```
    @register_runner(name='my_runner')
    def runner(source: str, source_type: nb_utils.KaggleSourceType, output_dir_path: str):
        ...
    ```

    where `source` is the source of the kaggle notebook, `source_type` is the type of source (ipython notebook,
    standard python file etc.), and `output_dir_path` is the path inside the Kaggle container which is synced with a
    directory on the host filesystem. This folder must be used for storing any persistent outputs that need to be
    communicated back to the host.
    """
    if name is None:
        raise ValueError(f"Name cannot be None for a runner")

    def wrapper(method: Callable):
        #  Attach metadata to the function itself for now.
        method.__dict__[_RUNNER_METADATA_KEY] = {
            "name": name
        }
        return method

    return wrapper


class BaseExecutor(ABC):
    """
    Base class for managing the execution of a Kaggle notebook.
    """
    KAGGLE_INPUT_DIR: str = "/kaggle/input"
    KAGGLE_WORKING_DIR: str = "/kaggle/working"

    @classmethod
    def _get_databutler_project_root(cls) -> str:
        """
        Returns the absolute path of the DataButler source.

        NOTE: This only works if the repository is cloned somewhere. This is guaranteed to happen though if you are
        running the scripts.
        """
        return os.path.join(os.path.dirname(__file__), *(os.pardir for _ in range(4)))

    @classmethod
    def _get_docker_client(cls, stdout_path: Optional[str] = None,
                           stderr_path: Optional[str] = None) -> DockerShellClient:
        """
        Creates a docker client using the BASH shell.
        """
        #  The timeout is not for commands that will be run, rather it is the timeout for setting up the client.
        return DockerShellClient(shell='/bin/bash', timeout=None,
                                 stdout_log_path=stdout_path, stderr_log_path=stderr_path)

    @classmethod
    def _download_image_if_not_available(cls, client: DockerShellClient, owner_username: str, kernel_slug: str):
        """
        Checks if image is available and downloads if unavailable.

        Args:
            client: A DockerShellClient instance.
            owner_username: A string corresponding to the username of the owner.
            kernel_slug: A string for the slug of the kernel.
        """
        image = nb_utils.get_docker_image_url(owner_username, kernel_slug)

        if not client.image_exists(image):
            client.pull_image(image, verbose=True)

    @classmethod
    def _get_modified_image_name(cls, owner_username: str, kernel_slug: str):
        """
        Returns a new name to use for the docker image created by running setup commands on the original Kaggle
        docker image for a Kaggle notebook.

        Args:
            owner_username: A string corresponding to the username of the owner.
            kernel_slug: A string for the slug of the kernel.

        Returns:
            (str): A string corresponding to the new image name.

        """
        image_digest = nb_utils.get_docker_image_digest(owner_username, kernel_slug)
        new_image_name = f"databutler-{image_digest}"

        return new_image_name

    @classmethod
    def _setup_image(cls, client: DockerShellClient, owner_username: str, kernel_slug: str):
        """
        Checks if setup commands have been run, and if not, creates a new image by creating a container, running setup,
        and then saving it.

        Args:
            client: A DockerShellClient instance.
            owner_username: A string corresponding to the username of the owner.
            kernel_slug: A string for the slug of the kernel.
        """
        new_image_name = cls._get_modified_image_name(owner_username, kernel_slug)

        if not client.image_exists(new_image_name):
            #  Run the setup commands and save the image.
            client = cls._get_docker_client()
            image = nb_utils.get_docker_image_url(owner_username, kernel_slug)

            container_id = client.create_container(image)

            cls._run_setup_commands(client, container_id)

            client.save_container_to_image(container_id, new_image_name)
            client.remove(container_id)

    @classmethod
    def _run_setup_commands(cls, client: DockerShellClient, container_id: str):
        """
        Sets up a freshly downloaded Kaggle image by running commands common to all Kaggle notebooks.

        This will include extra libraries that need to be installed for databutler, and a development install for
        databutler. Note that doing a development install allows for updating the source later by simply overwriting.

        This method can be overridden by subclasses to do any executor-specific setup.

        NOTE: Setup commands must be agnostic of the source competition of a Kaggle notebook. For example, the setup
        should not copy over data files, as notebooks from different competitions can end up using the same
        docker image.

        Args:
            client: A DockerShellClient instance.
            container_id: A string corresponding to the container ID.
        """
        with open(os.path.join(cls._get_databutler_project_root(), "requirements-kaggle-docker.txt"), "r") as f:
            requirements = f.read()

        client.write_file(container_id, filepath="/requirements.txt", contents=requirements)
        client.exec(container_id, cmd="pip install -r requirements.txt", on_error='raise')
        client.exec(container_id, cmd="rm requirements.txt", on_error='raise')

        #  Setup the kaggle directories as they are common to all notebooks.
        cls._create_kaggle_directories(client, container_id)

        #  Copy databutler and do a development install. The sources can still be updated later!
        cls._copy_databutler_sources(client, container_id)
        cls._install_databutler(client, container_id)

    @classmethod
    def _get_databutler_src_paths(cls) -> Set[str]:
        """
        Return the paths to include as source code of the project.
        Currently, we return all the files under git.
        """
        project_root_path = cls._get_databutler_project_root()
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

    @classmethod
    def _create_kaggle_directories(cls, client: DockerShellClient, container_id: str):
        client.exec(container_id, cmd=f"mkdir -p {cls.KAGGLE_INPUT_DIR}", on_error='raise')
        client.exec(container_id, cmd=f"mkdir -p {cls.KAGGLE_WORKING_DIR}", on_error='raise')

    @classmethod
    def _copy_databutler_sources(cls, client: DockerShellClient, container_id: str):
        client.cp_to_remote(container_id,
                            src_path=cls._get_databutler_project_root(),
                            dst_path=cls.KAGGLE_WORKING_DIR,
                            include=cls._get_databutler_src_paths())

    @classmethod
    def _install_databutler(cls, client: DockerShellClient, container_id: str):
        client.exec(container_id, cmd=f"pip install -e .", workdir=cls.KAGGLE_WORKING_DIR, on_error='raise')

    @classmethod
    def run_notebook(cls,
                     owner_username: str,
                     kernel_slug: str,
                     data_sources: List[nb_utils.KaggleDataSource],
                     output_dir_path: str,
                     timeout: Optional[int] = None) -> KaggleExecResult:

        """
        Runs a kaggle notebook with all the runners belonging to the executor class.

        This routine downloads and sets up the image if necessary, creates a container, copies over data sources,
        and runs the kaggle notebook.

        Args:
            owner_username: A string corresponding to the username of the owner.
            kernel_slug: A string for the slug of the kernel.
            data_sources: A list of data-sources for the notebook.
            output_dir_path: A string corresponding to a path on the host filesystem where all the output resulting
                from the execution of the Kaggle notebook or the corresponding analyses should be stored.
            timeout: Time-out to use for every runner, in seconds. Optional.

        Returns:
            A KaggleExecResult instance containing information about the run, including error details if an exception
            occurred or a timeout happened.
        """
        try:
            return cls._run_notebook_internal(
                owner_username=owner_username,
                kernel_slug=kernel_slug,
                data_sources=data_sources,
                output_dir_path=output_dir_path,
                timeout=timeout
            )

        except Exception as e:
            tb_msg = traceback.format_exc()
            return KaggleExecResult(
                status=KaggleExecStatus.ERROR,
                msg=tb_msg,
            )

    @classmethod
    def _run_notebook_internal(cls,
                               owner_username: str,
                               kernel_slug: str,
                               data_sources: List[nb_utils.KaggleDataSource],
                               output_dir_path: str,
                               timeout: Optional[int] = None) -> KaggleExecResult:
        #  Make sure the output dir exists on the host filesystem.
        os.makedirs(output_dir_path, exist_ok=True)
        #  Its counterpart on the container.
        container_output_path = "/host_output"

        #  Ensure the data_sources are locally available.
        for ds in data_sources:
            if not ds.is_locally_available():
                nb_utils.download_data_source(ds, force=False, quiet=True)

        client = cls._get_docker_client(stdout_path=f"{container_output_path}/stdout.log",
                                        stderr_path=f"{container_output_path}/stderr.log")

        #  Make sure the image is available before making a container.
        cls._download_image_if_not_available(client, owner_username, kernel_slug)
        #  Ensure setup is complete
        cls._setup_image(client, owner_username, kernel_slug)

        #  Create a fresh client for good measure.
        client = cls._get_docker_client(stdout_path=f"{container_output_path}/stdout.log",
                                        stderr_path=f"{container_output_path}/stderr.log")

        #  Initialize a container.

        #  First set up the volume mappings for the datasources and output dir.
        #  We create a separate directory for the datasources, and then symlink all files to the destination.
        #  This way, if the notebook makes changes to the destination, it won't mess with the host filesystem.
        ro_ds_mappings = [f"/host_{idx}" for idx in range(len(data_sources))]

        volumes = {
            **{
                os.path.abspath(ds.local_storage_path): {
                    "bind": vol,
                    "mode": "ro"
                }
                for ds, vol in zip(data_sources, ro_ds_mappings)
            },
            os.path.abspath(output_dir_path): {
                "bind": container_output_path,
                "mode": "rw"
            }
        }

        image = cls._get_modified_image_name(owner_username, kernel_slug)

        with client.create_container_context(image, volumes=volumes) as container_id:
            #  Copy over the data-sources using symlinks
            for ds, vol in zip(data_sources, ro_ds_mappings):
                #  NOTE: `cp -as` only works on linux containers.
                client.exec(container_id, cmd=f"cp -as {vol}/ {cls.KAGGLE_INPUT_DIR}/{ds.mount_slug}/")

            #  Update the source code for databutler. We do not need to install it again.
            cls._copy_databutler_sources(client, container_id)

            #  Finally write the Kaggle notebook source to a file
            source = nb_utils.get_source(owner_username, kernel_slug)
            source_type = nb_utils.get_source_type(owner_username, kernel_slug)

            if source_type == nb_utils.KaggleSourceType.IPYTHON_NOTEBOOK:
                script_name = "kaggle_script.ipynb"
            elif source_type == nb_utils.KaggleSourceType.PYTHON_SOURCE_FILE:
                script_name = "kaggle_script.py"
            else:
                raise NotImplementedError(f"Cannot run source of type {source_type}")

            client.write_file(container_id, filepath=f"{cls.KAGGLE_WORKING_DIR}/{script_name}", contents=source)

            runner_script_src = cls._create_runner_script(output_dir_path=container_output_path,
                                                          script_path=f"{cls.KAGGLE_WORKING_DIR}/{script_name}",
                                                          script_src_type=source_type)

            client.write_file(container_id, filepath=f"{cls.KAGGLE_WORKING_DIR}/kaggle_runner.py",
                              contents=runner_script_src)

            res = client.exec(container_id, cmd=f"python kaggle_runner.py", workdir=cls.KAGGLE_WORKING_DIR,
                              timeout=timeout, on_error='ignore')

            if res.get('timeout', False):
                return KaggleExecResult(
                    status=KaggleExecStatus.TIMEOUT,
                    msg="",
                )

        return KaggleExecResult(
            status=KaggleExecStatus.SUCCESS,
            msg="",
        )

    @classmethod
    def _create_runner_script(cls, output_dir_path: str, script_path: str,
                              script_src_type: nb_utils.KaggleSourceType) -> str:
        """
        Returns the entrypoint code to run in the container. This script invokes the `runner_main` method for the
        target executor class.

        Args:
            output_dir_path: A string corresponding to a path on the host filesystem where all the output resulting
                from the execution of the Kaggle notebook or the corresponding analyses should be stored.
            script_path: A string corresponding to a path on the container filesystem where the contents of the
                original Kaggle notebook are stored.
            script_src_type: An enum member of KaggleSourceType representing the source type of the original Kaggle
                notebook.

        Returns:
`           (str): A string corresponding to code to run.
        """
        class_file = inspect.getabsfile(cls)
        cls_import_path = os.path.relpath(class_file, cls._get_databutler_project_root()).replace("/", ".")
        cls_import_path = ".".join(cls_import_path.split('.')[:-1])  # Remove the .py

        nb_utils_file = os.path.dirname(inspect.getabsfile(nb_utils))  # This is already a module.
        nb_utils_import_path = os.path.relpath(nb_utils_file, cls._get_databutler_project_root()).replace("/", ".")

        return textwrap.dedent(f"""
        from {cls_import_path} import {cls.__name__}
        from {nb_utils_import_path} import nb_utils
        
        with open({script_path!r}, "r") as f_script:
            source = f_script.read()
            
        {cls.__name__}.runner_main(source=source, 
                                   source_type=nb_utils.{str(script_src_type)}, 
                                   output_dir_path={output_dir_path!r})
        """)

    @classmethod
    def _get_cls_runners(cls) -> List[Callable]:
        """
        Helper method to get all the runners defined within the executor using the `register_runner` method.
        """
        runners = []
        for k in dir(cls):
            v = getattr(cls, k)
            if inspect.ismethod(v) and hasattr(v, "__dict__") and _RUNNER_METADATA_KEY in v.__dict__:
                runners.append(v)

        return runners

    @classmethod
    def runner_main(cls, source: str, source_type: nb_utils.KaggleSourceType, output_dir_path: str):
        """
        Runs all the runners defined in the executor.

        This function is run inside the *container* - see `create_runner_script` above.

        Each runner receives the source of the original Kaggle notebook, its type, and a path to the output directory.
        The runner is free to run the source however it wants (and can even skip running it). Thus runners can be
        used to perform program slicing and other analyses on top of the kaggle notebook.

        Args:
            source: A string corresponding to the contents of the original Kaggle notebook.
            source_type: An enum member of KaggleSourceType representing the source type of the original Kaggle
                notebook.
            output_dir_path: A string corresponding to a path on the host filesystem where all the output resulting
                from the execution of the Kaggle notebook or the corresponding analyses should be stored.
        """
        runner_output_dict: Dict[str, Any] = {}

        for runner in cls._get_cls_runners():
            name = runner.__dict__[_RUNNER_METADATA_KEY]["name"]
            runner_output_dict[name] = runner(source, source_type, output_dir_path)

        for name, output in runner_output_dict.items():
            with open(os.path.join(output_dir_path, name), "wb") as f:
                pickle.dump(output, file=f)
