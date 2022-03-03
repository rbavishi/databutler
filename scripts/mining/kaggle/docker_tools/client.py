import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Optional, Collection, List

import attrs
import docker
import docker.models.containers
from docker.errors import ImageNotFound

from scripts.mining.kaggle.docker_tools.exceptions import ContainerStartError, CommandFailedError
from scripts.mining.kaggle.docker_tools.utils import _tarify_contents, _tarify_path
from databutler.utils.logging import logger


@attrs.define(eq=False, repr=False)
class DockerShellClient:
    """
    A convenient wrapper around the docker library offering a shell (such as /bin/bash) and
    high-level abstractions around file transfer shell command execution etc.
    """
    shell: str = '/bin/bash'
    timeout: Optional[int] = None

    stdout_log_path: Optional[str] = None
    stderr_log_path: Optional[str] = None

    _client: docker.DockerClient = attrs.field(init=False, default=None)
    _ids_to_containers: Dict[str, docker.models.containers.Container] = attrs.field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self._client = docker.from_env(timeout=self.timeout)

    def create_container(self, image: str, **kwargs) -> str:
        """
        Creates a container with the supplied image ID and returns the ID of the container as a string.
        Additional kwargs correspond to the arguments to `run` documented at
        https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run
        :param image:
        :param kwargs:
        :return:
        """
        self.pull_image(image)
        container = self._client.containers.run(
            image=image, entrypoint=self.shell, stdin_open=True, detach=True,
            **kwargs
        )
        self._ids_to_containers[container.id] = container
        return container.id

    @contextmanager
    def create_container_context(self, image: str, **kwargs) -> str:
        try:
            container_id = self.create_container(image, **kwargs)
        except:
            raise ContainerStartError(f"Could not start container with image {image}")

        try:
            yield container_id
        finally:
            try:
                self.remove(container_id)
            except:
                pass

    def image_exists(self, image: str) -> bool:
        """
        Checks whether the given image exists locally.

        Args:
            image: A string corresponding to the image ID.
        """
        try:
            self._client.api.inspect_image(image)
            return True
        except ImageNotFound:
            return False
        except:
            return True

    def get_images(self) -> List[Dict]:
        """
        Returns a list of attributes for each of the images available locally.
        """
        return [i.attrs for i in self._client.images.list()]

    def remove_image(self, image: str) -> None:
        """
        Delete local copy of the image.
        """
        self._client.images.remove(image, force=True)

    def pull_image(self, image: str, verbose: bool = False) -> bool:
        """
        Pull image if not available locally.

        If verbose is True, the output of `docker pull` is relayed.

        Args:
            image: A string corresponding to the image ID.
            verbose: If True, the output of `docker pull` is relayed. Defaults to False.

        Returns:
            True if successful else False.
        """
        #  Check if image exists.
        if self.image_exists(image):
            return True

        try:
            if verbose:
                active_pulls = {}
                for line in self._client.api.pull(image, stream=True, decode=True):
                    for _ in active_pulls.keys():
                        sys.stdout.write("\033[1A")
                        sys.stdout.write("\033[0K")

                    if "id" in line:
                        if "progress" in line:
                            active_pulls[line["id"]] = f"{line['status']}: {line['progress']}"
                        elif "status" in line:
                            if line["status"] == "Pull complete" or line["status"] == "Already exists":
                                if line["id"] in active_pulls:
                                    del active_pulls[line["id"]]
                            else:
                                active_pulls[line["id"]] = f"{line['status']}"
                        else:
                            active_pulls[line["id"]] = str(line)

                    for key in sorted(active_pulls.keys()):
                        sys.stdout.write(f"{key} - {active_pulls[key]}\n")
                    sys.stdout.flush()

            else:
                self._client.images.pull(image)

            return True

        except Exception as e:
            logger.warning(f"Could not pull image {image}")
            logger.exception(e)
            return False

    def exec(self, container_id: str, cmd: str, workdir: Optional[str] = None, timeout: Optional[int] = None,
             on_error: str = 'ignore'):
        """
        Runs the supplied command in a shell in the provided container.

        NOTE: If timeout is specified, the result will not contain the stdout and stderr.

        Args:
            container_id (str): A string corresponding to the container ID.
                Note that this is different than the image ID.
            cmd (str): A string representing the command to run.
            workdir (Optional[str]): A string for the path of a directory in which to run the command. Optional.
            timeout (Optional[int]): An integer corresponding to the timeout, in seconds. Optional.
            on_error (str): If 'raise', then non-zero exit-codes trigger an exception. Defaults to 'ignore'.

        Returns:
            A dictionary containing the exit_code, stdout and stderr, and elapsed_time if timeout not specified.
            A dictionary containing only the exit_code and elapsed_time if timeout is specified.
        """
        if container_id not in self._ids_to_containers:
            raise KeyError(f"No container found with id {container_id}")

        if self.stdout_log_path is not None and self.stderr_log_path is not None:
            cmd = f"{self.shell} -c \"({cmd}) 1>>{self.stdout_log_path} 2>>{self.stderr_log_path}\""
        elif self.stdout_log_path is not None:
            cmd = f"{self.shell} -c \"({cmd}) 1>>{self.stdout_log_path}\""
        elif self.stderr_log_path is not None:
            cmd = f"{self.shell} -c \"({cmd}) 2>>{self.stderr_log_path}\""

        container = self._ids_to_containers[container_id]
        if timeout is None:
            start_time = time.time()
            exit_code, (stdout, stderr) = container.exec_run(cmd, demux=True, workdir=workdir)
            elapsed_time = time.time() - start_time

            if on_error == 'raise' and exit_code != 0:
                raise CommandFailedError(f"Command {cmd} failed with exit-code {exit_code}")

            return {
                'exit_code': exit_code,
                'stdout': stdout.decode() if stdout else stdout,
                'stderr': stderr.decode() if stderr else stderr,
                'elapsed_time': elapsed_time,
            }

        else:
            sleep_time: int = timeout // 10
            if sleep_time == 0:
                sleep_time = 1
            elif sleep_time > 10:
                sleep_time = 10

            handle = self._client.api.exec_create(
                container=container_id, cmd=cmd, workdir=workdir
            )

            #  Start the execution
            start_time = time.time()
            self._client.api.exec_start(handle, detach=True)

            #  Keep polling every sleep_time seconds.
            #  NOTE : This solution is better than using signal handlers and the likes as this
            #         way we can adhere to the time-limit perfectly, which is crucial for our use-case.
            exit_code = self._client.api.exec_inspect(handle['Id']).get('ExitCode', None)
            elapsed_time = time.time() - start_time
            while (exit_code is None) and (elapsed_time < timeout):
                time.sleep(sleep_time)
                exit_code = self._client.api.exec_inspect(handle['Id']).get('ExitCode', None)
                elapsed_time = time.time() - start_time

            timeout = (exit_code is None and elapsed_time > timeout)

            if on_error == 'raise' and exit_code is not None and exit_code != 0:
                raise CommandFailedError(f"Command {cmd} failed with exit-code {exit_code}")

            return {
                'exit_code': exit_code,
                'elapsed_time': elapsed_time,
                'timeout': timeout
            }

    def save_container_to_image(self, container_id: str, new_image_name: str):
        """
        Saves the running container into a new image.

        Args:
            container_id (str): A string corresponding to the container ID.
                Note that this is different than the image ID.
            new_image_name (str): A string corresponding to the new image name.
        """
        if container_id not in self._ids_to_containers:
            raise KeyError(f"No container found with id {container_id}")

        container = self._ids_to_containers[container_id]
        container.commit(new_image_name)

    def write_file(self, container_id: str, filepath: str, contents: str):
        """
        Writes a file inside a container.

        Args:
            container_id (str): A string corresponding to the container ID.
                Note that this is different than the image ID.
            filepath (str): A string for the path inside the container where to write the file.
            contents (str): A string corresponding to the entire contents of the file.
        """
        if container_id not in self._ids_to_containers:
            raise KeyError(f"No container found with id {container_id}")

        container = self._ids_to_containers[container_id]
        dirpath = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        container.put_archive(dirpath, _tarify_contents(filename, contents))

    def cp_to_remote(self, container_id: str, src_path: str, dst_path: str,
                     include: Collection[str] = None, exclude: Collection[str] = None):
        """
        Copies a file from the host filesystem to the container.

        Args:
            container_id (str): A string corresponding to the container ID.
                Note that this is different than the image ID.
            src_path (str): Path to the file in the host filesystem to be copied over to the container.
            dst_path (str): Path in the container where to place the file.
            include (Optional[Collection[str]): If src_path is a directory, then only the files listed are copied over.
                Optional.
            exclude (Optional[Collection[str]): If src_path is a directory, the files listed are excluded. Optional.
        """
        if container_id not in self._ids_to_containers:
            raise KeyError(f"No container found with id {container_id}")

        container = self._ids_to_containers[container_id]
        container.put_archive(dst_path, _tarify_path(src_path, root_name='',
                                                     include=include, exclude=exclude))

    def remove(self, container_id: str):
        """
        Removes the given container. This is equivalent to `docker kill -9` followed by `docker rm -f`.

        Args:
            container_id (str): A string corresponding to the container ID.
                Note that this is different than the image ID.
        """

        if container_id not in self._ids_to_containers:
            raise KeyError(f"No container found with id {container_id}")

        container = self._ids_to_containers[container_id]
        self._ids_to_containers.pop(container_id)
        container.kill()
        container.remove(force=True)
