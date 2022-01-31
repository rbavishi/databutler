import os
import subprocess
import sys
from contextlib import contextmanager

from databutler.utils import paths as path_utils


def _get_lib_version_path(lib_name: str, version: str) -> str:
    """
    Get the pre-defined path for the library version.

    Args:
        lib_name (str): Name of the library.
        version (str): The version to use. This must be an exact version - wildcards are not supported as of now.
    """
    path = os.path.join(path_utils.get_user_home_dir_path(), ".databutler", f"{lib_name}-{version}")
    return path


def _check_if_lib_version_exists(lib_name: str, version: str) -> bool:
    """
    Check if the specified library version is already installed at the pre-defined path.

    Args:
        lib_name (str): Name of the library.
        version (str): The version to use. This must be an exact version - wildcards are not supported as of now.
    """
    path = _get_lib_version_path(lib_name, version)
    return os.path.exists(path)


def _download_lib_version(lib_name: str, version: str) -> None:
    """
    Installs the specified version at the pre-defined path for library versions.

    Args:
        lib_name (str): Name of the library.
        version (str): The version to use. This must be an exact version - wildcards are not supported as of now.
    """
    path = _get_lib_version_path(lib_name, version)

    #  As per guidelines at https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
    result = subprocess.check_call([sys.executable, '-m', 'pip', 'install', f"{lib_name}=={version}", '-t', path])
    if result != 0:
        raise RuntimeError(f"Could not install version {version} of library {lib_name}")


@contextmanager
def modified_lib_env(lib_name: str, version: str):
    """
    Modifies sys.path and sys.modules to allow a different library version to be picked up.

    NOTE: This is highly experimental and needs further testing.

    This **should** be run in a different process. Note that the process must be spawned, not forked.
    Since fork is the default on Unix, it might fail on Unix systems if the spawn multiprocess context is not specified.

    For multiple versions, chain the contexts together, as shown below:

    ```
    with modified_lib_env("seaborn", "0.11"), modified_lib_env("pandas", "0.25.0"):
        [your code here]
    ```

    This further means that the order must be carefully chosen by the user. The outer context call libraries will be
    added first to sys.path, and hence the libraries added later will be picked up first (stack model).

    Args:
        lib_name (str): Name of the library.
        version (str): The version to use. This must be an exact version - wildcards are not supported as of now.
    """
    #  First ensure the specific version is downloaded.
    if not _check_if_lib_version_exists(lib_name, version):
        _download_lib_version(lib_name, version)

    #  Modify sys.path temporarily
    orig_sys_path = list(sys.path)
    try:
        lib_path = _get_lib_version_path(lib_name, version)
        sys.path = [lib_path] + list(sys.path)

        #  Get currently active dependencies, and forcefully remove them.
        active = []
        for mod_name in sys.modules.keys():
            #  Note that we need to remove submodules as well, otherwise things won't work.
            key = mod_name.split('.')[0]
            if os.path.exists(os.path.join(lib_path, key)) or \
                    os.path.exists(os.path.join(lib_path, f"{key}.py")):
                active.append(mod_name)

        for i in active:
            sys.modules.pop(i)

        yield

    finally:
        sys.path = orig_sys_path[:]
