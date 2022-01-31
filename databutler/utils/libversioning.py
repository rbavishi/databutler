import os
import subprocess
import sys
from contextlib import contextmanager

from databutler.utils import paths as path_utils


def _get_lib_version_path(lib_name: str, version: str) -> str:
    path = os.path.join(path_utils.get_user_home_dir_path(), ".databutler", f"{lib_name}-{version}")
    return path


def _check_if_lib_version_exists(lib_name: str, version: str) -> bool:
    path = _get_lib_version_path(lib_name, version)
    return os.path.exists(path)


def _download_lib_version(lib_name: str, version: str) -> None:
    path = _get_lib_version_path(lib_name, version)

    #  As per guidelines at https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
    result = subprocess.check_call([sys.executable, '-m', 'pip', 'install', f"{lib_name}=={version}", '-t', path])
    if result != 0:
        raise RuntimeError(f"Could not install version {version} of library {lib_name}")


@contextmanager
def modified_lib_env(lib_name: str, version: str):
    #  First ensure the specific version is downloaded.
    if not _check_if_lib_version_exists(lib_name, version):
        _download_lib_version(lib_name, version)

    #  Modify sys.path temporarily
    orig_sys_path = list(sys.path)
    try:
        sys.path = [_get_lib_version_path(lib_name, version)] + list(sys.path)
        yield

    finally:
        sys.path = orig_sys_path[:]
