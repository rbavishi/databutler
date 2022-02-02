import os
import pathlib
import platform
import tempfile


def get_user_home_dir_path() -> str:
    return str(pathlib.Path.home().absolute())


def get_logging_dir_path() -> str:
    return os.path.join(get_user_home_dir_path(), ".databutler", "logs")


def get_tmpdir_path() -> str:
    #  tempfile.gettempdir() returns user-isolated paths on MacOS, which we do not want.
    return "/tmp" if platform.system().lower() == "darwin" else tempfile.gettempdir()
