import os
import pathlib


def get_user_home_dir_path() -> str:
    return str(pathlib.Path.home().absolute())
