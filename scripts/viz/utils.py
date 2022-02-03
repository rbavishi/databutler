import collections
import glob
import os
from typing import Callable, Dict

from databutler.utils import gdriveutils


_FIRE_COMMANDS: Dict[str, Dict[str, Callable]] = collections.defaultdict(dict)


def check_drive_contents_against_path(gdrive_folder_id: str, local_dir_path: str) -> bool:
    """
    Checks if the given local directory exactly matches the specified google drive folder in terms of the directory
    content.

    Args:
        gdrive_folder_id: ID of the google drive folder.
        local_dir_path: Path to the local directory.

    Returns:
        True if the two directories match, and False otherwise.
    """
    if not os.path.exists(local_dir_path):
        return False

    local_dir_paths = [os.path.relpath(p, os.path.dirname(local_dir_path))
                       for p in glob.glob(os.path.join(local_dir_path, "*"))]
    drive_paths = gdriveutils.get_folder_contents(gdrive_folder_id)

    return set(local_dir_paths) == set(drive_paths)


def fire_command(*, name: str = None, collection: str = None):
    if name is None:
        raise ValueError("Name for a fire command cannot be None")

    if collection is None:
        raise ValueError("Collection for a fire command cannot be None")

    def wrapper(func: Callable):
        _FIRE_COMMANDS[collection][name] = func
        return func

    return wrapper


def get_fire_commands_for_collection(collection: str) -> Dict[str, Callable]:
    return _FIRE_COMMANDS[collection]
