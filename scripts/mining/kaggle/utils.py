import collections
import os
import zipfile
from typing import Callable, Dict

from databutler.utils import paths
import ast

_FIRE_COMMANDS: Dict[str, Dict[str, Callable]] = collections.defaultdict(dict)


def get_working_dir_for_mining() -> str:
    """
    Returns the path of the directory to use for downloading and running notebooks.

    Returns:
        (str): A string corresponding to the path.
    """
    path = os.path.join(paths.get_user_home_dir_path(), ".databutler", "mining", "kaggle")
    os.makedirs(path, exist_ok=True)
    return path


def unzip(zip_path: str, dir_path: str, remove_zip: bool = False):
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dir_path)
    except zipfile.BadZipFile as e:
        raise ValueError(
            'Bad zip file, please report on '
            'www.github.com/kaggle/kaggle-api', e)

    if remove_zip:
        try:
            os.remove(zip_path)
        except OSError as e:
            print('Could not delete zip file, got %s' % e)


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

def is_library_used(code: str, qual_name: str):
    """
    Returns true if a piece of code `code` imports `qual_name`
    """
    c_ast = ast.parse(code)
    print(ast.dump(c_ast))

    vars_to_track = set()
    for node in ast.walk(c_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == qual_name:
                    if alias.asname is None:
                        vars_to_track.add(alias.name)
                    else:
                        vars_to_track.add(alias.asname)

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                full_name = node.module + "." + alias.name
                if full_name == qual_name:
                    vars_to_track.add(alias.asname or full_name)

    for node in ast.walk(c_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in vars_to_track:
            return True

    return False
