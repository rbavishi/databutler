import os
import zipfile

from databutler.utils import paths


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
