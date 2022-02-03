import json
import os
import shutil
import textwrap
from typing import Optional

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from databutler.utils import paths
from databutler.utils.logging import logger


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
    except ValueError:
        return False
    else:
        return True


def _get_settings_path() -> str:
    return os.path.join(paths.get_user_home_dir_path(), ".databutler", "gdrive", "settings.yaml")


def _get_client_secrets_path() -> str:
    return os.path.join(paths.get_user_home_dir_path(), ".databutler", "gdrive", "client_secrets.json")


def _get_credentials_path() -> str:
    return os.path.join(paths.get_user_home_dir_path(), ".databutler", "gdrive", "credentials.yaml")


def _get_success_path() -> str:
    return os.path.join(paths.get_user_home_dir_path(), ".databutler", "gdrive", "SUCCESS")


def _setup_credentials():
    url = "https://medium.com/analytics-vidhya/how-to-connect-google-drive-to-python-using-pydrive-9681b2a14f20"
    if os.path.exists(_get_success_path()):
        return

    while True:
        print(f"Please provide your client_secrets.json contents. Instructions can be found here - {url}")
        secrets = input()
        os.makedirs(os.path.dirname(_get_client_secrets_path()), exist_ok=True)
        with open(_get_client_secrets_path(), "w") as f:
            print(secrets, file=f, end='')

        #  The file needs to exist beforehand, otherwise it won't work.
        with open(_get_credentials_path(), "w") as f:
            pass

        settings_yaml = textwrap.dedent(f"""
        client_config_backend: file
        client_config_file: {_get_client_secrets_path()}
        
        save_credentials: True
        save_credentials_backend: file
        save_credentials_file: {_get_credentials_path()}

        get_refresh_token: True

        oauth_scope:
            - https://www.googleapis.com/auth/drive
            - https://www.googleapis.com/auth/drive.install
        """)

        with open(_get_settings_path(), "w") as f:
            print(settings_yaml, file=f)

        gauth = GoogleAuth(settings_file=_get_settings_path())
        try:
            gauth.LocalWebserverAuth()
        except:
            print("That did not work. Please try again.")
        else:
            #  All good, setup the SUCCESS file to indicate this has worked.
            with open(_get_success_path(), "w") as f:
                pass

            break


def _get_drive() -> GoogleDrive:
    _setup_credentials()
    g_auth = GoogleAuth(settings_file=_get_settings_path())
    g_auth.LoadCredentialsFile(_get_credentials_path())
    drive = GoogleDrive(g_auth)

    return drive


def _get_name(file_or_folder_id: str, drive: Optional[GoogleDrive] = None):
    if drive is None:
        drive = _get_drive()

    obj = drive.CreateFile({"id": file_or_folder_id})
    obj.FetchMetadata(fields="title")
    return obj.metadata["title"]


def download_file(file_id: str, path_dir=".") -> None:
    """
    Downloads a file from Google drive given the file ID. The file ID is the last segment of a Google Drive URL.

    Args:
        file_id: A string representing the ID of the file to download.
        path_dir: Path to directory in which to store the file. Defaults to the current folder.
            The folder is created if it does not exist.
    """
    drive = _get_drive()
    file_obj = drive.CreateFile({"id": file_id})

    #  Fetch the title of the file.
    name = _get_name(file_id, drive)

    #  Make sure the directory exists
    os.makedirs(path_dir, exist_ok=True)

    #  Download contents
    logger.info(f"Downloading `{name}` to `{path_dir}` ...")
    file_obj.GetContentFile(os.path.join(path_dir, name))


def download_folder(folder_id: str, path_dir=".", _indent: int = 0):
    """
    Downloads a folder recursively from Google Drive given the folder ID. The folder ID is the last segment of
    the URL of the folder.

    Args:
        folder_id: A string representing the ID of the file to download.
        path_dir: Path to directory in which to store the file. Defaults to the current folder.
            The folder is created if it does not exist.

        _indent: (Internal) Controls the indentation for the logger to represent folder structure.
    """
    drive = _get_drive()

    #  Get the name of the folder
    name = _get_name(folder_id, drive)

    #  Get a list of all the files
    file_list = drive.ListFile({"q": f"'{folder_id}' in parents and trashed=False"}).GetList()

    #  Make sure the directory exists
    os.makedirs(path_dir, exist_ok=True)
    target_dir = os.path.join(path_dir, name)
    #  Delete the existing contents to avoid potential issues.
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    logger.info(f"{_indent * ' '}Downloading `{name}` to `{path_dir}` ...")
    _indent += 2

    for obj in file_list:
        if obj.metadata["mimeType"].endswith("folder"):
            #  Recursively download directories.
            download_folder(obj.metadata["id"], target_dir, _indent=_indent)
        else:
            logger.info(f"{_indent * ' '}Downloading `{obj.metadata['title']}` to `{target_dir}` ...")
            obj.GetContentFile(os.path.join(target_dir, obj.metadata["title"]))
