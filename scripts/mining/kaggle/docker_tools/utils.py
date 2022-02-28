import io
import os
import tarfile
from typing import Collection


def _tarify_path(path: str, root_name: str,
                 include: Collection[str] = None,
                 exclude: Collection[str] = None) -> bytes:
    """
    Returns a tarball containing the file or the directory (recursively) at the given path as bytes.

    Args:
        path (str): A path to the file or directory to tar.
        root_name: A string corresponding to the root name.
        include (Optional[Collection[str]]): If path is a directory, then only the specified files are included.
            Optional.
        exclude (Optional[Collection[str]]): If path is a directory, then the specified files are excluded. Optional.

    Returns:
        (bytes): Bytes corresponding to the tarball.
    """
    if include is not None:
        include = set(include)
        include.add('')

    if exclude is not None:
        exclude = set(exclude)

    def path_filter(tinfo):
        if include is not None and tinfo.name not in include:
            return None
        if exclude is not None and tinfo.name in exclude:
            return None

        return tinfo

    output_bytes = io.BytesIO()
    tar = tarfile.TarFile(fileobj=output_bytes, mode="w")

    if os.path.isfile(path):
        root_name = os.path.basename(path)

    tar.add(path, arcname=root_name, filter=path_filter)
    tar.close()

    output_bytes.seek(0)
    return output_bytes.getvalue()


def _tarify_contents(file_name: str, contents: str) -> bytes:
    """
    Returns a tarball as a bytestream containing a single file with the given name and contents.

    Args:
        file_name: A string corresponding to the name of the file in the tarball.
        contents: A string corresponding to the Contents of the file in the tarball.

    Returns:
        (bytes): Bytes corresponding to the tarball.
    """
    output_bytes = io.BytesIO()
    tar = tarfile.TarFile(fileobj=output_bytes, mode="w")
    added_file = tarfile.TarInfo(name=file_name)

    encoded_string = str.encode(contents)
    added_file.size = len(encoded_string)
    tar.addfile(added_file, io.BytesIO(encoded_string))
    tar.close()

    output_bytes.seek(0)
    return output_bytes.getvalue()
