import os
import pickle
from typing import Any, Sequence, List, Union, Optional, BinaryIO

import attrs

from databutler.utils import lazyobjs


def smart_dump(obj: object, pickle_path: str) -> None:
    """

    Args:
        obj:
        pickle_path:

    Returns:

    """
    with open(pickle_path, "wb") as f:
        pickle.dump(obj, file=f)


def smart_load(pickle_path: str) -> Any:
    """

    Args:
        pickle_path:

    Returns:

    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


class _Unloaded:
    pass


@attrs.define(eq=False, repr=False)
class PickledCollectionWriter:
    path: str
    overwrite_existing: bool = True

    _file_obj: BinaryIO = attrs.field(init=False)
    _offset_map: List[int] = attrs.field(init=False)

    def __attrs_post_init__(self):
        if os.path.exists(self.path):
            if self.overwrite_existing:
                mode = "w"
            else:
                mode = "a"
        else:
            mode = "w"

        self._open(mode)

    def _open(self, mode: str):
        """

        """
        offset_map_path = self.get_offset_map_path(self.path)
        if mode == "a":
            self._file_obj = open(self.path, "ab")
            try:
                with open(offset_map_path, "rb") as f:
                    self._offset_map = pickle.load(f)
            except FileNotFoundError:
                if self._file_obj.tell() != 0:
                    raise FileNotFoundError("Offset map not found for supplied pickle file. "
                                            "Did you write it using PickledCollectionWriter?")
                else:
                    self._offset_map = []

        else:
            self._file_obj = open(self.path, "wb")
            if os.path.exists(offset_map_path):
                os.unlink(offset_map_path)

            self._offset_map = []

    def close(self):
        """

        """
        with open(self.get_offset_map_path(self.path), "wb") as f:
            pickle.dump(self._offset_map, file=f)

        self._file_obj.close()

    def append(self, obj):
        offset = self._file_obj.tell()
        pickle.dump(obj, self._file_obj)
        self._offset_map.append(offset)

    def __len__(self) -> int:
        return len(self._offset_map)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def get_offset_map_path(cls, path: str) -> str:
        return path + ".map"


@attrs.define(eq=False, repr=False)
class PickledCollectionReader(Sequence):
    path: str
    use_cache: bool = False

    _file_obj: BinaryIO = attrs.field(init=False)
    _offset_map: List[int] = attrs.field(init=False)
    _cache: List[Union[Any, _Unloaded]] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self._open()

    def _open(self):
        self._file_obj = open(self.path, "rb")
        with open(PickledCollectionWriter.get_offset_map_path(self.path), "rb") as f:
            self._offset_map = pickle.load(f)

        if self.use_cache:
            self._cache = [_Unloaded for _ in range(len(self._offset_map))]
        else:
            self._cache = []

    def close(self):
        self._file_obj.close()
        self._cache.clear()
        self._offset_map.clear()

    def __getitem__(self, i: int) -> Any:
        try:
            offset = self._offset_map[i]
        except IndexError:
            raise IndexError("Pickled collection index out of range")

        if self.use_cache and self._cache[i] is not _Unloaded:
            return self._cache[i]

        self._file_obj.seek(offset)
        obj = pickle.load(self._file_obj)
        if self.use_cache:
            self._cache[i] = obj

        return obj

    def __len__(self) -> int:
        return len(self._offset_map)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def delete_pickled_collection(path: str):
    if os.path.exists(path):
        os.unlink(path)

    offset_map_path = PickledCollectionWriter.get_offset_map_path(path)
    if os.path.exists(offset_map_path):
        os.unlink(offset_map_path)


@attrs.define(eq=False, repr=False)
class PickledRef(lazyobjs.ObjRef):
    #  Path to pickle file
    path: str
    #  If index is not None, this means the object belongs to a pickled collection (see PickledCollectionReader)
    index: Optional[int] = None

    def resolve(self) -> Any:
        if self.index is None:
            return smart_load(self.path)

        else:
            with PickledCollectionReader(self.path, use_cache=False) as reader:
                return reader[self.index]
