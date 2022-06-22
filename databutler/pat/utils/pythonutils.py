import os
import importlib
import pkgutil

from typing import Type, Set


def get_all_subclasses(cls) -> Set[Type]:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def load_module_complete(mod):
    for m, name, i in pkgutil.iter_modules([os.path.dirname(mod.__file__)]):
        try:
            importlib.import_module(f"{mod.__name__}.{name}")
        except:
            pass
