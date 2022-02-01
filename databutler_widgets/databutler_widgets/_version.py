import json
from pathlib import Path

__all__ = ["__version__"]


def _fetchVersion():
    here = Path(__file__).parent.resolve()

    for settings in here.rglob("package.json"):
        try:
            with settings.open() as f:
                return json.load(f)["version"]
        except FileNotFoundError:
            pass

    raise FileNotFoundError(f"Could not find package.json under dir {here!s}")


__version__ = _fetchVersion()
