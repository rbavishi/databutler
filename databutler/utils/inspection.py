import inspect
from typing import Optional, Callable, List


def _get_signature(func: Callable):
    try:
        sig = inspect.signature(func)
    except ValueError:
        raise ValueError("Could not obtain signature for the provided function")
    except TypeError:
        raise TypeError("Signature not supported for the provided object")
    else:
        return sig


def get_required_args(*, func: Optional[Callable] = None, sig: Optional[inspect.Signature] = None) -> List[str]:
    if func is None and sig is None:
        raise ValueError("One of func and sig must be supplied as a keyword arg")

    if sig is None:
        sig = _get_signature(func)

    result: List[str] = []
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind == inspect.Parameter.POSITIONAL_ONLY:
            #  We are looking for positional arguments with a default value.
            if param.default is inspect.Parameter.empty:
                result.append(param.name)

    return result


def get_optional_args(func: Optional[Callable] = None, sig: Optional[inspect.Signature] = None) -> List[str]:
    if func is None and sig is None:
        raise ValueError("One of func and sig must be supplied as a keyword arg")

    if sig is None:
        sig = _get_signature(func)

    result: List[str] = []
    for param in sig.parameters.values():
        #  We are looking for keyword arguments with a default value.
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind == inspect.Parameter.POSITIONAL_ONLY:
            if param.default is not inspect.Parameter.empty:
                result.append(param.name)

    return result


def get_fully_qualified_name(func: Optional[Callable]) -> Optional[str]:
    if inspect.isroutine(func) and hasattr(func, "__qualname__"):
        try:
            mod = inspect.getmodule(func)
            name = f"{mod.__name__}.{func.__qualname__}"
            return name
        except:
            pass

    return None
