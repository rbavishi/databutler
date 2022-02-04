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


def get_positional_only_args(*, func: Optional[Callable] = None, sig: Optional[inspect.Signature] = None) -> List[str]:
    """
    Returns the positional-only arguments, if any, in their definition order for the given function.

    This may not always work for builtin functions, especially for older Python versions.

    Args:
        func: Optional; a callable representing the function to be analyzed. Defaults to None.
            If not provided, `sig` must be provided.
        sig: Optional; the signature of the function to be analyzed. Defaults to None.
            The value of `func` is ignored if this argument is provided.

    Returns:
        A list of strings corresponding to the required arguments, in their definition order.

    Raises:
        ValueError: if the signature could not be obtained for the function.
        TypeError: if the provided function is invalid.
    """
    if func is None and sig is None:
        raise ValueError("One of func and sig must be supplied as a keyword arg")

    if sig is None:
        sig = _get_signature(func)

    result: List[str] = []
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            result.append(param.name)

    return result


def get_required_args(*, func: Optional[Callable] = None, sig: Optional[inspect.Signature] = None) -> List[str]:
    """
    Returns the required arguments, in their definition order, for the given function.

    This may not always work for builtin functions, especially for older Python versions.

    Args:
        func: Optional; a callable representing the function to be analyzed. Defaults to None.
            If not provided, `sig` must be provided.
        sig: Optional; the signature of the function to be analyzed. Defaults to None.
            The value of `func` is ignored if this argument is provided.

    Returns:
        A list of strings corresponding to the required arguments, in their definition order.

    Raises:
        ValueError: if the signature could not be obtained for the function.
        TypeError: if the provided function is invalid.
    """
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
    """
    Returns the optional keyword arguments, in their definition order, for the given function.

    This may not always work for builtin functions, especially for older Python versions.

    Args:
        func: Optional; a callable representing the function to be analyzed. Defaults to None.
            If not provided, `sig` must be provided.
        sig: Optional; the signature of the function to be analyzed. Defaults to None.
            The value of `func` is ignored if this argument is provided.

    Returns:
        A list of strings corresponding to the optional arguments, in their definition order.

    Raises:
        ValueError: if the signature could not be obtained for the function.
        TypeError: if the provided function is invalid.
    """
    if func is None and sig is None:
        raise ValueError("One of func and sig must be supplied as a keyword arg")

    if sig is None:
        sig = _get_signature(func)

    result: List[str] = []
    for param in sig.parameters.values():
        #  We are looking for keyword arguments with a default value.
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if param.default is not inspect.Parameter.empty:
                result.append(param.name)

    return result


def get_fully_qualified_name(func: Optional[Callable]) -> Optional[str]:
    """
    Returns the fully qualified name of a function, if applicable, and None otherwise.

    Note that None will be returned for builtin functions, or any function that cannot be traced back to a module.

    Args:
        func: A callable representing the function to be analyzed.

    Returns:
        A string if a fully qualified name could be obtained, and None otherwise.
    """
    if inspect.isroutine(func) and hasattr(func, "__qualname__"):
        try:
            mod = inspect.getmodule(func)
            name = f"{mod.__name__}.{func.__qualname__}"
            return name
        except:
            pass

    return None
