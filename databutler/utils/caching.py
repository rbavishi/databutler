"""Utilities for managing project cache as well as caches for functions"""


class _Missing:
    def __reduce__(self):
        return "_missing"

    def __repr__(self):
        return "no value"


_missing = _Missing()


def caching_method(method):
    """
    Caching decorator for a class method. The method *cannot* be static or be a classmethod.
    That is, its first argument needs to be self. The cache dictionary itself is attached to the object itself (self),
    so objects are cleaned up automatically.
    :param method:
    :return:
    """

    def wrapper(self, *args, **kwargs):
        memory_key = f"____method_cache_var{method.__name__}"
        if not hasattr(self, memory_key):
            setattr(self, memory_key, {})

        memory = getattr(self, memory_key)
        if len(kwargs) == 0:
            data_key = args
        else:
            data_key = (args, tuple(sorted(kwargs.items())))

        if data_key in memory:
            return memory[data_key]
        else:
            memory[data_key] = result = method(self, *args, **kwargs)
            return result

    wrapper.__name__ = method.__name__
    wrapper.__module__ = method.__module__
    wrapper.__doc__ = method.__doc__

    return wrapper


def caching_function(func):
    """
    Caching decorator for a function.
    The cache dictionary is attached to the function object.
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        memory_key = f"____func_cache_var{func.__name__}"
        if not hasattr(func, memory_key):
            setattr(func, memory_key, {})

        memory = getattr(func, memory_key)
        if len(kwargs) == 0:
            data_key = args
        else:
            data_key = (args, tuple(sorted(kwargs.items())))

        if data_key in memory:
            return memory[data_key]
        else:
            memory[data_key] = result = func(*args, **kwargs)
            return result

    wrapper.__name__ = func.__name__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__

    return wrapper


class cached_property(object):
    """
    NOTE : Directly copied from Werkzeug -
    https://github.com/pallets/werkzeug/blob/0e1b8c4fe598725b343085c5a9a867e90b966db6/werkzeug/utils.py#L35-L73

    A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Foo(object):

            @cached_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.
    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value
