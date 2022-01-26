import collections
import inspect

from typing import List, Union, Callable, Tuple, Any, Optional, Mapping

from databutler.pat.analysis.clock import LogicalClock
from databutler.pat.analysis.hierarchical_trace.builder_utils import TraceEventsCollector, TraceItemsCollector
from databutler.pat.analysis.hierarchical_trace.core import ObjWriteEvent, ObjReadEvent
from databutler.pat.utils import pythonutils
from databutler.utils.logging import logger

SPECS = collections.defaultdict(list)
_CLOCK: Optional[LogicalClock] = None
_TRACE_EVENTS_COLLECTOR: Optional[TraceEventsCollector] = None
_TRACE_ITEMS_COLLECTOR: Optional[TraceItemsCollector] = None


def read_write_spec(funcs: Union[Callable, List[Callable]]):
    """
    The spec is associated with all the functions in `funcs`.
    A spec must take as input a binding of arguments to that function, and return the names of parameters modified.
    The binding itself is a mapping from parameter names to values
    :param funcs:
    :return:
    """
    if not isinstance(funcs, list):
        funcs = [funcs]

    def wrapper(spec):
        for f in funcs:
            SPECS[f].append(spec)

        return spec

    return wrapper


# ------------------
#  SPECS START HERE
# ------------------


@read_write_spec([
    list.extend,
    list.append,
    list.insert,
    list.reverse,
    list.remove,
    list.sort,
    list.clear,
    list.pop,
    dict.pop,
    dict.update,
    dict.clear,
    dict.popitem,
])
def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
    #  The list (i.e. `self`) is always modified for these.
    return [(binding.arguments['self'], 'write')]


# --------------
#  PANDAS
# --------------

try:
    import pandas as pd
except ModuleNotFoundError:
    pass
else:
    try:
        pythonutils.load_module_complete(pd)

        #  Including checks to partly take care of version issues.
        @read_write_spec([
            *[getattr(pd.DataFrame, name) for name in ["pop", "update"] if hasattr(pd.DataFrame, name)],
            *[getattr(pd.Series, name) for name in ["pop", "update"] if hasattr(pd.Series, name)],
        ])
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            return [(binding.arguments['self'], 'write')]


        methods_with_inplace = []
        for elem in [pd.DataFrame, pd.Series]:
            for klass in elem.mro():
                for k in dir(klass):
                    m = getattr(klass, k)
                    try:
                        sig = inspect.signature(m)
                        if 'inplace' in sig.parameters:
                            methods_with_inplace.append(m)

                    except:
                        pass


        @read_write_spec(methods_with_inplace)
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            if binding.arguments['inplace'] is True:
                return [(binding.arguments['self'], 'write')]
            else:
                return []


        methods_with_copy = []
        for elem in [pd.DataFrame, pd.Series]:
            for klass in elem.mro():
                for k in dir(klass):
                    if k.startswith("_"):
                        continue

                    m = getattr(klass, k)
                    try:
                        sig = inspect.signature(m)
                        if 'copy' in sig.parameters:
                            methods_with_copy.append(m)

                    except:
                        pass


        @read_write_spec(methods_with_copy)
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            if binding.arguments['copy'] is False:
                return [(binding.arguments['self'], 'write')]
            else:
                return []


        pd_plotting_functions = [
            pd.DataFrame.plot,
            pd.Series.plot,
            *(v for k, v in vars(pd.DataFrame.plot).items() if not k.startswith("_")),
            *(v for k, v in vars(pd.Series.plot).items() if not k.startswith("_")),
            *(getattr(pd.DataFrame, k) for k in ['boxplot', 'hist'] if hasattr(pd.DataFrame, k)),
            *(getattr(pd.Series, k) for k in ['boxplot', 'hist'] if hasattr(pd.Series, k)),
        ]

        try:
            module = pd.plotting._core
            for k in ['boxplot', 'boxplot_frame', 'boxplot_frame_groupby', 'hist_frame', 'hist_series']:
                if hasattr(module, k):
                    pd_plotting_functions.append(getattr(module, k))

            pd_plotting_functions.append(module.PlotAccessor.__call__)

        except:
            pass

        try:
            @read_write_spec(pd_plotting_functions)
            def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
                #  Mark the current figure object as read and written, in that order.
                gcf = plt.gcf()
                return [(gcf, 'read'), (gcf, 'write')]

        except Exception as e:
            logger.info("Failed to load specs for pandas plotting backend.")
            logger.exception(e)

    except Exception as e:
        logger.info("Failed to load specs for pandas.")
        logger.exception(e)

# --------------
#  SKLEARN
# --------------

try:
    import sklearn
except ModuleNotFoundError:
    pass
else:
    try:
        pythonutils.load_module_complete(sklearn)

        estimators = set(pythonutils.get_all_subclasses(sklearn.base.BaseEstimator))
        classifiers = set(pythonutils.get_all_subclasses(sklearn.base.ClassifierMixin))
        regressors = set(pythonutils.get_all_subclasses(sklearn.base.RegressorMixin))
        transformers = set(pythonutils.get_all_subclasses(sklearn.base.TransformerMixin))
        try:
            classifiers.add(sklearn.pipeline.Pipeline)
            regressors.add(sklearn.pipeline.Pipeline)
        except:
            pass

        fit_and_fit_transform_methods = [
            *(getattr(estimator, "fit") for estimator in estimators
              if hasattr(estimator, "fit")),
            *(getattr(estimator, "fit_transform") for estimator in estimators
              if hasattr(estimator, "fit_transform")),
        ]


        @read_write_spec(fit_and_fit_transform_methods)
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            return [(binding.arguments['self'], 'write')]

    except Exception as e:
        logger.info("Failed to load specs for sklearn.")
        logger.exception(e)

# --------------
#  Matplotlib
# --------------

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
else:
    try:
        pythonutils.load_module_complete(matplotlib)

        class RcParamsWrapper(matplotlib.RcParams):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._obj_proxy_dict = collections.defaultdict(object)
                self._last_event_time = None
                self._keys_set_at_last_time = set()
                self._keys_get_at_last_time = set()

            def __setitem__(self, key, value):
                if _CLOCK is not None and _TRACE_ITEMS_COLLECTOR is not None:
                    cur_item = _TRACE_ITEMS_COLLECTOR.get_last_in_progress_item()
                    if cur_item is not None:
                        time = _CLOCK.get_time()
                        if self._last_event_time != time:
                            self._last_event_time = time
                            self._keys_get_at_last_time.clear()
                            self._keys_set_at_last_time.clear()

                        if key not in self._keys_set_at_last_time:
                            obj_id = id(self._obj_proxy_dict[key])
                            event = ObjWriteEvent(
                                timestamp=_CLOCK.get_time(),
                                owner=cur_item,
                                ast_node=cur_item.ast_node,
                                obj_id=obj_id,
                            )
                            _TRACE_EVENTS_COLLECTOR.add_event(event)
                            self._keys_set_at_last_time.add(key)

                return super().__setitem__(key, value)

            def __getitem__(self, key):
                if _TRACE_ITEMS_COLLECTOR is not None:
                    cur_item = _TRACE_ITEMS_COLLECTOR.get_last_in_progress_item()
                    if cur_item is not None:
                        time = _CLOCK.get_time()
                        if self._last_event_time != time:
                            self._last_event_time = time
                            self._keys_get_at_last_time.clear()
                            self._keys_set_at_last_time.clear()

                        if key not in self._keys_get_at_last_time:
                            obj_id = id(self._obj_proxy_dict[key])
                            event = ObjReadEvent(
                                timestamp=_CLOCK.get_time(),
                                owner=cur_item,
                                ast_node=cur_item.ast_node,
                                obj_id=obj_id,
                            )
                            _TRACE_EVENTS_COLLECTOR.add_event(event)
                            self._keys_get_at_last_time.add(key)

                return super().__getitem__(key)

        rc_params_wrapped = RcParamsWrapper(plt.rcParams)
        matplotlib.rcParams = rc_params_wrapped
        plt.rcParams = rc_params_wrapped

        plt_functions = [
            getattr(plt, k, None) for k in dir(plt)
            if inspect.isroutine(getattr(plt, k, None)) and k != 'show' and k != 'figure'
        ]


        @read_write_spec(plt_functions)
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            #  Mark the current figure object as read and written, in that order.
            gcf = plt.gcf()
            return [(gcf, 'read'), (gcf, 'write')]


        @read_write_spec([plt.figure])
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            return [(ret_val, 'read'), (ret_val, 'write')]

        figure_methods = []
        for k in dir(plt.Figure):
            if k.startswith("_") or k.startswith("get_"):
                continue

            m = getattr(plt.Figure, k)
            try:
                sig = inspect.signature(m)
                figure_methods.append(m)

            except:
                pass

        @read_write_spec(figure_methods)
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            fig = binding.arguments['self']
            return [(fig, 'read'), (fig, 'write')]

        subclasses = pythonutils.get_all_subclasses(matplotlib.artist.Artist)
        klass_access_subplot = next(k for k in subclasses if k.__name__.endswith("AxesSubplot"))

        subplot_functions = [
            v
            for klass in klass_access_subplot.mro()
            for k, v in klass.__dict__.items() if inspect.isroutine(v)
        ]

        @read_write_spec(subplot_functions)
        def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
            #  We mark the figure object as read and written, in that order.
            fig = binding.arguments['self'].figure
            return [(fig, 'read'), (fig, 'write')]

        try:
            import seaborn as sns

            pythonutils.load_module_complete(sns)

            plotting_modules = [
                'seaborn.categorical',
                'seaborn.distributions',
                'seaborn.relational',
                'seaborn.regression',
                'seaborn.axisgrid',
                'seaborn.matrix'
            ]

            themeing_modules = [
                'seaborn.rcmod'
            ]

            seaborn_functions = [
                getattr(sns, k, None) for k in dir(sns) if inspect.isroutine(getattr(sns, k, None))
            ]

            seaborn_plotting_functions = [f for f in seaborn_functions
                                          if (not hasattr(f, "__module__")) or f.__module__ in plotting_modules]

            seaborn_themeing_functions = [f for f in seaborn_functions
                                          if (not hasattr(f, "__module__")) or f.__module__ in themeing_modules]

            @read_write_spec(seaborn_plotting_functions)
            def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
                #  Mark the current figure object as read and written, in that order.
                gcf = plt.gcf()
                return [(gcf, 'read'), (gcf, 'write')]


            grid_subclasses = pythonutils.get_all_subclasses(sns.axisgrid.Grid)
            grid_functions = {
                v
                for par_klass in grid_subclasses
                for klass in par_klass.mro()
                for k, v in klass.__dict__.items() if inspect.isroutine(v)
            }

            @read_write_spec(list(grid_functions))
            def spec(binding: inspect.BoundArguments, ret_val: Any) -> List[Tuple[Any, str]]:
                try:
                    fig = binding.arguments['self'].fig
                    return [(fig, 'read'), (fig, 'write')]
                except:
                    return []

        except Exception as e:
            logger.info("Failed to load specs for seaborn.")
            logger.exception(e)

    except Exception as e:
        logger.info("Failed to load specs for matplotlib.")
        logger.exception(e)

