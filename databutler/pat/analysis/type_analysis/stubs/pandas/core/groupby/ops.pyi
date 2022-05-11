import numpy as np
from pandas._typing import FrameOrSeries as FrameOrSeries
from pandas.core.base import SelectionMixin as SelectionMixin
from pandas.core.dtypes.common import ensure_float64 as ensure_float64, ensure_int64 as ensure_int64, ensure_int_or_float as ensure_int_or_float, ensure_platform_int as ensure_platform_int, is_bool_dtype as is_bool_dtype, is_categorical_dtype as is_categorical_dtype, is_complex_dtype as is_complex_dtype, is_datetime64_any_dtype as is_datetime64_any_dtype, is_datetime64tz_dtype as is_datetime64tz_dtype, is_extension_array_dtype as is_extension_array_dtype, is_integer_dtype as is_integer_dtype, is_numeric_dtype as is_numeric_dtype, is_period_dtype as is_period_dtype, is_sparse as is_sparse, is_timedelta64_dtype as is_timedelta64_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.missing import isna as isna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby import base as base, grouper as grouper
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex, ensure_index as ensure_index
from pandas.core.series import Series as Series
from pandas.core.sorting import compress_group_index as compress_group_index, decons_obs_group_ids as decons_obs_group_ids, get_flattened_iterator as get_flattened_iterator, get_group_index as get_group_index, get_group_index_sorter as get_group_index_sorter, get_indexer_dict as get_indexer_dict
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import cache_readonly as cache_readonly
from typing import Any, List, Optional, Sequence, Tuple

class BaseGrouper:
    axis: Any = ...
    sort: Any = ...
    group_keys: Any = ...
    mutated: Any = ...
    indexer: Any = ...
    def __init__(self, axis: Index, groupings: Sequence[grouper.Grouping], sort: bool=..., group_keys: bool=..., mutated: bool=..., indexer: Optional[np.ndarray]=...) -> None: ...
    @property
    def groupings(self) -> List[grouper.Grouping]: ...
    @property
    def shape(self) -> Any: ...
    def __iter__(self) -> Any: ...
    @property
    def nkeys(self) -> int: ...
    def get_iterator(self, data: FrameOrSeries, axis: int=...) -> Any: ...
    def apply(self, f: Any, data: FrameOrSeries, axis: int=...) -> Any: ...
    def indices(self) -> Any: ...
    @property
    def codes(self) -> List[np.ndarray]: ...
    @property
    def levels(self) -> List[Index]: ...
    @property
    def names(self) -> Any: ...
    def size(self) -> Series: ...
    def groups(self) -> Any: ...
    def is_monotonic(self) -> bool: ...
    def group_info(self) -> Any: ...
    def codes_info(self) -> np.ndarray: ...
    def ngroups(self) -> int: ...
    @property
    def reconstructed_codes(self) -> List[np.ndarray]: ...
    def result_index(self) -> Index: ...
    def get_group_levels(self) -> Any: ...
    def aggregate(self, values: Any, how: str, axis: int=..., min_count: int=...) -> Tuple[np.ndarray, Optional[List[str]]]: ...
    def transform(self, values: Any, how: str, axis: int=..., **kwargs: Any) -> Any: ...
    def agg_series(self, obj: Series, func: Any) -> Any: ...

class BinGrouper(BaseGrouper):
    bins: Any = ...
    binlabels: Any = ...
    mutated: Any = ...
    indexer: Any = ...
    def __init__(self, bins: Any, binlabels: Any, filter_empty: bool=..., mutated: bool=..., indexer: Any = ...) -> None: ...
    def groups(self) -> Any: ...
    @property
    def nkeys(self) -> int: ...
    def get_iterator(self, data: FrameOrSeries, axis: int=...) -> Any: ...
    def indices(self) -> Any: ...
    def group_info(self) -> Any: ...
    def reconstructed_codes(self) -> List[np.ndarray]: ...
    def result_index(self) -> Any: ...
    @property
    def levels(self) -> Any: ...
    @property
    def names(self) -> Any: ...
    @property
    def groupings(self) -> List[grouper.Grouping]: ...
    def agg_series(self, obj: Series, func: Any) -> Any: ...

class DataSplitter:
    data: Any = ...
    labels: Any = ...
    ngroups: Any = ...
    axis: Any = ...
    def __init__(self, data: FrameOrSeries, labels: Any, ngroups: int, axis: int=...) -> None: ...
    def slabels(self) -> Any: ...
    def sort_idx(self) -> Any: ...
    def __iter__(self) -> Any: ...

class SeriesSplitter(DataSplitter): ...

class FrameSplitter(DataSplitter):
    def fast_apply(self, f: Any, names: Any) -> Any: ...

def get_splitter(data: FrameOrSeries, *args: Any, **kwargs: Any) -> DataSplitter: ...