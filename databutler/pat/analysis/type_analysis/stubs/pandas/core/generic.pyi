from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, AnyStr, overload

from pandas import datetime
from pandas.core.resample import Resampler
from pandas.core.window import ExponentialMovingWindow

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp

import pandas.core.indexing as indexing
from pandas._typing import Axis, Dtype, FilePathOrBuffer, FrameOrSeries, JSONSerializable, Level, Renamer, \
    ReplaceMethod, ReplaceValue, ToReplace, Frequency, Scalar, AxisOption, SortKind, JoinType, NaSortPosition, \
    ErrorsStrategy, ArrayLike, ValueKeyFunc, IndexKeyFunc
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.base import PandasObject, SelectionMixin
from pandas.core.indexes.api import Index
from pandas.core.internals import BlockManager

bool_t = bool

Self = TypeVar('Self', bound=NDFrame)
PipeReturn = TypeVar('PipeReturn')

class NDFrame(PandasObject, SelectionMixin, indexing.IndexingMixin):
    __array_priority__: int = ...
    __bool__: Any = ...

    def __init__(self, data: BlockManager, axes: Optional[List[Index]]=..., copy: bool_t=..., dtype: Optional[Dtype]=..., attrs: Optional[Mapping[Optional[Hashable], Any]]=..., fastpath: bool_t=...) -> None: ...
    def __neg__(self) -> Any: ...
    def __pos__(self) -> Any: ...
    def __invert__(self) -> Any: ...
    def __abs__(self) -> FrameOrSeries: ...
    def __round__(self, decimals: int=...) -> FrameOrSeries: ...
    def __hash__(self) -> Any: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: Any) -> bool_t: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __delitem__(self, key: Any) -> None: ...
    def __copy__(self, deep: bool_t=...) -> FrameOrSeries: ...
    def __deepcopy__(self, memo: Any = ...) -> FrameOrSeries: ...
    def __array__(self, dtype: Any = ...) -> np.ndarray: ...
    def __array_wrap__(self, result: Any, context: Optional[Any] = ...) -> Any: ...
    def keys(self) -> Any: ...
    def items(self) -> Any: ...
    def iteritems(self) -> Any: ...
    @property
    def index(self) -> Index: ...
    @index.setter
    def index(self, idx: Index) -> None: ...
    @property
    def attrs(self) -> Dict[Optional[Hashable], Any]: ...
    @attrs.setter
    def attrs(self, value: Mapping[Optional[Hashable], Any]) -> None: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def axes(self) -> List[Index]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> Any: ...
    def __nonzero__(self) -> None: ...
    def bool(self) -> Any: ...
    def set_axis(self, labels: Any, axis: int = ..., inplace: bool_t = ...) -> Any: ...
    def swapaxes(self, axis1: Any, axis2: Any, copy: Any = ...) -> FrameOrSeries: ...
    def droplevel(self, level: Any, axis: Any = ...) -> FrameOrSeries: ...
    def pop(self, item: Any) -> FrameOrSeries: ...
    def squeeze(self, axis: Optional[Any] = ...) -> Any: ...
    def swaplevel(self, i: Any = ..., j: Any = ..., axis: Any = ...) -> FrameOrSeries: ...
    @overload
    def rename(self, mapper: Optional[Renamer]=..., *, index: Optional[Renamer]=..., columns: Optional[Renamer]=..., axis: Optional[Axis]=..., copy: bool_t=..., inplace: Literal[False] = ..., level: Optional[Level]=..., errors: ErrorsStrategy=...) -> FrameOrSeries: ...
    @overload
    def rename(self, mapper: Optional[Renamer]=..., *, index: Optional[Renamer]=..., columns: Optional[Renamer]=..., axis: Optional[Axis]=..., copy: bool_t=..., inplace: Literal[True], level: Optional[Level]=..., errors: ErrorsStrategy=...) -> None: ...
    @overload
    def rename_axis(self: Self, mapper: Any = ..., inplace: Literal[False] = ..., **kwargs: Any) -> Self: ...
    @overload
    def rename_axis(self, inplace: Literal[True], mapper: Any = ..., **kwargs: Any) -> None: ...
    def equals(self, other: Any) -> Any: ...
    @property
    def empty(self) -> bool_t: ...
    def to_excel(self, excel_writer: Any, sheet_name: Any = ..., na_rep: Any = ..., float_format: Any = ..., columns: Any = ..., header: Any = ..., index: Any = ..., index_label: Any = ..., startrow: Any = ..., startcol: Any = ..., engine: Any = ..., merge_cells: Any = ..., encoding: Any = ..., inf_rep: Any = ..., verbose: Any = ..., freeze_panes: Any = ..., storage_options: Optional[Dict[str, Any]] = None) -> None: ...
    def to_json(self, path_or_buf: Optional[FilePathOrBuffer[AnyStr]]=..., orient: Optional[str]=..., date_format: Optional[str]=..., double_precision: int=..., force_ascii: bool_t=..., date_unit: str=..., default_handler: Optional[Callable[[Any], JSONSerializable]]=..., lines: bool_t=..., compression: Optional[str]=..., index: bool_t=..., indent: Optional[int]=..., storage_options: Optional[Dict[str, Any]] = None) -> Optional[str]: ...
    def to_hdf(self, path_or_buf: Any, key: str, mode: str=..., complevel: Optional[int]=..., complib: Optional[str]=..., append: bool_t=..., format: Optional[str]=..., index: bool_t=..., min_itemsize: Optional[Union[int, Dict[str, int]]]=..., nan_rep: Any = ..., dropna: Optional[bool_t]=..., data_columns: Optional[List[str]]=..., errors: str=..., encoding: str=...) -> None: ...
    def to_sql(self, name: str, con: Any, schema: Any = ..., if_exists: str=..., index: bool_t=..., index_label: Any = ..., chunksize: Any = ..., dtype: Any = ..., method: Any = ...) -> None: ...
    def to_pickle(self, path: Any, compression: Optional[str]=..., protocol: int=..., storage_options: Optional[Dict[str, Any]] = None) -> None: ...
    def to_clipboard(self, excel: bool_t=..., sep: Optional[str]=..., **kwargs: Any) -> None: ...
    def to_xarray(self) -> Any: ...
    def to_latex(self, buf: Optional[Any] = ..., columns: Optional[Any] = ..., col_space: Optional[Any] = ..., header: bool_t = ..., index: bool_t = ..., na_rep: str = ..., formatters: Optional[Any] = ..., float_format: Optional[Any] = ..., sparsify: Optional[Any] = ..., index_names: bool_t = ..., bold_rows: bool_t = ..., column_format: Optional[Any] = ..., longtable: Optional[Any] = ..., escape: Optional[Any] = ..., encoding: Optional[Any] = ..., decimal: str = ..., multicolumn: Optional[Any] = ..., multicolumn_format: Optional[Any] = ..., multirow: Optional[Any] = ..., caption: Union[str, Tuple[str, str]] = ..., label: Optional[Any] = ..., position: Optional[str] = ...) -> Any: ...
    def to_csv(self, path_or_buf: Optional[FilePathOrBuffer[AnyStr]]=..., sep: str=..., na_rep: str=..., float_format: Optional[str]=..., columns: Optional[Sequence[Optional[Hashable]]]=..., header: Union[bool_t, List[str]]=..., index: bool_t=..., index_label: Optional[Union[bool_t, str, Sequence[Optional[Hashable]]]]=..., mode: str=..., encoding: Optional[str]=..., compression: Optional[Union[str, Mapping[str, str]]]=..., quoting: Optional[int]=..., quotechar: str=..., line_terminator: Optional[str]=..., chunksize: Optional[int]=..., date_format: Optional[str]=..., doublequote: bool_t=..., escapechar: Optional[str]=..., decimal: Optional[str]=..., errors: str=..., storage_options: Optional[Dict[str, Any]] = None) -> Optional[str]: ...
    def take(self, indices: Any, axis: Any = ..., is_copy: Optional[bool_t]=..., **kwargs: Any) -> FrameOrSeries: ...
    def xs(self, key: Any, axis: Any = ..., level: Any = ..., drop_level: bool_t=...) -> Any: ...
    def get(self, key: Any, default: Optional[Any] = ...) -> Any: ...
    def reindex_like(self, other: Any, method: Optional[str]=..., copy: bool_t=..., limit: Any = ..., tolerance: Any = ...) -> FrameOrSeries: ...
    @overload
    def drop(self, labels: Any = ..., axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., errors: ErrorsStrategy=..., *, inplace: Literal[False] = ...) -> Any: ...
    @overload
    def drop(self, labels: Any = ..., axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., errors: ErrorsStrategy=..., *, inplace: Literal[True]) -> None: ...
    def add_prefix(self, prefix: str) -> FrameOrSeries: ...
    def add_suffix(self, suffix: str) -> FrameOrSeries: ...
    @overload
    def sort_values(self, by: Union[str, List[str]] = ..., axis: Union[int, str] = ..., ascending: bool_t = ..., kind: str = ..., na_position: str = ..., ignore_index: bool_t = ...,  key: ValueKeyFunc = ..., *, inplace: Literal[False] = ...) -> FrameOrSeries: ...
    @overload
    def sort_values(self, by: Union[str, List[str]] = ..., axis: Union[int, str] = ..., ascending: bool_t = ..., kind: str = ..., na_position: str = ..., ignore_index: bool_t = ...,  key: ValueKeyFunc = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def sort_index(self, axis: Any = ..., level: Any = ..., ascending: bool_t=..., kind: SortKind = ..., na_position: NaSortPosition = ..., sort_remaining: bool_t=..., ignore_index: bool_t=..., key: IndexKeyFunc = ..., *, inplace: Literal[False] = ...) -> Any: ...
    @overload
    def sort_index(self, axis: Any = ..., level: Any = ..., ascending: bool_t=..., kind: SortKind = ..., na_position: NaSortPosition = ..., sort_remaining: bool_t=..., ignore_index: bool_t=..., key: IndexKeyFunc = ..., *, inplace: Literal[True]) -> None: ...
    def reindex(self, *args: Any, **kwargs: Any) -> FrameOrSeries: ...
    def filter(self, items: Any = ..., like: Optional[str]=..., regex: Optional[str]=..., axis: Any = ...) -> FrameOrSeries: ...
    def head(self, n: int = ...) -> FrameOrSeries: ...
    def tail(self, n: int = ...) -> FrameOrSeries: ...
    def sample(self, n: int = ..., frac: float = ..., replace: bool_t = ..., weights: Union[str, ArrayLike] = ..., random_state: Union[int, np.random.RandomState] = ..., axis: Optional[AxisOption] = ...) -> FrameOrSeries: ...
    @overload
    def pipe(self: Any, func: Union[Callable[..., PipeReturn], Tuple[Callable[..., PipeReturn], str]], *args: Any, **kwargs: Any) -> PipeReturn: ...
    @overload
    def pipe(self: Any, func: PipeReturn, *args: Any, **kwargs: Any) -> PipeReturn: ...
    def __finalize__(self, other: Any, method: Any = ..., **kwargs: Any) -> FrameOrSeries: ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def dtypes(self) -> Any: ...
    def astype(self, dtype: Any, copy: bool_t=..., errors: str=...) -> FrameOrSeries: ...
    def copy(self: Self, deep: bool_t=...) -> Self: ...
    def infer_objects(self) -> FrameOrSeries: ...
    def convert_dtypes(self, infer_objects: bool_t=..., convert_string: bool_t=..., convert_integer: bool_t=..., convert_boolean: bool_t=...) -> FrameOrSeries: ...
    @overload
    def fillna(self, value: Any = ..., method: Any = ..., axis: Any = ..., limit: Any = ..., downcast: Any = ..., *, inplace: Literal[False] = ...) -> FrameOrSeries: ...
    @overload
    def fillna(self, value: Any = ..., method: Any = ..., axis: Any = ..., limit: Any = ..., downcast: Any = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def ffill(self, axis: Any = ..., limit: Any = ..., downcast: Any = ..., *, inplace: Literal[False] = ...) -> FrameOrSeries: ...
    @overload
    def ffill(self, axis: Any = ..., limit: Any = ..., downcast: Any = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def bfill(self, axis: Any = ..., limit: Any = ..., downcast: Any = ..., *, inplace: Literal[False] = ...) -> FrameOrSeries: ...
    @overload
    def bfill(self, axis: Any = ..., limit: Any = ..., downcast: Any = ..., *, inplace: Literal[True]) -> None: ...
    @overload
    def replace(self, to_replace: Optional[ToReplace] = ..., value: Optional[ReplaceValue] = ..., limit: Optional[int] = ..., regex: bool_t = ..., method: ReplaceMethod = ..., *, inplace: Literal[False] = ...) -> FrameOrSeries: ...
    @overload
    def replace(self, to_replace: Optional[ToReplace] = ..., value: Optional[ReplaceValue] = ..., limit: Optional[int] = ..., regex: bool_t = ..., method: ReplaceMethod = ..., *, inplace: Literal[True]) -> None: ...
    def interpolate(self, method: str = ..., axis: int = ..., limit: Optional[Any] = ..., inplace: bool_t = ..., limit_direction: str = ..., limit_area: Optional[Any] = ..., downcast: Optional[Any] = ..., **kwargs: Any) -> Optional[FrameOrSeries]: ...
    def asof(self, where: Any, subset: Optional[Any] = ...) -> Any: ...
    def isna(self) -> FrameOrSeries: ...
    def isnull(self) -> FrameOrSeries: ...
    def notna(self) -> FrameOrSeries: ...
    def notnull(self) -> FrameOrSeries: ...
    def clip(self, lower: Any = ..., upper: Any = ..., axis: Any = ..., inplace: bool_t=..., *args: Any, **kwargs: Any) -> Optional[FrameOrSeries]: ...
    def asfreq(self, freq: Any, method: Any = ..., how: Optional[str]=..., normalize: bool_t=..., fill_value: Any = ...) -> FrameOrSeries: ...
    def at_time(self, time: Any, asof: bool_t=..., axis: Any = ...) -> FrameOrSeries: ...
    def between_time(self, start_time: Any, end_time: Any, include_start: bool_t=..., include_end: bool_t=..., axis: Any = ...) -> FrameOrSeries: ...
    def resample(self, rule: Any, axis: Any = ..., closed: Optional[str]=..., label: Optional[str]=..., convention: str=..., kind: Optional[str]=..., loffset: Any = ..., base: int=..., on: Any = ..., level: Any = ..., origin: Union[Timestamp, str] = ..., offset: Union[Timedelta, str] = ...) -> Resampler: ...
    def first(self, offset: Any) -> FrameOrSeries: ...
    def last(self, offset: Any) -> FrameOrSeries: ...
    def rank(self, axis: Any = ..., method: str=..., numeric_only: Optional[bool_t]=..., na_option: str=..., ascending: bool_t=..., pct: bool_t=...) -> FrameOrSeries: ...
    def align(self, other: Any, join: JoinType = ..., axis: Optional[Any] = ..., level: Optional[Any] = ..., copy: bool_t = ..., fill_value: Optional[Any] = ..., method: Optional[Any] = ..., limit: Optional[int] = ..., fill_axis: AxisOption = ..., broadcast_axis: Optional[AxisOption] = ...) -> Any: ...
    def where(self, cond: Any, other: Any = ..., inplace: bool_t = ..., axis: Optional[Any] = ..., level: Optional[Any] = ..., errors: str = ..., try_cast: bool_t = ...) -> Any: ...
    def mask(self, cond: Any, other: Any = ..., inplace: bool_t = ..., axis: Optional[Any] = ..., level: Optional[Any] = ..., errors: str = ..., try_cast: bool_t = ...) -> Any: ...
    def shift(self, periods: int = ..., freq: Optional[Frequency] = ..., axis: AxisOption = ..., fill_value: Scalar = ...) -> FrameOrSeries: ...
    def slice_shift(self, periods: int=..., axis: Any = ...) -> FrameOrSeries: ...
    def tshift(self, periods: int=..., freq: Any = ..., axis: Any = ...) -> FrameOrSeries: ...
    def truncate(self, before: Any = ..., after: Any = ..., axis: Any = ..., copy: bool_t=...) -> FrameOrSeries: ...
    def tz_convert(self, tz: Any, axis: Any = ..., level: Any = ..., copy: bool_t=...) -> FrameOrSeries: ...
    def tz_localize(self, tz: Any, axis: Any = ..., level: Any = ..., copy: bool_t=..., ambiguous: Any = ..., nonexistent: str=...) -> FrameOrSeries: ...
    def abs(self) -> FrameOrSeries: ...
    def describe(self, percentiles: Any = ..., include: Any = ..., exclude: Any = ..., datetime_is_numeric: bool_t = ...) -> FrameOrSeries: ...
    def pct_change(self, periods: Any = ..., fill_method: Any = ..., limit: Any = ..., freq: Any = ..., **kwargs: Any) -> FrameOrSeries: ...
    def transform(self, func: Any, *args: Any, **kwargs: Any) -> Any: ...
    def first_valid_index(self) -> Any: ...
    def last_valid_index(self) -> Any: ...
    def set_flags(self, *, copy: bool_t = ..., allows_duplicate_labels: Optional[bool_t] = ...) -> FrameOrSeries: ...
    def ewm(self, com: Optional[float] = ..., span: Optional[float] = ..., halflife: Optional[Union[float, str, datetime.timedelta]] = ..., alpha: Optional[float] = ..., min_periods:int = ..., adjust: bool_t = ..., ignore_na: bool_t = ..., axis: Literal[0, 1] = ..., times: Optional[Union[str, np.ndarray, FrameOrSeries]] = ...) -> ExponentialMovingWindow: ...