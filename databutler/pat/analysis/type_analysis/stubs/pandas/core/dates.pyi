import numpy as np
from pandas.core.base import NoNewAttributesMixin as NoNewAttributesMixin
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from typing import Any, List, Optional, Union


class DateTimeMethods(NoNewAttributesMixin):
    def __init__(self, data: Any) -> None: ...
    def __iter__(self) -> Any: ...

    @property
    def date(self) -> Series: ...
    @property
    def time(self) -> Series: ...
    @property
    def timetz(self) -> Series: ...
    @property
    def year(self) -> Series: ...
    @property
    def month(self) -> Series: ...
    @property
    def day(self) -> Series: ...
    @property
    def hour(self) -> Series: ...
    @property
    def minute(self) -> Series: ...
    @property
    def second(self) -> Series: ...
    @property
    def microsecond(self) -> Series: ...
    @property
    def nanosecond(self) -> Series: ...
    @property
    def week(self) -> Series: ...
    @property
    def weekofyear(self) -> Series: ...
    @property
    def dayofweek(self) -> Series: ...
    @property
    def day_of_week(self) -> Series: ...
    @property
    def weekday(self) -> Series: ...
    @property
    def dayofyear(self) -> Series: ...
    @property
    def day_of_year(self) -> Series: ...
    @property
    def quarter(self) -> Series: ...
    @property
    def is_month_start(self) -> Series: ...
    @property
    def is_month_end(self) -> Series: ...
    @property
    def is_quarter_start(self) -> Series: ...
    @property
    def is_quarter_end(self) -> Series: ...
    @property
    def is_year_start(self) -> Series: ...
    @property
    def is_year_end(self) -> Series: ...
    @property
    def is_leap_year(self) -> Series: ...
    @property
    def daysinmonth(self) -> Series: ...
    @property
    def days_in_month(self) -> Series: ...
    @property
    def tz(self) -> Series: ...
    @property
    def freq(self) -> Series: ...

    def strftime(self, *args: Any, **kwargs: Any) -> Series: ...
    def to_period(self, *args: Any, **kwargs: Any) -> Series: ...
    def to_pydatetime(self, *args: Any, **kwargs: Any) -> Series: ...
    def tz_localize(self, *args: Any, **kwargs: Any) -> Series: ...
    def tz_convert(self, *args: Any, **kwargs: Any) -> Series: ...
    def round(self, *args: Any, **kwargs: Any) -> Series: ...
    def floor(self, *args: Any, **kwargs: Any) -> Series: ...
    def ceil(self, *args: Any, **kwargs: Any) -> Series: ...
    def month_name(self, *args: Any, **kwargs: Any) -> Series: ...
