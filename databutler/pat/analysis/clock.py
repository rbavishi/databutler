import attr


@attr.s(cmp=False, repr=False)
class LogicalClock:
    """
    A basic clock implementation supporting getters and increment methods.
    """
    _time: int = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.reset()

    def reset(self, init_time: int = 0):
        self._time = init_time

    def get_time(self) -> int:
        return self._time

    def increment(self, step: int = 1) -> int:
        self._time += step
        return self._time
