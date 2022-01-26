import attr


@attr.s
class NodePosition:
    line_start: int = attr.ib()
    column_start: int = attr.ib()
    line_end: int = attr.ib()
    column_end: int = attr.ib()
