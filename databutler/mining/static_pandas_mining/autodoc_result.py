from typing import List

import attrs

from databutler.mining.static_pandas_mining.autodoc_strategies import CanonicalAutodocDescription, AutodocDescription


@attrs.define(eq=False, repr=False)
class AutodocResult:
    uid: str
    code: str
    template: str
    success: bool
    canonical_descs: List[CanonicalAutodocDescription]
    additional_descs: List[AutodocDescription]
    failed_descs: List[AutodocDescription]
    is_derived: bool = False
