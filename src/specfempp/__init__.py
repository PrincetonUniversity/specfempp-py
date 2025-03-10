import atexit

from specfempp_core import _initialize, _finalize

from .config import Config
from .execute import execute

if _initialize([]):
    # atexit.register(_finalize)


__all__ = [
    "Config",
    "execute",
]
