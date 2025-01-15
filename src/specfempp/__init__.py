import atexit

from specfempp_core import _initialize, _finalize

from .main import (
    set_par,
    set_default_par,
    load_par,
    load_default_par,
    save_par,
    save_default_par,
    get_par,
    get_default_par,
    execute,
)

if _initialize([]):
    atexit.register(_finalize)


__all__ = [
    "set_par",
    "set_default_par",
    "load_par",
    "load_default_par",
    "save_par",
    "save_default_par",
    "get_par",
    "get_default_par",
    "execute",
]
