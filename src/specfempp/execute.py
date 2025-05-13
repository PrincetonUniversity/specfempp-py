from __future__ import annotations

import sys
import yaml
from typing import TYPE_CHECKING

from specfempp_core import _execute

if TYPE_CHECKING:
    from .config import Config
    

def execute(config: Config):
    """Execute the simulation."""

    config._load_default()
    
    # SPECFEM++ will raise errors when it receives a signal
    # We catch these errors and raise the appropriate exception
    # See specfem::periodic_tasks::check_signals
    
    try:
        _execute(
            yaml.dump(config._runtime_config, sort_keys=False),
            yaml.dump(config._default_config, sort_keys=False)
            )
    except Exception as e:
        if "Signal" in str(e):
            print("RuntimeError: Simulation interrupted by user", file=sys.stderr)
        else:
            raise e
