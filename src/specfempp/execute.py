from __future__ import annotations

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

    except RuntimeError as e:
        print(e)
        print(e)
        print(e)
        if "SIGNAL" in str(e):
            code = int(str(e).rsplit(" ", 1)[1])
            
            if code == "SIGINT" or code == "2":
                print("Simulation was interrupted.")
                raise KeyboardInterrupt
            elif code == "SIGKILL" or code == "9":
                print("Simulation was killed.")
                raise KeyboardInterrupt
            elif code == "SIGTERM" or code == "15":
                print("Simulation was terminated.")
                raise KeyboardInterrupt
            elif code == "SIGSEGV" or code == "11":
                print("Simulation was terminated due to a segmentation fault.")
                raise KeyboardInterrupt
            elif code == "SIGABRT" or code == "6":
                print("Simulation was terminated due to an abort signal.")
                raise KeyboardInterrupt
        else:
            raise e
        
