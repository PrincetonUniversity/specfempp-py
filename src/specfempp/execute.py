from __future__ import annotations

import yaml
from typing import TYPE_CHECKING

from specfempp_core import _execute

if TYPE_CHECKING:
    from .config import Config


def execute(config: Config):
    """Execute the simulation."""

    config._load_default()
    _execute(
        yaml.dump(config._runtime_config, sort_keys=False),
        yaml.dump(config._default_config, sort_keys=False),
    )
