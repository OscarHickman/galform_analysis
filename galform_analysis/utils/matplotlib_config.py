"""Matplotlib runtime configuration helpers.

This module provides a reusable plotting configuration and a convenient
``mpl.setconfig()`` entrypoint once imported.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional

import matplotlib as mpl


ConfigDict = Dict[str, Dict[str, Any]]

DEFAULT_FRAME_COLOR = (0.0, 0.0, 0.0, 0.8)

DEFAULT_CONFIG_DICT: ConfigDict = {
    'font': {'family': 'serif', 'weight': 'normal', 'size': 20},
    'mathtext': {'fontset': 'cm', 'default': 'it'},
    'figure': {'figsize': (8, 6)},
    'figure.subplot': {'wspace': 0.25, 'hspace': 0.25},
    'axes': {
        'labelsize': 16,
        'labelweight': 'normal',
        'linewidth': 1,
        'edgecolor': DEFAULT_FRAME_COLOR,
        'grid': True,
    },
    'xtick': {
        'color': DEFAULT_FRAME_COLOR,
        'labelsize': 13,
        'direction': 'in',
        'top': True,
    },
    'xtick.major': {'width': 1.8, 'size': 9},
    'xtick.minor': {'width': 1, 'size': 4, 'visible': True},
    'ytick': {
        'color': DEFAULT_FRAME_COLOR,
        'labelsize': 13,
        'direction': 'in',
        'right': True,
    },
    'ytick.major': {'width': 1.8, 'size': 9},
    'ytick.minor': {'width': 1, 'size': 4, 'visible': True},
    'grid': {
        'color': 'gray',
        'alpha': 0.5,
        'linestyle': (0, (1.6, 1.6)),
        'linewidth': 0.8,
    },
    'patch': {'edgecolor': DEFAULT_FRAME_COLOR},
    'lines': {'linewidth': 2.5},
    'errorbar': {'capsize': 4},
    'legend': {
        'numpoints': 2,
        'scatterpoints': 3,
        'markerscale': 1.0,
        'fontsize': 20,
        'title_fontsize': 22,
        'loc': 'best',
        'handlelength': 2.0,
        'handleheight': 0.5,
        'labelspacing': 0.1,
        'handletextpad': 0.5,
        'borderpad': 0.25,
        'borderaxespad': 0.8,
        'columnspacing': 1.0,
        'shadow': False,
        'labelcolor': None,
        'frameon': False,
        'framealpha': 0.8,
    },
}


class RuntimeConfig:
    """Apply runtime matplotlib configuration."""

    def __init__(self, config_dict: Optional[ConfigDict] = None) -> None:
        self.config_dict: ConfigDict = deepcopy(DEFAULT_CONFIG_DICT)
        self.update(config_dict)

    def update(self, config_dict: Optional[ConfigDict] = None) -> None:
        if config_dict is None:
            return
        for section, values in config_dict.items():
            if section not in self.config_dict:
                self.config_dict[section] = dict(values)
            else:
                self.config_dict[section].update(values)

    def set_global(self, mpl_module: Any = None) -> None:
        module = mpl if mpl_module is None else mpl_module
        for key, detail in self.config_dict.items():
            module.rc(key, **detail)


def setconfig(
    config_dict: Optional[ConfigDict] = None,
    mpl_module: Any = None,
) -> None:
    """Apply default or overridden plotting configuration."""
    RuntimeConfig(config_dict=config_dict).set_global(mpl_module=mpl_module)


def register_matplotlib_setconfig(mpl_module: Any = None):
    """Register ``setconfig`` as ``mpl.setconfig`` on a matplotlib module."""
    module = mpl if mpl_module is None else mpl_module
    module.setconfig = partial(setconfig, mpl_module=module)
    return module.setconfig


register_matplotlib_setconfig()
