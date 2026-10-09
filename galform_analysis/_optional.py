"""Lazy imports for optional dependencies, with install hints.

The core package (I/O, configuration, mass functions, aggregation) installs from
wheels alone.  Clustering needs Corrfunc, which compiles from source (C compiler,
OpenMP and GSL), and theoretical predictions need hmf/CAMB/colossus, so these
are imported only when a function that needs them is called.
"""

import importlib
from types import ModuleType

# Top-level package name -> pip extra that installs it.
_EXTRA_FOR_PACKAGE = {
    "Corrfunc": "clustering",
    "hmf": "science",
    "astropy": "science",
    "camb": "science",
    "colossus": "science",
    "scipy": "science",
}


def import_optional(module: str) -> ModuleType:
    """Import ``module``, or raise ImportError saying which extra provides it.

    Args:
        module: Dotted module path, e.g. ``"Corrfunc.theory.DD"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module (or its top-level package) is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        package = module.split(".")[0]
        extra = _EXTRA_FOR_PACKAGE.get(package)
        hint = (
            f"pip install 'galform_analysis[{extra}]'"
            if extra
            else f"pip install {package}"
        )
        raise ImportError(
            f"This function requires the optional dependency '{package}'. "
            f"Install it with: {hint}"
        ) from exc
