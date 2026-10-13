"""Dust attenuation of GALFORM galaxy luminosities.

This package contains no dust model. It runs the diffuse-dust code of your
own GALFORM checkout (set ``GALFORM_DIR``), so GALFORM itself stays private.
Not included: the extra attenuation of very young stars inside their birth
clouds, which needs tables that are not in galaxies.hdf5.
"""

from .galaxy import band_attenuation, inclinations, resolve_dust_model
from .galform import (
    GalformBuild,
    dust_driver,
    dust_table,
    find_galform,
    run_dust_driver,
)
from .models import BAUGH05, DUST_MODELS, LACEY16, DustModel

__all__ = [
    "DustModel",
    "LACEY16",
    "BAUGH05",
    "DUST_MODELS",
    "band_attenuation",
    "inclinations",
    "resolve_dust_model",
    "GalformBuild",
    "find_galform",
    "dust_driver",
    "dust_table",
    "run_dust_driver",
]
