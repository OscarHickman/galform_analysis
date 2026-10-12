"""Dust attenuation of GALFORM galaxy luminosities.

Implements the diffuse-dust part of GALFORM's dust model (Ferrara et al. 1999
radiative-transfer tables, as in Cole et al. 2000 and Lacey et al. 2016),
reproducing GALFORM's own attenuation of the luminosities in galaxies.hdf5.
The attenuation of young stars inside molecular clouds is not included: it
needs GALFORM's ``neta`` tables, which are not part of galaxies.hdf5.
"""

from .diffuse import CV_DUST, diffuse_attenuation, dust_atten
from .galaxy import band_attenuation, inclinations, resolve_dust_model
from .models import BAUGH05, DUST_MODELS, LACEY16, DustModel
from .table import AttenuationTable, find_dust_table, read_attenuation_table

__all__ = [
    "DustModel",
    "LACEY16",
    "BAUGH05",
    "DUST_MODELS",
    "AttenuationTable",
    "read_attenuation_table",
    "find_dust_table",
    "dust_atten",
    "diffuse_attenuation",
    "band_attenuation",
    "inclinations",
    "resolve_dust_model",
    "CV_DUST",
]
