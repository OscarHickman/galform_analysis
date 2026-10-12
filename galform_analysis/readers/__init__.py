"""I/O subpackage for reading GALFORM outputs."""

from .bands import find_band, list_bands, luminosity_to_magnitude
from .loaders import (
    close_snapshot,
    get_completed_subvolumes,
    get_output_group,
    open_galaxies_hdf5,
    read_snapshot_data,
)
from .magnitudes import read_band_magnitudes

__all__ = [
    "read_snapshot_data",
    "close_snapshot",
    "get_completed_subvolumes",
    "open_galaxies_hdf5",
    "get_output_group",
    "list_bands",
    "find_band",
    "luminosity_to_magnitude",
    "read_band_magnitudes",
]
