"""Centralized readers for galaxies.hdf5 files.

This module provides reusable helpers for loading galaxy data from a single
GALFORM subvolume into NumPy arrays or pandas DataFrames, with consistent
filtering and metadata handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import N_SUBVOLUMES
from ..io.loaders import (
	open_galaxies_hdf5,
	get_output_group,
	_get_first_array,
	_get_redshift_from_file,
	_get_redshift_from_zsnap,
)


def _normalize_arrays(arrays: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
	"""Ensure arrays are 1D and trimmed to a common length."""
	arrays = {k: np.ravel(v) for k, v in arrays.items() if v is not None}
	if not arrays:
		return {}, 0
	lengths = [len(v) for v in arrays.values()]
	n = min(lengths)
	arrays = {k: v[:n] for k, v in arrays.items()}
	return arrays, n


def _apply_mask(arrays: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
	"""Apply a boolean mask to all arrays."""
	return {k: v[mask] for k, v in arrays.items()}


def read_galaxy_arrays(
	iz_path: str,
	ivol: int = 0,
	fields: Optional[Iterable[str]] = None,
	include_positions: bool = True,
	include_derived: bool = True,
	centrals_only: bool = False,
	mhalo_min: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
	"""Read galaxy data arrays from galaxies.hdf5 for one subvolume.

	Args:
		iz_path: Path to snapshot directory (e.g., /.../iz207)
		ivol: Subvolume index
		fields: Optional iterable of dataset names to pull directly from the output group.
		include_positions: Include x,y,z positions (xgal/ygal/zgal)
		include_derived: Include derived fields (mstar, mhalo, sfr, is_central)
		centrals_only: If True, filter to is_central == 1
		mhalo_min: Minimum halo mass (mhalo) threshold; None = no cut

	Returns:
		Tuple of (arrays, metadata)
	"""
	f = open_galaxies_hdf5(iz_path, ivol=ivol)
	if f is None:
		raise FileNotFoundError(f"Missing or unreadable galaxies.hdf5 at {iz_path}/ivol{ivol}")

	try:
		g = get_output_group(f)
		if g is None:
			raise RuntimeError("No OutputNNN group found in HDF5 file")

		arrays: Dict[str, np.ndarray] = {}

		if include_positions:
			for key, alias in (('xgal', 'x'), ('ygal', 'y'), ('zgal', 'z')):
				if key not in g:
					raise KeyError("Could not find xgal/ygal/zgal position arrays in Output group")
				arrays[alias] = np.asarray(g[key])

		if include_derived:
			m_disk = _get_first_array(g, ['mstars_disk'])
			m_bulge = _get_first_array(g, ['mstars_bulge'])
			if m_disk.size and m_bulge.size:
				arrays['mstar'] = m_disk + m_bulge
			else:
				arrays['mstar'] = _get_first_array(g, ['mstars', 'StellarMass', 'Mstar', 'mstars_allburst'])

			arrays['mhalo'] = _get_first_array(g, ['mhalo', 'mchalo', 'Mhalo', 'M_Halo'])
			arrays['sfr'] = _get_first_array(g, ['mstardot', 'Sfr', 'sfr', 'sfr_disk'])

			if 'is_central' in g:
				arrays['is_central'] = np.asarray(g['is_central'])

		if fields:
			for name in fields:
				if name in arrays:
					continue
				if name in g:
					arrays[name] = np.asarray(g[name])

		arrays, n = _normalize_arrays(arrays)

		mask = np.ones(n, dtype=bool)
		if centrals_only:
			if 'is_central' not in arrays:
				raise KeyError("is_central field not found - cannot filter for centrals")
			mask &= arrays['is_central'] == 1

		if mhalo_min is not None:
			if 'mhalo' not in arrays:
				raise KeyError("mhalo field not found - cannot apply halo mass cut")
			mask &= arrays['mhalo'] >= mhalo_min

		arrays = _apply_mask(arrays, mask)

		# Metadata
		meta: Dict[str, Any] = {
			'iz': Path(iz_path).name,
			'ivol': ivol,
			'z': _get_redshift_from_file(f) or _get_redshift_from_zsnap(iz_path, ivol),
			'V_total': None,
			'V_ivol': None,
		}

		if 'Parameters' in f and 'volume' in f['Parameters']:
			V_ivol = float(np.array(f['Parameters']['volume']))
			meta['V_ivol'] = V_ivol
			n_subvol = int(np.array(f['Parameters'].get('n_subvolumes', N_SUBVOLUMES)))
			meta['V_total'] = V_ivol * n_subvol if n_subvol and n_subvol > 0 else V_ivol

		return arrays, meta
	finally:
		try:
			f.close()
		except Exception:
			pass


def read_galaxies_dataframe(
	iz_path: str,
	ivol: int = 0,
	fields: Optional[Iterable[str]] = None,
	include_positions: bool = True,
	include_derived: bool = True,
	centrals_only: bool = False,
	mhalo_min: Optional[float] = None,
	return_metadata: bool = False,
):
	"""Read galaxies.hdf5 and return a pandas DataFrame.

	Args are the same as read_galaxy_arrays. If return_metadata is True,
	returns (df, metadata).
	"""

	arrays, meta = read_galaxy_arrays(
		iz_path=iz_path,
		ivol=ivol,
		fields=fields,
		include_positions=include_positions,
		include_derived=include_derived,
		centrals_only=centrals_only,
		mhalo_min=mhalo_min,
	)

	df = pd.DataFrame(arrays)
	df.attrs.update(meta)
	return (df, meta) if return_metadata else df


def read_galaxy_positions(
	iz_path: str,
	ivol: int,
	centrals_only: bool = False,
	mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
	"""Load galaxy positions and redshift for a subvolume.

	Returns:
		positions: (N,3) array
		z: redshift (if available)
	"""
	arrays, meta = read_galaxy_arrays(
		iz_path=iz_path,
		ivol=ivol,
		fields=None,
		include_positions=True,
		include_derived=True,
		centrals_only=centrals_only,
		mhalo_min=mhalo_min,
	)

	if not all(k in arrays for k in ('x', 'y', 'z')):
		raise KeyError("Missing position columns in galaxies.hdf5")

	pos = np.vstack([arrays['x'], arrays['y'], arrays['z']]).T.astype(np.float64, copy=False)
	return pos, meta.get('z')
