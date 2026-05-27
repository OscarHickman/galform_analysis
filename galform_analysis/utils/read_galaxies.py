"""Centralized readers for galaxies.hdf5 files.

This module provides reusable helpers for loading galaxy data from a single
GALFORM subvolume into NumPy arrays or pandas DataFrames, with consistent
filtering and metadata handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import polars as pl

from galform_analysis.config import N_SUBVOLUMES
from galform_analysis.readers.loaders import (
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
	centrals_only: bool = True,
	mhalo_min: Optional[float] = None,
	mstar_min: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
	"""Read galaxy data arrays from galaxies.hdf5 for one subvolume.

	When centrals_only=True, filters to central galaxies (is_central == 1).
	When centrals_only=False, returns all galaxies (centrals + satellites).
	For dark matter halos, use read_halo_arrays() instead.

	Args:
		iz_path: Path to snapshot directory (e.g., /.../iz207)
		ivol: Subvolume index
		fields: Optional iterable of dataset names to pull directly from the output group.
		include_positions: Include x,y,z positions (xgal/ygal/zgal)
		include_derived: Include derived fields (mstar, mhalo, sfr, is_central)
		centrals_only: If True, keep only central galaxies (is_central==1)
		mhalo_min: Minimum subhalo mass (mhalo) threshold; None = no cut
		mstar_min: Minimum stellar mass (mstar) threshold in M_sun/h; None = no cut

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
				raise KeyError("is_central field not found - cannot filter for central galaxies")
			mask &= arrays['is_central'] == 1

		if mhalo_min is not None:
			if 'mhalo' not in arrays:
				raise KeyError("mhalo field not found - cannot apply halo mass cut")
			mask &= arrays['mhalo'] >= mhalo_min

		if mstar_min is not None:
			if 'mstar' not in arrays:
				raise KeyError("mstar field not found - cannot apply stellar mass cut")
			mask &= arrays['mstar'] >= mstar_min

		arrays = _apply_mask(arrays, mask)
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
	centrals_only: bool = True,
	mhalo_min: Optional[float] = None,
	return_metadata: bool = False,
):
	"""Read galaxies.hdf5 and return a Polars DataFrame.

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

	df = pl.DataFrame(arrays)
	df.attrs = meta
	return (df, meta) if return_metadata else df


def read_halo_arrays(
	iz_path: str,
	ivol: int = 0,
	fields: Optional[Iterable[str]] = None,
	include_positions: bool = True,
	include_derived: bool = True,
	mhhalo_min: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
	"""Read DM halo data from galaxies.hdf5 for one subvolume.
	
	DM halos are represented by central galaxies (is_central=1), which includes both
	main FOF halos and subhalos. Each central galaxy represents the center of its
	(sub)halo. This gives ~96k halos matching the number of central galaxies.
	Uses host halo mass (mhhalo) for filtering.

	Args:
		iz_path: Path to snapshot directory (e.g., /.../iz207)
		ivol: Subvolume index
		fields: Optional iterable of dataset names to pull directly from the output group.
		include_positions: Include x,y,z positions (xgal/ygal/zgal)
		include_derived: Include derived fields (mhhalo, is_central)
		mhhalo_min: Minimum host halo mass (mhhalo) threshold; None = no cut

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
			arrays['mhhalo'] = _get_first_array(g, ['mhhalo', 'mhalo_host'])
			
			if 'is_central' in g:
				arrays['is_central'] = np.asarray(g['is_central'])

		if fields:
			for name in fields:
				if name in arrays:
					continue
				if name in g:
					arrays[name] = np.asarray(g[name])

		arrays, n = _normalize_arrays(arrays)

		# Filter to DM halos: use all central galaxies as halo representatives
		# Each central galaxy (is_central==1) represents its (sub)halo center
		# This includes both main FOF halos and subhalos within larger structures
		mask = np.ones(n, dtype=bool)
		if 'is_central' not in arrays:
			raise KeyError('is_central field required for halo sample')
		mask &= arrays['is_central'] == 1

		if mhhalo_min is not None:
			if 'mhhalo' not in arrays:
				raise KeyError("mhhalo field not found - cannot apply halo mass cut")
			mask &= arrays['mhhalo'] >= mhhalo_min

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


def read_halo_positions(
	iz_path: str,
	ivol: int,
	mhhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
	"""Load DM halo (halo center) positions and redshift for a subvolume.

	Returns:
		positions: (N,3) array
		z: redshift (if available)
	"""
	arrays, meta = read_halo_arrays(
		iz_path=iz_path,
		ivol=ivol,
		fields=None,
		include_positions=True,
		include_derived=True,
		mhhalo_min=mhhalo_min,
	)

	if not all(k in arrays for k in ('x', 'y', 'z')):
		raise KeyError("Missing position columns in galaxies.hdf5")

	pos = np.vstack([arrays['x'], arrays['y'], arrays['z']]).T.astype(np.float64, copy=False)
	return pos, meta.get('z')


def read_galaxy_positions(
	iz_path: str,
	ivol: int,
	centrals_only: bool = True,
	mhalo_min: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
	"""Load galaxy positions and redshift for a subvolume.

	When centrals_only=True, returns only central galaxies (is_central == 1).
	When centrals_only=False, returns all galaxies (centrals + satellites).

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
