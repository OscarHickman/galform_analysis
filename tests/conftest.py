"""Shared pytest fixtures for galform_analysis tests.

Mock GALFORM HDF5 files are built from real-data value ranges sampled from
L800/lc16/iz155/ivol0/galaxies.hdf5, so fixtures match the actual schema
without requiring access to the 2 GB simulation outputs.

Real-data ranges used for mock values:
  mhalo:      1.59e9  – 4.14e13  M_sun/h
  mhhalo:     1.91e9  – 4.14e13  M_sun/h
  mstars_disk:    0   – 3.24e10  M_sun/h
  mstars_bulge: ~20   – 7.37e10  M_sun/h
  xgal/ygal/zgal: 0   – 542.16   Mpc/h
  vxgal/vy/vzgal: ±1600 km/s
  volume (per subvol): 155626.09 Mpc^3/h^3
  N_SUBVOLUMES: 1024
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

_BOXSIZE = 542.16
_VOLUME_PER_SUBVOL = 155626.09375
_N_SUBVOLS = 1024
_MHALO_MIN = 1.59e9
_MHALO_MAX = 4.14e13
_MSTAR_DISK_MAX = 3.24e10
_MSTAR_BULGE_MAX = 7.37e10
_VEL_MAX = 1600.0


def write_galaxy_hdf5(
    filepath: Path,
    n_gals: int = 100,
    seed: int = 42,
    completion_flag: int = 1,
) -> None:
    """Write a minimal galaxies.hdf5 matching the real GALFORM schema."""
    rng = np.random.default_rng(seed)
    n_cen = n_gals // 2

    mstar_disk = rng.uniform(1e6, _MSTAR_DISK_MAX, n_gals).astype(np.float32)
    mstar_bulge = rng.uniform(1e6, _MSTAR_BULGE_MAX, n_gals).astype(np.float32)
    # Log-uniform halo masses matching real distribution
    mhalo = np.exp(
        rng.uniform(np.log(_MHALO_MIN), np.log(_MHALO_MAX), n_gals)
    ).astype(np.float32)
    # Host halo mass >= subhalo mass
    mhhalo = (mhalo * rng.uniform(1.0, 3.0, n_gals)).astype(np.float32)
    x = rng.uniform(0.0, _BOXSIZE, n_gals).astype(np.float32)
    y = rng.uniform(0.0, _BOXSIZE, n_gals).astype(np.float32)
    z_pos = rng.uniform(0.0, _BOXSIZE, n_gals).astype(np.float32)
    vz = rng.uniform(-_VEL_MAX, _VEL_MAX, n_gals).astype(np.float32)
    is_central = np.zeros(n_gals, dtype=np.int32)
    is_central[:n_cen] = 1
    sfr = rng.uniform(0.0, 10.0, n_gals).astype(np.float32)
    dhalo_id = rng.integers(0, 100000, n_gals, dtype=np.int64)
    tree_id = rng.integers(0, 100000, n_gals, dtype=np.int64)

    with h5py.File(str(filepath), "w") as f:
        f.create_dataset("CompletionFlag", data=np.int32(completion_flag))

        # Parameters — real file stores volume per subvol (no n_subvolumes key)
        params = f.create_group("Parameters")
        params.create_dataset("volume", data=np.float64(_VOLUME_PER_SUBVOL))

        # Redshifts — real file uses a Group with string float keys
        redz = f.create_group("Redshifts")
        redz.create_dataset("0.0000", data=np.int32(0))

        # Trees — one mass per FOF group (denominator for HOD)
        n_trees = n_cen
        trees = f.create_group("Trees")
        trees.create_dataset(
            "mphalo",
            data=np.exp(
                rng.uniform(np.log(_MHALO_MIN), np.log(_MHALO_MAX), n_trees)
            ).astype(np.float32),
        )

        # GALFORM writes multiple outputs; get_output_group selects the highest
        # index. Use Output001 so the loader finds exactly one.
        g = f.create_group("Output001")
        g.create_dataset("mstars_disk", data=mstar_disk)
        g.create_dataset("mstars_bulge", data=mstar_bulge)
        g.create_dataset("mhalo", data=mhalo)
        g.create_dataset("mhhalo", data=mhhalo)
        g.create_dataset("xgal", data=x)
        g.create_dataset("ygal", data=y)
        g.create_dataset("zgal", data=z_pos)
        g.create_dataset("vzgal", data=vz)
        g.create_dataset("is_central", data=is_central)
        g.create_dataset("mstardot", data=sfr)
        g.create_dataset("DHaloID", data=dhalo_id)
        g.create_dataset("TreeID", data=tree_id)


@pytest.fixture
def galform_iz_dir(tmp_path):
    """Temp dir with iz155/ivol{0,1}/galaxies.hdf5 mock files."""
    iz_dir = tmp_path / "iz155"
    for ivol in range(2):
        ivol_dir = iz_dir / f"ivol{ivol}"
        ivol_dir.mkdir(parents=True)
        write_galaxy_hdf5(ivol_dir / "galaxies.hdf5", seed=42 + ivol)
    return str(iz_dir)


@pytest.fixture
def galform_base_dir(tmp_path):
    """Temp base dir with iz155 and iz207 snapshots, 2 ivols each."""
    for iz_num in (155, 207):
        iz_dir = tmp_path / f"iz{iz_num}"
        for ivol in range(2):
            ivol_dir = iz_dir / f"ivol{ivol}"
            ivol_dir.mkdir(parents=True)
            write_galaxy_hdf5(ivol_dir / "galaxies.hdf5", seed=iz_num + ivol)
    return str(tmp_path)
