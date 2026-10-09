"""Galaxy bias from the ratio of galaxy to matter correlation functions.

    b(r, z) = sqrt( xi_gal(r, z) / xi_m(r, z) )

xi_m must come from compute_matter_xi (CAMB linear theory), NOT from a halo
correlation function — halos are themselves biased tracers of the matter field.
"""

from typing import List

import numpy as np
import polars as pl


def _same_grid(a, b) -> bool:
    """True if two 1-D grids have the same length and values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b))


def compute_galaxy_bias(
    xi_galaxy: pl.DataFrame,
    xi_matter: pl.DataFrame,
) -> pl.DataFrame:
    """Compute scale-dependent galaxy bias b(r) = sqrt(xi_gal / xi_m).

    Args:
        xi_galaxy: DataFrame with columns 'r' and 'xi' from galaxy 2PCF.
        xi_matter: DataFrame with columns 'r' and 'xi' from compute_matter_xi.

    Returns:
        DataFrame with columns 'r' and 'bias'. The bias is sqrt(|ratio|), so
        it is always non-negative.

    Raises:
        ValueError: If the two inputs were not measured on the same bins.

    When both inputs carry matching ``attrs['rbins']`` but different ``r``
    (e.g. pair-weighted mean separations vs bin centres), xi_matter is
    interpolated onto the galaxy ``r``.
    """
    rbins_gal = xi_galaxy.attrs.get("rbins", None)
    rbins_mat = xi_matter.attrs.get("rbins", None)

    same_r = _same_grid(xi_galaxy["r"], xi_matter["r"])
    if rbins_gal is not None and rbins_mat is not None:
        bins_match = _same_grid(rbins_gal, rbins_mat)
    else:
        bins_match = same_r

    if not bins_match:
        raise ValueError(
            "Radial bins of xi_galaxy and xi_matter do not match. Galaxy xi(r) "
            "drops bin edges >= boxsize/2; compute xi_matter with the bins in "
            "xi_galaxy.attrs['rbins']."
        )

    if not same_r:
        xi_matter_interp = np.interp(xi_galaxy["r"], xi_matter["r"], xi_matter["xi"])
    else:
        xi_matter_interp = xi_matter["xi"].to_numpy()

    bias = np.sqrt(np.abs(xi_galaxy["xi"].to_numpy() / xi_matter_interp))

    df = pl.DataFrame({"r": xi_galaxy["r"], "bias": bias})
    df.attrs = {
        "method": "sqrt(xi_gal / xi_matter_linear)",
        "xi_galaxy_metadata": xi_galaxy.attrs,
        "xi_matter_metadata": xi_matter.attrs,
    }
    return df


def avg_galaxy_bias_over_subvolumes(
    xi_gal_list: List[pl.DataFrame],
    xi_matter: pl.DataFrame,
) -> pl.DataFrame:
    """Compute mean galaxy bias averaged over multiple subvolumes.

    xi_matter is shared across all subvolumes (it is a theoretical prediction
    at a fixed redshift, not measured per-subvolume).

    Args:
        xi_gal_list: List of per-subvolume galaxy xi DataFrames.
        xi_matter: Single matter xi DataFrame from compute_matter_xi.

    Returns:
        DataFrame with columns 'r', 'bias', 'bias_std'.

    Raises:
        ValueError: If xi_gal_list is empty or bins do not match.
    """
    if not xi_gal_list:
        raise ValueError("xi_gal_list is empty.")
    biases = [
        compute_galaxy_bias(xg, xi_matter)["bias"].to_numpy() for xg in xi_gal_list
    ]
    bias_arr = np.vstack(biases)
    r = xi_gal_list[0]["r"].to_numpy()
    return pl.DataFrame(
        {"r": r, "bias": bias_arr.mean(axis=0), "bias_std": bias_arr.std(axis=0)}
    )
