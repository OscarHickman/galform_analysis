"""
Galaxy bias computation utilities.
"""

import numpy as np
import polars as pl


def compute_galaxy_bias(xi_galaxy: pl.DataFrame, xi_halo: pl.DataFrame) -> pl.DataFrame:
    """
    Compute galaxy bias b(r) = sqrt(xi_galaxy / xi_halo).

    Args:
        xi_galaxy: DataFrame with columns 'r' and 'xi' from galaxy correlation.
        xi_halo: DataFrame with columns 'r' and 'xi' from halo correlation.

    Returns:
        DataFrame with columns 'r' and 'bias'.
    """
    # Ensure radial bins match
    if not np.allclose(xi_galaxy["r"], xi_halo["r"]):
        raise ValueError("Radial bins for galaxy and halo correlations do not match.")

    bias = np.sqrt(xi_galaxy["xi"] / xi_halo["xi"])

    df = pl.DataFrame({"r": xi_galaxy["r"], "bias": bias})

    df.attrs = {
        "method": "sqrt(xi_gal / xi_halo)",
        "xi_galaxy_metadata": xi_galaxy.attrs,
        "xi_halo_metadata": xi_halo.attrs,
    }

    return df


def avg_galaxy_bias_over_subvolumes(
    xi_gal_list: list[pl.DataFrame], xi_halo_list: list[pl.DataFrame]
) -> pl.DataFrame:
    """
    Compute average galaxy bias over a list of subvolumes.
    """
    if len(xi_gal_list) != len(xi_halo_list):
        raise ValueError("List lengths must match.")

    biases = []
    for xg, xh in zip(xi_gal_list, xi_halo_list):
        biases.append(compute_galaxy_bias(xg, xh)["bias"].to_numpy())

    bias_arr = np.vstack(biases)
    r = xi_gal_list[0]["r"].to_numpy()

    return pl.DataFrame(
        {"r": r, "bias": bias_arr.mean(axis=0), "bias_std": bias_arr.std(axis=0)}
    )
