"""Shared helpers for mass function averaging."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def _avg_phi_over_snapshots(
    single_fn: Callable,
    ivol: int,
    iz_nums: List[int],
    bins: np.ndarray,
    base_dir: str,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """Average mass function phi across multiple snapshots for one subvolume.

    Calls single_fn(iz_path, ivol, bins=bins, **kwargs) for each snapshot,
    then averages the resulting phi arrays.
    """
    per_phi: List[np.ndarray] = []
    iz_list: List[str] = []
    z_list: List[Optional[float]] = []
    centers_ref = None

    for iz_num in iz_nums:
        iz_path = os.path.join(base_dir, f"iz{iz_num}")
        if not os.path.isdir(iz_path):
            continue
        res = single_fn(iz_path, ivol, bins=bins, **kwargs)
        if res is None:
            continue
        if centers_ref is None:
            centers_ref = res["centers"]
        per_phi.append(res["phi"])
        iz_list.append(f"iz{iz_num}")
        z_list.append(res["z"])

    if not per_phi:
        return None

    per_phi_arr = np.array(per_phi)
    centers = centers_ref if centers_ref is not None else 0.5 * (bins[1:] + bins[:-1])

    return {
        "ivol": ivol,
        "iz_list": iz_list,
        "z_list": z_list,
        "centers": centers,
        "phi": per_phi_arr.mean(axis=0),
        "phi_std": per_phi_arr.std(axis=0),
        "n_used": per_phi_arr.shape[0],
        "n_requested": len(iz_nums),
    }
