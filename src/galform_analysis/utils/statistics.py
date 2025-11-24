#!/usr/bin/env python
"""
A collection of utility functions for statistical analysis and data manipulation.

This module provides optimized functions for common tasks such as creating
histograms, binning data in 2D, calculating percentiles within bins, and
performing logical operations on arrays.
"""

from typing import Optional

import numpy as np
from scipy import stats


def count_occurrences(
    x: np.ndarray, y: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Counts unique values in an array.

    If only `x` is provided, it returns the unique values in `x` and their
    counts. If `y` is also provided, it counts how many times each value
    from `y` appears in `x`.

    Args:
        x: An array of integers.
        y: An optional array of unique integer values to search for in `x`.

    Returns:
        A tuple containing:
        - An array of unique values.
        - An array of corresponding counts.
    """
    if y is None:
        # More direct way to get counts of unique elements in x
        return np.unique(x, return_counts=True)
    else:
        # Count occurrences of each element of y within x
        bins = np.append(y, np.max(y) + 1)
        counts, _ = np.histogram(x, bins=bins)
        return y, counts


def create_mask_inside_range(
    x: np.ndarray,
    low: Optional[float] = None,
    upp: Optional[float] = None,
    include_low: bool = True,
    include_upp: bool = True,
) -> np.ndarray:
    """
    Creates a boolean mask for values within a specified range.

    Args:
        x: The input data array.
        low: The lower bound of the range. Defaults to the minimum of `x`.
        upp: The upper bound of the range. Defaults to the maximum of `x`.
        include_low: Whether to include the lower bound (>=).
        include_upp: Whether to include the upper bound (<=).

    Returns:
        A boolean numpy array where True indicates the value is inside the range.
    """
    if low is None and upp is None:
        raise ValueError("At least one of 'low' or 'upp' must be specified.")
    if low is None:
        low = np.min(x)
    if upp is None:
        upp = np.max(x)

    mask_low = x >= low if include_low else x > low
    mask_upp = x <= upp if include_upp else x < upp
    return np.logical_and(mask_low, mask_upp)


def create_random_sample_mask(n: int, percent: float) -> np.ndarray:
    """
    Generates a boolean mask to select a random percentage of items.

    Args:
        n: The total number of items (length of the mask array).
        percent: The percentage of items to select (0-100).

    Returns:
        A boolean mask of length `n` with the specified percentage of
        True values at random positions.
    """
    if not 0 <= percent <= 100:
        raise ValueError("Percent must be between 0 and 100.")

    frac = percent / 100.0
    num_to_select = int(np.ceil(frac * n))

    mask = np.zeros(n, dtype=bool)
    if num_to_select > 0:
        indices = np.random.choice(n, size=num_to_select, replace=False)
        mask[indices] = True
    return mask


def digitize_2d_statistic(
    x: np.ndarray,
    y: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
    z: Optional[np.ndarray] = None,
    statistic: str = "median",
) -> tuple[np.ndarray, list[float]]:
    """
    Calculates a 2D statistic on binned data, replacing inefficient loops.

    This function bins the `z` values based on the `x` and `y` coordinates
    and computes a specified statistic for each bin. It is a highly efficient
    replacement for manual 2D digitization.

    Args:
        x, y: 1D arrays of coordinates for the data points.
        xbins, ybins: 1D arrays defining the bin edges for x and y axes.
        z: Optional 1D array of values to be binned. If None, the count
           of points in each bin is returned.
        statistic: The statistic to compute for each bin. Examples include
                   'mean', 'median', 'count', 'sum', 'std'.

    Returns:
        A tuple containing:
        - A 2D array with the calculated statistic for each bin.
        - The extent of the bins as [xmin, xmax, ymin, ymax].
    """
    if z is None:
        statistic = "count"
        values = x  # `values` are not used for 'count' but needed as arg
    else:
        values = z

    binned_stat, xedges, yedges, _ = stats.binned_statistic_2d(
        x, y, values=values, statistic=statistic, bins=[xbins, ybins]
    )

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return binned_stat.T, extent  # Transpose to match (row, col) convention


def calculate_percentile_in_bins(
    x: np.ndarray,
    y: np.ndarray,
    xbins: np.ndarray,
    percentile: float = 50.0,
    min_count: int = 10,
) -> np.ndarray:
    """
    Calculates a given percentile of y-values for data binned by x-values.

    Args:
        x: Array of values to bin by.
        y: Array of values to calculate percentiles from.
        xbins: The bin edges for the x-array.
        percentile: The percentile to compute (0-100).
        min_count: The minimum number of points required in a bin to compute
                   the percentile. Bins with fewer points will have a NaN result.

    Returns:
        An array of the calculated percentile for each bin.
    """

    def percentile_func(bin_data):
        return np.percentile(bin_data, percentile)

    # Calculate counts in each bin first
    counts = stats.binned_statistic(x, y, statistic="count", bins=xbins).statistic

    # Calculate percentiles only where counts are sufficient
    result = stats.binned_statistic(
        x, y, statistic=percentile_func, bins=xbins
    ).statistic
    result[counts < min_count] = np.nan
    return result


def chi_squared(observed: np.ndarray, expected: np.ndarray, error: np.ndarray) -> float:
    """Calculates the chi-squared statistic."""
    return np.sum(((observed - expected) / error) ** 2)


# Example usage block to demonstrate the modernized functions
if __name__ == "__main__":
    print("--- Demonstrating Statistical Utilities ---")

    # --- Generate Sample Data ---
    num_points = 5000
    x_coords = np.random.uniform(0, 10, num_points)
    y_coords = np.random.uniform(0, 20, num_points)
    z_values = x_coords + y_coords * 2 + np.random.randn(num_points)

    # --- 1. Demonstrate digitize_2d_statistic ---
    print("\n1. Calculating 2D binned median...")
    xbins = np.linspace(0, 10, 11)  # 10 bins
    ybins = np.linspace(0, 20, 21)  # 20 bins

    median_grid, grid_extent = digitize_2d_statistic(
        x_coords, y_coords, xbins, ybins, z=z_values, statistic="median"
    )
    print(f"   - Resulting grid shape: {median_grid.shape}")
    print(f"   - Grid extent: {grid_extent}")

    # --- 2. Demonstrate calculate_percentile_in_bins ---
    print("\n2. Calculating 25th and 75th percentiles in bins...")
    p25_values = calculate_percentile_in_bins(
        x_coords, z_values, xbins, percentile=25
    )
    p75_values = calculate_percentile_in_bins(
        x_coords, z_values, xbins, percentile=75
    )
    print("   - 25th Percentile per bin:", np.round(p25_values, 2))
    print("   - 75th Percentile per bin:", np.round(p75_values, 2))

    # --- 3. Demonstrate random sampling mask ---
    print("\n3. Creating a 15% random sample mask...")
    total_items = 20
    sample_mask = create_random_sample_mask(total_items, 15.0)
    print(f"   - Mask ({total_items} items): {sample_mask}")
    print(f"   - Number selected: {np.sum(sample_mask)} (expected ~3)")

