#!/usr/bin/env python
"""
A collection of utility functions to enhance and simplify plotting with Matplotlib.

This module provides helpers for common plotting tasks, such as:
- Setting global font sizes.
- Creating complex layouts like plots with attached histograms or residual panels.
- Automatically managing minor ticks.
- Drawing 3D shapes.
"""

from decimal import Decimal
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def change_axes_fontsize(fs: float = 10.0) -> None:
    """
    Sets the font size for axes labels and tick labels globally.

    Args:
        fs: The desired font size. Defaults to 10.0.
    """
    mpl.rcParams["axes.labelsize"] = fs
    mpl.rcParams["xtick.labelsize"] = fs
    mpl.rcParams["ytick.labelsize"] = fs


def print_rounded_value(value: float, uncertainty: float) -> str:
    """
    Rounds a value to the same number of decimal places as its uncertainty.

    Args:
        value: The numerical value to format.
        uncertainty: The uncertainty, which determines the rounding precision.

    Returns:
        A string representation of the rounded value.
    """
    return str(Decimal(str(value)).quantize(Decimal(str(uncertainty))))


def set_minor_ticks(ax_obj: Axes) -> None:
    """
    Automatically displays minor tick marks on a plot axis.

    This is a modern, simplified replacement for manually calculating intervals.

    Args:
        ax_obj: The axis object to modify (e.g., ax.xaxis or ax.yaxis).
    """
    ax_obj.set_minor_locator(AutoMinorLocator())


def set_minor_ticks_by_scale(ax_obj: Axes, scale: float = 2.0) -> None:
    """
    Displays minor ticks based on a scaling factor of the major ticks.

    Args:
        ax_obj: The axis object to modify (e.g., ax.xaxis or ax.yaxis).
        scale: The ratio of major_interval / minor_interval. For example,
               a scale of 5 means there will be 4 minor ticks between
               each major tick. Defaults to 2.0.
    """
    major_ticks = ax_obj.get_majorticklocs()
    if len(major_ticks) < 2:
        return  # Cannot determine interval
    major_interval = major_ticks[1] - major_ticks[0]
    ax_obj.set_minor_locator(MultipleLocator(base=major_interval / scale))


def get_position_in_axes(ax: Axes, x_frac: float, y_frac: float) -> tuple[float, float]:
    """
    Calculates data coordinates at a fractional position within the axes limits.

    Args:
        ax: The axis object.
        x_frac: The fractional distance along the x-axis (0.0 to 1.0).
        y_frac: The fractional distance along the y-axis (0.0 to 1.0).

    Returns:
        A tuple (x_pos, y_pos) in data coordinates.
    """
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    dx = xlims[1] - xlims[0]
    dy = ylims[1] - ylims[0]
    x_pos = xlims[0] + x_frac * dx
    y_pos = ylims[0] + y_frac * dy
    return x_pos, y_pos


def create_residual_axes(
    fig: Optional[Figure] = None, ratio: int = 3, **kwargs
) -> tuple[Axes, Axes]:
    """
    Creates a two-panel plot with a main panel and a smaller residuals panel below.

    Args:
        fig: The figure to draw on. If None, a new one is created.
        ratio: The height ratio of the main panel to the residuals panel.
        **kwargs: Keyword arguments passed to format_axes (e.g., x_lim, y_label).

    Returns:
        A tuple (ax_main, ax_sub) for the main and residual axes.
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[ratio, 1])

    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1], sharex=ax_main)
    fig.subplots_adjust(hspace=0.0)

    plt.setp(ax_main.get_xticklabels(), visible=False)
    _format_axes(ax_main, ax_sub, **kwargs)

    return ax_main, ax_sub


def _format_axes(ax_main: Axes, ax_sub: Axes, **kwargs) -> None:
    """Helper function to apply formatting to residual axes."""
    if "x_lim" in kwargs:
        ax_main.set_xlim(kwargs["x_lim"])
        ax_sub.set_xlim(kwargs["x_lim"])
        set_minor_ticks(ax_main.xaxis)
        set_minor_ticks(ax_sub.xaxis)
    if "y_lim" in kwargs:
        ax_main.set_ylim(kwargs["y_lim"])
        set_minor_ticks(ax_main.yaxis)
    if "sub_ylim" in kwargs:
        ax_sub.set_ylim(kwargs["sub_ylim"])
        set_minor_ticks(ax_sub.yaxis)
    if "x_label" in kwargs:
        ax_sub.set_xlabel(kwargs["x_label"])
    if "y_label" in kwargs:
        ax_main.set_ylabel(kwargs["y_label"])
    if "sub_ylabel" in kwargs:
        ax_sub.set_ylabel(kwargs["sub_ylabel"])


def draw_cuboid_3d(
    ax: Axes,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    size: Sequence[float] = (1.0, 1.0, 1.0),
    **kwargs,
) -> None:
    """
    Draws a 3D cuboid on a 3D axes object.

    Args:
        ax: A 3D axis object (created with projection='3d').
        position: The [x, y, z] coordinates of the cuboid's corner.
        size: The [dx, dy, dz] dimensions of the cuboid.
        **kwargs: Keyword arguments passed to Poly3DCollection (e.g.,
                  facecolor, edgecolor, lw, alpha).
    """
    x0, y0, z0 = position
    dx, dy, dz = size

    # Define the 8 vertices of the cuboid
    vertices = np.array([
        [x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
        [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz],
        [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz],
    ])

    # Define the 6 faces using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
        [vertices[7], vertices[6], vertices[2], vertices[3]],  # Top
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Front
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Back
    ]

    # Create the 3D polygon collection and add it to the axes
    poly3d = Poly3DCollection(faces, **kwargs)
    ax.add_collection3d(poly3d)


# Example usage block
if __name__ == "__main__":
    # --- Demonstrate create_residual_axes ---
    # Create some sample data
    x_data = np.linspace(0, 10, 50)
    y_data = 2 * x_data + 1 + np.random.normal(0, 1, size=x_data.shape)
    y_fit = 2 * x_data + 1
    residuals = y_data - y_fit

    # Create the axes using the utility function
    fig_res = plt.figure(figsize=(7, 7))
    ax_main, ax_sub = create_residual_axes(
        fig=fig_res,
        ratio=3,
        x_lim=(0, 10),
        y_lim=(0, 22),
        sub_ylim=(-3, 3),
        x_label="X-axis",
        y_label="Y-axis",
        sub_ylabel="Residuals",
    )
    fig_res.suptitle("Residuals Plot Example", fontsize=16)

    # Plot data on the axes
    ax_main.plot(x_data, y_data, "ko", label="Data")
    ax_main.plot(x_data, y_fit, "r-", label="Fit")
    ax_sub.plot(x_data, residuals, "bo")
    ax_sub.axhline(0, color="gray", linestyle="--")

    ax_main.legend()
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_sub.grid(True, linestyle=":", alpha=0.6)

    plt.savefig("residuals_example.png")
    plt.show()
    plt.close(fig_res)

    # --- Demonstrate draw_cuboid_3d ---
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    # Draw a semi-transparent blue cuboid
    draw_cuboid_3d(
        ax_3d,
        position=[1, 1, 1],
        size=[2, 3, 4],
        facecolor="cyan",
        edgecolor="k",
        lw=1,
        alpha=0.2,
    )
    # Draw another cuboid
    draw_cuboid_3d(
        ax_3d, position=[4, 5, 2], size=[1, 1, 1], facecolor="magenta", alpha=0.5
    )

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_xlim(0, 8)
    ax_3d.set_ylim(0, 8)
    ax_3d.set_zlim(0, 8)
    ax_3d.set_title("3D Cuboid Example")

    plt.savefig("cuboid_3d_example.png")
    plt.show()
    plt.close(fig_3d)