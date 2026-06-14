"""Linear matter correlation function from CAMB.

xi_m(r, z) is the correct reference field for galaxy bias:
    b(r, z) = sqrt( xi_gal(r, z) / xi_m(r, z) )

The matter xi is computed from the linear power spectrum P(k, z) via a
direct Hankel transform.  It is not the same as the halo xi — halos are
themselves biased tracers of the matter field.
"""

from typing import Optional

import numpy as np
import polars as pl

from galform_analysis.config import DEFAULT_RBINS, SimulationConfig


def compute_matter_xi(
    sim: SimulationConfig,
    z: float,
    rbins: Optional[np.ndarray] = None,
    ns: float = 0.961,
    kmax: float = 1e3,
    nk: int = 4096,
) -> pl.DataFrame:
    """Compute the linear matter correlation function xi_m(r, z) via CAMB.

    Cosmological parameters (omega_m, omega_b, h, sigma_8) are taken from
    the SimulationConfig.  The amplitude is rescaled so that sigma_8 at z=0
    matches the config value.

    Args:
        sim: SimulationConfig for the simulation family.
        z: Redshift at which to evaluate xi_m.
        rbins: Radial bin edges in Mpc/h. Defaults to DEFAULT_RBINS.
        ns: Scalar spectral index. Use 0.961 for L800/WMAP; 1.0 for Millennium.
        kmax: Maximum wavenumber [h/Mpc] for the P(k) grid.
        nk: Number of k points for the Hankel integration.

    Returns:
        DataFrame with columns ['r', 'xi'] and attrs {z, sim, linear, ns}.

    Raises:
        ImportError: If camb is not installed (pip install camb).
    """
    try:
        import camb
    except ImportError:
        raise ImportError(
            "compute_matter_xi requires CAMB: pip install camb  (or: pip install hmf)"
        )

    if rbins is None:
        rbins = DEFAULT_RBINS
    r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=sim.h0,
        ombh2=sim.omega_b * sim.h**2,
        omch2=(sim.omega_m - sim.omega_b) * sim.h**2,
        omk=0,
    )
    pars.InitPower.set_params(As=2.1e-9, ns=ns)
    # Always include z=0 so get_sigma8_0() is available for normalisation
    redshifts_camb = sorted({float(z), 0.0}, reverse=True)
    pars.set_matter_power(redshifts=redshifts_camb, kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)

    # Rescale the power spectrum amplitude so that sigma_8(z=0) matches the sim
    sigma8_camb = results.get_sigma8_0()
    pk_rescale = (sim.sigma_8 / sigma8_camb) ** 2

    # k in h/Mpc, pk in (Mpc/h)^3; rows ordered by descending redshift
    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=kmax, npoints=nk
    )
    idx = int(np.argmin(np.abs(np.array(z_out) - float(z))))
    pk_z = pk[idx] * pk_rescale

    # Hankel transform: xi(r) = 1/(2pi^2) * int k^2 P(k) sin(kr)/(kr) dk
    # numpy sinc(x) = sin(pi*x)/(pi*x), so sin(kr)/(kr) = sinc(kr/pi)
    try:
        _trapz = np.trapezoid  # NumPy >= 2.0
    except AttributeError:
        _trapz = np.trapz  # NumPy < 2.0

    xi_vals = np.array(
        [
            _trapz(kh**2 * pk_z * np.sinc(kh * r / np.pi), kh) / (2.0 * np.pi**2)
            for r in r_centers
        ]
    )

    df = pl.DataFrame({"r": r_centers.astype(np.float64), "xi": xi_vals})
    df.attrs = {
        "z": float(z),
        "sim": sim.name,
        "linear": True,
        "ns": ns,
        "rbins": rbins,
    }
    return df
