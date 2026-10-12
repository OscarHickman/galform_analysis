"""Small synthetic dust tables in GALFORM's ``dust_*.dat`` format."""

import numpy as np

INC = [10.0, 45.0, 90.0]
LAMBDA = [1500.0, 5500.0, 20000.0]
TAUV = [0.5, 2.0, 10.0]
RE_H = [0.1, 1.0]


def disk_atten(tau, lam, inc):
    """Attenuation of disk light: more dust, bluer light and edge-on all dim."""
    return np.exp(-0.1 * tau * (5500.0 / lam) * (1.0 + inc / 90.0))


def bulge_atten(tau, lam, inc, reh):
    """Bulge light is dimmed less when the bulge is large relative to the disk."""
    return disk_atten(tau, lam, inc) ** (1.0 / (1.0 + reh))


def _block(lines, inc, atten_fn):
    lines += [f"inc={inc:7.1f}", " "]
    lines.append("tauV  " + "".join(f"{lam:6.0f}" for lam in LAMBDA))
    for tau in TAUV:
        vals = "".join(f"{atten_fn(tau, lam, inc):6.3f}" for lam in LAMBDA)
        lines.append(f"{tau:4.1f}  {vals}")
    lines.append(" ")


def write_dust_table(path):
    """Write the synthetic table to ``path`` and return the path."""
    lines = [
        "synthetic dust",
        "hz/h=     8.75e-2",
        "hzd/hz=   1.0",
        " ",
        f"ninc=     {len(INC):2d}",
        f"nlam=     {len(LAMBDA):2d}",
        f"ntauv=    {len(TAUV):2d}",
        f"nreh=     {len(RE_H):2d}",
        " ",
        "-" * 40,
        "disk",
        " ",
    ]
    for inc in INC:
        _block(lines, inc, disk_atten)
    lines += ["-" * 40, "bulge", " "]
    for reh in RE_H:
        lines += [" ", f"re/h=     {reh}"]
        for inc in INC:
            _block(lines, inc, lambda t, la, i, r=reh: bulge_atten(t, la, i, r))
    path.write_text("\n".join(lines) + "\n")
    return path
