"""Stand-in for the GALFORM dust driver, for tests that have no GALFORM.

Speaks the same file protocol as ``galform_dust_driver.F90`` but uses a
made-up attenuation: more dust, bluer light and more edge-on all dim more,
bulge light is dimmed less than disk light, and burst dust dims only bulge
light. A table named ``fail.dat`` makes it exit with an error.
"""

import sys
from pathlib import Path

import numpy as np

AVERAGE_INCLINATION = 60.0  # stands in for GALFORM's inclination average


def attenuation(lam, mdisk, zdisk, inc, mburst, zburst, burst):
    """Disk and bulge attenuation for each galaxy at each wavelength."""
    colour = 5500.0 / lam[:, None]
    tau = 1e-8 * mdisk * zdisk * colour * (1.0 + inc / 90.0)
    disk = np.exp(-tau)
    bulge = disk**0.5 * np.where(
        burst == 1, np.exp(-1e-8 * mburst * zburst * colour), 1.0
    )
    return disk, bulge


def make_fake_galform(root: Path) -> Path:
    """Create a GALFORM checkout layout that ``find_galform`` accepts."""
    build = root / "build"
    (build / "include").mkdir(parents=True)
    (build / "CMakeFiles" / "galform2.dir").mkdir(parents=True)
    (build / "CMakeFiles" / "galform2.dir" / "link.txt").write_text("")
    (build / "CMakeCache.txt").write_text("")
    (build / "libgalform.a").write_text("")
    (root / "Data" / "dust").mkdir(parents=True)
    for name in ("dust_MW_hz1.0.dat", "fail.dat"):
        (root / "Data" / "dust" / name).write_text("")
    return root


def main(table, infile, outfile):
    if table.endswith("fail.dat"):
        print("fake failure", file=sys.stderr)
        sys.exit(2)
    with open(infile, "rb") as fh:
        n, nlam, ave = np.fromfile(fh, dtype=np.int32, count=3)
        h0 = np.fromfile(fh, dtype=np.float32, count=1)[0]
        lam = np.fromfile(fh, dtype=np.float32, count=nlam).astype(float)
        cols = (
            np.fromfile(fh, dtype=np.float32, count=8 * n).reshape(8, n).astype(float)
        )
        burst = np.fromfile(fh, dtype=np.int32, count=n)
    mdisk, zdisk, rdisk, rbulge, mburst, zburst, rburst, inc = cols
    if ave:
        inc = np.full(n, AVERAGE_INCLINATION)
    disk, bulge = attenuation(lam, mdisk, zdisk, inc, mburst, zburst, burst)
    atten = np.stack([disk, bulge])  # (2, nlam, n)
    tauv0 = np.full(n, h0, dtype=np.float32)  # echoes h0 so tests can check it
    with open(outfile, "wb") as fh:
        # Fortran order: component fastest, then wavelength, then galaxy.
        atten.transpose(2, 1, 0).astype(np.float32).tofile(fh)
        tauv0.tofile(fh)


if __name__ == "__main__":
    main(*sys.argv[1:4])
