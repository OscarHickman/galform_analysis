"""Run the dust routines of a local GALFORM checkout.

GALFORM is not public, so this package contains no dust model. Instead it
compiles a small driver program (``galform_dust_driver.F90``) against your own
GALFORM build and runs it. Point ``GALFORM_DIR`` at your GALFORM checkout,
which must have been built with CMake (``<checkout>/build`` by default, or set
``GALFORM_BUILD_DIR``). The driver is compiled once per GALFORM build and
cached under ``$XDG_CACHE_HOME/galform_analysis`` (default ``~/.cache``).
"""

import hashlib
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

DRIVER_SOURCE = Path(__file__).with_name("galform_dust_driver.F90")
_LINK_TARGET = "galform2"  # an existing GALFORM executable whose link line we reuse


@dataclass(frozen=True)
class GalformBuild:
    """A GALFORM checkout and the CMake build directory made from it."""

    source: Path
    build: Path

    @property
    def data_dir(self) -> Path:
        return self.source / "Data"


def find_galform(
    galform_dir: Optional[Union[str, Path]] = None,
    build_dir: Optional[Union[str, Path]] = None,
) -> GalformBuild:
    """Locate the GALFORM checkout and its build.

    Args:
        galform_dir: GALFORM checkout; defaults to ``$GALFORM_DIR``.
        build_dir: CMake build directory; defaults to ``$GALFORM_BUILD_DIR``,
            then ``<galform_dir>/build``.

    Raises:
        FileNotFoundError: if no checkout is configured, or it has not been
            built.
    """
    galform_dir = galform_dir or os.environ.get("GALFORM_DIR")
    if not galform_dir:
        raise FileNotFoundError(
            "Dust needs your GALFORM checkout: set GALFORM_DIR to its path "
            "(it must have been built, e.g. in <checkout>/build)."
        )
    source = Path(galform_dir).expanduser().resolve()
    build = Path(
        build_dir or os.environ.get("GALFORM_BUILD_DIR") or source / "build"
    ).expanduser()
    needed = [
        build / "libgalform.a",
        build / "include",
        build / "CMakeCache.txt",
        build / "CMakeFiles" / f"{_LINK_TARGET}.dir" / "link.txt",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"GALFORM build not found or incomplete in {build} (missing: "
            f"{', '.join(missing)}). Build GALFORM with CMake, or set "
            "GALFORM_BUILD_DIR to its build directory."
        )
    return GalformBuild(source=source, build=build.resolve())


def _cmake_cache(build: Path) -> dict:
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and ":" in line.split("=", 1)[0] and not line.startswith("#"):
            key_type, value = line.split("=", 1)
            values[key_type.split(":", 1)[0]] = value
    return values


def _driver_path(galform: GalformBuild) -> Path:
    lib = galform.build / "libgalform.a"
    stamp = f"{DRIVER_SOURCE.read_bytes()!r}{lib}{lib.stat().st_mtime_ns}"
    key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "galform_analysis" / f"dust_driver_{key}" / "galform_dust_driver"


def _run(cmd, cwd) -> None:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError as exc:  # e.g. the compiler GALFORM was built with is gone
        raise RuntimeError(
            f"building the GALFORM dust driver failed:\n$ {shlex.join(cmd)}\n{exc}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"building the GALFORM dust driver failed:\n$ {shlex.join(cmd)}\n"
            f"{(result.stdout + result.stderr)[-2000:]}"
        )


def dust_driver(galform: GalformBuild) -> Path:
    """Path to the driver for this GALFORM build, compiling it if needed.

    The compiler, flags and libraries are those GALFORM's own CMake build
    used, read from ``CMakeCache.txt`` and the link line of ``galform2``.
    """
    exe = _driver_path(galform)
    if exe.exists():
        return exe
    cache = _cmake_cache(galform.build)
    compiler = cache["CMAKE_Fortran_COMPILER"]
    flags = shlex.split(cache.get("CMAKE_Fortran_FLAGS", "")) + shlex.split(
        cache.get("CMAKE_Fortran_FLAGS_RELEASE", "-O2")
    )
    link = shlex.split(
        (galform.build / "CMakeFiles" / f"{_LINK_TARGET}.dir" / "link.txt").read_text()
    )
    exe.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=exe.parent) as tmp:
        obj = Path(tmp) / "driver.o"
        _run(
            [compiler, *flags, f"-I{galform.build / 'include'}", f"-J{tmp}"]
            + ["-c", str(DRIVER_SOURCE), "-o", str(obj)],
            cwd=tmp,
        )
        # Reuse GALFORM's link line, swapping in our object and output.
        out = Path(tmp) / exe.name
        cmd = [
            str(obj) if arg.endswith(f"{_LINK_TARGET}.F90.o") else arg for arg in link
        ]
        cmd[cmd.index("-o") + 1] = str(out)
        _run(cmd, cwd=galform.build)
        os.replace(out, exe)  # atomic, so concurrent jobs never see a partial file
    return exe


def dust_table(galform: GalformBuild, table: str) -> Path:
    """Path of a dust table: as given if it has a directory, otherwise in
    ``<GALFORM>/Data/dust``."""
    path = (
        Path(table) if len(Path(table).parts) > 1 else galform.data_dir / "dust" / table
    )
    if not path.is_file():
        raise FileNotFoundError(f"GALFORM dust table not found: {path}")
    return path


def run_dust_driver(
    exe: Path,
    table: Path,
    *,
    lam: np.ndarray,
    mdisk: np.ndarray,
    zdisk: np.ndarray,
    rdisk: np.ndarray,
    rbulge: np.ndarray,
    mburst: np.ndarray,
    zburst: np.ndarray,
    rburst: np.ndarray,
    burst: np.ndarray,
    h0: float,
    inc_deg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the driver on n galaxies at the rest wavelengths ``lam`` [A].

    ``inc_deg`` None means GALFORM's average over inclinations.

    Returns:
        ``(atten, tauv0)``: attenuation factors of shape (2, nlam, n), for
        disk light then bulge light, and each galaxy's central V-band optical
        depth.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float32))
    n, nlam = len(mdisk), len(lam)
    inc = np.zeros(n) if inc_deg is None else inc_deg
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        with open(infile, "wb") as fh:
            np.array([n, nlam, int(inc_deg is None)], dtype=np.int32).tofile(fh)
            np.array([h0], dtype=np.float32).tofile(fh)
            lam.tofile(fh)
            for arr in (mdisk, zdisk, rdisk, rbulge, mburst, zburst, rburst, inc):
                np.asarray(arr, dtype=np.float32).tofile(fh)
            np.asarray(burst, dtype=np.int32).tofile(fh)
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        result = subprocess.run(
            [str(exe), str(table), str(infile), str(outfile)],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0 or not outfile.exists():
            log = (result.stdout + result.stderr)[-2000:]
            raise RuntimeError(
                f"GALFORM dust driver failed (exit {result.returncode}):\n{log}"
            )
        out = np.fromfile(outfile, dtype=np.float32)
    atten = out[: 2 * nlam * n].reshape(n, nlam, 2).transpose(2, 1, 0)
    return atten.astype(np.float64), out[2 * nlam * n :].astype(np.float64)


__all__ = [
    "GalformBuild",
    "find_galform",
    "dust_driver",
    "dust_table",
    "run_dust_driver",
    "DRIVER_SOURCE",
]
