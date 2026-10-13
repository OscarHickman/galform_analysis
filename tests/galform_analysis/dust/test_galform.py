"""Tests for running the dust code of a GALFORM checkout."""

import os
from pathlib import Path

import numpy as np
import pytest

import galform_analysis.dust.galform as gf
from galform_analysis.dust.galform import (
    dust_driver,
    dust_table,
    find_galform,
    run_dust_driver,
)
from tests.fake_dust_driver import (
    AVERAGE_INCLINATION,
    attenuation,
    make_fake_galform,
)

REFERENCE = (
    Path(__file__).parents[2] / "data" / "galform_lc16_diffuse_dust_reference.npz"
)


class TestFindGalform:
    def test_needs_galform_dir(self, monkeypatch):
        monkeypatch.delenv("GALFORM_DIR", raising=False)
        with pytest.raises(FileNotFoundError, match="GALFORM_DIR"):
            find_galform()

    def test_from_environment(self, tmp_path, monkeypatch):
        root = make_fake_galform(tmp_path / "galform")
        monkeypatch.setenv("GALFORM_DIR", str(root))
        monkeypatch.delenv("GALFORM_BUILD_DIR", raising=False)
        galform = find_galform()
        assert galform.source == root.resolve()
        assert galform.build == (root / "build").resolve()
        assert galform.data_dir == root.resolve() / "Data"

    def test_build_dir_override(self, tmp_path, monkeypatch):
        root = make_fake_galform(tmp_path / "galform")
        (root / "build").rename(root / "build_other")
        monkeypatch.setenv("GALFORM_BUILD_DIR", str(root / "build_other"))
        assert find_galform(root).build == (root / "build_other").resolve()

    def test_unbuilt_checkout(self, tmp_path):
        (tmp_path / "galform").mkdir()
        with pytest.raises(FileNotFoundError, match="libgalform.a"):
            find_galform(tmp_path / "galform")


def test_dust_table(tmp_path):
    galform = find_galform(make_fake_galform(tmp_path / "galform"))
    assert dust_table(galform, "dust_MW_hz1.0.dat").name == "dust_MW_hz1.0.dat"
    other = tmp_path / "mine.dat"
    other.write_text("")
    assert dust_table(galform, str(other)) == other
    with pytest.raises(FileNotFoundError, match="nope.dat"):
        dust_table(galform, "nope.dat")


class TestDustDriverBuild:
    def fake_build(self, tmp_path):
        root = make_fake_galform(tmp_path / "galform")
        build = root / "build"
        (build / "CMakeCache.txt").write_text(
            "# comment\n"
            "CMAKE_Fortran_COMPILER:FILEPATH=/opt/gcc/bin/gfortran\n"
            "CMAKE_Fortran_FLAGS:STRING=-ffree-line-length-none\n"
            "CMAKE_Fortran_FLAGS_RELEASE:STRING=-O3\n"
        )
        (build / "CMakeFiles" / "galform2.dir" / "link.txt").write_text(
            "/opt/gcc/bin/gfortran -O3 CMakeFiles/galform2.dir/galform2.F90.o "
            "-o galform2 libgalform.a /opt/hdf5/lib/libhdf5_fortran.so -lm\n"
        )
        return find_galform(root)

    def test_compiles_with_galforms_own_settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        galform = self.fake_build(tmp_path)
        calls = []

        def fake_run(cmd, cwd):
            calls.append((cmd, Path(cwd)))
            Path(cmd[cmd.index("-o") + 1]).write_text("binary")

        monkeypatch.setattr(gf, "_run", fake_run)
        exe = dust_driver(galform)
        assert exe.read_text() == "binary"
        assert exe.is_relative_to(tmp_path / "cache" / "galform_analysis")

        (compile_cmd, _), (link_cmd, link_cwd) = calls
        assert compile_cmd[:4] == [
            "/opt/gcc/bin/gfortran",
            "-ffree-line-length-none",
            "-O3",
            f"-I{galform.build / 'include'}",
        ]
        assert str(gf.DRIVER_SOURCE) in compile_cmd
        # GALFORM's link line, with our object and output swapped in.
        assert link_cwd == galform.build
        assert "libgalform.a" in link_cmd and "-lm" in link_cmd
        assert not any("galform2.F90.o" in arg for arg in link_cmd)
        assert compile_cmd[compile_cmd.index("-o") + 1] in link_cmd

        # Built once: a second call reuses the cached driver.
        calls.clear()
        assert dust_driver(galform) == exe and not calls

    def test_rebuilds_when_galform_changes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        galform = self.fake_build(tmp_path)
        first = gf._driver_path(galform)
        lib = galform.build / "libgalform.a"
        os.utime(lib, ns=(0, lib.stat().st_mtime_ns + 10**9))
        assert gf._driver_path(galform) != first

    def test_compile_errors_are_reported(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        galform = self.fake_build(tmp_path)
        with pytest.raises(RuntimeError, match="building the GALFORM dust driver"):
            dust_driver(galform)  # /opt/gcc/bin/gfortran does not exist


class TestRunDustDriver:
    def inputs(self, n=4):
        return dict(
            lam=np.array([1500.0, 5500.0]),
            mdisk=np.linspace(1e9, 1e10, n),
            zdisk=np.full(n, 0.02),
            rdisk=np.full(n, 0.005),
            rbulge=np.full(n, 0.002),
            mburst=np.array([0.0, 2e9, 0.0, 0.0][:n]),
            zburst=np.full(n, 0.02),
            rburst=np.full(n, 0.002),
            burst=np.array([False, True, False, False][:n]),
            h0=0.6777,
        )

    @pytest.fixture
    def driver(self, fake_galform):
        import galform_analysis.dust.galaxy as galaxy

        return galaxy.dust_driver(None), fake_galform / "Data" / "dust"

    def test_round_trip(self, driver):
        exe, tables = driver
        kw = self.inputs()
        inc = np.array([0.0, 30.0, 60.0, 90.0])
        atten, tauv0 = run_dust_driver(
            exe, tables / "dust_MW_hz1.0.dat", inc_deg=inc, **kw
        )
        disk, bulge = attenuation(
            kw["lam"],
            kw["mdisk"],
            kw["zdisk"],
            inc,
            kw["mburst"],
            kw["zburst"],
            kw["burst"].astype(int),
        )
        assert atten.shape == (2, 2, 4)
        np.testing.assert_allclose(atten[0], disk, rtol=1e-6)
        np.testing.assert_allclose(atten[1], bulge, rtol=1e-6)
        np.testing.assert_allclose(tauv0, 0.6777, rtol=1e-6)  # h0 reached it

    def test_average_inclination(self, driver):
        exe, tables = driver
        kw = self.inputs()
        atten, _ = run_dust_driver(exe, tables / "dust_MW_hz1.0.dat", **kw)
        disk, _ = attenuation(
            kw["lam"],
            kw["mdisk"],
            kw["zdisk"],
            np.full(4, AVERAGE_INCLINATION),
            kw["mburst"],
            kw["zburst"],
            kw["burst"].astype(int),
        )
        np.testing.assert_allclose(atten[0], disk, rtol=1e-6)

    def test_failure(self, driver):
        exe, tables = driver
        with pytest.raises(RuntimeError, match="fake failure"):
            run_dust_driver(exe, tables / "fail.dat", **self.inputs())


class TestAgainstGalform:
    """Compare with GALFORM's own dust-attenuated magnitudes.

    The reference magnitudes were produced by GALFORM itself for 444 galaxies
    of an L800/lc16 z = 0 subvolume with the Lacey16 settings, in bands from
    GALEX-FUV to IRAC 8 micron, without burst dust. Needs a built GALFORM
    checkout in GALFORM_DIR.
    """

    @pytest.fixture(scope="class")
    def galform(self):
        try:
            galform = find_galform()
        except FileNotFoundError:
            pytest.skip("GALFORM_DIR not set to a built GALFORM checkout")
        return galform, dust_driver(galform)

    @pytest.fixture(scope="class")
    def ref(self):
        return dict(np.load(REFERENCE))

    def run(self, galform, ref, band, inc):
        build, exe = galform
        n = ref["mcold"].size
        zcold = np.zeros(n)
        has_gas = ref["mcold"] > 1.0
        zcold[has_gas] = ref["cold_metal"][has_gas] / ref["mcold"][has_gas]
        zeros = np.zeros(n)
        atten, tauv0 = run_dust_driver(
            exe,
            dust_table(build, "dust_MW_hz1.0.dat"),
            lam=np.array([float(ref[f"lam_{band}"])]),
            mdisk=0.5 * ref["mcold"],
            zdisk=zcold,
            rdisk=ref["rdisk"],
            rbulge=ref["rbulge"],
            mburst=zeros,
            zburst=zeros,
            rburst=zeros,
            burst=zeros.astype(bool),
            h0=float(ref["h0"]),
            inc_deg=inc,
        )
        ld = ref[f"Ld_{band}"].astype(float) * atten[0, 0]
        lb = ref[f"Lb_{band}"].astype(float) * atten[1, 0]
        mags = [-2.5 * np.log10(np.maximum(lum, 1e-20)) for lum in (ld + lb, ld, lb)]
        return mags, tauv0

    @pytest.mark.parametrize("band", ["GA", "U", "B", "I", "IR"])
    def test_inclined_magnitudes(self, galform, ref, band):
        (total, _, _), _ = self.run(galform, ref, band, ref["inc"])
        np.testing.assert_allclose(total, ref[f"mag_{band}_tot_ext"], atol=1e-5)

    def test_components_face_on_and_tauv0(self, galform, ref):
        (_, disk, bulge), tauv0 = self.run(galform, ref, "B", ref["inc"])
        np.testing.assert_allclose(disk, ref["mag_B_disk_ext"], atol=1e-5)
        np.testing.assert_allclose(bulge, ref["mag_B_bulge_ext"], atol=1e-5)
        np.testing.assert_allclose(tauv0, ref["tauv0"], rtol=1e-5, atol=1e-12)
        (face_on, _, _), _ = self.run(galform, ref, "B", np.zeros_like(ref["inc"]))
        np.testing.assert_allclose(face_on, ref["mag_B_tot_ext0"], atol=1e-5)
