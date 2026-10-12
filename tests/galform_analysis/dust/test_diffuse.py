"""Tests for the diffuse-dust attenuation port."""

from pathlib import Path

import numpy as np
import pytest

from galform_analysis.dust.diffuse import CV_DUST, diffuse_attenuation, dust_atten
from galform_analysis.dust.table import find_dust_table, read_attenuation_table
from tests.dust_table import write_dust_table

REFERENCE = (
    Path(__file__).parents[2] / "data" / "galform_lc16_diffuse_dust_reference.npz"
)
H0 = 0.6777


@pytest.fixture
def table(tmp_path):
    return read_attenuation_table(write_dust_table(tmp_path / "dust.dat"))


def gas_disk(n=1, mcold=1e10, z=0.02, rdisk=0.005, rbulge=0.002):
    """Arrays for n identical galaxies (masses Msun/h, radii Mpc/h)."""
    return [np.full(n, v, dtype=float) for v in (mcold, z, rdisk, rbulge)]


def test_cv_dust_matches_galform():
    # GALFORM's Dust_Read prints C_V = 2.1893 for these constants.
    np.testing.assert_allclose(CV_DUST, 2.1893, rtol=1e-4)


class TestDustAtten:
    def test_tauv0(self, table):
        mcold, z, rdisk, rbulge = gas_disk()
        _, _, tauv0 = dust_atten(table, mcold, z, rdisk, rbulge, 5500.0, H0, [30.0])
        h = rdisk / 1.678346990
        sigma = (mcold / H0) / (2 * np.pi * (1e6 * h / H0) ** 2)
        np.testing.assert_allclose(tauv0, CV_DUST * z * sigma)

    def test_no_gas_no_attenuation(self, table):
        mcold, z, rdisk, rbulge = gas_disk(mcold=0.0)
        disk, bulge, tauv0 = dust_atten(table, mcold, z, rdisk, rbulge, 3000.0, H0)
        np.testing.assert_allclose([disk, bulge, tauv0], [[1.0], [1.0], [0.0]])

    def test_non_positive_wavelength_is_not_attenuated(self, table):
        disk, bulge, _ = dust_atten(table, *gas_disk(), -999.0, H0, [30.0])
        np.testing.assert_array_equal([disk, bulge], [[1.0], [1.0]])

    def test_more_dust_bluer_and_edge_on_attenuate_more(self, table):
        def disk_atten(mcold=1e10, lam=5500.0, inc=30.0):
            return dust_atten(table, *gas_disk(mcold=mcold), lam, H0, [inc])[0][0]

        assert disk_atten(mcold=1e10) < disk_atten(mcold=1e9) < 1.0
        assert disk_atten(lam=2000.0) < disk_atten(lam=5500.0)
        assert disk_atten(inc=85.0) < disk_atten(inc=10.0)

    def test_average_between_face_on_and_edge_on(self, table):
        args = (table, *gas_disk(), 5500.0, H0)
        face = dust_atten(*args, [0.0])[0]
        edge = dust_atten(*args, [90.0])[0]
        ave = dust_atten(*args, None)[0]
        assert edge < ave < face

    def test_large_bulge_is_not_attenuated(self, table):
        # re/h above the table range: GALFORM sets bulge attenuation to 1.
        _, bulge, _ = dust_atten(
            table, *gas_disk(rdisk=0.001, rbulge=0.05), 5500.0, H0, [30.0]
        )
        np.testing.assert_allclose(bulge, 1.0)

    def test_power_law_extrapolation_above_grid(self, table):
        disk, _, tauv0 = dust_atten(table, *gas_disk(mcold=1e13), 5500.0, H0, [30.0])
        assert tauv0[0] > table.tauv0[-1]
        edge_of_grid = np.exp(-table.taueff[-1, 2, 2, 0, 0])
        assert 0.0 < disk[0] < edge_of_grid

    @pytest.mark.parametrize("lam", [50.0, 2.0e5])
    def test_wavelength_outside_table_raises(self, table, lam):
        with pytest.raises(ValueError, match="outside the dust table"):
            dust_atten(table, *gas_disk(), lam, H0, [30.0])

    @pytest.mark.parametrize("inc", [-5.0, 95.0])
    def test_inclination_outside_range_raises(self, table, inc):
        with pytest.raises(ValueError, match="between 0 and 90"):
            dust_atten(table, *gas_disk(), 5500.0, H0, [inc])

    def test_negative_metallicity_raises(self, table):
        with pytest.raises(ValueError, match="negative"):
            dust_atten(table, *gas_disk(z=-0.01), 5500.0, H0, [30.0])


class TestDiffuseAttenuation:
    def common(self, **overrides):
        kwargs = dict(
            mcold=np.array([1e10]),
            cold_metal=np.array([2e8]),
            rdisk=np.array([0.005]),
            rbulge=np.array([0.002]),
            mcold_burst=np.array([5e9]),
            metals_burst=np.array([1e8]),
            burst=np.array([False]),
            fcloud=0.5,
            rfacburst=1.0,
            lam=5500.0,
            h0=H0,
            inc_deg=np.array([30.0]),
        )
        kwargs.update(overrides)
        return kwargs

    def test_only_diffuse_gas_counts(self, table):
        disk, _ = diffuse_attenuation(table, **self.common(fcloud=0.5))
        direct, _, _ = dust_atten(
            table, [5e9], [0.02], [0.005], [0.002], 5500.0, H0, [30.0]
        )
        np.testing.assert_allclose(disk, direct)

    def test_burst_dims_only_bulge(self, table):
        quiet_disk, quiet_bulge = diffuse_attenuation(table, **self.common())
        disk, bulge = diffuse_attenuation(table, **self.common(burst=np.array([True])))
        np.testing.assert_allclose(disk, quiet_disk)
        assert bulge[0] < quiet_bulge[0]

    def test_rfacburst_zero_ignores_burst_dust(self, table):
        quiet = diffuse_attenuation(table, **self.common())
        off = diffuse_attenuation(
            table, **self.common(burst=np.array([True]), rfacburst=0.0)
        )
        np.testing.assert_allclose(off, quiet)

    def test_tiny_gas_mass_is_metal_free(self, table):
        disk, bulge = diffuse_attenuation(
            table,
            **self.common(mcold=np.array([0.5]), cold_metal=np.array([0.01])),
        )
        np.testing.assert_allclose([disk, bulge], [[1.0], [1.0]])


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


def _galform_table():
    try:
        return read_attenuation_table(find_dust_table("dust_MW_hz1.0.dat"))
    except FileNotFoundError:
        pytest.skip("GALFORM dust table dust_MW_hz1.0.dat not available")


class TestAgainstGalform:
    """GALFORM's own diffuse-dust magnitudes for 444 real lc16 galaxies.

    Reference values come from GALFORM's insert_gal_props run on an L800/lc16
    z = 0 subvolume with the Lacey16 dust parameters (fcloud = 0.5), in bands
    from GALEX-FUV to IRAC 8 micron (beyond the table: extrapolated); burst
    dust is excluded because that tool leaves it unset.
    """

    def magnitudes(self, table, ref, band, inc):
        disk, bulge = diffuse_attenuation(
            table,
            mcold=ref["mcold"],
            cold_metal=ref["cold_metal"],
            rdisk=ref["rdisk"],
            rbulge=ref["rbulge"],
            mcold_burst=np.zeros_like(ref["mcold"]),
            metals_burst=np.zeros_like(ref["mcold"]),
            burst=np.zeros(ref["mcold"].shape, dtype=bool),
            fcloud=0.5,
            rfacburst=1.0,
            lam=float(ref[f"lam_{band}"]),
            h0=float(ref["h0"]),
            inc_deg=inc,
        )
        ld = ref[f"Ld_{band}"].astype(float) * disk
        lb = ref[f"Lb_{band}"].astype(float) * bulge
        return [-2.5 * np.log10(np.maximum(lum, 1e-20)) for lum in (ld + lb, ld, lb)]

    @pytest.mark.parametrize("band", ["GA", "U", "B", "I", "IR"])
    def test_inclined_magnitudes(self, ref, band):
        table = _galform_table()
        total, _, _ = self.magnitudes(table, ref, band, ref["inc"])
        np.testing.assert_allclose(total, ref[f"mag_{band}_tot_ext"], atol=1e-5)

    def test_components_and_face_on(self, ref):
        table = _galform_table()
        _, disk, bulge = self.magnitudes(table, ref, "B", ref["inc"])
        np.testing.assert_allclose(disk, ref["mag_B_disk_ext"], atol=1e-5)
        np.testing.assert_allclose(bulge, ref["mag_B_bulge_ext"], atol=1e-5)
        face_on, _, _ = self.magnitudes(table, ref, "B", np.zeros_like(ref["inc"]))
        np.testing.assert_allclose(face_on, ref["mag_B_tot_ext0"], atol=1e-5)

    def test_tauv0(self, ref):
        table = _galform_table()
        with np.errstate(invalid="ignore", divide="ignore"):
            zcold = np.where(ref["mcold"] > 1.0, ref["cold_metal"] / ref["mcold"], 0.0)
        _, _, tauv0 = dust_atten(
            table,
            0.5 * ref["mcold"],
            zcold,
            ref["rdisk"],
            ref["rbulge"],
            5500.0,
            float(ref["h0"]),
            ref["inc"],
        )
        np.testing.assert_allclose(tauv0, ref["tauv0"], rtol=1e-5, atol=1e-12)
