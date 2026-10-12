"""Tests for dust-attenuated magnitudes read from galaxies.hdf5."""

import dataclasses

import h5py
import numpy as np
import pytest

from galform_analysis.analysis.luminosity_functions import (
    avg_lf_given_redshift_and_subvolumes,
    avg_lf_given_redshifts_and_subvolume,
    lf_given_redshift_and_subvolume,
)
from galform_analysis.dust import LACEY16, band_attenuation, inclinations
from galform_analysis.dust.galaxy import resolve_dust_model, rest_wavelength
from galform_analysis.readers import read_band_magnitudes
from tests.band_catalogue import magnitudes_to_luminosity, write_band_catalogue
from tests.dust_table import write_dust_table

N = 200
K_REST, R_REST, SPECIAL, K_OBS = 1, 2, 3, 4
LAMBDA_EFF = [22000.0, 6200.0, 5500.0, 22000.0, 6200.0]


def gas(n=N, **overrides):
    fields = dict(
        mcold=np.full(n, 1e10),
        cold_metal=np.full(n, 2e8),
        rdisk=np.full(n, 0.005),
        rbulge=np.full(n, 0.002),
        mcold_burst=np.zeros(n),
        metals_burst=np.zeros(n),
        tburst=np.full(n, 10.0),
        tsfburst=np.full(n, 1.0),
    )
    fields.update(overrides)
    return fields


@pytest.fixture
def model(tmp_path):
    table = write_dust_table(tmp_path / "dust.dat")
    return dataclasses.replace(LACEY16, table=str(table))


def write(tmp_path, ivol=0, n=N, redshift=0.0, gas_fields=None, **kwargs):
    lum = magnitudes_to_luminosity(np.full(n, -20.0))
    lums = {i: 0.5 * lum for i in range(1, 6)}
    return write_band_catalogue(
        tmp_path,
        271,
        ivol,
        lum_disk=lums,
        lum_bulge=lums,
        redshift=redshift,
        gas=gas(n) if gas_fields is None else gas_fields,
        parameters={"h0": 0.6777, "efold": 3.0},
        lambda_eff=LAMBDA_EFF,
        **kwargs,
    )


class TestInclinations:
    def test_random_is_isotropic_and_seeded(self):
        inc = inclinations("random", 100_000, seed=3)
        np.testing.assert_array_equal(inc, inclinations("random", 100_000, seed=3))
        assert not np.array_equal(inc, inclinations("random", 100_000, seed=4))
        assert np.all((inc >= 0) & (inc <= 90))
        # cos(i) is uniform, so its mean is 1/2.
        assert abs(np.cos(np.radians(inc)).mean() - 0.5) < 0.01

    def test_fixed_modes(self):
        assert inclinations("average", 3) is None
        np.testing.assert_array_equal(inclinations("face-on", 2), [0.0, 0.0])
        np.testing.assert_array_equal(inclinations(60.0, 2), [60.0, 60.0])

    @pytest.mark.parametrize("bad", ["edge-on", -1.0, 91.0])
    def test_invalid(self, bad):
        with pytest.raises(ValueError, match="inclination"):
            inclinations(bad, 3)


def test_resolve_dust_model():
    assert resolve_dust_model("lacey16") is LACEY16
    assert resolve_dust_model(LACEY16) is LACEY16
    with pytest.raises(ValueError, match="unknown dust model"):
        resolve_dust_model("calzetti")


def test_dust_model_validation():
    with pytest.raises(ValueError, match="fcloud"):
        dataclasses.replace(LACEY16, fcloud=1.5)
    with pytest.raises(ValueError, match="rfacburst"):
        dataclasses.replace(LACEY16, rfacburst=-1.0)


class TestBandAttenuation:
    def open(self, iz_dir):
        f = h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "r")
        return f, f["Output001"]

    def test_rest_wavelength(self, tmp_path):
        f, _ = self.open(write(tmp_path))
        with f:
            assert rest_wavelength(f, K_REST, 1.0) == 22000.0
            assert rest_wavelength(f, K_OBS, 1.0) == pytest.approx(11000.0)
            assert rest_wavelength(f, SPECIAL, 1.0) < 0
            with pytest.raises(ValueError, match="redshift"):
                rest_wavelength(f, K_OBS, None)

    def test_stellar_pop_z_ref_blocks_observer_frame(self, tmp_path):
        iz_dir = write(tmp_path)
        with h5py.File(iz_dir / "ivol0" / "galaxies.hdf5", "a") as f:
            f["Bands"].create_dataset("stellar_pop_z_ref", data=np.float32(0.5))
            with pytest.raises(ValueError, match="stellar_pop_z_ref"):
                rest_wavelength(f, K_OBS, 0.5)

    def test_bluer_band_attenuated_more(self, tmp_path, model):
        f, g = self.open(write(tmp_path))
        with f:
            k_disk, _ = band_attenuation(f, g, K_REST, model, 0.0)
            r_disk, _ = band_attenuation(f, g, R_REST, model, 0.0)
            special = band_attenuation(f, g, SPECIAL, model, 0.0)
        assert np.all(r_disk < k_disk) and np.all(k_disk < 1.0)
        np.testing.assert_array_equal(special, np.ones((2, N)))

    def test_ongoing_burst_dims_bulge(self, tmp_path, model):
        bursting = gas(
            mcold_burst=np.full(N, 5e9),
            metals_burst=np.full(N, 1e8),
            tburst=np.full(N, 0.1),
        )
        f, g = self.open(write(tmp_path, gas_fields=bursting))
        with f:
            disk, bulge = band_attenuation(f, g, R_REST, model, 0.0, "face-on")
        quiet_dir = write(tmp_path / "quiet")
        f, g = self.open(quiet_dir)
        with f:
            quiet_disk, quiet_bulge = band_attenuation(
                f, g, R_REST, model, 0.0, "face-on"
            )
        np.testing.assert_allclose(disk, quiet_disk)
        assert np.all(bulge < quiet_bulge)

    def test_missing_gas_fields(self, tmp_path, model):
        fields = gas()
        del fields["rdisk"]
        f, g = self.open(write(tmp_path, gas_fields=fields))
        with f, pytest.raises(KeyError, match="rdisk"):
            band_attenuation(f, g, K_REST, model, 0.0)

    def test_missing_table(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GALFORM_DATA_DIR", raising=False)
        f, g = self.open(write(tmp_path))
        with f, pytest.raises(FileNotFoundError, match="GALFORM_DATA_DIR"):
            band_attenuation(f, g, K_REST, "lacey16", 0.0)


class TestReadBandMagnitudes:
    def test_intrinsic_by_default(self, tmp_path):
        res = read_band_magnitudes(str(write(tmp_path)), 0, "SDSS-r")
        np.testing.assert_allclose(res["mag"], -20.0, atol=1e-5)
        assert res["dust"] is None and res["inclination"] is None

    @pytest.mark.parametrize("component", ["total", "disk", "bulge"])
    def test_dust_makes_galaxies_fainter(self, tmp_path, model, component):
        iz = str(write(tmp_path))
        bare = read_band_magnitudes(iz, 0, "SDSS-r", component=component)
        dusty = read_band_magnitudes(iz, 0, "SDSS-r", component=component, dust=model)
        assert np.all(dusty["mag"] > bare["mag"])
        assert dusty["dust"] == "lacey16"
        assert dusty["inclination"] == "random"

    def test_same_orientation_in_every_band_and_new_per_subvolume(
        self, tmp_path, model
    ):
        iz = str(write(tmp_path, ivol=0))
        write(tmp_path, ivol=1)
        r0 = read_band_magnitudes(iz, 0, "SDSS-r", dust=model)["mag"]
        k0 = read_band_magnitudes(iz, 0, "UKIRT-K", dust=model)["mag"]
        r1 = read_band_magnitudes(iz, 1, "SDSS-r", dust=model)["mag"]
        # Identical galaxies differ only by orientation: the most attenuated
        # galaxy in r is also the most attenuated in K.
        assert np.argmax(r0) == np.argmax(k0)
        assert not np.array_equal(r0, r1)

    def test_face_on_is_least_attenuated(self, tmp_path, model):
        iz = str(write(tmp_path))
        mags = {
            inc: read_band_magnitudes(iz, 0, "SDSS-r", dust=model, inclination=inc)[
                "mag"
            ][0]
            for inc in ("face-on", "average", 90.0)
        }
        assert mags["face-on"] < mags["average"] < mags[90.0]

    def test_invalid_options_raise_before_reading(self, tmp_path):
        missing = str(tmp_path / "iz1")
        with pytest.raises(ValueError, match="dust model"):
            read_band_magnitudes(missing, 0, "SDSS-r", dust="calzetti")
        with pytest.raises(ValueError, match="inclination"):
            read_band_magnitudes(
                missing, 0, "SDSS-r", dust="lacey16", inclination="sideways"
            )


class TestLuminosityFunctionWithDust:
    BINS = np.arange(-21.0, -18.0, 0.25)

    def test_counts_move_faintwards(self, tmp_path, model):
        iz = str(write(tmp_path))
        bare = lf_given_redshift_and_subvolume(iz, 0, "SDSS-r", bins=self.BINS)
        dusty = lf_given_redshift_and_subvolume(
            iz, 0, "SDSS-r", bins=self.BINS, dust=model
        )
        assert bare["counts"].sum() == dusty["counts"].sum() == N
        centre = lambda r: np.average(r["centers"], weights=r["counts"])  # noqa: E731
        assert centre(dusty) > centre(bare)
        assert dusty["dust"] == "lacey16"
        assert dusty["inclination"] == "random"

    def test_averages_pass_dust_through(self, tmp_path, model):
        write(tmp_path, ivol=0)
        write(tmp_path, ivol=1)
        kwargs = dict(bins=self.BINS, dust=model, base_dir=str(tmp_path))
        over_ivols = avg_lf_given_redshift_and_subvolumes(
            271, [0, 1], "SDSS-r", inclination="face-on", **kwargs
        )
        over_iz = avg_lf_given_redshifts_and_subvolume(
            0, [271], "SDSS-r", inclination="face-on", **kwargs
        )
        assert over_ivols["dust"] == over_iz["dust"] == "lacey16"
        assert over_ivols["inclination"] == "face-on"
        np.testing.assert_allclose(over_ivols["phi"], over_iz["phi"])

    def test_invalid_dust_raises_without_data(self, tmp_path):
        with pytest.raises(ValueError, match="dust model"):
            lf_given_redshift_and_subvolume(
                str(tmp_path / "iz1"), 0, "SDSS-r", dust="calzetti"
            )
