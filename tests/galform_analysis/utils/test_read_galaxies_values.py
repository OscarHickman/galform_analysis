"""Value-level tests for utils/read_galaxies.py against hand-built files."""

import h5py
import numpy as np
import pytest

from galform_analysis.utils.read_galaxies import (
    read_galaxies_dataframe,
    read_galaxy_arrays,
    read_galaxy_positions,
    read_halo_arrays,
    read_halo_positions,
)

# Six galaxies: three centrals (is_central == 1) and three satellites.
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
IS_CENTRAL = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
DISK = np.array([1e9, 2e9, 3e9, 4e9, 5e9, 6e9])
BULGE = np.array([1e8, 1e8, 1e8, 1e8, 1e8, 1e8])
MHALO = np.array([1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
MHHALO = MHALO * 2.0


def write_catalogue(iz_dir, ivol=0, drop=(), extra=None, volume=1000.0):
    ivol_dir = iz_dir / f"ivol{ivol}"
    ivol_dir.mkdir(parents=True, exist_ok=True)
    path = ivol_dir / "galaxies.hdf5"
    columns = {
        "xgal": X,
        "ygal": X + 10.0,
        "zgal": X + 20.0,
        "mstars_disk": DISK,
        "mstars_bulge": BULGE,
        "mhalo": MHALO,
        "mhhalo": MHHALO,
        "mstardot": X / 10.0,
        "is_central": IS_CENTRAL,
        **(extra or {}),
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("CompletionFlag", data=np.int32(1))
        if volume is not None:
            f.create_group("Parameters").create_dataset("volume", data=volume)
        g = f.create_group("Output001")
        g.create_dataset("redshift", data=1.5)
        for name, data in columns.items():
            if name not in drop:
                g.create_dataset(name, data=data)
    return path


@pytest.fixture
def iz_dir(tmp_path):
    d = tmp_path / "iz100"
    write_catalogue(d)
    return d


class TestReadGalaxyArrays:
    def test_values_for_all_galaxies(self, iz_dir):
        arrays, meta = read_galaxy_arrays(str(iz_dir), 0, centrals_only=False)
        np.testing.assert_array_equal(arrays["x"], X)
        np.testing.assert_array_equal(arrays["y"], X + 10.0)
        np.testing.assert_array_equal(arrays["z"], X + 20.0)
        np.testing.assert_array_equal(arrays["mstar"], DISK + BULGE)
        np.testing.assert_array_equal(arrays["mhalo"], MHALO)
        np.testing.assert_array_equal(arrays["sfr"], X / 10.0)
        assert meta == {
            "iz": "iz100",
            "ivol": 0,
            "z": 1.5,
            "V_ivol": 1000.0,
            "V_total": 1000.0 * 1024,
            "n_subvolumes": 1024,
        }

    def test_centrals_and_cuts_select_expected_rows(self, iz_dir):
        arrays, _ = read_galaxy_arrays(
            str(iz_dir), 0, centrals_only=True, mhalo_min=1e12, mstar_min=4e9
        )
        np.testing.assert_array_equal(arrays["x"], [5.0])  # central, 1e14, 5.1e9

    def test_missing_derived_field_does_not_drop_galaxies(self, tmp_path):
        """A missing optional column (here the SFR) must not empty the sample."""
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("mstardot",))
        arrays, _ = read_galaxy_arrays(str(d), 0, centrals_only=False)
        assert "sfr" not in arrays
        np.testing.assert_array_equal(arrays["x"], X)
        pos, _ = read_galaxy_positions(str(d), 0, centrals_only=True)
        assert pos.shape == (3, 3)

    def test_empty_catalogue_returns_empty_arrays(self, tmp_path):
        d = tmp_path / "iz100" / "ivol0"
        d.mkdir(parents=True)
        with h5py.File(d / "galaxies.hdf5", "w") as f:
            g = f.create_group("Output001")
            for name in ("xgal", "ygal", "zgal", "mstars_disk", "mstars_bulge"):
                g.create_dataset(name, data=np.zeros(0))
            g.create_dataset("is_central", data=np.zeros(0, dtype=np.int32))
        arrays, _ = read_galaxy_arrays(str(tmp_path / "iz100"), 0, mstar_min=1.0)
        assert set(arrays) == {"x", "y", "z", "mstar", "is_central"}
        assert all(len(v) == 0 for v in arrays.values())

    def test_mass_cut_on_missing_field_raises(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("mhalo",))
        with pytest.raises(KeyError, match="mhalo"):
            read_galaxy_arrays(str(d), 0, mhalo_min=1e11)
        write_catalogue(d, drop=("mstars_disk", "mstars_bulge"))
        with pytest.raises(KeyError, match="mstar"):
            read_galaxy_arrays(str(d), 0, mstar_min=1e9)

    def test_inconsistent_lengths_raise(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, extra={"mhalo": MHALO[:4]})
        with pytest.raises(ValueError, match="lengths"):
            read_galaxy_arrays(str(d), 0, centrals_only=False)

    def test_extra_fields_are_read_and_filtered(self, iz_dir):
        arrays, _ = read_galaxy_arrays(
            str(iz_dir), 0, fields=["mhhalo", "mhalo", "not_a_field"]
        )
        np.testing.assert_array_equal(arrays["mhhalo"], MHHALO[IS_CENTRAL == 1])
        assert "not_a_field" not in arrays

    def test_fields_only(self, iz_dir):
        arrays, _ = read_galaxy_arrays(
            str(iz_dir),
            0,
            fields=["mhhalo"],
            include_positions=False,
            include_derived=False,
            centrals_only=False,
        )
        assert list(arrays) == ["mhhalo"]
        np.testing.assert_array_equal(arrays["mhhalo"], MHHALO)

    def test_centrals_only_without_is_central_raises(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("is_central",))
        with pytest.raises(KeyError, match="is_central"):
            read_galaxy_arrays(str(d), 0, centrals_only=True)

    def test_missing_positions_raise(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("zgal",))
        with pytest.raises(KeyError, match="position"):
            read_galaxy_arrays(str(d), 0)

    def test_missing_output_group_raises(self, tmp_path):
        d = tmp_path / "iz100" / "ivol0"
        d.mkdir(parents=True)
        h5py.File(d / "galaxies.hdf5", "w").close()
        with pytest.raises(RuntimeError, match="OutputNNN"):
            read_galaxy_arrays(str(tmp_path / "iz100"), 0)
        with pytest.raises(RuntimeError, match="OutputNNN"):
            read_halo_arrays(str(tmp_path / "iz100"), 0)

    def test_metadata_without_volume(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, volume=None)
        _, meta = read_galaxy_arrays(str(d), 0)
        assert meta["V_ivol"] is None and meta["V_total"] is None


class TestDataFrame:
    def test_columns_and_mstar_cut(self, iz_dir):
        df, meta = read_galaxies_dataframe(
            str(iz_dir), 0, centrals_only=False, mstar_min=3e9, return_metadata=True
        )
        np.testing.assert_array_equal(df["x"].to_numpy(), [3.0, 4.0, 5.0, 6.0])
        assert df.attrs == meta
        assert meta["z"] == 1.5


class TestHalos:
    def test_halo_arrays_are_centrals_with_host_mass(self, iz_dir):
        arrays, meta = read_halo_arrays(str(iz_dir), 0, mhhalo_min=1e12)
        np.testing.assert_array_equal(arrays["x"], [3.0, 5.0])
        np.testing.assert_array_equal(arrays["mhhalo"], [2e12, 2e14])
        assert meta["V_total"] == pytest.approx(1000.0 * 1024)

    def test_halo_positions(self, iz_dir):
        pos, z = read_halo_positions(str(iz_dir), 0)
        expected = np.column_stack([X, X + 10.0, X + 20.0])[IS_CENTRAL == 1]
        np.testing.assert_array_equal(pos, expected)
        assert pos.dtype == np.float64
        assert z == 1.5

    def test_halo_extra_fields(self, iz_dir):
        arrays, _ = read_halo_arrays(str(iz_dir), 0, fields=["mhalo", "mhhalo"])
        np.testing.assert_array_equal(arrays["mhalo"], MHALO[IS_CENTRAL == 1])

    def test_halo_requires_is_central(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("is_central",))
        with pytest.raises(KeyError, match="is_central"):
            read_halo_arrays(str(d), 0)

    def test_halo_mass_cut_without_host_mass_raises(self, tmp_path):
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("mhhalo",))
        with pytest.raises(KeyError, match="mhhalo"):
            read_halo_arrays(str(d), 0, mhhalo_min=1e12)

    def test_halo_missing_file_and_positions(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_halo_arrays(str(tmp_path), 0)
        d = tmp_path / "iz100"
        write_catalogue(d, drop=("xgal",))
        with pytest.raises(KeyError, match="position"):
            read_halo_arrays(str(d), 0)
