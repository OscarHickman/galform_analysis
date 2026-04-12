import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


class TestCorrelationModule(unittest.TestCase):
	def setUp(self):
		# Common rbins for tests
		self.rbins = np.logspace(np.log10(0.1), np.log10(5.0), 8)

	@patch('galform_analysis.analysis.correlation.halo_sampling_correction.marked_tpcf')
	@patch('galform_analysis.analysis.correlation.halo_sampling_correction.npairs_3d')
	def test_compute_halo_sampling_corrected_xi(self, mock_npairs, mock_marked):
		from galform_analysis.analysis.correlation.halo_sampling_correction import compute_halo_sampling_corrected_xi

		gals = pd.DataFrame(
			{
				'xgal': [0.0, 0.5, 5.0, 5.5],
				'ygal': [0.0, 0.0, 0.0, 0.0],
				'zgal': [0.0, 0.0, 0.0, 0.0],
				'mstar': [10.0, 10.1, 10.2, 10.3],
				'halo_id': [1, 1, 2, 3],
			}
		)
		rbins = np.array([0.0, 1.0, 2.0])

		mock_npairs.return_value = np.array([0.0, 10.0, 16.0])
		mock_marked.return_value = np.array([0.4, 0.5])

		res = compute_halo_sampling_corrected_xi(
			gals,
			rbins=rbins,
			sampling_fraction=0.5,
			boxsize=10.0,
			mstar_min_log10=9.0,
			num_threads=1,
		)

		dd_all = np.array([10.0, 6.0])
		dd_1h = np.array([4.0, 3.0])
		dd_2h = dd_all - dd_1h
		dd_hat = dd_1h / 0.5 + dd_2h / (0.5 ** 2)
		n_full = 4.0 / 0.5
		shell = np.diff((4.0 / 3.0) * np.pi * rbins ** 3)
		expected = (dd_hat / (n_full ** 2)) / (shell / (10.0 ** 3)) - 1.0

		np.testing.assert_allclose(res, expected)

	@patch('galform_analysis.analysis.correlation.halo_sampling_correction.halo_mass_to_halo_radius')
	def test_load_halo_sampled_galaxies_prefers_treeid_when_no_valid_dhaloid(self, mock_rhalo):
		from galform_analysis.analysis.correlation.halo_sampling_correction import load_halo_sampled_galaxies

		mock_rhalo.return_value = np.ones(2)

		with tempfile.TemporaryDirectory() as tmpdir:
			gal_path = Path(tmpdir) / 'iz155' / 'ivol0'
			gal_path.mkdir(parents=True)
			with h5py.File(gal_path / 'galaxies.hdf5', 'w') as f:
				g = f.create_group('Output001')
				g.create_dataset('mhhalo', data=np.array([1e12, 2e12]))
				g.create_dataset('TreeID', data=np.array([101, 101]))
				g.create_dataset('DHaloID', data=np.array([-1, -1]))
				g.create_dataset('is_central', data=np.array([1, 0]))
				g.create_dataset('xgal', data=np.array([1.0, 1.1]))
				g.create_dataset('ygal', data=np.array([2.0, 2.1]))
				g.create_dataset('zgal', data=np.array([3.0, 3.1]))
				g.create_dataset('mstars_bulge', data=np.array([1e10, 5e9]))
				g.create_dataset('mstars_disk', data=np.array([1e10, 4e9]))

			gals = load_halo_sampled_galaxies(tmpdir, 155, np.array([0]), boxsize=542.16)

			self.assertEqual(len(gals), 2)
			self.assertTrue((gals['halo_id'].values == np.array([101, 101])).all())
			self.assertTrue((gals['halo_id_source'].values == np.array(['TreeID', 'TreeID'])).all())

	@patch('galform_analysis.analysis.correlation.group_sampling_correlation.halo_mass_to_halo_radius')
	def test_load_notebook_style_galaxies_uses_latest_output_group(self, mock_rhalo):
		from galform_analysis.analysis.correlation.group_sampling_correlation import load_notebook_style_galaxies

		mock_rhalo.return_value = np.ones(2)

		with tempfile.TemporaryDirectory() as tmpdir:
			gal_path = Path(tmpdir) / 'iz207' / 'ivol0'
			gal_path.mkdir(parents=True)
			with h5py.File(gal_path / 'galaxies.hdf5', 'w') as f:
				# Intentionally only create a higher-numbered output group.
				g = f.create_group('Output005')
				g.create_dataset('mhhalo', data=np.array([1e12, 1.2e12]))
				g.create_dataset('vhhalo', data=np.array([1.0, 2.0]))
				g.create_dataset('is_central', data=np.array([1, 0]))
				g.create_dataset('xgal', data=np.array([1.0, 1.1]))
				g.create_dataset('ygal', data=np.array([2.0, 2.1]))
				g.create_dataset('zgal', data=np.array([3.0, 3.1]))
				g.create_dataset('mstars_bulge', data=np.array([1e10, 5e9]))
				g.create_dataset('mstars_disk', data=np.array([1e10, 4e9]))

			gals = load_notebook_style_galaxies(
				base_dir=tmpdir,
				iz_num=207,
				ivols=np.array([0]),
				boxsize=542.16,
				mhalo_min=1e11,
			)

			self.assertEqual(len(gals), 2)
			self.assertTrue(np.isfinite(gals['dr_norm'].values).all())

	@patch('galform_analysis.analysis.correlation.group_sampling_correlation.halo_mass_to_halo_radius')
	def test_load_notebook_style_galaxies_falls_back_when_vhhalo_missing(self, mock_rhalo):
		from galform_analysis.analysis.correlation.group_sampling_correlation import load_notebook_style_galaxies

		mock_rhalo.return_value = np.ones(2)

		with tempfile.TemporaryDirectory() as tmpdir:
			gal_path = Path(tmpdir) / 'iz155' / 'ivol0'
			gal_path.mkdir(parents=True)
			with h5py.File(gal_path / 'galaxies.hdf5', 'w') as f:
				g = f.create_group('Output001')
				g.create_dataset('mhhalo', data=np.array([1e12, 2e12]))
				g.create_dataset('TreeID', data=np.array([77, 77]))
				g.create_dataset('is_central', data=np.array([1, 0]))
				g.create_dataset('xgal', data=np.array([1.0, 1.1]))
				g.create_dataset('ygal', data=np.array([2.0, 2.1]))
				g.create_dataset('zgal', data=np.array([3.0, 3.1]))
				g.create_dataset('mstars_bulge', data=np.array([1e10, 5e9]))
				g.create_dataset('mstars_disk', data=np.array([1e10, 4e9]))

			gals = load_notebook_style_galaxies(
				base_dir=tmpdir,
				iz_num=155,
				ivols=np.array([0]),
				boxsize=542.16,
				mhalo_min=1e11,
			)

			self.assertEqual(len(gals), 2)
			self.assertEqual(int(gals['igrp'].nunique()), 1)

	@patch('galform_analysis.analysis.correlation.group_sampling_correlation.halo_mass_to_halo_radius')
	def test_load_notebook_style_galaxies_handles_missing_centrals(self, mock_rhalo):
		from galform_analysis.analysis.correlation.group_sampling_correlation import load_notebook_style_galaxies

		mock_rhalo.return_value = np.ones(2)

		with tempfile.TemporaryDirectory() as tmpdir:
			gal_path = Path(tmpdir) / 'iz271' / 'ivol0'
			gal_path.mkdir(parents=True)
			with h5py.File(gal_path / 'galaxies.hdf5', 'w') as f:
				g = f.create_group('Output001')
				g.create_dataset('mhhalo', data=np.array([1e12, 1e12]))
				g.create_dataset('TreeID', data=np.array([88, 88]))
				# No flagged central survives this test sample.
				g.create_dataset('is_central', data=np.array([0, 0]))
				g.create_dataset('xgal', data=np.array([1.0, 1.1]))
				g.create_dataset('ygal', data=np.array([2.0, 2.1]))
				g.create_dataset('zgal', data=np.array([3.0, 3.1]))
				g.create_dataset('mstars_bulge', data=np.array([6e9, 2e9]))
				g.create_dataset('mstars_disk', data=np.array([7e9, 3e9]))

			gals = load_notebook_style_galaxies(
				base_dir=tmpdir,
				iz_num=271,
				ivols=np.array([0]),
				boxsize=542.16,
				mhalo_min=1e11,
			)

			self.assertEqual(len(gals), 2)
			self.assertTrue(np.isfinite(gals['dr_norm'].values).all())

	@patch('galform_analysis.analysis.correlation.correlation.corrfunc_DD')
	def test_compute_xi_corrfunc_basic(self, mock_dd):
		# Prepare Corrfunc DD mock output
		def dd_stub(**kwargs):
			# Return list of dicts with npairs and ravg per bin
			binfile = kwargs['binfile']
			r1 = np.array(binfile[:-1])
			r2 = np.array(binfile[1:])
			ravg = 0.5 * (r1 + r2)
			# Simple model: npairs proportional to shell volume
			vshell = (4.0/3.0) * np.pi * (r2**3 - r1**3)
			return [{'npairs': float(v), 'ravg': float(r)} for v, r in zip(vshell, ravg)]

		mock_dd.side_effect = dd_stub

		from galform_analysis.analysis.correlation.correlation import compute_xi_corrfunc

		# Create a small set of positions in a box
		rng = np.random.default_rng(123)
		pos = rng.uniform(0, 100.0, size=(500, 3))

		res = compute_xi_corrfunc(positions=pos, boxsize=100.0, rbins=self.rbins, nthreads=1)
		self.assertIn('r', res.columns)
		self.assertIn('xi', res.columns)
		self.assertEqual(len(res['r']), len(self.rbins) - 1)
		self.assertEqual(res['xi'].shape, res['r'].shape)
		self.assertEqual(res.attrs.get('ngal'), pos.shape[0])

	@patch('galform_analysis.analysis.correlation.correlation.read_snapshot_data')
	@patch('galform_analysis.analysis.correlation.correlation.read_galaxy_positions')
	@patch('galform_analysis.analysis.correlation.correlation.corrfunc_DD')
	def test_correlation_given_redshift_and_subvolume(self, mock_dd, mock_read_pos, mock_read_meta):
		# Mock positions and metadata
		pos = np.stack([np.linspace(0, 10, 100)] * 3, axis=1)
		mock_read_pos.return_value = (pos, 0.5)
		mock_read_meta.return_value = {'V_ivol': 100.0**3, 'z': 0.5}

		# Corrfunc stub
		def dd_stub(**kwargs):
			binfile = kwargs['binfile']
			r1 = np.array(binfile[:-1])
			r2 = np.array(binfile[1:])
			ravg = 0.5 * (r1 + r2)
			vshell = (4.0/3.0) * np.pi * (r2**3 - r1**3)
			return [{'npairs': float(v), 'ravg': float(r)} for v, r in zip(vshell, ravg)]
		mock_dd.side_effect = dd_stub

		from galform_analysis.analysis.correlation.correlation import correlation_given_redshift_and_subvolume

		res = correlation_given_redshift_and_subvolume(iz_path='/tmp/iz207', ivol=0, rbins=self.rbins, nthreads=1)
		self.assertIsNotNone(res)
		self.assertIn('r', res.columns)
		self.assertIn('xi', res.columns)
		self.assertAlmostEqual(res.attrs.get('z'), 0.5)
		self.assertEqual(res.attrs.get('ngal'), 100)

	@patch('galform_analysis.analysis.correlation.correlation.read_snapshot_data')
	@patch('galform_analysis.analysis.correlation.correlation.read_galaxy_positions')
	@patch('galform_analysis.analysis.correlation.correlation.corrfunc_DD')
	@patch('galform_analysis.analysis.correlation.correlation.get_base_dir')
	def test_avg_correlation_given_redshift_and_subvolumes(self, mock_base, mock_dd, mock_read_pos, mock_read_meta):
		mock_base.return_value = '/tmp/base'
		# Mock filesystem presence
		with patch('os.path.isdir', return_value=True):
			pos = np.stack([np.linspace(0, 10, 50)] * 3, axis=1)
			mock_read_pos.return_value = (pos, 1.01)
			mock_read_meta.return_value = {'V_ivol': 100.0**3, 'z': 1.01}

			# Corrfunc stub
			def dd_stub(**kwargs):
				binfile = kwargs['binfile']
				r1 = np.array(binfile[:-1])
				r2 = np.array(binfile[1:])
				ravg = 0.5 * (r1 + r2)
				vshell = (4.0/3.0) * np.pi * (r2**3 - r1**3)
				return [{'npairs': float(v), 'ravg': float(r)} for v, r in zip(vshell, ravg)]
			mock_dd.side_effect = dd_stub

			from galform_analysis.analysis.correlation.correlation import avg_correlation_given_redshift_and_subvolumes

			res = avg_correlation_given_redshift_and_subvolumes(
				iz_num=207,
				ivols=[0, 1, 2],
				rbins=self.rbins,
				nthreads=1,
				base_dir='/tmp/base'
			)
			self.assertIsNotNone(res)
			self.assertIn('xi', res.columns)
			self.assertIn('xi_std', res.columns)
			self.assertEqual(res.attrs.get('n_used'), 3)
			self.assertEqual(res.attrs.get('n_requested'), 3)

	@patch('galform_analysis.analysis.correlation.correlation.read_snapshot_data')
	@patch('galform_analysis.analysis.correlation.correlation.read_galaxy_positions')
	@patch('galform_analysis.analysis.correlation.correlation.corrfunc_DD')
	@patch('galform_analysis.analysis.correlation.correlation.get_base_dir')
	def test_correlations_given_redshifts_and_subvolume(self, mock_base, mock_dd, mock_read_pos, mock_read_meta):
		mock_base.return_value = '/tmp/base'
		with patch('os.path.isdir', return_value=True):
			pos = np.stack([np.linspace(0, 10, 60)] * 3, axis=1)
			mock_read_pos.return_value = (pos, 0.5)
			mock_read_meta.side_effect = lambda iz_path, ivol: {'V_ivol': 100.0**3, 'z': 0.5 if 'iz100' in iz_path else 3.05}

			def dd_stub(**kwargs):
				binfile = kwargs['binfile']
				r1 = np.array(binfile[:-1])
				r2 = np.array(binfile[1:])
				ravg = 0.5 * (r1 + r2)
				vshell = (4.0/3.0) * np.pi * (r2**3 - r1**3)
				return [{'npairs': float(v), 'ravg': float(r)} for v, r in zip(vshell, ravg)]
			mock_dd.side_effect = dd_stub

			from galform_analysis.analysis.correlation.correlation import correlations_given_redshifts_and_subvolume

			res_list = correlations_given_redshifts_and_subvolume(
				iz_nums=[100, 120], ivol=3, rbins=self.rbins, nthreads=1, base_dir='/tmp/base'
			)
			self.assertGreaterEqual(len(res_list), 2)
			zs = [r.attrs.get('z') for r in res_list]
			self.assertIn(0.5, zs)
			self.assertIn(3.05, zs)

	@patch('galform_analysis.analysis.correlation.correlation.read_snapshot_data')
	@patch('galform_analysis.analysis.correlation.correlation.read_galaxy_positions')
	@patch('galform_analysis.analysis.correlation.correlation.corrfunc_DD')
	@patch('galform_analysis.analysis.correlation.correlation.get_base_dir')
	def test_avg_correlation_given_subvolume_and_redshifts(self, mock_base, mock_dd, mock_read_pos, mock_read_meta):
		mock_base.return_value = '/tmp/base'
		with patch('os.path.isdir', return_value=True):
			pos = np.stack([np.linspace(0, 10, 40)] * 3, axis=1)
			mock_read_pos.return_value = (pos, 2.0)
			mock_read_meta.return_value = {'V_ivol': 100.0**3, 'z': 2.0}

			def dd_stub(**kwargs):
				binfile = kwargs['binfile']
				r1 = np.array(binfile[:-1])
				r2 = np.array(binfile[1:])
				ravg = 0.5 * (r1 + r2)
				vshell = (4.0/3.0) * np.pi * (r2**3 - r1**3)
				return [{'npairs': float(v), 'ravg': float(r)} for v, r in zip(vshell, ravg)]
			mock_dd.side_effect = dd_stub

			from galform_analysis.analysis.correlation.correlation import avg_correlation_given_subvolume_and_redshifts

			res = avg_correlation_given_subvolume_and_redshifts(
				iz_nums=[100, 142, 207], ivol=5, rbins=self.rbins, nthreads=1, base_dir='/tmp/base'
			)
			self.assertIsNotNone(res)
			self.assertIn('xi', res.columns)
			self.assertIn('xi_std', res.columns)
			self.assertEqual(res.attrs.get('n_used'), 3)
			self.assertEqual(res.attrs.get('used_iz'), ['iz100', 'iz142', 'iz207'])


if __name__ == '__main__':
	unittest.main()

