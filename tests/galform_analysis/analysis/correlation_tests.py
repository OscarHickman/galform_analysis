import os
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np


class TestCorrelationModule(unittest.TestCase):
	def setUp(self):
		# Common rbins for tests
		self.rbins = np.logspace(np.log10(0.1), np.log10(5.0), 8)

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

