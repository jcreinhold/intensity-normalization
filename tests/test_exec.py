#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_exec

test the intensity normalization command line interfaces for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 06, 2018
"""

import os
import shutil
import tempfile
import unittest

try:
    import ants
except ImportError:
    ants = None


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.out_dir = tempfile.mkdtemp()
        self.args = f'-i {self.data_dir} -m {self.mask_dir} -o {self.out_dir}'.split()

    def test_zscore_normalization_cli(self):
        from intensity_normalization.exec.zscore_normalize import main as zscore
        args = self.args
        retval = zscore(args)
        self.assertEqual(retval, 0)

    def test_fcm_normalization_cli(self):
        from intensity_normalization.exec.fcm_normalize import main as fcm
        args = f'-i {self.data_dir}/test.nii.gz -m {self.mask_dir}/mask.nii.gz -s -o {self.out_dir}'.split()
        retval = fcm(args)
        self.assertEqual(retval, 0)

    def test_fcm_single_img_normalization_cli(self):
        from intensity_normalization.exec.fcm_normalize import main as fcm
        args = self.args
        retval = fcm(args)
        self.assertEqual(retval, 0)

    def test_gmm_normalization_cli(self):
        from intensity_normalization.exec.gmm_normalize import main as gmm
        args = self.args
        retval = gmm(args)
        self.assertEqual(retval, 0)

    def test_kde_normalization_cli(self):
        from intensity_normalization.exec.kde_normalize import main as kde
        args = self.args
        retval = kde(args)
        self.assertEqual(retval, 0)

    def test_lsq_normalization_cli(self):
        from intensity_normalization.exec.lsq_normalize import main as lsq
        args = self.args
        retval = lsq(args)
        self.assertEqual(retval, 0)

    def test_hm_normalization_cli(self):
        from intensity_normalization.exec.nyul_normalize import main as nyul
        args = self.args
        retval = nyul(args)
        self.assertEqual(retval, 0)

    def test_nyul_normalization_save_sh_cli(self):
        from intensity_normalization.exec.nyul_normalize import main as nyul
        args = self.args + f'-sh {self.out_dir}/sh.npy'.split()
        retval = nyul(args)
        self.assertEqual(retval, 0)
        retval = nyul(args)
        self.assertEqual(retval, 0)

    def test_ws_normalization_cli(self):
        from intensity_normalization.exec.ws_normalize import main as ws
        args = self.args
        retval = ws(args)
        self.assertEqual(retval, 0)

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_ravel_normalization_cli_no_register(self):
        from intensity_normalization.exec.ravel_normalize import main as ravel
        args = self.args + ['--no-registration', '-s', '0.1']
        retval = ravel(args)
        self.assertEqual(retval, 0)

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_ravel_normalization_cli_register(self):
        from intensity_normalization.exec.ravel_normalize import main as ravel
        args = self.args
        retval = ravel(args)
        self.assertEqual(retval, 0)

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_ravel_normalization_cli_fcm(self):
        from intensity_normalization.exec.ravel_normalize import main as ravel
        args = self.args + ['--no-registration', '--use-atropos']
        retval = ravel(args)
        self.assertEqual(retval, 0)

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_coregister_cli(self):
        from intensity_normalization.exec.coregister import main as coregister
        args = f'-i {self.data_dir} -t {self.data_dir} -o {self.out_dir}'.split()
        retval = coregister(args)
        self.assertEqual(retval, 0)

    def test_norm_quality_cli(self):
        from intensity_normalization.exec.norm_quality import main as norm_quality
        args = f'-i {self.data_dir} -m {self.mask_dir} -o {self.out_dir}/pairwisejsd.png'.split()
        retval = norm_quality(args)
        self.assertEqual(retval, 0)

    def test_plot_hists_cli(self):
        from intensity_normalization.exec.plot_hists import main as plot_hists
        args = f'-i {self.data_dir} -m {self.mask_dir} -o {self.out_dir}/hist.png'.split()
        retval = plot_hists(args)
        self.assertEqual(retval, 0)

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_preprocess_cli(self):
        from intensity_normalization.exec.preprocess import main as preprocess
        args = self.args
        retval = preprocess(args)
        self.assertEqual(retval, 0)

    def test_tissue_mask_cli(self):
        from intensity_normalization.exec.tissue_mask import main as tissue_mask
        args = self.args + ['--memberships']
        retval = tissue_mask(args)
        self.assertEqual(retval, 0)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
