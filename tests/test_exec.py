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

from intensity_normalization.exec.fcm_normalize import main as fcm
from intensity_normalization.exec.gmm_normalize import main as gmm
from intensity_normalization.exec.hm_normalize import main as hm
from intensity_normalization.exec.kde_normalize import main as kde
from intensity_normalization.exec.ravel_normalize import main as ravel
from intensity_normalization.exec.ws_normalize import main as ws
from intensity_normalization.exec.zscore_normalize import main as zscore

from intensity_normalization.exec.coregister import main as coregister
from intensity_normalization.exec.plot_hists import main as plot_hists
from intensity_normalization.exec.preprocess import main as preprocess
from intensity_normalization.exec.tissue_mask import main as tissue_mask


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.out_dir = tempfile.mkdtemp()
        self.args = f'-i {self.data_dir} -m {self.mask_dir} -o {self.out_dir}'.split()

    def test_zscore_normalization_cli(self):
        args = self.args
        retval = zscore(args)
        self.assertEqual(retval, 0)

    def test_fcm_normalization_cli(self):
        args = self.args
        retval = fcm(args)
        self.assertEqual(retval, 0)

    def test_gmm_normalization_cli(self):
        args = self.args
        retval = gmm(args)
        self.assertEqual(retval, 0)

    def test_kde_normalization_cli(self):
        args = self.args
        retval = kde(args)
        self.assertEqual(retval, 0)

    def test_hm_normalization_cli(self):
        args = self.args
        retval = hm(args)
        self.assertEqual(retval, 0)

    def test_ws_normalization_cli(self):
        args = self.args
        retval = ws(args)
        self.assertEqual(retval, 0)

    def test_ravel_normalization_cli_no_register(self):
        args = self.args + ['--no-registration', '-s', '0.1']
        retval = ravel(args)
        self.assertEqual(retval, 0)

    def test_ravel_normalization_cli_register(self):
        args = self.args
        retval = ravel(args)
        self.assertEqual(retval, 0)

    def test_ravel_normalization_cli_fcm(self):
        args = self.args + ['--no-registration', '--use-fcm']
        retval = ravel(args)
        self.assertEqual(retval, 0)

    def test_coregister_cli(self):
        args = f'-i {self.data_dir} -t {self.data_dir} -o {self.out_dir}'.split()
        retval = coregister(args)
        self.assertEqual(retval, 0)

    def test_plot_hists_cli(self):
        args = f'-i {self.data_dir} -m {self.mask_dir} -o {self.out_dir}/hist.png'.split()
        retval = plot_hists(args)
        self.assertEqual(retval, 0)

    def test_preprocess_cli(self):
        args = self.args
        retval = preprocess(args)
        self.assertEqual(retval, 0)

    def test_tissue_mask_cli(self):
        args = self.args + ['--memberships']
        retval = tissue_mask(args)
        self.assertEqual(retval, 0)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
