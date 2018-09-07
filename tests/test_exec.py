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
        zscore(args)

    def test_fcm_normalization_cli(self):
        args = self.args
        fcm(args)

    def test_gmm_normalization_cli(self):
        args = self.args
        gmm(args)

    def test_kde_normalization_cli(self):
        args = self.args
        kde(args)

    def test_hm_normalization_cli(self):
        args = self.args
        hm(args)

    def test_ws_normalization_cli(self):
        args = self.args
        ws(args)

    def test_ravel_normalization_cli_no_register(self):
        args = self.args
        ravel(args)

    def test_ravel_normalization_cli_register(self):
        args = self.args + ['--do-registration']
        ravel(args)

    def test_coregister_cli(self):
        args = f'-i {self.data_dir} -t {self.data_dir} -o {self.out_dir}'.split()
        coregister(args)

    def test_plot_hists_cli(self):
        args = f'-i {self.data_dir} -m {self.mask_dir} -o {self.out_dir}/hist.png'.split()
        plot_hists(args)

    def test_preprocess_cli(self):
        args = self.args
        preprocess(args)

    def test_tissue_mask_cli(self):
        args = self.args + ['--memberships']
        tissue_mask(args)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
