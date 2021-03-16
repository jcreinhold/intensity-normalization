#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_normalization

test the intensity normalization functions for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 02, 2018
"""

import os
import unittest

import numpy as np

try:
    import ants
except ImportError:
    ants = None

from intensity_normalization.normalize import fcm
from intensity_normalization.utilities import io


class TestNormalization(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.img = io.open_nii(os.path.join(self.data_dir, 'test.nii.gz'))
        self.brain_mask = io.open_nii(os.path.join(self.mask_dir, 'mask.nii.gz'))
        self.template_mask = os.path.join(self.mask_dir, 'mask.nii.gz')
        self.wm_mask = fcm.find_tissue_mask(self.img, self.brain_mask)
        self.norm_val = 1000

    def test_zscore_normalization(self):
        from intensity_normalization.normalize import zscore
        normalized = zscore.zscore_normalize(self.img, self.brain_mask)
        self.assertAlmostEqual(np.mean(normalized.get_fdata()[self.brain_mask.get_fdata() > 0]), 0, places=4)

    def test_fcm_normalization(self):
        normalized = fcm.fcm_normalize(self.img, self.wm_mask, norm_value=self.norm_val)
        self.assertAlmostEqual(normalized.get_fdata()[self.wm_mask.get_fdata() > 0.].mean(), self.norm_val, places=3)

    def test_gmm_normalization(self):
        from intensity_normalization.normalize import gmm
        normalized = gmm.gmm_normalize(self.img, self.brain_mask, norm_value=self.norm_val)
        self.assertAlmostEqual(normalized.get_fdata()[self.wm_mask.get_fdata() > 0.].mean(), self.norm_val, delta=20)

    def test_kde_normalization(self):
        from intensity_normalization.normalize import kde
        normalized = kde.kde_normalize(self.img, self.brain_mask, contrast='T1', norm_value=self.norm_val)
        self.assertAlmostEqual(normalized.get_fdata()[self.wm_mask.get_fdata() > 0.].mean(), self.norm_val, delta=20)

    def test_nyul_normalization(self):
        from intensity_normalization.normalize import nyul
        normalized = nyul.nyul_normalize(self.data_dir, write_to_disk=False)

    def test_lsq_normalization(self):
        from intensity_normalization.normalize import lsq
        normalized = lsq.lsq_normalize(self.data_dir, write_to_disk=False)

    def test_ws_normalization(self):
        from intensity_normalization.normalize import whitestripe
        normalized = whitestripe.ws_normalize(self.data_dir, 'T1', mask_dir=self.mask_dir, write_to_disk=False)
        self.assertEqual(np.sum(normalized.get_fdata().shape), np.sum(self.img.get_fdata().shape))

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_ravel_normalization(self):
        from intensity_normalization.normalize.ravel import ravel_normalize
        normalized = ravel_normalize(self.data_dir, self.mask_dir, 'T1', write_to_disk=False)

    @unittest.skipIf(ants is None, "ANTsPy is not installed on this system")
    def test_ravel_normalization_csf_masks(self):
        from intensity_normalization.normalize.ravel import ravel_normalize
        normalized = ravel_normalize(self.data_dir, self.mask_dir, 'T1', write_to_disk=False, csf_masks=True)

    def tearDown(self):
        del self.img, self.brain_mask


if __name__ == '__main__':
    unittest.main()
