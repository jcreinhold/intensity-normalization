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

from intensity_normalization.normalize import zscore, fcm, gmm, kde, hm, whitestripe, ravel
from intensity_normalization.utilities import io


class TestNormalization(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.control_dir = os.path.join(wd, 'test_data', 'control')
        self.img = io.open_nii(os.path.join(self.data_dir, 'test.nii.gz'))
        self.brain_mask = io.open_nii(os.path.join(self.mask_dir, 'mask.nii.gz'))
        self.template_mask = os.path.join(self.mask_dir, 'mask.nii.gz')
        self.wm_mask = fcm.find_wm_mask(self.img, self.brain_mask)
        self.csf_mask = os.path.join(self.control_dir, 'csf_mask.nii.gz')
        self.norm_val = 1000

    def test_zscore_normalization(self):
        normalized = zscore.zscore_normalize(self.img, self.brain_mask)
        self.assertAlmostEqual(np.mean(normalized.get_data()[self.brain_mask.get_data() == 1]), 0, places=4)

    def test_fcm_normalization(self):
        normalized = fcm.fcm_normalize(self.img, self.wm_mask, norm_value=self.norm_val)
        self.assertAlmostEqual(normalized.get_data()[self.wm_mask.get_data()].mean(), self.norm_val, places=3)

    def test_gmm_normalization(self):
        normalized = gmm.gmm_normalize(self.img, self.brain_mask, norm_value=self.norm_val)
        self.assertAlmostEqual(normalized.get_data()[self.wm_mask.get_data()].mean(), self.norm_val, delta=20)

    def test_kde_normalization(self):
        normalized = kde.kde_normalize(self.img, self.brain_mask, contrast='T1', norm_value=self.norm_val)
        self.assertAlmostEqual(normalized.get_data()[self.wm_mask.get_data()].mean(), self.norm_val, delta=20)

    def test_hm_normalization(self):
        normalized = hm.hm_normalize(self.data_dir, self.template_mask, 'T1', write_to_disk=False)

    def test_ws_normalization(self):
        normalized = whitestripe.ws_normalize(self.data_dir, 'T1', mask_dir=self.mask_dir, write_to_disk=False)
        self.assertEqual(np.sum(normalized.get_data().shape), np.sum(self.img.get_data().shape))

    def test_ravel_normalization(self):
        normalized = ravel.ravel_normalize(self.data_dir, self.template_mask, self.csf_mask, 'T1', write_to_disk=False)

    def tearDown(self):
        del self.img, self.brain_mask


if __name__ == '__main__':
    unittest.main()
