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

from intensity_normalization.normalize import fcm, gmm, kde
from intensity_normalization.utilities import io


class TestNormalization(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.img = io.open_nii(os.path.join(wd, 'test_data/test.nii.gz'))
        self.brain_mask = io.open_nii(os.path.join(wd, 'test_data/mask.nii.gz'))
        self.norm_val = 1000

    def test_fcm_normalization(self):
        wm_mask = fcm.find_wm_mask(self.img, self.brain_mask)
        normalized = fcm.fcm_normalize(self.img, wm_mask, norm_value=self.norm_val)
        self.assertEqual(np.max(normalized.get_data()), self.norm_val)

    def test_gmmm_normalization(self):
        normalized = gmm.gmm_normalize(self.img, self.brain_mask, norm_value=self.norm_val)
        self.assertEqual(np.max(normalized.get_data()), self.norm_val)

    def test_kde_normalization(self):
        normalized = kde.kde_normalize(self.img, self.brain_mask, contrast='T1', norm_value=self.norm_val)
        # testing data only has one voxel at maximum intensity, so peak found at "GM"
        self.assertAlmostEqual(np.max(normalized.get_data()), 1498.0831, places=4)

    def tearDown(self):
        del self.img, self.brain_mask


if __name__ == '__main__':
    unittest.main()
