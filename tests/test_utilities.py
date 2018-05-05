#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_utilities

test the functions located in utilities submodule for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 01, 2018
"""

import os
import unittest

import numpy as np

from intensity_normalization.utilities import io, mask


class TestUtilities(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.img = io.open_nii(os.path.join(self.data_dir, 'test.nii.gz'))
        self.brain_mask = io.open_nii(os.path.join(self.mask_dir, 'mask.nii.gz'))

    def test_fcm_mask(self):
        m = mask.fcm_class_mask(self.img, self.brain_mask, hard_seg=True)
        self.assertEqual(len(np.unique(m)), 4)

    def test_gmm_mask(self):
        wm_peak = mask.gmm_class_mask(self.img, self.brain_mask, return_wm_peak=True)
        self.assertAlmostEqual(wm_peak, 300)
        m = mask.gmm_class_mask(self.img, self.brain_mask, return_wm_peak=False, hard_seg=True)
        self.assertEqual(len(np.unique(m)), 4)
        m = mask.gmm_class_mask(self.img, self.brain_mask, return_wm_peak=False, hard_seg=False)
        self.assertEqual(m.shape[3], 3)

    def test_bg_mask(self):
        bkgd = mask.background_mask(self.img, seed=0).get_data()
        self.assertEqual(np.sum(bkgd), np.size(bkgd))

    def tearDown(self):
        del self.img, self.brain_mask


if __name__ == '__main__':
    unittest.main()
