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

from intensity_normalization.normalize import fcm
from intensity_normalization.utilities import io


class TestNormalization(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.img = io.open_nii(os.path.join(wd, 'test_data/test.nii.gz'))
        self.brain_mask = io.open_nii(os.path.join(wd, 'test_data/mask.nii.gz'))

    def test_fcm_normalization(self):
        norm_val = 1000
        wm_mask = fcm.find_wm_mask(self.img, self.brain_mask)
        normalized = fcm.fcm_normalize(self.img, wm_mask, norm_value=norm_val)
        self.assertEqual(np.max(normalized.get_data()), norm_val)

    def tearDown(self):
        del self.img, self.brain_mask


if __name__ == '__main__':
    unittest.main()
