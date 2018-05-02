#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_mask


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 01, 2018
"""

import os
import unittest

import numpy as np

from intensity_normalization.utilities import io, mask


class TestMask(unittest.TestCase):

    def test_mask(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        img = io.open_nii(os.path.join(wd, 'test_data/test.nii.gz'))
        brain_mask = io.open_nii(os.path.join(wd, 'test_data/mask.nii.gz'))
        m = mask.class_mask(img, brain_mask, hard_seg=True)
        self.assertEqual(len(np.unique(m)), 4)


if __name__ == '__main__':
    unittest.main()
