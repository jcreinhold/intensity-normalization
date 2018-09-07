#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_plot

test the functions located in plot submodule for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 06, 2018
"""

import os
import unittest

from intensity_normalization.plot.hist import all_hists


class TestPlot(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')

    def test_all_hist(self):
        _ = all_hists(self.data_dir, self.mask_dir)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
