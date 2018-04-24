#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.io

assortment of input/output utilities for the project

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import os


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

