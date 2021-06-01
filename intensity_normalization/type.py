#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.type

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "Array",
    "NiftiImage",
]

import nibabel as nib
import numpy as np

Array = np.ndarray
NiftiImage = nib.Nifti1Image
