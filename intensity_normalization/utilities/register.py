#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register


Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 08, 2018
"""

from glob import glob
import logging
import os

import ants
import numpy as np

from intensity_normalization.utilities import io

logger = logging.getLogger()


def register_imgs(img_dir, out_dir=None, tx_dir=None, template_img=0):
    img_fns = glob(os.path.join(img_dir, '*.nii*'))
    if isinstance(template_img, int):
        template_img = img_fns[template_img]
    template = ants.image_read(template_img)
    for fn in img_fns[1:]:
        img = ants.image_read(fn)
        tx = ants.registration(template, img, type_of_transform='SyN')
