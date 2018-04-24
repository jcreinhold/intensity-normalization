#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.kde

use kernel density estimation to find the peak of the histogram
associated with the WM and move this to peak to a (standard) value

Author: Blake Dewey
        Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import logging

import numpy as np
from scipy.signal import argrelextrema
import statsmodels.api as sm

from errors import NormalizationError

logger = logging.getLogger()


def kde_normalize(vol, contrast):
    temp = vol[np.nonzero(vol)]
    if contrast.upper() == 'T1C' or contrast.upper() == 'FLC':
        q = np.percentile(temp, 96.0)
    else:
        q = np.percentile(temp, 99.0)
    temp = temp[temp <= q]
    temp = np.asarray(temp, dtype=float).reshape(-1, 1)
    bw = float(q) / 80
    logger.info("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    X = 100.0 * kde.density
    Y = kde.support

    indx = argrelextrema(X, np.greater)
    indx = np.asarray(indx, dtype=int)
    H = X[indx]
    H = H[0]
    p = Y[indx]  # p is array of an array, so p=p[0] is necessary. I don't know why this happened!!!
    p = p[0]
    logger.info("%d peaks found." % (len(p)))
    if contrast.upper() in ("T1", "T1C"):
        x = p[-1]
        logger.info("Peak found at %.4f for %s" % (x, contrast))
    elif contrast.upper() in ("T2", "FL", "PD", "FLC"):
        x = np.amax(H)
        j = np.where(H == x)
        x = p[j]
        logger.info("Peak found at %.4f for %s" % (x, contrast))
    else:
        raise NormalizationError("Contrast must be one of T1,T1C,T2,PD,FL,FLC.")
    return x
