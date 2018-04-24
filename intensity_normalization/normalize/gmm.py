#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.gmm

fit three gaussians to the histogram of
skull-stripped image and normalize the WM mean
to some standard value

Author: Blake Dewey
        Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Apr 24, 2018
"""

from __future__ import print_function, division

import argparse
import logging
import math
import os
import sys

import nibabel as nib
import numpy as np
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    from sklearn.mixture import GMM as GaussianMixture
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import (binary_closing, binary_fill_holes, generate_binary_structure, iterate_structure,
                                      binary_dilation)

from utilities.io import split_filename

logger = logging.getLogger()


def otsu(img, bins=64):
    steplength = (img.max() - img.min()) / float(bins)
    initial_threshold = img.min() + steplength
    
    best_bcv = 0
    best_threshold = initial_threshold
    
    for threshold in np.arange(initial_threshold, img.max(), steplength):
        mask_fg = (img >= threshold)
        mask_bg = (img < threshold)
        
        wfg = np.count_nonzero(mask_fg)
        wbg = np.count_nonzero(mask_bg)
        
        if 0 == wfg or 0 == wbg:
            continue
        
        mfg = img[mask_fg].mean()
        mbg = img[mask_bg].mean()
        
        bcv = wfg * wbg * math.pow(mbg - mfg, 2)
        
        if bcv > best_bcv:
            best_bcv = bcv
            best_threshold = threshold
    
    return best_threshold


def fill_2p5d(img):
    out_img = np.zeros_like(img)
    for slice_num in range(img.shape[2]):
        out_img[:, :, slice_num] = binary_fill_holes(img[:, :, slice_num])
    return out_img


def gmm_normalize(img_filename, mask_filename=None, norm_value=1000, contrast='t1', keep_bg=False,
                  bg_mask=None, wm_peak=None):
    logger.info('Loading image:', img_filename)
    path, base, ext = split_filename(img_filename)
    obj = nib.load(img_filename)
    img_data = obj.get_data().astype(np.float32)
    
    if mask_filename is None and wm_peak is None:
        raise RuntimeError('Mask or WM Peak must be specified.')
    
    if wm_peak is None:
        logger.info('Loading Mask:', mask_filename)
        mask_data = nib.load(mask_filename).get_data().astype(np.float32)
        
        logger.info('Fitting GMM...')
        gmm = GaussianMixture(3)
        gmm.fit(np.expand_dims(img_data[mask_data == 1].flatten(), 1))
        
        means = gmm.means_.T.tolist()[0]
        weights = gmm.weights_.tolist()
        
        wm_peak = max(means) if contrast == 't1' else max(zip(means, weights), key=lambda x: x[1])[0]
        logger.info('WM Peak Found at', wm_peak)
        
        logger.info('Saving WM Peak...')
        np.save(os.path.join(path, base + '_wmpeak.npy'), wm_peak)
        
    else:
        logger.info('WM Peak loaded: ', wm_peak)
    
    logger.info('Normalizing Data...')
    norm_data = img_data/wm_peak*norm_value
    norm_data[norm_data < 0.1] = 0.0
    
    if keep_bg:
        masked_image = norm_data
    else:
        if bg_mask is None:
            logger.info('Finding background...')
            # threshold = otsu(img_data)
            # raw_mask = img_data > threshold
            km = KMeans(4)
            rand_mask = np.random.rand(*img_data.shape) > 0.75
            logger.info('Fitting KMeans...')
            km.fit(np.expand_dims(img_data[rand_mask], 1))
            logger.info('Generating Mask...')
            classes = km.predict(np.expand_dims(img_data.flatten(), 1)).reshape(img_data.shape)
            means = [np.mean(img_data[classes == i]) for i in range(4)]
            raw_mask = (classes == np.argmin(means)) == 0.0
            # noinspection PyTypeChecker
            filled_raw_mask = fill_2p5d(raw_mask)
            dist2_5by5_kernel = iterate_structure(generate_binary_structure(3, 1), 2)
            closed_mask = binary_closing(filled_raw_mask, dist2_5by5_kernel, 5)
            filled_closed_mask = fill_2p5d(np.logical_or(closed_mask, filled_raw_mask)).astype(np.float32)
            final_mask = binary_dilation(filled_closed_mask, generate_binary_structure(3, 1), 2)
            
            logger.info('Saving background mask...')
            nib.Nifti1Image(final_mask, obj.affine, obj.header).to_filename(os.path.join(path, base + '_bgmask' + ext))
        else:
            logger.info('Loading background mask: ', bg_mask)
            final_mask = nib.load(bg_mask).get_data().astype(np.float32)
        
        logger.info('Applying background mask...')
        masked_image = norm_data * final_mask
    
    logger.info('Saving output...')
    nib.Nifti1Image(masked_image, obj.affine, obj.header).to_filename(os.path.join(path, base + '_norm' + ext))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True)
    parser.add_argument('-m', '--mask', type=str)
    parser.add_argument('-b', '--background-mask', type=str)
    parser.add_argument('-w', '--wm-peak', type=str)
    parser.add_argument('--norm-value', type=float, default=1000)
    parser.add_argument('--contrast', type=str, choices=['t1', 't2'], default='t1')
    parser.add_argument('--keep-bg', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    peak = None
    if args.wm_peak is not None:
        peak = float(np.load(args.wm_peak))
    try:
        gmm_normalize(args.image, args.mask, args.norm_value, args.contrast, args.keep_bg, args.background_mask, peak)
        return 0
    except Exception as exc:
        logger.exception(exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
