#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity-normalization.normalize.normalize

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jun 01, 2021
"""

__all__ = [
    "NormalizeBase",
]

from typing import Optional, Union

import nibabel as nib

from intensity_normalization.type import Array, NiftiImage


class NormalizeBase:
    def __init__(self, norm_value: float = 1.0):
        self.norm_value = norm_value

    def __call__(
        self,
        data: Union[Array, NiftiImage],
        mask: Optional[Union[Array, NiftiImage]] = None,
        modality: Optional[str] = None,
    ) -> Union[Array, NiftiImage]:
        if isinstance(data, nib.Nifti1Image):
            return self.normalize_nifti(data, mask, modality)
        else:
            return self.normalize_array(data, mask, modality)

    def normalize_array(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None,
    ) -> Array:
        self.setup(data, mask, modality)
        loc = self.calculate_location(data, mask, modality)
        scale = self.calculate_scale(data, mask, modality)
        self.teardown()
        return ((data - loc) / scale) * self.norm_value

    def normalize_nifti(
        self,
        image: NiftiImage,
        mask_image: Optional[NiftiImage] = None,
        modality: Optional[str] = None,
    ) -> NiftiImage:
        data = image.get_fdata()
        mask = mask_image and mask_image.get_fdata()
        normalized = self.normalize_array(data, mask, modality)
        return nib.Nifti1Image(normalized, image.affine, image.header)

    def calculate_location(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> float:
        raise NotImplementedError

    def calculate_scale(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> float:
        raise NotImplementedError

    def setup(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ):
        return

    def teardown(self):
        return

    def _get_mask(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ):
        if mask is None:
            mask = data > 0.0
        return mask

    def _get_voi(
        self, data: Array, mask: Optional[Array] = None, modality: Optional[str] = None
    ) -> Array:
        return data[self._get_mask(data, mask, modality)]

    def _get_modality(self, modality: Optional[str]) -> str:
        return "t1" if modality is None else modality.lower()


class NormalizeSetBase(NormalizeBase):
    pass
