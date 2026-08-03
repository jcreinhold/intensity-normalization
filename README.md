# intensity-normalization

[![PyPI](https://img.shields.io/pypi/v/intensity-normalization.svg)](https://pypi.org/project/intensity-normalization/)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://jcreinhold.github.io/intensity-normalization/)

Normalize the intensities of magnetic resonance (MR) images — T1-w, T2-w,
FLAIR, PD-w — across scanners, sites, and sessions.

MR images have no consistent intensity scale; the inconsistency is an
acquisition artifact that breaks downstream processing (especially ML). This
package implements the standard fixes:

- **Individual methods** — `zscore`, `fcm`, `kde`, `whitestripe`: plain
  functions of one image.
- **Population methods** — `nyul`, `lsq`, `ravel`: learn a transform from a set
  of images; save it, apply it to new scans.
- **Tools** — tissue membership maps, histogram plotting (validation),
  N4 preprocessing, co-registration (ANTs).

## Install

```bash
pip install intensity-normalization            # or: uv pip install intensity-normalization
pip install "intensity-normalization[ants]"  # ravel registration, preprocess, coregister
pip install "intensity-normalization[plot]"  # histogram plotting
```

The CLI also runs without installing: `uvx intensity-normalize --help`.

## Quickstart

```python
import intensity_normalization as inorm

normed = inorm.whitestripe(t1w_image, mask=brain_mask)  # numpy or nibabel in → same type out

tx = inorm.nyul.fit(train_images, masks=train_masks)  # population: fit once...
tx.save("nyul.npz")
normed_new = tx(new_image)  # ...apply to new scans
```

```bash
intensity-normalize fcm t1w.nii.gz -m brain_mask.nii.gz -p
intensity-normalize nyul images/ -m masks/ -o normalized/ --save-state nyul.npz
```

**[Documentation](https://jcreinhold.github.io/intensity-normalization/)** —
[quickstart](https://jcreinhold.github.io/intensity-normalization/quickstart/),
[how-to guides](https://jcreinhold.github.io/intensity-normalization/how-to/),
[choosing a method](https://jcreinhold.github.io/intensity-normalization/methods/),
[algorithms](https://jcreinhold.github.io/intensity-normalization/algorithms/),
[CLI](https://jcreinhold.github.io/intensity-normalization/cli/),
[API](https://jcreinhold.github.io/intensity-normalization/api/),
[migrating to v4](https://jcreinhold.github.io/intensity-normalization/migration/).

## Reference

If you use this package, please cite the accompanying
[pre-print](https://arxiv.org/abs/1812.04652):

```bibtex
@article{reinhold2019evaluating,
  title={Evaluating the impact of intensity normalization on MR image synthesis},
  author={Reinhold, Jacob C and Dewey, Blake E and Carass, Aaron and Prince, Jerry L},
  journal={Medical Imaging 2019: Image Processing},
  year={2019}
}
```

## License

MIT
