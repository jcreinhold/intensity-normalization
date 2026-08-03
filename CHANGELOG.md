# Changelog

## 4.0.0

Complete redesign of the Python API and CLI; restores the v2 feature set dropped in the short-lived v3 rewrite, on a new
architecture.

**Restored features**

- RAVEL normalization (WhiteStripe + CSF control-voxel correction), with optional deformable registration via antspy
- Savable population transforms: fit `nyul`/`lsq` on a training set, persist (`--save-state` / `tx.save`), apply to new
  scans (`--load-state` / `Transform.load`)
- `plot-histograms` tool and `-p/--plot` flag for visual validation
- `tissue-membership` tool (FCM CSF/GM/WM membership maps)
- `preprocess` (N4 bias correction + resample/reorient) and `coregister` behind the `[ants]` extra
- Batch CLI processing with naming conventions, progress bars, `-j` parallelism

**Breaking changes**

- One CLI, `intensity-normalize <command>`; the eleven per-method scripts are gone (see the migration guide)
- Python API redesigned: individual methods are plain functions (`inorm.whitestripe(img, mask)`); population methods
  return fitted transform objects (`inorm.nyul.fit(images)`)
- Saved transform state is now stamped `.npz`; old `.npy` files cannot be loaded
- `pymedio` and `Modality`/`TissueType` enums removed; pass numpy arrays or nibabel images and modality/tissue as
  strings
- v3's `normalize_image(method=...)` dispatcher, service/config/adapter layers removed

**Other**

- scikit-fuzzy dependency replaced with an in-house, seeded fuzzy c-means (fixes Python 3.12+ incompatibility)
- All stochastic steps are deterministic by default (`seed=0`)
- Fitted transforms are immutable (frozen, write-protected arrays), so a saved transform always matches the in-memory
  one
- `LSQTransform.reference_membership` always holds the reference image's CSF/GM/WM membership map (persisted in the
  saved state)
- `nyul.fit` takes a single `landmarks` sequence instead of five percentile grid parameters
- RAVEL warps masks into template space alongside images; `masks_are_csf` now composes with registration
- Errors are validated upfront and actionable (empty foreground, mask/image shape or space mismatch, unco-registered
  RAVEL input); normalized nibabel images always store float32 data with the source affine and codes intact
- Docs moved to MkDocs Material on GitHub Pages

## 3.0.1 and earlier

See git history; the v3 line was an experimental simplification and is superseded by 4.0.0. The v2.2.4 feature set is
the baseline that 4.0.0 restores.
