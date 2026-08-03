# Which method should I use?

There is no gold standard for MR intensity normalization — the right choice
depends on your data and downstream task. This guide is the practical summary;
see [How the methods work](algorithms.md) for the math.

## Decision guide

**One T1-w image per subject (the common case)**
:   Start with **FCM** (`inorm.fcm`). It scales the white-matter mean to 1 using
    fuzzy tissue memberships and is robust in practice. If results look odd in
    non-WM tissue (check with a histogram plot!), try **LSQ** across your set,
    which balances CSF/GM/WM means jointly.

**No T1-w image, or non-brain anatomy**
:   Use **zscore** (any anatomy, any modality) or **kde** with the right
    `--modality` so the correct histogram peak is used.

**A dataset for machine learning (train/test split)**
:   Use **nyul**: fit the standard histogram on the training set, save the
    transform, and apply the *same* transform to validation/test/inference
    data. Histogram matching tends to equalize the whole distribution, not just
    one landmark — often what ML pipelines want. RAVEL and LSQ transforms are
    likewise savable.

**Multi-scanner/multi-site study, co-registered images, visible batch effects**
:   Use **ravel**: WhiteStripe + removal of latent technical factors estimated
    from CSF control voxels. It's the most involved (needs co-registration,
    batch-only) and the most aggressive at removing scanner effects.

**Principled per-image standardization with a paper behind it**
:   Use **whitestripe**: z-score within the normal-appearing white matter.

## Modality matters

`kde` and `whitestripe` anchor on a tissue peak of the smoothed foreground
histogram, and which peak is the right one depends on the modality:

| modality | peak used | rationale |
|---|---|---|
| `t1` | `last` | WM is the brightest of the three tissues |
| `t2`, `flair` | `largest` | global maximum of the histogram |
| `pd`, `md` | `first` | lowest-intensity tissue peak |
| `other` | `last` (override with `--peak`) | escape hatch for non-standard data |

For contrast-enhanced or otherwise non-standard images, read
[How the methods work](algorithms.md), pick the peak that matches your tissue of
interest, and pass `--peak` / `peak=` explicitly. Then **validate with a
histogram plot** before batch-processing hundreds of scans.

## What every method assumes

- **Brain masks** (or skull-stripped input) for everything except `zscore` and
  `nyul`, which tolerate full-head images better. Bad masks are the top cause of
  bad results. The package fails loudly with an actionable message, but
  garbage in is still garbage in.
- All brain-specific methods (`fcm`, `kde`, `whitestripe`, `lsq`, `ravel`)
  assume three tissue classes (CSF/GM/WM) exist. Heavy pathology, infants, or
  non-human primates can violate that.
- Population methods assume all input images share **one modality** and
  comparable acquisition.
