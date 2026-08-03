# CLI reference

One entry point, `intensity-normalize`, with subcommands mirroring the Python
API. Every command has `--help` with the full option list. Global options:
`--version`, `--debug` (full tracebacks).

```text
intensity-normalize <command> --help
```

## Individual methods

```text
intensity-normalize zscore|fcm|kde|whitestripe IMG [IMG ...]
    [-m MASK | --mask-dir DIR]
    [-o OUTPUT | --output-dir DIR]
    [--modality t1|t2|flair|pd|md|other] [--peak last|largest|first]
    [--norm-value X] [--seed N] [-j N] [-p] [-q]
```

- One image → file semantics (`-o out.nii.gz`); several images or a directory →
  batch semantics with a progress bar (`--output-dir`, masks matched by
  filename from `--mask-dir`).
- Default output: `<name>_<method>.nii.gz` next to each input.
- `-j/--jobs N`: process a batch in parallel.
- `-p/--plot`: show foreground histograms before/after (needs `[plot]`).
- Method-specific options: `fcm --tissue csf|gm|wm`; `whitestripe --width`.

Examples:

```bash
intensity-normalize whitestripe t1w.nii.gz -m mask.nii.gz -p
intensity-normalize fcm images/ --mask-dir masks/ --output-dir out/ -j 8
intensity-normalize kde t2w.nii.gz -m mask.nii.gz --modality t2
```

## Population methods

```text
intensity-normalize nyul|lsq DIR [-m MASK_DIR] [-o OUT_DIR]
    [--save-state tx.npz] [--load-state tx.npz] [-q]
```

- Default: fit on `DIR`, transform those images, write `<name>_<method>.nii.gz`.
- `--save-state`: persist the fitted transform; `--load-state`: skip fitting
  and apply a saved transform to the inputs.
- `lsq --save-tissue-maps` also writes the reference image's membership map.

```text
intensity-normalize ravel DIR [-m MASK_DIR] [-o OUT_DIR]
    [-b N] [--membership-threshold X] [--no-registration] [--sparse-svd]
    [--masks-are-csf] [--quantile-to-label-csf X] [--save-state artifacts.npz]
```

- Requires same-shape, co-registered images; by default it deformably registers
  to the first image internally (needs `[ants]`). With `--no-registration` the
  images must already be deformably co-registered.
- RAVEL is batch-only: there is no `--load-state` (see
  [How the methods work](algorithms.md#ravel)).

## Tools

```text
intensity-normalize tissue-membership IMG [-m MASK] [-o OUT] [--hard] [--seed N]
intensity-normalize plot-histograms DIR [-m MASK_DIR] [-o FIG.png] [--linear]
intensity-normalize preprocess IMG [-m MASK] [-o OUT] [-r X Y Z]
                      [--orientation RAS] [--single-n4]         # [ants]
intensity-normalize coregister IMG [IMG ...] [--template TPL]
                      [--type-of-transform SyN] [-o DIR]        # [ants]
```

## Errors

The CLI validates inputs before doing heavy work and prints one-line,
actionable errors (`error: No mask named sub3.nii.gz in masks/ …`) instead of
tracebacks. Pass `--debug` to get the full traceback.
