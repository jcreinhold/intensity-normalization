# Plan: intensity-normalization v4 — restore dropped features on a sound architecture

## Context

The v3.0 "total overhaul" (commit `ad689cc`, written with early Claude Code) cut the
codebase from ~4,500 to ~940 lines and dropped the features that made this package
distinctive, while layering on a shallow "enterprise" architecture (service layer,
adapter layer, stringly-typed config dispatch, `inspect.signature` introspection).
The v2.2.4 code (commit `07641d0`) had the right features but poor factoring
(CLI/algorithm mixin soup, 437-line `normalize/base.py` doing I/O, argparse, and math).

Goal: restore the v2 feature set and user experience on a deep-module architecture
(per *A Philosophy of Software Design* + module-design skill) that a solo part-time
maintainer can sustain. Plus a modern, simple docs site on GitHub Pages.

## What was dropped (feature inventory from `07641d0`)

1. **RAVEL normalization** (`normalize/ravel.py`, 432 lines) — WhiteStripe + CSF control-voxel
   correction for co-registered populations. The flagship population method; cited in the paper.
2. **Save/load of fitted normalizer state** — Nyul (`-ssh/-lsh`) and LSQ (`-sstm/-lstm`) CLIs
   could persist fitted histograms/tissue means and apply them to new images. This is the
   *fit-on-training-set, apply-to-new-data* workflow that every serious user needs.
3. **Histogram plotting** (`plot/histogram.py`, `plot-histograms` CLI, `-p` flags) — the
   recommended validation workflow ("plot the foreground intensities to validate").
4. **Tissue membership** (`util/tissue_membership.py`, `tissue-membership` CLI) — FCM
   tissue probability maps / hard segmentation, useful standalone.
5. **Preprocessing** (`util/preprocess.py`) — N4 bias-field correction + resampling (ants).
6. **Co-registration** (`util/coregister.py`) — ANTs-based registration to template (ants).
7. **Proper directory CLIs** — each method had its own script with image/mask directory
   handling, output naming conventions, modality flags, verbose logging, `-p` plotting.
8. **Histogram tools** (`util/histogram_tools.py`) — shared KDE tissue-mode estimation
   (partially re-inlined into the new whitestripe/kde as duplicated code).

## What's wrong with the v3 architecture (PoSD red flags)

- **Shallow adapters**: `ImageProtocol`/`NumpyImageAdapter`/`NibabelImageAdapter` re-wrap
  what nibabel already provides; `create_image` duck-types. Interface ≈ implementation.
- **Pass-through layers**: free function `normalize_image` → `NormalizationService` →
  normalizer — three layers doing one dispatch.
- **Stringly-typed config**: `NormalizationConfig(method="fcm", ...)` + registry +
  `inspect.signature` filtering + `_create_normalizer_from_instance` `hasattr` chains.
  Fragile, hides parameters, pushes complexity upward.
- **Config parameters exported** that the library can default internally.
- **Legacy shims** (`PEAK`, `VALID_PEAKS`) kept without purpose.

## Domain analysis (from GitHub issues, literature, and git history)

**Who the users are**: neuroimaging researchers and ML engineers normalizing datasets of
NIfTI brain scans before segmentation/synthesis/radiomics. They bring brain masks from
ROBEX/HD-BET/SynthStrip of varying quality, on data of varying quality (thick slices,
slice gaps, pathology). They are not software engineers.

**What the issue tracker and commit history teach**:

- **The #1 failure class is bad/empty foregrounds** — `IndexError: cannot do a non-empty
  take from an empty axes` (nyul with an empty or misaligned mask); LSQ's weighted-average
  `TypeError` when tissue maps don't match the image (issue #59: 5mm slices + HD-BET
  masks). v2/v3 let these surface as cryptic numpy errors deep in the stack. v4 must
  **validate upfront and fail with actionable messages** ("mask contains no foreground
  voxels — check the mask aligns with the image").
- **Modality semantics confuse users** ("What is the MD modality?"). Tissue-mode selection
  depends on modality (T1: WM = last mode; T2/FLAIR: largest; PD/MD: first). Help text
  and docs must spell this out; the old escape hatch (`--modality other` + explicit peak,
  commit `5a871fc`) survives.
- **skfuzzy broke Python 3.12** (issue #82) and is effectively unmaintained. Fuzzy c-means
  is ~40 lines of numpy. **Implement FCM in-house; drop skfuzzy entirely.** One less
  fragile dep, and it removes the input-mutation bug class (commit `a80f5a9` defensively
  copied arrays against skfuzzy).
- **antspy import fragility** (commit `139b949`) and **pymedio/protocol pain**
  (commits `02eb23c`, `c34b11b`) justify: lazy optional imports, no home-grown image
  protocol.
- **Users want intermediates**: LSQ gained "save tissue memberships" (commit `f692319`).
  v4 exposes fitted-transform internals read-only; LSQ/RAVEL can optionally return tissue
  maps / control-voxel diagnostics.
- **RAVEL misuse is common**: it requires co-registered, same-shape images
  (commits `c06d089`, `8934e2e`). Validate shapes upfront; error message points at
  `coregister`.
- **No gold standard exists** in the literature; method choice is application-dependent.
  The tool's job is to make trying and *validating* methods easy — histogram plotting is
  first-class, and the docs carry a "which method?" guide (FCM for T1-w brain; nyul for
  cross-scanner ML datasets; whitestripe for principled per-image standardization;
  ravel/lsq when you have a population and want joint correction).

## What the API must serve (actual user workflows)

1. **One-off normalization in a pipeline** — the most common case. User has a numpy array
   or nibabel image, wants the normalized image back, one line.
2. **Batch a dataset** — directory of images + masks, normalize all with a naming
   convention. CLI job.
3. **Fit on a training set, apply at inference time** — learn a transform from N images,
   persist it, apply to new scans later (population methods; the old `-ssh/-lsh`,
   `-sstm/-lstm` flags).
4. **Validate visually** — plot foreground histograms before/after (the old recommended
   workflow; every method's docs told users to do this).
5. **Supporting tools** — tissue membership maps, N4 preprocess, co-registration.

The domain fact the v2 and v3 APIs both obscured: **individual methods
(zscore/fcm/kde/whitestripe) are pure functions of one image.** Their parameters are
estimated from the image being normalized and discarded; an unfitted `WhiteStripe()`
object is a meaningless intermediate state (v3 invented `is_fitted` bookkeeping for it,
v2 invented setup/teardown hooks). **Population methods (nyul/lsq/ravel) genuinely learn
a reusable transform.** The redesign makes this distinction structural.

## Design-it-twice

- **A — uniform sklearn-style `fit`/`transform`/`is_fitted` for everything** (my first
  sketch). Rejected: false abstraction for individual methods; unfitted-object state
  machine; `save()` on an individual normalizer is nonsense; callers write two lines
  where one suffices.
- **B — individual methods are functions; population methods are fitted transform
  objects.** Chosen. Construction *is* fitting, so an unfitted state cannot exist
  (PoSD ch. 10: define errors out of existence).
- **C — v2 callable classes** (`FCMNormalize()(img)`). Rejected: constructor/call split
  hides nothing over a function; base-class mixin soup coupled math, I/O, and argparse.

## Target architecture (deep modules, small interfaces)

```
src/intensity_normalization/
├── __init__.py          # curated public API: flat re-exports (inorm.whitestripe, ...)
├── _image.py            # PRIVATE: the only module that knows numpy vs nibabel;
│                        # type-preserving extract/restore used by every public function
├── errors.py            # one base exception; bad args surface as ValueError/TypeError
├── histogram.py         # KDE smoothing, tissue modes, modality→mode policy (public)
├── io.py                # load/save NIfTI et al. for the CLI (paths never enter math)
├── methods/             # THE NORMALIZATION ALGORITHMS — the reason the package exists
│   ├── __init__.py      # re-exports; defines the individual/population split in docstring
│   ├── _fcm.py          # PRIVATE: in-house seeded fuzzy c-means (replaces skfuzzy)
│   ├── _transform.py    # PRIVATE: FittedTransform base — save/load npz, stamping
│   ├── zscore.py        # individual methods: one function each
│   ├── fcm.py
│   ├── kde.py
│   ├── whitestripe.py
│   ├── nyul.py          # population methods: fit() -> fitted transform object
│   ├── lsq.py
│   └── ravel.py         # lazy ants import [ants extra]
├── tools/               # supporting utilities — useful standalone, not normalizers
│   ├── __init__.py
│   ├── tissue.py        # tissue_membership(): FCM probability maps / hard segmentation
│   ├── plot.py          # plot_histograms(); lazy matplotlib [plot extra]
│   └── ants.py          # preprocess (N4 + resample), coregister; lazy ants [ants extra]
└── cli/                 # typer app; the only stringly-typed layer, by necessity
    ├── __init__.py      # app definition + shared options + upfront validation
    ├── normalize.py     # the 7 method subcommands (individual + population)
    └── tools.py         # tissue-membership, plot-histograms, preprocess, coregister
```

Organization rationale (split by independent change; alphabetical order now groups
correctly): `methods/` is everything a developer touches when fixing or adding an
algorithm — the individual/population split lives in the *shape of the API* (function
vs. fitted object), signaled in each module's docstring rather than by deeper nesting
(`methods/individual/` would be one-item-per-folder fragmentation). `tools/` is stuff
users call around normalization; it never imports from `methods` except `tissue` →
`_fcm`. `cli/` depends on everything; nothing depends on it. The public surface stays
flat (`inorm.whitestripe`, `inorm.nyul.fit`, `inorm.tissue_membership`) via
`__init__.py` re-exports — internal moves never break user code.

Where functionality belongs, and why:

1. **Math takes numpy, returns numpy.** No image-library coupling in algorithms —
   general-purpose modules are deeper (ch. 6), trivially testable, usable from any
   caller. `methods/` groups the 7 algorithms with their shared private machinery
   (`_fcm`, `_transform`); everything else orbits it.
2. **Type preservation happens exactly once**, in private `_image.py`: numpy in → numpy
   out; nibabel in → nibabel out (affine/header preserved). Every public function routes
   through it. This replaces the entire v3 `adapters/` layer (shallow wrappers duplicating
   nibabel) — one deep module, zero public surface.
3. **Histogram logic in one public module** shared by kde/whitestripe/lsq and useful to
   end users validating results (the old `histogram_tools`, now including the
   modality→tissue-mode policy computed *inside* — pull complexity downward; callers pass
   `modality="t1"` and nothing else).
4. **Configuration = keyword arguments with computed defaults.** Export a parameter only
   where the user genuinely knows better (`width`, `norm_value`, `tissue`); everything
   else is derived internally. No config objects, no registries, no
   `inspect.signature` filtering, no stringly-typed `normalize(method="...")` dispatcher —
   `inorm.whitestripe(img)` is shorter than `inorm.normalize(img, method="whitestripe")`
   and statically checkable. The CLI owns the single name→callable table, because parsing
   strings is its job.
5. **Serialization is a property of fitted transforms only.** `save(path)` /
   `Transform.load(path)` as `.npz` stamped with method name + format version.
   Individual functions have nothing to save — the API makes that obvious instead of
   inheriting a vestigial method.
6. **Errors**: one `IntensityNormalizationError` base carrying *actionable* messages
   (empty foreground, degenerate histogram, shape mismatch — see Domain analysis);
   bad arguments surface as ordinary `TypeError`/`ValueError`. No `except Exception`
   re-wrap chains (v3 whitestripe swallows the real error); no `NotFittedError` —
   that state cannot exist. Validation happens at the boundary, before compute.
7. **Optional deps stay lazy**: ants and matplotlib imported inside functions, behind
   `[ants]` / `[plot]` extras, as in v2 — but now without import-time module failures.
8. **Performance by design** (ch. 20, applied where the domain says it matters):
   - float32 internally (nibabel `get_fdata` returns float64 — halves memory);
   - KDE/tissue-mode estimation on a seeded subsample of foreground voxels
     (~50k: statistically identical mode, O(n²) `gaussian_kde` made tractable);
     every stochastic step takes `seed=`;
   - population `fit` **streams**: one image at a time, retaining only per-image
     statistics (nyul landmarks, lsq tissue means) — dataset size never bounds RAM.
     RAVEL's voxel×image matrix is inherent: float32 + documented memory note;
   - CLI batch individual methods are embarrassingly parallel: `-j/--jobs` process
     pool (paths cross the boundary; images load in workers). Default 1.

## Python API (design B)

```python
import intensity_normalization as intnorm

# --- Individual methods: plain functions ---
# numpy or nibabel in → same type out; mask optional (foreground estimated otherwise)

normed = intnorm.zscore(img, mask=mask)
normed = intnorm.fcm(img, tissue="wm", seed=0)              # modality="t1" default
normed = intnorm.kde(img, modality="t2")
normed = intnorm.whitestripe(img, mask=mask, width=0.05, norm_value=1.0)
# escape hatch for non-standard data: modality="other" + explicit peak choice
normed = intnorm.kde(img, modality="other", peak="largest")

# --- Population methods: fit returns a reusable, savable transform ---

tx = intnorm.nyul.fit(train_imgs, masks=train_masks)      # construction is fitting
normed = tx(new_img)                                      # callable, type-preserving
normed_train = [tx(i) for i in train_imgs]
tx.save("nyul.npz")
tx = intnorm.NyulTransform.load("nyul.npz")               # only fitted states exist
tx.landmarks                                              # learned data, read-only

tx, normed = intnorm.nyul.fit_transform(train_imgs)       # fit + apply in one call

# ravel: registration to the template happens inside fit/transform (lazy ants)
tx = intnorm.ravel.fit(imgs, masks=masks, template="mni", template_mask=None)

tx = intnorm.lsq.fit(imgs, masks=masks, return_tissue_maps=True)  # optional intermediates

# --- Supporting tools ---
probs = intnorm.tissue_membership(img)                    # FCM probability maps
intnorm.plot_histograms(imgs, masks=masks)                # validation workflow [plot]
intnorm.preprocess(img, n4=True, resample=(1, 1, 1))      # [ants]
intnorm.coregister(img, template=tpl)                     # [ants]
```

Why this is ergonomic:

- **One line for the common case** (workflow 1); no objects to instantiate for an
  operation that has no reusable state.
- **No invalid states**: you cannot call `transform` before `fit` because there is no
  pre-fit object; you cannot `save` an individual normalizer because it isn't a thing.
- **Type preservation is invisible**: users pass whatever they have (numpy array,
  nibabel image) and get the same kind back, affine intact.
- **The discoverable surface is small**: `dir(inorm)` shows ~10 callables + 3 transform
  classes + `Modality`/`Tissue` enums. Compare v3: service, config, registry, protocols,
  adapters, and two parallel class hierarchies to do the same work.
- **The `inorm` alias** replaces v2's `intnorm` — 5 characters, reads as "i-norm",
  unambiguous next to `np`/`nib` in the same file. Docs and examples use it consistently.

## CLI (cargo-style, typer + rich)

**Framework: typer + rich.** This is justified, not decorative:

- **Help quality**: cargo-like subcommand help generated from typed signatures and
  docstrings — per-subcommand `--help` with examples in the epilog, modality meanings
  spelled out (directly addresses the "what is MD?" confusion). argparse subparser
  boilerplate for 11 subcommands with shared flags is exactly the kind of code that rots.
- **Progress**: batch normalization of hundreds of scans is a minutes-long operation;
  rich progress bars (per-image) and a spinner during population `fit` are real UX.
  Respects `NO_COLOR`/non-TTY automatically.
- **Error UX**: rich-formatted errors with the actionable message ("mask is empty …",
  "RAVEL requires same-shape, co-registered images — see `intensity-normalize coregister`")
  instead of tracebacks; `--debug` re-enables tracebacks.
- Shell completions come free with typer. `-q/--quiet`, `-v/--verbose`.

**Command layout** — one binary, subcommands mirror the Python API names; the v2 split
between "single-image CLI" and "directory CLI" is defined out of existence:

```
intensity-normalize zscore|fcm|kde|whitestripe IMG [IMG ...]
    [-m MASK | --mask-dir DIR] [-o OUT | --output-dir DIR]
    [--modality t1|t2|flair|pd|md|other] [--peak last|largest|first]
    [--seed N] [-j/--jobs N] [-p] [-v] [-q]
# one input → file semantics; several inputs or a directory → batch with progress bar

intensity-normalize nyul|lsq|ravel DIR [-m MASK_DIR] [-o OUT_DIR]
    [--save-state tx.npz] [--load-state tx.npz] [--save-tissue-maps] [-v]
# default: fit on DIR, transform, write outputs (old behavior);
# --load-state: skip fitting, apply a saved transform to the inputs

intensity-normalize tissue-membership IMG [-o OUT] [--hard]
intensity-normalize plot-histograms DIR [-m MASK_DIR] [-o hist.png]
intensity-normalize preprocess IMG [-o OUT] [--n4] [--resample 1x1x1]   # [ants]
intensity-normalize coregister IMG [IMG ...] --template TPL [-o DIR]    # [ants]
```

Upfront validation before any heavy work: inputs exist, masks match image shapes,
output dirs writable, `--load-state` file is a stamped npz for the right method.

## Docs (new requirement)

- **MkDocs Material + mkdocstrings** — far simpler than the old Sphinx/RTD setup
  (`docs/conf.py` was 187 lines), standard today, deploys to GitHub Pages via a single
  GitHub Actions workflow (`mkdocs gh-deploy` or actions pages).
- Port and update the old material:
  - `docs/algorithm.rst` → `docs/algorithms.md` (MathJax via `pymdownx.arithmatex` —
    keep the math, update for API changes)
  - `tutorials/5min_tutorial.rst` → quickstart page + Python API examples
  - installation / usage / CLI reference / API reference (auto via mkdocstrings)
  - the histogram illustration figure (`docs/_static/imgs/intnorm_illustration.png`)
- README trimmed to pitch + install + one example + link to the Pages site.

## Files to modify / create

- Rewrite: `src/intensity_normalization/{__init__,cli}.py`, all of `normalizers/`
- Delete: `adapters/`, `domain/`, `services/` (fold the useful bits into the above)
- New: `_image.py`, `errors.py`, `histogram.py`, `io.py`,
  `methods/{__init__,_fcm,_transform,zscore,fcm,kde,whitestripe,nyul,lsq,ravel}.py`
  (population trio ported from `07641d0`),
  `tools/{__init__,tissue,plot,ants}.py`, `cli/{__init__,normalize,tools}.py`
- New: `mkdocs.yml`, `docs/**`, `.github/workflows/docs.yml`
- Update: `pyproject.toml` (deps: numpy/scipy/nibabel + typer/rich; **drop scikit-fuzzy**;
  extras `[ants]`,`[plot]`,`[docs]`; single `intensity-normalize` entry point;
  **no v2 script-name aliases**), `README.md`, `tests/**`

## Reuse

- v2.2.4 algorithm implementations as the correctness reference:
  `git show 07641d0:intensity_normalization/normalize/{ravel,nyul,lsq,fcm}.py`,
  `util/{histogram_tools,tissue_membership,preprocess,coregister}.py`,
  `plot/histogram.py`, `docs/algorithm.rst`, `tutorials/5min_tutorial.rst`
- Current v3 whitestripe/kde/fcm/zscore math is largely correct — keep the math,
  drop the scaffolding around it.

## Decisions (from user)

1. **Clean break at 4.0.0** — no v2 import-path shims, no old script names (`fcm-normalize`,
   …). One entry point: `intensity-normalize <subcommand>`. Migration notes in docs.
2. **Restore all three ants features now** (RAVEL, preprocess/N4+resample, coregister)
   behind the `[ants]` extra with lazy imports.
3. **Docs: MkDocs Material** (mdBook considered; rejected because it has no Python API-doc
   generation — mkdocstrings is the deciding feature).
4. **Version 4.0.0**, leaving 3.x untouched on PyPI.

## Steps

- [x] In-house FCM (`methods/_fcm.py`, numpy, seeded) — replaces skfuzzy; shared by
      `fcm.py`, `lsq.py`, `tools/tissue.py`
- [x] `methods/_transform.py` (save/load npz base) + port individual normalizers
      (zscore, fcm, kde, whitestripe) as numpy-in/out functions with upfront
      validation and `seed=`
- [x] Port population normalizers (lsq, nyul) as streaming fits returning savable
      transform objects; optional tissue-map returns
- [x] Port RAVEL (float32 image matrix, shape validation, lazy ants) + `[ants]` extra
- [x] `_image.py` type-preserving wrap + `io.py` path I/O for the CLI
- [x] `tools/tissue.py` and `tools/plot.py` (`[plot]` extra)
- [x] `tools/ants.py` (preprocess + coregister)
- [x] `cli/`: typer app, subcommands, rich progress/errors, `-j`, save/load state, `-p`,
      upfront validation
- [x] Tests: synthetic-phantom correctness (known modes/means), serialization
      round-trips, CLI smoke tests, regression vs v2.2.4 outputs for deterministic
      methods (zscore, nyul, lsq on fixed seeds)
- [x] MkDocs Material site + GitHub Pages workflow; port algorithm/tutorial material
- [x] README rewrite; bump to 4.0.0; changelog entry

## Verification

- `pytest` green: unit correctness (synthetic phantom with known tissue modes),
  round-trip save/load equality, CLI end-to-end on generated NIfTI fixtures
- Numerical regression: run v2.2.4 (git worktree + venv) and v4 on identical fixtures;
  deterministic methods must match to float tolerance; stochastic (FCM) with fixed seed
- Determinism: same seed → bit-identical output; float32 memory ceiling test on a
  large synthetic image; KDE subsample invariance (mode within tolerance at 50k samples)
- `ruff check` + `mypy` clean; `mkdocs build --strict` in CI; Pages deploy on tag
- Manual: `intensity-normalize whitestripe img.nii -m mask.nii -p` shows histograms
