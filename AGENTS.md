# intensity-normalization

Guidance for coding agents working in this repository.

## What this is

A Python package to normalize MR image intensities (T1-w, T2-w, FLAIR, PD-w).
v4.0.0 is a clean-slate redesign; see `PLAN.md` for the full design rationale
and `docs/migration.md` for what changed from v2/v3.

## Architecture (read this before editing)

Design rule: **individual methods are functions; population methods are fitted
transforms.** Don't blur it.

```
src/intensity_normalization/
├── _image.py        # PRIVATE: the ONLY module that knows numpy vs nibabel;
│                    # type-preserving unwrap/restore for every public function
├── _ants.py         # PRIVATE: lazy antspy import + numpy/nibabel → ANTs bridge
├── errors.py        # one base exception, actionable messages
├── histogram.py     # KDE smoothing, tissue modes, modality→peak policy
├── io.py            # path I/O for the CLI (paths never enter the math API)
├── methods/         # the normalization algorithms
│   ├── _fcm.py      # PRIVATE in-house fuzzy c-means (replaces scikit-fuzzy)
│   ├── _transform.py# PRIVATE FittedTransform base + stamped .npz save/load
│   ├── zscore.py fcm.py kde.py whitestripe.py   # individual: one function each
│   ├── nyul.py lsq.py                           # population: fit() -> transform
│   └── ravel.py                                 # batch-only, no apply-to-new
├── tools/           # tissue.py, plot.py (lazy matplotlib), ants.py (lazy ants)
└── cli/             # typer app; the only stringly-typed layer
```

Invariants to preserve:

- Methods are split into **array cores and image wrappers**: the `*_array`
  functions (e.g. `zscore_array`, `fcm_array`, `fit_array`, `transform_array`)
  are the actual methods — pure numpy, no nibabel. The plain-named functions
  (`zscore`, `fcm`, `fit`, `transform`) are thin convenience wrappers:
  unwrap → core → restore. Naming rule: a core is its wrapper's name +
  `_array` (`zscore_array`, `fit_array`, `fit_transform_array`).
- Decomplected boundaries (see `DECOMPLECTING.md`): no core accepts `None`
  masks, modality strings, or `**kwargs`. Foreground resolution
  (`resolve_foreground`), mask binarization (`unwrap_mask`), and modality→peak
  policy (`histogram.resolve_peak`) each happen once, at the boundary; policy
  flows down as values (`BinaryMask`, `Peak`, `WhiteStripeSpec`). The affine
  apply step is shared (`methods/_common.standardize`). Image context is a
  value (`ImageMeta`), never a closure.
- Type aliases live in `_image.py`: `IntensityArray` (float image data),
  `ForegroundIntensities` (1-D in-mask samples), `MaskArray` (bool mask array),
  `Image`/`Mask` (user-facing unions incl. nibabel). PEP 695 `type` statements
  — but never for aliases consumed at runtime (typer `Annotated` options,
  `histogram.Peak` used with `typing.get_args`): those stay plain assignments.
- Math takes numpy, returns numpy. Type preservation (nibabel in → nibabel out)
  happens only in `_image.py`. No adapter/protocol layers — they were deleted
  on purpose.
- Construction *is* fitting for population methods: no unfitted states, no
  `is_fitted` flags. RAVEL is batch-only by design (no single-image transform).
  Fitted transforms are frozen dataclasses with write-protected arrays.
- The `.npz` file format is owned solely by `_transform.py`
  (`_save_stamped`/`_load_stamped`); transforms and `RavelResult` only supply
  state dicts.
- `methods/` never imports from `tools/`; shared ants infrastructure lives in
  the private root `_ants.py`.
- The CLI's shared option vocabulary is defined once as `Annotated` aliases at
  the top of `cli/normalize.py`; command functions are thin dispatchers.
- All stochastic steps take `seed=` and default to `seed=0` (deterministic).
- ants and matplotlib are optional and imported lazily inside functions.
- Errors are validated at the boundary with actionable messages; no broad
  `except Exception` re-wrap chains.

## Development commands

```bash
uv sync --dev                    # setup
uv run pytest                    # tests (phantom-based correctness, CLI e2e)
uv run ruff check src tests      # lint
uv run ruff format src tests     # format (CI checks this)
uv run ty check src             # types
uv run mkdocs build --strict     # docs
```

ants-dependent paths need a separate venv (antspyx has no cp314 wheel):

```bash
uv venv --python 3.12 .venv-ants
VIRTUAL_ENV=.venv-ants uv pip install -e ".[ants,plot]" pytest pytest-cov
.venv-ants/bin/python -m pytest tests/
```

## Testing philosophy

Test the **public API contract**, never private internals (`_fcm.py`,
`_image.py` internals). Suites by kind:

- `tests/test_individual.py`, `test_population.py`, `test_ravel.py` — oracle
  tests on phantoms with *known* tissue statistics (`conftest.make_phantom`).
- `tests/test_laws.py` — hypothesis property tests: type/shape preservation,
  finiteness, scale equivariance, determinism, nyul monotonicity,
  serialization round-trips, rejection laws. Assertions on standardized
  outputs use `atol`, never `rtol` (values cross zero).
- `tests/test_metadata.py` — affine/header/dtype preservation: identical
  affine and qform/sform codes, unmutated source header, inexact-dtype
  preservation (float64 kept, else float32), save/reload round-trips, scaled (scl_slope) sources, mask/image
  affine mismatch rejection.
- `tests/test_regressions.py` — smallest reproducer per fixed bug; each
  docstring names the failure it guards.
- `tests/test_cli.py` — typer `CliRunner` end-to-end on generated NIfTI
  fixtures: exit codes, output naming, save/load state equivalence.

Hard-won invariants the suite guards (do not regress):

- nibabel restore must copy the header and set its dtype to the data's —
  an int16 source header truncates normalized floats on save otherwise.
  The image data path preserves float64 (float64 in -> float64 out) and
  canonicalizes everything else to float32; derived/internal data (RAVEL's
  image matrix, membership maps, learned parameters) stays float32 for
  memory.
- Mask/image affine mismatch must raise (same shape, different space is a
  silent-corruption trap).
- Statistics over foregrounds are computed in float64 (float32 accumulation
  makes near-constant foregrounds look non-degenerate).
- Constant/near-constant foregrounds raise actionable errors, never NaN or
  scipy `LinAlgError` leaks.
- Fitted transforms and `RavelResult` are frozen dataclasses with
  write-protected arrays: a saved transform always matches the in-memory one.
- RAVEL's `_WorkSpace` owns images *and masks* in the working space (masks
  are warped nearest-neighbor in template space), so a native-space mask can
  never index a template-space matrix.
