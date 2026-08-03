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
├── errors.py        # one base exception, actionable messages
├── histogram.py     # KDE smoothing, tissue modes, modality→peak policy
├── io.py            # path I/O for the CLI (paths never enter the math API)
├── methods/         # the normalization algorithms
│   ├── _fcm.py      # PRIVATE in-house fuzzy c-means (replaces scikit-fuzzy)
│   ├── _transform.py# PRIVATE FittedTransform base (save/load stamped .npz)
│   ├── zscore.py fcm.py kde.py whitestripe.py   # individual: one function each
│   ├── nyul.py lsq.py                           # population: fit() -> transform
│   └── ravel.py                                 # batch-only, no apply-to-new
├── tools/           # tissue.py, plot.py (lazy matplotlib), ants.py (lazy ants)
└── cli/             # typer app; the only stringly-typed layer
```

Invariants to preserve:

- Math takes numpy, returns numpy. Type preservation (nibabel in → nibabel out)
  happens only in `_image.py`. No adapter/protocol layers — they were deleted
  on purpose.
- Construction *is* fitting for population methods: no unfitted states, no
  `is_fitted` flags. RAVEL is batch-only by design (no single-image transform).
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
uv run mypy src                  # types
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
  affine and qform/sform codes, unmutated source header, float32 storage
  dtype, save/reload round-trips, scaled (scl_slope) sources, mask/image
  affine mismatch rejection.
- `tests/test_regressions.py` — smallest reproducer per fixed bug; each
  docstring names the failure it guards.
- `tests/test_cli.py` — typer `CliRunner` end-to-end on generated NIfTI
  fixtures: exit codes, output naming, save/load state equivalence.

Hard-won invariants the suite guards (do not regress):

- nibabel restore must copy the header and set its dtype to float32 — an
  int16 source header truncates normalized floats on save otherwise.
- Mask/image affine mismatch must raise (same shape, different space is a
  silent-corruption trap).
- Statistics over foregrounds are computed in float64 (float32 accumulation
  makes near-constant foregrounds look non-degenerate).
- Constant/near-constant foregrounds raise actionable errors, never NaN or
  scipy `LinAlgError` leaks.
