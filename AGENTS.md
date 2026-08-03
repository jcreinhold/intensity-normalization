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

Correctness is tested against synthetic phantoms with **known** tissue
statistics (see `tests/conftest.py::make_phantom`), not against snapshots of
previous versions. Serialization round-trips must be bit-exact. CLI tests use
typer's `CliRunner` on generated NIfTI fixtures.
