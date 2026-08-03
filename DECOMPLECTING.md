# Decomplecting intensity-normalization — a design for ease of reasoning

## Context

v4 (see `PLAN.md`) gave the package a sound *module* architecture: methods are
array cores with thin image wrappers, `_image.py` alone knows nibabel, fitted
transforms are values. The remaining complexity is not in the module graph but
in the *braiding* inside and across those modules. This document applies two
lenses:

- **Hickey, "Simple Made Easy"**: *complect* means to interleave/braid
  independent concerns so they can no longer be reasoned about independently.
  Simple = one strand, one role. State complects everything it touches;
  closures complect context with function; conditionals braid policy into
  mechanism; **data and pure functions are simple**. Easy ≠ simple: an
  optional parameter is *easy* (near at hand, familiar) while braiding a
  policy into a mechanism — complexity that compounds.
- **Ousterhout, *A Philosophy of Software Design***: complexity is incremental
  (ch. 2); deep modules with small interfaces (ch. 4); different layers should
  carry different abstractions (ch. 7); pull complexity downwards so callers
  stay simple (ch. 8); bring together what shares information, separate what
  doesn't (ch. 9); define errors out of existence (ch. 10); names should
  create the right picture (ch. 14); consistency creates cognitive leverage
  (ch. 17).

The two agree on the diagnosis and mostly on the cure. Where they pull in
different directions (Hickey would dissolve the type system; Ousterhout would
keep more convenience in interfaces) the design below says which we follow and
why.

## The braids (what's complected today)

Each item: the strands involved, and why the braid compounds.

1. **Foreground definition ⊂ every core.** `get_mask` does four things —
   None-estimation ("positive voxels" policy), binarization (`> 0`), shape
   validation, emptiness validation — and is called by every core. The
   `mask: BinaryMask | None` parameter braids a *guess about the user's data*
   (policy) into every mechanism. Hickey: an optional argument is *easy*, not
   *simple* — it saves the caller one line and costs every reader a branch.
   PoSD ch. 10: the None case should be defined out of the cores, not handled
   in all of them.

2. **Estimate ⊂ apply, in every method.** zscore/whitestripe/kde/fcm/lsq each
   re-braid the same three steps: resolve foreground → estimate reference
   statistic(s) → affine-scale the image. The apply step is one line, but it
   carries real subtleties (float64 statistics, float32 storage, zero-spread
   rejection) that currently live in seven slightly different copies. There is
   no place where "normalization is an affine map of the intensities" is
   *said*. PoSD ch. 9 red flag: repetition of the same pattern means the
   abstraction hasn't been found.

3. **Modality policy resolved at the bottom of the stack.** `modality: str`
   and `peak` travel through `kde_array`/`whitestripe_array` into
   `histogram.tissue_mode`, where `MODALITY_PEAKS` (data!) is finally
   consulted. Policy should be resolved at the boundary and flow down as a
   value (`Peak`), so cores never branch on strings. PoSD ch. 7: different
   layers, different abstractions — the wrapper speaks "modality", the core
   speaks "peak".

4. **Type-preservation context hidden in a closure.** `unwrap` returns
   `(data, restore)` where `restore` secretly captures the source image's
   class, affine, and header. You cannot inspect it, log it, compare it, or
   test it independently. Hickey: context should be *data*. A frozen
   `ImageMeta` value makes the same information examinable and `restore` a
   pure function of it.

5. **Validation braided with conversion in `unwrap_mask`.** None-passthrough +
   binarization + affine agreement + (deferred) shape check, one function.
   Each is a separate strand; only the affine check genuinely needs nibabel.

6. **kwargs tunnels.** `ravel.fit_transform(whitestripe_kwargs: dict)` braids
   ravel's signature to whitestripe's, untyped: typos vanish, readers must
   hold both signatures in mind. The base `**kwargs` chain
   (`__call__` → `transform` → `transform_array`) exists only so LSQ's
   `membership` can ride along — one real parameter forcing a tunnel through
   three layers.

7. **Reference selection braided into LSQ fitting.** "The first image is the
   reference" is a policy silently encoded in `fit_array`'s loop. Making it
   explicit (`fit_reference(data, ...)` + `fit` composing it) separates *what*
   is learned from *which data* teaches it.

What is already simple (keep): frozen-dataclass transforms and `.npz` state
dicts are values with no time component; `io.py` keeps paths out of math;
`MODALITY_PEAKS` is data; seeds are explicit parameters, not ambient state;
the CLI is a thin dispatch layer.

## Target design

### The one-sentence version

> A normalization method is: **resolve a foreground (data) → estimate a
> reference (pure function) → apply an affine map (one shared function)**.
> Everything else — nibabel, modality names, mask conveniences, defaults —
> is boundary, and lives at the boundary.

### Strand 1 — foreground resolution has exactly one owner

```python
# _image.py — the only place these rules exist

def unwrap_mask(image: Image, mask: Mask | None) -> BinaryMask | None:
    """Boundary: nibabel-aware. Affine validation + binarization (> 0)."""

def resolve_foreground(data: IntensityArray, mask: BinaryMask | None) -> BinaryMask:
    """The one owner of foreground semantics.

    None → estimated as positive voxels (the *only* place that policy lives).
    Shape mismatch → actionable error. Empty → actionable error.
    """
```

**Rule: no core ever sees `mask=None`.** Wrappers resolve:

```python
def zscore(image: Image, mask: Mask | None = None, *, norm_value: float = 1.0) -> Image:
    data, meta = _image.unwrap(image)
    foreground = _image.resolve_foreground(data, _image.unwrap_mask(image, mask))
    return _image.restore(meta, zscore_array(data, foreground, norm_value=norm_value))
```

Cores take the foreground as **required data**:

```python
def zscore_array(data: IntensityArray, foreground: BinaryMask, *, norm_value: float = 1.0) -> IntensityArray: ...
def whitestripe_array(data, foreground: BinaryMask, *, peak: Peak, ...) -> IntensityArray: ...
def fcm_array(data, foreground: BinaryMask, *, modality, tissue, membership, ...) -> IntensityArray: ...
```

Population cores take per-item resolved masks: `fit_array(datas,
foregrounds: Sequence[BinaryMask], ...)`. The lone, *documented* exception is
`ravel_array`: foregrounds must be estimated on WhiteStripe-normalized arrays
(inside the algorithm), so it accepts `Sequence[BinaryMask | None]` and
resolves per item — through the same `resolve_foreground`, never its own
logic.

Consequences: the "positive voxels" estimation policy, the shape/empty
validation, and their error messages exist **once**. Deleting the None branch
from every core is PoSD ch. 10 applied mechanically: an entire case defined
out of the mechanisms.

### Strand 2 — the affine application is one shared function

```python
# methods/_common.py

def standardize(data: IntensityArray, center: float, spread: float, norm_value: float) -> IntensityArray:
    """(data - center) / spread * norm_value, stored float32.

    Spread validity is the *caller's* check: the zero-spread error must name
    the statistic that failed ('white stripe has zero standard deviation'),
    so each core validates before calling and keeps its actionable message.
    """
```

Every individual-method core ends in `return standardize(data, c, s, norm_value)`.
KDE/FCM/LSQ are the `center=0` special case; zscore/whitestripe use their
estimated (mean, std). The float64-stats-then-float32-store discipline gets
one home instead of seven copies. `_common.py` earns its module by ch. 9's
rule (it eliminates a repeated pattern with a simpler signature); it is
deliberately tiny and grows only when a *second* method shares a pattern.

### Strand 3 — policy resolves at the boundary and flows down as values

```python
# histogram.py — owns the modality→peak policy (it is histogram semantics)

def resolve_peak(modality: str, peak: Peak | None) -> Peak: ...
def tissue_mode(intensities: ForegroundIntensities, peak: Peak, ...) -> float: ...
```

Wrappers call `resolve_peak(modality, peak)`; cores take `peak: Peak` and
never see a modality string. `tissue_mode` loses its `modality` parameter —
a **public API change** (histogram is a public module); see migration notes.
`MODALITY_PEAKS` stays data, now consulted at exactly one point in the call
graph. The FCM modality rule ("memberships are only meaningful on T1-w") is
*algorithm* semantics, not peak policy — it stays in `fcm_array`.

### Strand 4 — image context is a value, not a closure

```python
# _image.py

@dataclasses.dataclass(frozen=True)
class ImageMeta:
    """Everything needed to rebuild the user's image from an array."""
    cls: type            # np.ndarray or the nibabel image class
    affine: IntensityArray | None
    header: typing.Any | None   # nibabel header copy (duck-typed by nibabel)

def unwrap(image: Image) -> tuple[IntensityArray, ImageMeta]: ...
def restore(meta: ImageMeta, data: IntensityArray) -> Image: ...
```

The header copy happens at unwrap time (same immutability guarantee as today:
the source header is never mutated). `Restorer` is deleted. Gains: the
context is inspectable in a debugger, printable in logs, comparable in tests
(`assert meta.affine == ...`), and `restore` is a pure function. This is the
most Hickey-motivated change and the least Ousterhout-motivated (the closure
*was* a deep little interface); we take it because the closure's opacity has
real costs (untestable restoration logic — the int16-header truncation bug
class lives here) and near-zero API cost, since callers already treated
`restore` opaquely.

### Strand 5 — no kwargs tunnels; parameters are explicit data

- `ravel.fit_transform` replaces `whitestripe_kwargs: dict[str, Any]` with a
  frozen value:

  ```python
  @dataclasses.dataclass(frozen=True)
  class WhiteStripeSpec:
      modality: str = "t1"
      peak: Peak | None = None
      width: float = 0.05
      width_l: float | None = None
      width_u: float | None = None
  ```

  `fit_transform(..., whitestripe: WhiteStripeSpec | None = None)`. Typed,
  documented in one place, typo-proof.

- LSQ's `membership` stops riding the base-class `**kwargs`. The base chain
  becomes `__call__(image, mask)` → `transform(image, mask)` →
  `transform_array(data, foreground)` with **no `**kwargs` anywhere**.
  `LSQTransform` overrides `transform(image, mask, *, membership=None)` —
  an LSP-legal override (an added keyword with a default) — so the one real
  extra parameter is explicit on the one class that has it.

## Module layout after the refactor

```
_image.py        # type aliases; ImageMeta; unwrap/restore; unwrap_mask;
                 # resolve_foreground; foreground_values  (the boundary)
histogram.py     # KDE, modes; MODALITY_PEAKS + resolve_peak (policy data)
methods/_common.py   # standardize (+ future shared math, when 2+ methods share it)
methods/*.py         # cores: required BinaryMask, resolved Peak, standardize()
                     # wrappers: modality/mask conveniences, unwrap → core → restore
methods/_transform.py# __call__ → transform → transform_array, no **kwargs
```

## Standing rules (to be added to AGENTS.md when implemented)

1. No core accepts `None` masks, modality strings, or `**kwargs`.
2. Foreground estimation, mask binarization, and mask/image validation happen
   once each, in `_image.py`.
3. Policy (modality→peak, reference selection, whitestripe parameters) is
   resolved at the boundary and flows down as values (`Peak`, `BinaryMask`,
   `WhiteStripeSpec`).
4. Context is data (`ImageMeta`, state dicts, frozen dataclasses), never
   closures or dict tunnels.
5. Errors are validated at the boundary with actionable messages; cores raise
   only statistic-specific failures ("white stripe has zero std").

## Phased migration

Each phase leaves `ty`, `ruff`, and the full test suite green. The suite
exercises the public wrappers, so phases 1–3 should need no test changes;
phases 4–5 touch internals only.

1. **Strand 1** — `resolve_foreground` (renamed `get_mask`, None-policy
   extracted); cores take required `BinaryMask`; wrappers resolve.
2. **Strand 3** — `resolve_peak`; `tissue_mode` drops `modality`; cores take
   `peak: Peak`. *Public API change*: `histogram.tissue_mode(intensities,
   modality=...)` callers must switch to `resolve_peak` first. Noted in
   `docs/migration.md`.
3. **Strand 2** — `methods/_common.standardize`; rewire the five cores.
4. **Strand 5** — `WhiteStripeSpec`; LSQ `transform` override; delete
   `**kwargs` from the transform chain. *Public API change*:
   `ravel.fit_transform(whitestripe_kwargs=...)` → `whitestripe=WhiteStripeSpec(...)`.
5. **Strand 4** — `ImageMeta`; `Restorer` deleted (private, no API change).

## Non-goals

- Not dissolving the type aliases into `Any` (Hickey's data-over-types stance
  serves a dynamically-typed host; here the aliases are documentation the
  checker enforces — they *aid* reasoning).
- Not splitting `_image.py` into mask/image modules: the boundary knowledge
  (nibabel, affines, binarization thresholds) is shared information —
  ch. 9 says bring it together.
- Not making seeds ambient: explicit `seed=` parameters are data passed in
  the open, and determinism-by-default is a feature.
