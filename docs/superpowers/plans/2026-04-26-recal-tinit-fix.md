# Recal T_init Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `T_init` calibration for the four `recal_X` strategies so they start at the correct base-kind leverage (≈2.14x for `recal_static` in the user's primary scenario) instead of bottoming out at 1.0x.

**Architecture:** Two parts.
1. **JIT plumbing**: add `init_strat_idx` int parameter to `_simulate_core` / `_simulate_core_grid` (used by `meta_recal` to initialize `strat_active` so between-recal logic for years 0 → first recal uses the chosen strategy's update rule).
2. **Calibration logic in `app.py compute()`**: for `recal_static` / `recal_hybrid` / `recal_adaptive_dd`, replace the binary search over the recal trajectory with a calibration of the corresponding **base kind**. For `meta_recal`, calibrate all four base candidates `{static, dd_decay, adaptive_dd, hybrid}` and pick `argmax(T_rec)`; the winner's index in that ordered list becomes `init_strat_idx`.

**Tech Stack:** Python 3, numpy, numba, scipy, streamlit. Run scripts with `.venv/bin/python`. Numba cache lives at `__pycache__/` — first run after a JIT signature change recompiles, subsequent runs use the cache.

---

## File Structure

**Modified:**
- `project_portfolio.py` — JIT cores (`_simulate_core`, `_simulate_core_grid`) and Python wrappers (`simulate`, `call_rate`, `find_max_safe_T_grid`, `find_max_safe_T`)
- `app.py` — `compute()` calibration block (~line 497–525), `compute()` projection block (~line 600+), calibration display section

**Created:**
- `verify_recal_tinit.py` — regression-style verification script (compares plain base T_rec with what `recal_X` should produce post-fix)

---

## Tasks

### Task 1: Add `init_strat_idx` parameter to `_simulate_core`

**Files:**
- Modify: `project_portfolio.py:361-395` (signature + strat_active init)

- [ ] **Step 1: Update the function signature**

Find `_simulate_core` at `project_portfolio.py:361`. Edit the signature to append `init_strat_idx` as the final parameter:

```python
@numba.njit(cache=True, fastmath=True)
def _simulate_core(ret, tsy, cpi, kind_code, T_init, C, S, T_yrs, S2, max_days,
                   avail, F, floor, checkpoint_days, cap_real, wealth_X,
                   vol_factor, dip_threshold, dip_bonus,
                   rate_threshold, rate_factor,
                   recal_period_days, t_recal_table, e_recal_grid,
                   h_recal_grid_days,
                   t_recal_tables_meta, meta_strategy_codes,
                   init_strat_idx):
```

- [ ] **Step 2: Update `strat_active` initialization**

Find the existing line at `project_portfolio.py:395`:

```python
    strat_active = np.zeros(K, dtype=np.int64)   # meta_recal: index of selected strategy
```

Replace with:

```python
    strat_active = np.full(K, init_strat_idx, dtype=np.int64)   # meta_recal: index of selected strategy
```

- [ ] **Step 3: Verify the module still imports**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && rm -rf __pycache__ && .venv/bin/python -c "import project_portfolio; print('OK')"
```

Expected output:
```
OK
```

(No Numba `TypingError` or signature mismatch errors. The `rm -rf __pycache__` clears stale JIT cache from the old signature.)

- [ ] **Step 4: Commit**

```bash
git add project_portfolio.py && git commit -m "$(cat <<'EOF'
feat(simulate): add init_strat_idx param to _simulate_core

Initializes strat_active[k] for meta_recal so the years-before-first-recal
phase applies the chosen strategy's update logic. Default 0 (set in the
Python wrapper) preserves existing behavior for non-meta kinds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add `init_strat_idx` parameter to `_simulate_core_grid`

**Files:**
- Modify: `project_portfolio.py:1043-1066` (signature + strat_active init)

- [ ] **Step 1: Update the function signature**

Find `_simulate_core_grid` at `project_portfolio.py:1043`. Edit the signature to append `init_strat_idx`:

```python
@numba.njit(cache=True, fastmath=True)
def _simulate_core_grid(ret, tsy, cpi, kind_code, T_inits, C, S, T_yrs, S2,
                        max_days, avail, F, floor, cap_real, wealth_X,
                        vol_factor, dip_threshold, dip_bonus,
                        rate_threshold, rate_factor,
                        recal_period_days, t_recal_table, e_recal_grid,
                        h_recal_grid_days,
                        t_recal_tables_meta, meta_strategy_codes,
                        init_strat_idx):
```

- [ ] **Step 2: Update `strat_active` initialization**

Find at `project_portfolio.py:1066`:

```python
    strat_active = np.zeros((K, T_count), dtype=np.int64)
```

Replace with:

```python
    strat_active = np.full((K, T_count), init_strat_idx, dtype=np.int64)
```

- [ ] **Step 3: Verify imports**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && rm -rf __pycache__ && .venv/bin/python -c "import project_portfolio; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add project_portfolio.py && git commit -m "$(cat <<'EOF'
feat(simulate): add init_strat_idx param to _simulate_core_grid

Mirrors the change in _simulate_core for the grid-vectorized path used by
find_max_safe_T_grid. Default 0 (set in the Python wrapper) preserves
existing behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Thread `init_strat_idx` through Python wrappers

**Files:**
- Modify: `project_portfolio.py:950+` (`simulate`), `1017+` (`call_rate`), `1597+` (`find_max_safe_T_grid`), `1674+` (`find_max_safe_T`)

- [ ] **Step 1: Update `simulate()` wrapper**

In `project_portfolio.py`, find `simulate(...)` at line 950. Add `init_strat_idx=0` as a kwarg at the end of the signature. Pass it to `_simulate_core`.

The signature edit (around line 950):
```python
def simulate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
             avail=None, F=1.5, floor=1.0, checkpoint_days=None,
             cap_real=float("inf"), wealth_X=float("inf"),
             vol_factor=0.0, dip_threshold=0.0, dip_bonus=0.0,
             rate_threshold=float("inf"), rate_factor=0.0,
             recal_period_days=0, t_recal_table=None,
             e_recal_grid=None, h_recal_grid_days=None,
             t_recal_tables_meta=None, meta_strategy_codes=None,
             init_strat_idx=0):
```

Find the `_simulate_core(...)` call inside `simulate` (it's the one that returns `(real_eq, called, peak_lev, lev_at_cp)`). Append `int(init_strat_idx)` as the final positional argument matching the new JIT signature.

If the existing call ends like:
```python
                                    t_recal_tables_meta, meta_strategy_codes)
```

Change it to:
```python
                                    t_recal_tables_meta, meta_strategy_codes,
                                    int(init_strat_idx))
```

- [ ] **Step 2: Update `call_rate()` wrapper**

In `project_portfolio.py:1017`, edit `call_rate` similarly:

Add `init_strat_idx=0` kwarg to the signature.

Add `init_strat_idx=init_strat_idx` to the inner `simulate(...)` call.

The full edit replaces the existing `call_rate` body. The new signature:
```python
def call_rate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
              avail=None, F=1.5, cap_real=float("inf"), wealth_X=float("inf"),
              vol_factor=0.0, dip_threshold=0.0, dip_bonus=0.0,
              rate_threshold=float("inf"), rate_factor=0.0,
              recal_period_days=0, t_recal_table=None,
              e_recal_grid=None, h_recal_grid_days=None,
              t_recal_tables_meta=None, meta_strategy_codes=None,
              init_strat_idx=0):
    _, called, _, _ = simulate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
                            avail=avail, F=F, cap_real=cap_real, wealth_X=wealth_X,
                            vol_factor=vol_factor, dip_threshold=dip_threshold,
                            dip_bonus=dip_bonus, rate_threshold=rate_threshold,
                            rate_factor=rate_factor,
                            recal_period_days=recal_period_days,
                            t_recal_table=t_recal_table,
                            e_recal_grid=e_recal_grid,
                            h_recal_grid_days=h_recal_grid_days,
                            t_recal_tables_meta=t_recal_tables_meta,
                            meta_strategy_codes=meta_strategy_codes,
                            init_strat_idx=init_strat_idx)
    return float(called.mean())
```

- [ ] **Step 3: Update `find_max_safe_T_grid()` wrapper**

In `project_portfolio.py:1597`, edit `find_max_safe_T_grid`:

Add `init_strat_idx=0` kwarg to the signature (after `meta_strategy_codes=None`).

Inside, in the `_eval(T_grid)` inner function, append `int(init_strat_idx)` to the `_simulate_core_grid(...)` call as the final positional argument.

The `_eval` body becomes:
```python
    def _eval(T_grid):
        called = _simulate_core_grid(ret, tsy, cpi, kind_code, T_grid,
                                     float(C), float(S), float(T_yrs), float(S2),
                                     int(max_days), avail, float(F), float(floor),
                                     float(cap_real), float(wealth_X),
                                     float(vol_factor), float(dip_threshold),
                                     float(dip_bonus), float(rate_threshold),
                                     float(rate_factor),
                                     int(recal_period_days), t_recal_table,
                                     e_recal_grid, h_recal_grid_days,
                                     t_recal_tables_meta, meta_strategy_codes,
                                     int(init_strat_idx))
        return called.mean(axis=0)
```

- [ ] **Step 4: Confirm `find_max_safe_T()` does not need the param**

`find_max_safe_T` (the legacy non-grid binary search at `project_portfolio.py:1674`) does not accept recal kwargs and is only used for plain kinds via `run()` / CLI mode. Since `call_rate(init_strat_idx=0)` is the default and `find_max_safe_T` never passes recal tables, the parameter is irrelevant there. Leave this function unchanged.

- [ ] **Step 5: Verify the module still imports and runs a quick sim**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && rm -rf __pycache__ && .venv/bin/python -c "
import numpy as np
from project_portfolio import simulate, TD
# Minimal sanity: 1 path, 1 day, kind=static, default init_strat_idx
ret = np.zeros((1, 2), dtype=np.float32)
tsy = np.full((1, 2), 0.04, dtype=np.float32)
cpi = np.full((1, 2), 100.0, dtype=np.float32)
real_eq, called, peak_lev, lev_at_cp = simulate(
    ret, tsy, cpi, 'static', 1.5, 100.0, 0.0, 1.0, 0.0, 1,
    avail=np.array([1], dtype=np.int64))
print(f'real_eq[0,0]={real_eq[0,0]:.2f} (expect 100.00)')
print(f'real_eq[0,1]={real_eq[0,1]:.2f}')
print(f'called[0]={called[0]}')
print('OK')
"
```

Expected output (final line):
```
OK
```

(The exact real_eq[0,1] value depends on tsy compounding; it should be slightly less than 100 since static is taking a 0.5x loan and rates are non-zero.)

- [ ] **Step 6: Commit**

```bash
git add project_portfolio.py && git commit -m "$(cat <<'EOF'
feat(simulate): thread init_strat_idx through Python wrappers

simulate(), call_rate(), find_max_safe_T_grid() now accept init_strat_idx
(default 0). Used by app.py to set the meta_recal initial strategy in the
years-before-first-recal phase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Write the verification script

**Files:**
- Create: `verify_recal_tinit.py`

This script computes plain base-kind T_rec values for the user's primary scenario. Post-fix, the app's `recal_X` strategies should display these same numbers.

- [ ] **Step 1: Write the script**

Create `verify_recal_tinit.py`:

```python
"""Verify recal_X T_init calibration matches plain base-kind T_rec.

Run with: .venv/bin/python verify_recal_tinit.py

This script computes well-defended T_rec for plain {static, hybrid,
adaptive_dd, dd_decay} using the same calibration pattern as app.py
(historical 0% calls + bootstrap <= 1% + stretched 0% calls).

Post-fix expectation:
- recal_static    T_rec in the streamlit calibration table ≈ T_static here
- recal_hybrid    T_rec ≈ T_hybrid
- recal_adaptive_dd T_rec ≈ T_adaptive
- meta_recal      T_rec ≈ max(T_static, T_hybrid, T_adaptive, T_dd_decay)
- meta_recal      init_base = argmax(...) (one of static/dd_decay/adaptive_dd/hybrid)

Pre-fix: recal_X T_rec collapses to 1.0x (binary search bottoms out).
"""
import numpy as np

from data_loader import load
from project_portfolio import (
    build_historical_paths, build_bootstrap_paths,
    find_max_safe_T_grid, stretch_returns, TD,
)

# User's primary scenario — keep in sync with the streamlit defaults
C = 160_000.0
S = 180_000.0
T_yrs = 5.0
S2 = 30_000.0
HORIZON_YEARS = 30
F = 1.5
WEALTH_X = 3_000_000.0
BOOT_TARGET = 0.01
STRETCH_F = 1.1
N_BOOT = 500

max_days = HORIZON_YEARS * TD

print(f"Scenario: C={C:,.0f}  S={S:,.0f}/yr×{T_yrs:g}y  S2={S2:,.0f}/yr  "
      f"X={WEALTH_X:,.0f}  H={HORIZON_YEARS}y  F={F}  stretch={STRETCH_F}  "
      f"boot_target={BOOT_TARGET:.0%}  N_boot={N_BOOT}")

# Load data + paths
dates, px, tsy, cpi = load(with_cpi=True)
ret_c, tsy_c, cpi_c, avail_c, _entry_dates = build_historical_paths(
    dates, px, tsy, cpi, max_days)

rng = np.random.default_rng(42)
ret_b, tsy_b, cpi_b = build_bootstrap_paths(
    dates, px, tsy, cpi, max_days, n_paths=N_BOOT,
    block_days=int(1.0 * TD), rng=rng)

ret_s = stretch_returns(ret_c, STRETCH_F) if STRETCH_F > 1.0 else None


def calibrate(kind):
    """Plain well-defended calibration: min(T_hist, T_boot, T_stretch)."""
    T_hist = find_max_safe_T_grid(
        ret_c, tsy_c, cpi_c, kind, 0.0,
        C, S, T_yrs, S2, max_days, avail=avail_c,
        F=F, wealth_X=WEALTH_X)
    T_boot = find_max_safe_T_grid(
        ret_b, tsy_b, cpi_b, kind, BOOT_TARGET,
        C, S, T_yrs, S2, max_days,
        F=F, wealth_X=WEALTH_X)
    if ret_s is not None:
        T_stretch = find_max_safe_T_grid(
            ret_s, tsy_c, cpi_c, kind, 0.0,
            C, S, T_yrs, S2, max_days, avail=avail_c,
            F=F, wealth_X=WEALTH_X)
    else:
        T_stretch = float("inf")
    return min(T_hist, T_boot, T_stretch), T_hist, T_boot, T_stretch


print("\nCalibrating plain base kinds (well-defended)...")
results = {}
for kind in ["static", "hybrid", "adaptive_dd", "dd_decay"]:
    T_rec, T_h, T_b, T_s = calibrate(kind)
    results[kind] = T_rec
    print(f"  {kind:14s}  T_rec={T_rec:.3f}x  "
          f"(hist={T_h:.3f}x  boot={T_b:.3f}x  "
          f"stretch={'inf' if T_s == float('inf') else f'{T_s:.3f}x'})")

# meta_recal = argmax over the four
meta_kinds = ["static", "dd_decay", "adaptive_dd", "hybrid"]   # MUST match app.py order
meta_T = [results[k] for k in meta_kinds]
winner_idx = int(np.argmax(meta_T))
winner_kind = meta_kinds[winner_idx]
T_meta = max(meta_T)

print(f"\nPost-fix expected T_rec values:")
print(f"  recal_static       T_rec = {results['static']:.3f}x")
print(f"  recal_hybrid       T_rec = {results['hybrid']:.3f}x")
print(f"  recal_adaptive_dd  T_rec = {results['adaptive_dd']:.3f}x")
print(f"  meta_recal         T_rec = {T_meta:.3f}x  "
      f"(winner: {winner_kind}, init_strat_idx={winner_idx})")
```

- [ ] **Step 2: Run pre-fix to capture baseline**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && .venv/bin/python verify_recal_tinit.py
```

Expected output (approximate, depends on data):
```
Scenario: C=160,000  S=180,000/yr×5y  S2=30,000/yr  ...

Calibrating plain base kinds (well-defended)...
  static          T_rec=2.140x  (hist=...x  boot=...x  stretch=...x)
  hybrid          T_rec=1.661x  (...)
  adaptive_dd     T_rec=...
  dd_decay        T_rec=...

Post-fix expected T_rec values:
  recal_static       T_rec = 2.140x
  recal_hybrid       T_rec = 1.661x
  recal_adaptive_dd  T_rec = ...
  meta_recal         T_rec = 2.140x  (winner: static, init_strat_idx=0)
```

The user's scenario shows `recal_static` should produce ~2.140x post-fix. Save this output mentally — it's the target.

- [ ] **Step 3: Commit the script**

```bash
git add verify_recal_tinit.py && git commit -m "$(cat <<'EOF'
test: verification script for recal T_init calibration

Computes plain well-defended T_rec for {static, hybrid, adaptive_dd,
dd_decay} on the user's primary scenario. Used to verify the recal_X
strategies in the streamlit app calibrate to matching values after the fix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update `compute()` calibration — `recal_static` / `recal_hybrid` / `recal_adaptive_dd`

**Files:**
- Modify: `app.py:497-525` (the per-strategy calibration loop)

The current loop calibrates every strategy by passing `kind=spec["kind"]` directly to `find_max_safe_T_grid` along with `recal_kw_for(name)`. For `recal_X` kinds, this triggers the recal-trajectory binary search that bottoms out at 1.0x.

The fix: when iterating, detect `recal_X` kinds and substitute the corresponding base kind for the `find_max_safe_T_grid` calls. Don't pass `recal_kw_for(name)` to those calibration calls (base kinds don't use the lookup table).

- [ ] **Step 1: Add a helper mapping near the top of `compute()` body**

In `app.py compute()`, after the `strategies` reconstruction block and before the calibration loop, add a mapping (place it just before `calibrated = {}` at ~line 497):

```python
    # Map recal_X strategies to their base kind for T_init calibration.
    # The base kind's well-defended T_rec is used as T_init for the recal
    # simulation: at year 0, the recal strategy behaves like the plain
    # base kind. At each recal event, lookup-table values take over.
    BASE_KIND_FOR_RECAL = {
        "recal_static": "static",
        "recal_hybrid": "hybrid",
        "recal_adaptive_dd": "adaptive_dd",
    }
```

- [ ] **Step 2: Replace the calibration loop body**

Find the loop at `app.py:497`:

```python
    calibrated = {}
    for name, spec in strategies:
        kind = spec["kind"]
        F = spec.get("F", 1.5)
        T_hist = find_max_safe_T_grid(ret_c, tsy_c, cpi_c, kind, 0.0,
                                       C, S, T, S2, max_days, avail=avail_c, F=F,
                                       cap_real=cap_real, wealth_X=wealth_X,
                                       **overlay_kw, **recal_kw_for(name))
        boot_at_hist = call_rate(ret_b, tsy_b, cpi_b, kind, T_hist,
                                  C, S, T, S2, max_days, F=F, cap_real=cap_real,
                                  wealth_X=wealth_X, **overlay_kw, **recal_kw_for(name))
        T_boot = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, kind, boot_target,
                                       C, S, T, S2, max_days, F=F,
                                       cap_real=cap_real, wealth_X=wealth_X,
                                       **overlay_kw, **recal_kw_for(name))
        if ret_s is not None:
            T_stress = find_max_safe_T_grid(ret_s, tsy_c, cpi_c, kind, 0.0,
                                             C, S, T, S2, max_days, avail=avail_c,
                                             F=F, cap_real=cap_real,
                                             wealth_X=wealth_X, **overlay_kw,
                                             **recal_kw_for(name))
        else:
            T_stress = float("inf")
        T_rec = min(T_hist, T_boot, T_stress)
        calibrated[name] = dict(spec=spec, T_hist=T_hist,
                                boot_at_hist=boot_at_hist,
                                T_boot=T_boot,
                                T_stress=T_stress if ret_s is not None else None,
                                T_rec=T_rec)
```

Replace with:

```python
    calibrated = {}
    for name, spec in strategies:
        kind = spec["kind"]
        F = spec.get("F", 1.5)

        # For recal_static / recal_hybrid / recal_adaptive_dd: calibrate
        # using the base kind, NOT the recal trajectory. The recal lookup
        # table is well-defended per cell already; T_init equals what the
        # plain base kind would give at year 0. The years-0-to-first-recal
        # phase therefore behaves like the plain base.
        # meta_recal is handled in the next block.
        if kind in BASE_KIND_FOR_RECAL:
            calib_kind = BASE_KIND_FOR_RECAL[kind]
            calib_recal_kw = {}   # base kinds don't use the lookup table
        elif kind == "meta_recal":
            # Skip: handled in the meta_recal block below
            continue
        else:
            calib_kind = kind
            calib_recal_kw = recal_kw_for(name)

        T_hist = find_max_safe_T_grid(ret_c, tsy_c, cpi_c, calib_kind, 0.0,
                                       C, S, T, S2, max_days, avail=avail_c, F=F,
                                       cap_real=cap_real, wealth_X=wealth_X,
                                       **overlay_kw, **calib_recal_kw)
        boot_at_hist = call_rate(ret_b, tsy_b, cpi_b, calib_kind, T_hist,
                                  C, S, T, S2, max_days, F=F, cap_real=cap_real,
                                  wealth_X=wealth_X, **overlay_kw,
                                  **calib_recal_kw)
        T_boot = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, calib_kind, boot_target,
                                       C, S, T, S2, max_days, F=F,
                                       cap_real=cap_real, wealth_X=wealth_X,
                                       **overlay_kw, **calib_recal_kw)
        if ret_s is not None:
            T_stress = find_max_safe_T_grid(ret_s, tsy_c, cpi_c, calib_kind, 0.0,
                                             C, S, T, S2, max_days, avail=avail_c,
                                             F=F, cap_real=cap_real,
                                             wealth_X=wealth_X, **overlay_kw,
                                             **calib_recal_kw)
        else:
            T_stress = float("inf")
        T_rec = min(T_hist, T_boot, T_stress)
        calibrated[name] = dict(spec=spec, T_hist=T_hist,
                                boot_at_hist=boot_at_hist,
                                T_boot=T_boot,
                                T_stress=T_stress if ret_s is not None else None,
                                T_rec=T_rec,
                                init_base_kind=BASE_KIND_FOR_RECAL.get(kind),
                                init_strat_idx=0)
```

(The `init_base_kind` field is informational; `init_strat_idx=0` is the default for non-meta kinds and gets used in the projection step.)

- [ ] **Step 3: Smoke test that compute still runs (Streamlit)**

Run the streamlit app:
```bash
cd /Users/dnissim/projects/margin_simulator && .venv/bin/streamlit run app.py
```

In the sidebar:
- Set scenario to `C=160000, S=180000, T=5, S2=30000, X=3000000`, horizon 30y
- Enable strategies: `static`, `hybrid`, `adaptive_dd`, `recal_static`, `recal_hybrid`, `recal_adaptive_dd` (NOT meta_recal yet — that comes in Task 6)
- Set `recal period (months)` to 60
- Click "Run / refresh"

Expected calibration table values (approximate):
- `static` T_rec ≈ 2.14x
- `hybrid` T_rec ≈ 1.66x
- `adaptive_dd` T_rec ≈ (some value)
- `recal_static` T_rec ≈ 2.14x  ← was 1.0x before fix
- `recal_hybrid` T_rec ≈ 1.66x  ← was 1.0x before fix
- `recal_adaptive_dd` T_rec ≈ (matches plain adaptive_dd)  ← was 1.0x before fix

If `recal_X` rows still show 1.0x, the fix didn't take — recheck that `BASE_KIND_FOR_RECAL` is being consulted before the fall-through branch.

Stop the streamlit process with Ctrl+C when done.

- [ ] **Step 4: Commit**

```bash
git add app.py && git commit -m "$(cat <<'EOF'
fix(app): calibrate recal_X T_init using base kind

For recal_static / recal_hybrid / recal_adaptive_dd, T_init now equals
the corresponding plain base kind's well-defended T_rec. Previously the
binary search ran over the full recal trajectory and bottomed out at
1.0x because the recal events themselves produce ~3% bootstrap calls
regardless of T_init.

Effect: recal strategies now match plain-base behavior in the years
before the first recal event (e.g., recal_static at T_init=2.14x for
the user's primary scenario instead of 1.0x).

meta_recal handled separately in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Update `compute()` calibration — `meta_recal`

**Files:**
- Modify: `app.py:525+` (add meta_recal calibration block after the main loop)

Calibrate all four base candidates `{static, dd_decay, adaptive_dd, hybrid}` (the `meta_kinds` list, in that fixed order). The winner's index becomes `init_strat_idx`. The winner's T_rec becomes `T_init` for the projection.

- [ ] **Step 1: Add the meta_recal calibration block**

After the main `for name, spec in strategies:` loop (which now `continue`s past `meta_recal`), and before any subsequent code that uses `calibrated`, add:

```python
    # meta_recal: calibrate all four base candidates and pick argmax(T_rec).
    # The winner's index in meta_kinds becomes init_strat_idx so the
    # years-before-first-recal phase applies the chosen strategy's logic.
    META_KINDS = ["static", "dd_decay", "adaptive_dd", "hybrid"]   # MUST match the app.py meta_kinds ordering
    meta_strategies = [(n, s) for n, s in strategies if s["kind"] == "meta_recal"]
    for name, spec in meta_strategies:
        F = spec.get("F", 1.5)
        per_base = {}
        for base_kind in META_KINDS:
            T_h = find_max_safe_T_grid(ret_c, tsy_c, cpi_c, base_kind, 0.0,
                                        C, S, T, S2, max_days, avail=avail_c,
                                        F=F, cap_real=cap_real, wealth_X=wealth_X,
                                        **overlay_kw)
            T_bo = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, base_kind, boot_target,
                                         C, S, T, S2, max_days, F=F,
                                         cap_real=cap_real, wealth_X=wealth_X,
                                         **overlay_kw)
            if ret_s is not None:
                T_st = find_max_safe_T_grid(ret_s, tsy_c, cpi_c, base_kind, 0.0,
                                             C, S, T, S2, max_days, avail=avail_c,
                                             F=F, cap_real=cap_real,
                                             wealth_X=wealth_X, **overlay_kw)
            else:
                T_st = float("inf")
            per_base[base_kind] = dict(T_hist=T_h, T_boot=T_bo, T_stress=T_st,
                                        T_rec=min(T_h, T_bo, T_st))

        # argmax over base kinds
        T_recs = [per_base[k]["T_rec"] for k in META_KINDS]
        winner_idx = int(np.argmax(T_recs))
        winner_kind = META_KINDS[winner_idx]
        T_rec = T_recs[winner_idx]
        winner = per_base[winner_kind]

        # Strategy-level boot rate at the chosen T (for display only)
        boot_at_hist = call_rate(ret_b, tsy_b, cpi_b, "meta_recal", T_rec,
                                  C, S, T, S2, max_days, F=F, cap_real=cap_real,
                                  wealth_X=wealth_X, **overlay_kw,
                                  **recal_kw_for(name),
                                  init_strat_idx=winner_idx)

        calibrated[name] = dict(
            spec=spec,
            T_hist=winner["T_hist"],
            boot_at_hist=boot_at_hist,
            T_boot=winner["T_boot"],
            T_stress=(winner["T_stress"] if ret_s is not None else None),
            T_rec=T_rec,
            init_base_kind=winner_kind,
            init_strat_idx=winner_idx,
            per_base=per_base,   # full table for display
        )
```

(`numpy` is already imported as `np` at the top of `app.py`.)

- [ ] **Step 2: Smoke test in Streamlit**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && .venv/bin/streamlit run app.py
```

Enable `meta_recal` (along with `static`, `hybrid`, `adaptive_dd` for comparison). Click Run / refresh.

Expected: `meta_recal` row T_rec ≈ max over the plain base T_rec values shown in the same table. For the user's primary scenario, this is `static` at ~2.14x.

If the initialization is wrong, T_rec might still come out as 1.0x or as a different base kind's value. Verify by comparing: `meta_recal T_rec == max(static T_rec, hybrid T_rec, adaptive_dd T_rec, dd_decay T_rec)`.

If `dd_decay` is not in the displayed strategies but is the actual winner, the user won't see that row — that's still fine; the calibration internally evaluated all four. (`init_base_kind` field will reflect the actual winner.)

Stop streamlit with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add app.py && git commit -m "$(cat <<'EOF'
fix(app): calibrate meta_recal T_init = argmax over base candidates

meta_recal now calibrates all four base candidates {static, dd_decay,
adaptive_dd, hybrid} as plain strategies, picks argmax(T_rec), and stores
the winner's index as init_strat_idx. Mirrors the recal-event semantics
where each event picks the strategy with the highest lookup-table T.

The years-before-first-recal phase now applies the winning strategy's
update logic (via _simulate_core's strat_active initialization).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update `compute()` projection block to pass `init_strat_idx`

**Files:**
- Modify: `app.py:594-665` (the `for name, spec in strategies` projection loop)

Each `simulate(...)` and `call_rate(...)` invocation in the projection loop needs to receive the calibrated `init_strat_idx` (only meaningful for `meta_recal`; defaults to 0 for everything else).

- [ ] **Step 1: Pull `init_strat_idx` from `calibrated[name]` before the simulate calls**

In `app.py:594`, find the projection loop:

```python
    for name, spec in strategies:
        c = calibrated[name]
        T_target = c["T_rec"]
        kind = spec["kind"]
        F = spec.get("F", 1.5)
        # Projection on historical (variable horizon) — uses cap_real
        real_eq, called_h, peak_lev_h, lev_at_cp = simulate(
            ret_h, tsy_h, cpi_h, kind, T_target,
            C, S, T, S2, max_days, avail=avail_h, F=F,
            checkpoint_days=cp_days, cap_real=cap_real, wealth_X=wealth_X,
            **overlay_kw, **recal_kw_for(name))
```

Add a line after `F = spec.get("F", 1.5)` to pull init_strat_idx:

```python
        init_strat_idx = c.get("init_strat_idx", 0)
```

- [ ] **Step 2: Pass `init_strat_idx` to each `simulate(...)` and `call_rate(...)` call in the loop**

There are several `simulate(...)` and `call_rate(...)` calls inside the loop body (between line 600 and ~670). Each one needs `init_strat_idx=init_strat_idx` added to its kwargs.

Specifically, find every block matching `**overlay_kw, **recal_kw_for(name))` inside this loop and append `, init_strat_idx=init_strat_idx` before the closing paren. Examples:

Original:
```python
        real_eq, called_h, peak_lev_h, lev_at_cp = simulate(
            ret_h, tsy_h, cpi_h, kind, T_target,
            C, S, T, S2, max_days, avail=avail_h, F=F,
            checkpoint_days=cp_days, cap_real=cap_real, wealth_X=wealth_X,
            **overlay_kw, **recal_kw_for(name))
```

After:
```python
        real_eq, called_h, peak_lev_h, lev_at_cp = simulate(
            ret_h, tsy_h, cpi_h, kind, T_target,
            C, S, T, S2, max_days, avail=avail_h, F=F,
            checkpoint_days=cp_days, cap_real=cap_real, wealth_X=wealth_X,
            **overlay_kw, **recal_kw_for(name),
            init_strat_idx=init_strat_idx)
```

Apply the same `init_strat_idx=init_strat_idx` addition to:
- The "no-cap" `simulate(...)` call (at ~line 611)
- The bootstrap `simulate(...)` for safety (at ~line 638)
- The calibration paths `simulate(...)` (at ~line 647)
- The stress test `simulate(...)` (at ~line 654)

(In total: 5 simulate calls in this loop body, plus any future similar calls. Use grep to find them.)

Run a quick grep to confirm:
```bash
grep -n "recal_kw_for(name)" /Users/dnissim/projects/margin_simulator/app.py
```

Each match inside the projection loop body needs the `init_strat_idx=init_strat_idx` kwarg added. (Calls in `compute()`'s calibration block in Task 5/6 already handle init_strat_idx via direct calls or via their own bookkeeping — those should NOT be edited here.)

- [ ] **Step 3: Smoke test in Streamlit**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && .venv/bin/streamlit run app.py
```

Enable `meta_recal` and click Run / refresh.

Expected: The `meta_recal` projection chart should show real_eq/leverage trajectories that reflect the chosen base strategy's logic in years 0–5 (e.g., if `static` is the winner, leverage should drift naturally without rebalancing in years 0–5). At year 5, the recal event should fire and the lookup-table T should kick in.

Sanity check: at year 0, `meta_recal`'s starting leverage should equal its T_rec (≈2.14x for static-winner case), not 1.0x.

The leverage chart shows pre-rebalance leverage; for the static-winner case in years 0–5 there are no rebalances, so the chart should show natural drift from 2.14x downward toward 1.0x as wealth grows.

Stop streamlit with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add app.py && git commit -m "$(cat <<'EOF'
fix(app): thread init_strat_idx through projection sims

Each simulate/call_rate call in compute()'s projection loop now passes
init_strat_idx from calibrated[name]. For meta_recal this initializes
strat_active to the winning base kind's index, so years 0 → first recal
apply the chosen strategy's between-recal update rule. Default 0 for
all non-meta strategies preserves their existing behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Display winning base kind for `meta_recal`

**Files:**
- Modify: `app.py:780-800` (the Calibration section's row builder)

- [ ] **Step 1: Add `Init base` column to the calibration table**

In `app.py`, find the calibration row builder around line 789–799. The current code is:

```python
    calib_data = []
    for name, c in res["calibrated"].items():
        row = {
            "Strategy": name,
            "T_hist_safe": f"{c['T_hist']:.3f}x",
            "Boot @ hist": f"{100 * c['boot_at_hist']:.2f}%",
            "T_boot_safe": f"{c['T_boot']:.3f}x",
        }
        if stretch_F_used > 1.0 and c.get("T_stress") is not None:
            row[f"T_stress (F={stretch_F_used:.2f})"] = f"{c['T_stress']:.3f}x"
        row["T_recommended"] = f"{c['T_rec']:.3f}x"
        calib_data.append(row)
```

Replace with:

```python
    calib_data = []
    any_meta = any(c.get("init_base_kind") for c in res["calibrated"].values()
                   if c["spec"]["kind"] == "meta_recal")
    for name, c in res["calibrated"].items():
        row = {
            "Strategy": name,
            "T_hist_safe": f"{c['T_hist']:.3f}x",
            "Boot @ hist": f"{100 * c['boot_at_hist']:.2f}%",
            "T_boot_safe": f"{c['T_boot']:.3f}x",
        }
        if stretch_F_used > 1.0 and c.get("T_stress") is not None:
            row[f"T_stress (F={stretch_F_used:.2f})"] = f"{c['T_stress']:.3f}x"
        row["T_recommended"] = f"{c['T_rec']:.3f}x"
        if any_meta:
            # Only meta_recal sets init_base_kind to a non-empty string;
            # other recal_X strategies have init_base_kind set but their
            # init is conceptually fixed (not a choice), so show blank.
            if c["spec"]["kind"] == "meta_recal":
                row["Init base"] = c.get("init_base_kind") or ""
            else:
                row["Init base"] = ""
        calib_data.append(row)
```

The `any_meta` guard means the `Init base` column only appears in the table when at least one `meta_recal` strategy is enabled, keeping the table tidy for non-meta scenarios.

- [ ] **Step 2: Smoke test in Streamlit**

Run streamlit:
```bash
cd /Users/dnissim/projects/margin_simulator && .venv/bin/streamlit run app.py
```

Enable `meta_recal` (along with the four base strategies for comparison). Click Run / refresh.

Expected: the calibration table now has an `Init base` column. The `meta_recal` row shows the winning base kind name (e.g., `static`). Other rows show empty `Init base`.

Now disable `meta_recal` (keep the others). Refresh. The `Init base` column should disappear entirely (since `any_meta` is False).

Stop streamlit with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add app.py && git commit -m "$(cat <<'EOF'
feat(app): show winning base kind for meta_recal in calibration table

Adds an "Init base" column to the calibration table when any meta_recal
strategy is enabled. The meta_recal row shows the argmax base kind name
(e.g., static); non-meta rows are left blank.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Final cross-check + CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md` (add to §5f or create a new sub-section about the T_init fix)

- [ ] **Step 1: Re-run the verification script**

Run:
```bash
cd /Users/dnissim/projects/margin_simulator && .venv/bin/python verify_recal_tinit.py
```

Note the printed `static`, `hybrid`, `adaptive_dd`, `dd_decay` T_rec values.

- [ ] **Step 2: Open Streamlit, capture calibration values**

Run streamlit, enable all base strategies + all four recal strategies, click Run / refresh. Capture the T_rec column.

Verify:
- `recal_static` T_rec matches `static` T_rec (within 0.01x)
- `recal_hybrid` T_rec matches `hybrid` T_rec
- `recal_adaptive_dd` T_rec matches `adaptive_dd` T_rec
- `meta_recal` T_rec equals the max of the four base T_rec values
- `meta_recal` displayed `init_base` matches the argmax base name

If all match → fix is verified. If not, recheck Task 5/6 logic and the kwargs threading in Task 7.

- [ ] **Step 3: Update CLAUDE.md**

Append to §5f's "Open questions for next session" the resolution of question 1, and update key empirical observations. Edit CLAUDE.md to add or update content describing:

- Open question #1 ("fix T_init calibration") is resolved.
- The fix: recal_X T_init = base kind's well-defended T_rec. meta_recal T_init = argmax over four candidates.
- Updated empirical numbers for the user's primary scenario (replace the table that shows T_rec=1.000x with the new values from Step 2).

Add a section heading like `### T_init calibration fix (added in sixth session)` and capture:
- Before/after T_rec values
- Before/after p50@30y wealth values (capture from streamlit)
- Note that strategy-level boot call rate is still ~3% by design (per-cell ≤1%; multi-event compounding accepted)
- Reference the spec at `docs/superpowers/specs/2026-04-26-recal-tinit-fix-design.md`

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md && git commit -m "$(cat <<'EOF'
docs(CLAUDE.md): record recal T_init calibration fix

Updates §5f with the resolution of open question #1 and the new
empirical values for the user's primary scenario after the fix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Spec Coverage Check

Spec requirements ↔ tasks:

| Spec section | Task |
|---|---|
| `_simulate_core` adds `init_strat_idx` | Task 1 |
| `_simulate_core_grid` adds `init_strat_idx` | Task 2 |
| `simulate()`, `find_max_safe_T_grid()`, `find_max_safe_T()`, `call_rate()` accept `init_strat_idx` | Task 3 |
| Calibrate `recal_static`/`recal_hybrid`/`recal_adaptive_dd` using base kind | Task 5 |
| Calibrate `meta_recal` as argmax over four base kinds; store `init_strat_idx` | Task 6 |
| Pass `T_init`, `init_strat_idx` to projection sims | Task 7 |
| Display winning base kind for `meta_recal` | Task 8 |
| Acceptance: `recal_static` T_init ≈ plain static T_rec | Task 5 Step 3, Task 9 |
| Acceptance: `meta_recal` T_init = argmax base, `init_strat_idx` correct | Task 6 Step 2, Task 9 |
| Acceptance: day-0 leverage matches base | Task 7 Step 3 |
| Acceptance: regression — recal_static p50@30y ≥ plain static p50@30y | Task 9 Step 2 |

All spec requirements have corresponding tasks. ✓

## Out of scope reminders (deferred work)

These remain open after this plan completes — listed in the spec, intentionally not addressed here:

- "Option B": preserve `max_dd` ratchet across recal events
- `recal_period_months < T_yrs` lookup-table-DCA mismatch
- Adaptive `recal_period`
- Joint-defended sizing (strategy-level ≤1% boot)
- Checkpoint capture pre-vs-post-rebalance for recal events
- Documentation of recal strategies in `pages/Documentation.py`
- Test on different scenarios (low-DCA, longer horizon)
