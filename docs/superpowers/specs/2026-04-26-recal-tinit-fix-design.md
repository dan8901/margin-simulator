# Recal strategies: T_init calibration fix

**Date:** 2026-04-26
**Scope:** Fix the `T_init` calibration for the four re-calibration strategies (`recal_static`, `recal_hybrid`, `recal_adaptive_dd`, `meta_recal`) so the years-before-first-recal phase uses the correct leverage instead of bottoming out at 1.0x.

## Problem

The recal strategies model the user's intended workflow: every N months, open the simulator app, observe current portfolio state, decide on a strategy/target for the next N months, repeat.

Currently, `app.py compute()` calibrates each `recal_X` strategy by binary-searching `T_init` over the *entire 30y trajectory including all recal events*:

```python
T_hist = find_max_safe_T_grid(ret_c, tsy_c, cpi_c, kind="recal_static", 0.0, ...)
T_boot = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, kind="recal_static", boot_target, ...)
T_stress = find_max_safe_T_grid(ret_s, tsy_c, cpi_c, kind="recal_static", 0.0, ...)
T_rec = min(T_hist, T_boot, T_stress)
```

The recal events themselves produce ~3% bootstrap calls regardless of `T_init` (they jump leverage at well-defended cell values, and the cumulative call probability across ~6 events compounds). The binary search therefore bottoms out at the lower bound (1.0x).

Result: `recal_static` runs unleveraged for years 0 through `recal_period`, missing the heavy-DCA early window where plain `static` can safely take 2.14x. The strategy is strictly dominated by plain `static` for the user's scenario.

## Intent

The recal strategies should simulate this workflow:

1. **Year 0**: User opens simulator with `(C, S, T, S2, horizon)`. The simulator recommends optimal T per strategy via the standard well-defended calibration (≤1% boot, hist 0%, stretch 0%). User picks one and follows it.
2. **Year `recal_period`**: User opens simulator again. New current state, updated remaining horizon, updated remaining DCA schedule. Simulator re-recommends. User adjusts.
3. Repeat.

The `T_init` for a `recal_X` strategy should equal what plain `X` would calibrate to at year 0 with the same scenario. Each recal-event lookup is well-defended at ≤1% boot per cell. Strategy-level call rate (over all events) is whatever falls out — empirically ~3%, accepted as the cost of multi-event exposure.

## Design

### Approach A (chosen): Calibrate `T_init` using the base kind

For each recal strategy, replace `find_max_safe_T_grid(kind="recal_X", ...)` with `find_max_safe_T_grid(kind=BASE_X, ...)`. The recal simulation then runs with that base-derived `T_init` — at year 0 it behaves identically to the plain base strategy; at recal events it deviates via the lookup table.

**Mapping:**
| Recal strategy | Base kind for T_init calibration |
|---|---|
| `recal_static` | `static` |
| `recal_hybrid` | `hybrid` |
| `recal_adaptive_dd` | `adaptive_dd` |
| `meta_recal` | `argmax({static, dd_decay, adaptive_dd, hybrid})` of T_rec |

For `meta_recal`, the winner's **index** within `meta_kinds = ["static", "dd_decay", "adaptive_dd", "hybrid"]` (0–3) becomes `init_strat_idx`. This initializes `strat_active[k]` so the between-recals logic for years 0 → first recal uses the chosen strategy's update rule (the JIT loop dereferences `meta_strategy_codes[strat_active[k]]` to get the actual kind_code).

### Why not Approach B / C

- **Approach B** (force a recal event at d=0, drive T_init from the lookup table): the lookup table is built with `S2` DCA throughout. Cell `(E=C, H=full_horizon)` for the user's scenario gives a conservative T because it ignores the high-DCA early phase. To fix correctly, the lookup table would need to depend on `(S, T, S2)` — a 4D table or per-path on-the-fly compute. Significant refactor for marginal benefit.
- **Approach C** (read T_init from the existing lookup cell at `(E=C, H=full)`): same `S2`-only-DCA problem as B; conservative T_init for the high-DCA early phase.

Approach A is the smallest-diff fix and gives correct semantics for the typical case `recal_period_months ≈ T_yrs` (user's scenario: both = 5y), where the lookup-table cells used at recal events are correct from year `T_yrs` onward.

### Code changes

**1. `project_portfolio.py` — JIT inner loops**

Add `init_strat_idx` parameter (int, default 0) to:
- `_simulate_core(...)`
- `_simulate_core_grid(...)`

Inside both, replace:
```python
strat_active = np.zeros(K, dtype=np.int64)
```
with:
```python
strat_active = np.full(K, init_strat_idx, dtype=np.int64)
```

**2. `project_portfolio.py` — Python wrappers**

Thread `init_strat_idx` through:
- `simulate(...)`: add kwarg `init_strat_idx=0`, pass to `_simulate_core`.
- `find_max_safe_T_grid(...)`: add kwarg `init_strat_idx=0`, pass to `_simulate_core_grid`.
- `find_max_safe_T(...)`: same.
- Any other call sites that invoke the JIT cores (e.g., `call_rate`).

**3. `app.py compute()` — calibration block (around line 497)**

For each strategy in the `strategies` list:
- If `kind` ∈ {`static`, `relever`, `dd_decay`, `wealth_decay`, `hybrid`, `r_hybrid`, `vol_hybrid`, `dip_hybrid`, `rate_hybrid`, `adaptive_dd`, `adaptive_hybrid`}: existing logic unchanged.
- If `kind` ∈ {`recal_static`, `recal_hybrid`, `recal_adaptive_dd`}: calibrate using the corresponding base kind (`static`, `hybrid`, `adaptive_dd`) with the user's full `(C, S, T, S2, max_days)` scenario. Use the resulting `T_rec` as `T_init` for the recal simulation. Skip the recal_kw_for(name) plumbing on the calibration calls (since base kinds don't need recal tables).
- If `kind == meta_recal`: calibrate all four base candidates `meta_kinds = ["static", "dd_decay", "adaptive_dd", "hybrid"]` (each well-defended: `T_rec = min(T_hist, T_boot, T_stress)` per the standard rest-of-app calibration). Pick `argmax`. Store both the winning T (`T_init`) and the winner's index in `meta_kinds` (`init_strat_idx`, integer 0–3). Also store the winning base name for display.

The displayed `T_rec`, `T_hist`, `T_boot`, `T_stress` for a `recal_X` strategy are the base kind's calibration values — same numbers a user would see on the plain base row, by design.

**4. `app.py compute()` — projection block**

When invoking `simulate(..., kind="recal_X", T_init=...)` for the projection step:
- Pass `T_init = base T_rec`.
- Pass `init_strat_idx = winner_idx` (only relevant for `meta_recal`; default 0 for the others).

The recal simulation now starts at the correct base-strategy leverage on day 0, then re-picks at each recal event via the lookup tables.

**5. UI display**

- Calibration table for `recal_X` shows the same `T_hist / T_boot / T_stress / T_rec` numbers as the plain base row (identical values by construction). For `meta_recal`, also display the winning base name (e.g., "init: static").
- *(Optional, deferred):* add a "trajectory boot %" column showing `call_rate(ret_b, ..., kind="recal_X", T_init=base_T_rec, init_strat_idx=...)` so the ~3% strategy-level call rate is visible alongside the per-cell ≤1% defense.

## Expected empirical impact

For the user's scenario (`C=160k, S=180k×5y, S2=30k, X=$3M, 30y, F=1.5`):

| Strategy | Before fix | After fix |
|---|---|---|
| `recal_static` T_init | 1.000x | ≈2.14x (= plain static) |
| `recal_static` p50@30y | $13.31M | ≥ $13.31M (likely higher: years 0-5 capture leverage) |
| `recal_static` boot rate | 3.0–3.2% | similar (~3%, may shift slightly) |

Years 0 → first recal now run as plain static at full safe leverage. Recal events from year `recal_period` onward continue to re-pick from the lookup table.

`recal_static` should now Pareto-dominate plain `static` (same year-0 start, additional re-leveraging in later years), at the cost of the strategy-level ~3% call rate vs. plain static's 0%.

## Out of scope (deferred)

These open questions remain after this fix:

1. **Option B: preserve `max_dd` across recal events.** Currently each recal resets `max_dd=0` and `hwm=current_eq`. Preserving these would propagate dd-ratchet learning across cycles. Predicted to lower bootstrap fragility from ~3% → ~1% at the cost of some IRR uplift on quiet paths.
2. **`recal_period_months < T_yrs` edge case.** Lookup table cells use `S2` DCA, so a recal at year 2 (when actual remaining DCA is still S=180k) underestimates safe T. Not relevant in user's primary case (`recal_period = T_yrs = 5y`).
3. **Adaptive `recal_period`** — longer after observed crashes, shorter during calm.
4. **Joint-defended sizing** — find T_init such that the strategy *as a whole* satisfies ≤1% boot, instead of per-cell ≤1%. Would lower IRR; matches rest-of-app convention more cleanly.
5. **Checkpoint capture** at recal events: currently captures pre-rebalance leverage so recal lifts are invisible on the chart. Worth capturing post-rebalance for recal kinds, or both.
6. **Documentation in `pages/Documentation.py`** — the strategy catalog there does not yet include the four recal kinds.
7. **Test on different scenarios** (low-DCA, longer horizon, different `wealth_X`) to find regimes where the four recal strategies diverge from each other.

## Acceptance criteria

- For the user's primary scenario, `recal_static` calibrates `T_init` to approximately the same value as plain `static` (e.g., 2.14x ± 0.01x).
- `recal_hybrid`, `recal_adaptive_dd` calibrate to their corresponding plain-base values.
- `meta_recal` calibrates `T_init` to the maximum over the four base candidates and starts in that strategy's mode.
- Day-0 leverage on a freshly run path matches the base strategy: e.g., `recal_static` at year 0 takes a loan of `(T_init - 1) * C`.
- p50@30y for `recal_static` is no worse than plain `static` (regression check).
- Strategy-level boot call rate for the recal strategies is in the same ballpark as before the fix (~3%, not 0% — that would indicate the recal events aren't firing).
