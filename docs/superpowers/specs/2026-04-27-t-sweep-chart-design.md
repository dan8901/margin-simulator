# T-sweep wealth chart

**Date:** 2026-04-27
**Scope:** Add a Streamlit chart that sweeps the savings-duration parameter `T` (years contributing `S`) from 0 to 15 and renders a p10/p50/p90 whisker per `T`, showing wealth at a chosen target year. Each `T` value gets its own well-defended calibration of the selected strategy.

## Problem

The simulator currently fixes `T` (years saving the higher contribution rate `S`) at the sidebar value and reports wealth percentiles only at that value. To answer "how does my year-N wealth depend on how long I save aggressively?" the user has to manually re-run with different `T` values and assemble the picture themselves.

A T-sweep chart that computes the answer in one shot — re-calibrating at each `T` for an apples-to-apples honest leverage target — directly answers this question.

## Intent

For the user's selected strategy and chosen target year, render wealth at year `target_year` as a whisker per `T ∈ [0, 15]`:

- **x-axis**: `T` (years saving `S`)
- **y-axis**: wealth at `target_year` in millions, real or nominal per global toggle
- **at each x**: vertical line p10 → p90, horizontal tick at p50

The chart is its own self-contained "what-if explorer." Changing the in-chart S override or target year does not propagate to the sidebar or the existing fan / calibration tables.

## Design

### Architecture

A new section in `app.py`, placed immediately after the existing wealth-fan chart (currently `app.py:1354–1381`).

**Inputs** (above the chart):
- **Target year** — `st.selectbox` over `res["checkpoints"]`. Default = sidebar's `target_year`.
- **Annual savings (S) for sweep** — `st.number_input`. Default = sidebar's `S`.
- **"Run T sweep" button** — `st.button`. Explicit trigger because each fresh sweep is ~80–160s.

**Strategy selection**: reuses `strategy_to_fan` (the variable from the existing fan-chart selectbox at `app.py:1357`). No new dropdown.

**State**: sweep results live in `st.session_state["t_sweep"]`. Only invalidated by changes to the sweep's input params (S, S2, C, max_days, strategy spec, F, stretch, boot_target, broker_bump, n_bootstrap, block_years, seed). Changing target year, real/nominal mode, or the strategy selectbox post-sweep does **not** trigger recompute — only re-renders.

### Sweep function

New cached function in `app.py`:

```python
@st.cache_data(show_spinner=False)
def compute_t_sweep(
    C, S, S2, max_days,
    strategy_spec_tuple,        # ("hybrid", {"F": 1.5, ...})
    stretch_F, boot_target_pct, broker_bump_days,
    wealth_X,
    n_bootstrap, block_years_int, seed,
    _paths_key, _paths,
    t_values=tuple(range(16)),
):
    """For each T in t_values, run full well-defended calibration (hist + boot
    + stretch) for the given single strategy, then project on historical paths
    and capture wealth arrays at every checkpoint year.

    Returns: dict[int T] -> {
        "T_rec": float,
        "T_hist": float, "T_boot": float, "T_stress": float,
        "init_strat_idx": int,           # for meta_recal
        "init_base_kind": str | None,
        "per_cp": dict[float year -> {"real": np.ndarray, "nominal": np.ndarray}],
    }
    """
```

**Internals** reuse existing primitives:
- `find_max_safe_T_grid(ret_c, ...)` for `T_hist` (calibration paths, full-horizon entries)
- `find_max_safe_T_grid(ret_b, ...)` for `T_boot`
- `find_max_safe_T_grid(stretch_returns(ret_h_full, stretch_F), ...)` for `T_stress` (only if stretch_F > 1)
- `simulate(ret_h_full, ..., checkpoint_days=cp_days)` for the projection
- `per_checkpoint_arrays`-equivalent capture at every checkpoint year (real & nominal)

**Single strategy, single horizon**: avoids the multi-strategy loop in the main `compute()`. Per-T cost ≈ 3 calibration legs + 1 projection ≈ 5–10s. Total: ~80–160s for a 16-point sweep.

**Calibration handling**:
- `unlev`: skip calibration, set `T_rec = 1.0`, project unleveraged
- `meta_recal`: pick `init_base_kind` by `argmax(p50 real terminal wealth at next recal event)` over `META_KINDS = ["static", "hybrid", "adaptive_hybrid"]` (matches `app.py:556` and the v4 myopic-score rule from §5g of CLAUDE.md). The winner's index in this list (0, 1, or 2) becomes `init_strat_idx`. The JIT loop uses `meta_strategy_codes` (= `[0, 4, 10]` for the three META_KINDS) to map the index to the real kind_code at runtime — same machinery as the main `compute()`.
- `recal_X`: calibrate `T_init` using the base kind (per §5g fix), build the appropriate recal lookup table for projection, pass via `recal_kw_for(...)`.

### Rendering

matplotlib chart, same style as the existing fan chart at `app.py:1372–1381`:

- Whisker plot via `ax.vlines(T_arr, p10_arr, p90_arr)` for the p10–p90 segments
- p50 ticks via `ax.hlines(p50_arr, T_arr - 0.3, T_arr + 0.3)` (small horizontal mark)
- Color: matplotlib `C0` (matches existing fan chart)
- Optional dashed horizontal at `target_wealth_X / 1e6` (sidebar param) for context
- Optional dashed vertical at sidebar's `T` value
- x-axis: `T`, integer ticks 0..15
- y-axis: wealth (M USD), label includes mode ("real" or "nominal")
- Title: `f"Wealth at year {int(target_year)} vs. years saving ${S:,} — {strategy_name}"`
- Caption: `f"Each whisker = p10–p90 across N historical paths; tick = p50. Strategy re-calibrated per T (well-defended: hist=0%, boot ≤ {boot_target_pct}%, stretch F={stretch_F})."`

### Edge cases

- `len(arr) == 0` at a (T, target_year) cell: skip that whisker — no marker rendered.
- `target_year × 252 > max_days`: filter target_year selectbox to checkpoints with `int(y * TD) <= max_days` (existing pattern).
- `S = 0` in the sweep: chart degenerates to a flat line (T is irrelevant when there are no high-rate contributions). Still renders correctly.
- `T = 0`: no high-rate phase, all years use S2. The simulator already supports this (the `S` phase has duration 0).
- Strategy = `unlev`: chart shows a single horizontal line (T-independent at unlev wealth).

### What's not included (YAGNI)

- No interpolation between integer T values
- No per-strategy comparison overlay (one strategy at a time, like the existing fan)
- No nominal/real toggle local to the chart (follows global toggle)
- No export to CSV
- No re-rendering of the existing chart sections — purely additive
- No support for free-form target years off the checkpoint grid

## Verification

This project has no automated tests. Verification is manual:

1. Run `.venv/bin/streamlit run app.py`, default scenario.
2. Click "Run / refresh" to populate `res`.
3. Scroll to new T-sweep section, click "Run T sweep" with default S override = sidebar S.
4. Confirm sweep takes ~1–3 minutes and renders a 16-whisker chart.
5. Check that p50 at the sidebar's current T value matches (within rendering tolerance) the existing fan chart's p50 at the same target year.
6. Change target year selectbox → chart should re-render instantly (no recompute).
7. Change strategy in fan-chart selectbox → T-sweep chart should re-render instantly with the new strategy's data IF the new strategy's sweep is cached, otherwise show stale data with a notice. (Caching is per strategy spec; first run for a new strategy will require re-clicking "Run T sweep".)
8. Change S override → "Run T sweep" button needs to be re-clicked.
9. Sanity check: at T=15 with the user's primary scenario (C=160k, S=180k, S2=30k), p50 wealth at year 30 should be substantially higher than at T=0 (more cumulative DCA).

## Open questions / future work

- Should "Run T sweep" auto-trigger when the user first scrolls to the section, gated on a confirm dialog? Current spec: explicit click.
- Should the chart support comparing two strategies (overlay)? Current spec: one at a time, matching the existing fan.
- Should T_rec(T) itself be plotted as a secondary axis or separate small chart? Useful diagnostic, not in current spec.
- For meta_recal, the chosen `init_base_kind` may differ across T values. Currently not surfaced in the chart; could be shown as colored x-tick labels or a sub-caption.
