"""
app.py — Streamlit UI for the calibrated portfolio projection.

Run with:
    .venv/bin/streamlit run app.py

Sliders in the sidebar configure the scenario. The "Run / refresh" button
in the main pane triggers a calibration + projection cycle (~10-15 seconds
the first time on cold paths, ~5-10 seconds afterwards once path arrays
are cached).

Path arrays (historical and bootstrap) are cached via @st.cache_resource so
they only rebuild when the horizon or bootstrap settings change. Calibration
re-runs on every press of the button.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load
from project_portfolio import (
    TD,
    build_bootstrap_paths,
    build_historical_paths,
    call_rate,
    compute_recal_table,
    compute_recal_tables_multi,
    find_max_safe_T_grid,
    percentiles_at,
    simulate,
    stretch_returns,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Portfolio Projection", layout="wide")
st.title("Leveraged portfolio projection")
st.caption(
    "Real-dollar projections with leverage strategies calibrated to your scenario "
    "(historical and bootstrap-defended)."
)

with st.expander("About this tool / disclaimers"):
    st.markdown(
        """
**What this is**

A backtest-driven simulator for leveraged S&P 500 strategies in a long-horizon
hold-forever taxable account. Given your savings pattern and a horizon, it:

1. **Calibrates** safe leverage targets per strategy across post-1932 historical
   paths AND synthetic block-bootstrap paths (so the answer doesn't depend on
   the specific historical sequence).
2. **Projects** real-dollar wealth percentiles (CPI-adjusted) at the checkpoint
   years you choose.
3. **Verifies safety** by reporting actual margin-call counts and peak-leverage
   distributions at the recommended target.

Sources: SPX-TR daily 1927-2026 (Bloomberg + yfinance), 3M Treasury (FRED
DGS3MO), CPI-U (FRED CPIAUCNS). Margin rate modeled as box-spread financing
(3M Tsy + 15bps), no tax benefit assumed — Section-1256 60/40 capital
losses are only useful if you have offsetting capital gains, which a
strict hold-forever SPX investor does not. Margin call threshold = 4.0x
leverage (Reg-T 25% maintenance).

**What this isn't**

Not financial advice. Past performance is not predictive — the historical
sample includes one ~83% drawdown (1929-32) deliberately excluded from
calibration, so your tail risk in the real future could exceed what's
modeled. Tax model is intentionally simplified. Real margin calls can
happen intraday, not just at daily close. Use this to think, not to act.

**Source code**: [github.com/dan8901/margin-simulator](https://github.com/dan8901/margin-simulator)
"""
    )


# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    # SCENARIO — most-tweaked, default open
    with st.expander("Scenario", expanded=True):
        C = st.number_input(
            "Current portfolio C ($)", value=160_000, step=10_000, min_value=0,
            help="Total liquid investable assets you have today. Real $ throughout.")
        S = st.number_input(
            "Annual savings S ($/yr, today's $)", value=180_000, step=10_000, min_value=0,
            help="Real-dollar contributions per year while you're saving. Grows with "
                 "CPI along each historical path so purchasing power stays constant.")
        T = st.slider(
            "Years saving S (T)", 0.0, 30.0, 5.0, step=1.0,
            help="After T years the savings rate switches from S to S2.")
        S2 = st.number_input(
            "Savings after T (S2 $/yr)", value=30_000, step=5_000, min_value=0,
            help="Real-dollar savings rate once you stop saving at the higher rate. "
                 "Set to 0 if you'll stop saving entirely.")

    # STRATEGIES — second-most-tweaked
    with st.expander("Strategies", expanded=True):
        show_static = st.checkbox(
            "static", value=False,
            help="Set leverage at day 0, never rebalance. Drifts down naturally as "
                 "DCA dilutes leverage. Architecturally the cleanest, lowest tail risk.")
        show_relever = st.checkbox(
            "relever (monthly)", value=False,
            help="Monthly re-lever back to the target. Captures leverage on every "
                 "contribution dollar but most exposed to bootstrap path-overfitting.")
        show_dd = st.checkbox(
            "dd_decay (drawdown decay)", value=False,
            help="Targets T_init initially; ratchets target DOWN by F × max-drawdown "
                 "observed (lifetime). Pareto-dominates wealth/time decay per project §5d.")
        dd_F = st.slider(
            "dd_decay F (decay factor)", 0.5, 3.0, 1.5, step=0.1,
            help="target = max(1.0, T_init − F × max_dd_observed). Higher F = more "
                 "aggressive deleveraging once a drawdown is observed. Default 1.5.")
        show_adaptive = st.checkbox(
            "adaptive_dd (cushion-coupled F)", value=False,
            help="Like dd_decay but F scales with current cushion: F_eff = F × "
                 "(L_now − 1) / (T_init − 1). Aggressive when leverage is near "
                 "T_init, mild when already deleveraged. Monotonic-down ratchet.")
        show_wealth = st.checkbox(
            "wealth_decay (current-equity glide)", value=False,
            help="Target linearly interpolates T_init → 1.0x as REAL equity grows "
                 "from C to wealth_X. Uses CURRENT equity (not HWM): a drawdown "
                 "raises target back, recovery lowers it again. Pure utility-glide; "
                 "less safety-architected than dd_decay.")
        show_adaptive_hybrid = st.checkbox(
            "adaptive_hybrid (adaptive_dd + wealth)", value=False,
            help="Combines adaptive_dd's cushion-coupled F + monotonic ratchet "
                 "with wealth_decay's glide path. Inherits adaptive's IRR-per-"
                 "safety efficiency and hybrid's unlever-at-wealth_X property.")
        show_hybrid = st.checkbox(
            "hybrid (dd + wealth)", value=True,
            help="target = min(dd_decay_target, wealth_decay_target). dd handles "
                 "risk events (asymmetric ratchet on drawdowns); wealth enforces "
                 "the glide path so you reach 1.0x at wealth_X. Either signal can "
                 "lower target; the more conservative wins.")
        show_r_hybrid = st.checkbox(
            "r_hybrid (ratcheted hybrid)", value=False,
            help="Same as hybrid but wealth_progress ratchets UP only — once "
                 "target has been lowered by the wealth-glide it stays lowered "
                 "even if equity drops. Strictly-monotonic-down target.")
        show_vol_hybrid = st.checkbox(
            "vol_hybrid (hybrid + vol haircut)", value=False,
            help="target = hybrid_target − vol_factor × realized_60d_annualized_vol. "
                 "Deleverages when vol spikes (often leads drawdowns).")
        show_dip_hybrid = st.checkbox(
            "dip_hybrid (hybrid + dip floor)", value=False,
            help="When current drawdown exceeds dip_threshold, target is floored "
                 "at T_init + dip_bonus (allowed ABOVE T_init temporarily). "
                 "Contrarian to dd_decay's ratchet during deep drawdowns.")
        show_rate_hybrid = st.checkbox(
            "rate_hybrid (hybrid + rate haircut)", value=False,
            help="target = hybrid_target − rate_factor × max(0, tsy_3m − rate_threshold). "
                 "Deleverages when carry trade economics deteriorate (high rates).")
        wealth_X_M = st.slider(
            "wealth_X target ($M, real)", 1.0, 20.0, 3.0, step=0.5,
            help="Real-dollar wealth at which wealth_decay / hybrid hit 1.0x. "
                 "Linear interpolation from T_init at C to 1.0x at wealth_X. "
                 "Below C: target = T_init. Above wealth_X: target = 1.0x.")
        vol_factor = st.slider(
            "vol_hybrid: vol_factor", 0.0, 3.0, 1.0, step=0.1,
            help="Multiplier on annualized 60d realized vol. 1.0 = a 30% vol "
                 "regime drops target by 0.30x.")
        dip_threshold_pct = st.slider(
            "dip_hybrid: trigger drawdown (%)", 10, 50, 30, step=5,
            help="Current drawdown above this triggers the dip-buy floor.")
        dip_bonus = st.slider(
            "dip_hybrid: bonus leverage", 0.0, 0.5, 0.2, step=0.05,
            help="Target floored at T_init + dip_bonus while in dip.")
        rate_threshold_pct = st.slider(
            "rate_hybrid: rate threshold (% nominal)", 2.0, 10.0, 5.0, step=0.5,
            help="3M Tsy yield above this triggers haircut.")
        rate_factor = st.slider(
            "rate_hybrid: rate_factor", 0.0, 20.0, 5.0, step=1.0,
            help="With factor=5 and yield 3% above threshold, target drops 0.15x.")
        show_recal = st.checkbox(
            "recal_static (periodic re-calibration)", value=False,
            help="Behaves like static between re-cal events. Every N months, "
                 "looks up new T_max from a pre-computed table (based on current "
                 "equity + remaining horizon) and takes additional loan to "
                 "reach it. Models the 'every N months, re-evaluate' workflow. "
                 "First run takes ~5s to precompute the table.")
        show_recal_hybrid = st.checkbox(
            "recal_hybrid (re-cal + dd ratchet + wealth glide)", value=True,
            help="Like recal_static but uses hybrid-table values + applies "
                 "hybrid logic (dd ratchet + wealth glide) between recals. "
                 "Re-cal events reset state to fresh.")
        show_recal_adaptive_dd = st.checkbox(
            "recal_adaptive_dd (re-cal + adaptive_dd between)", value=False,
            help="Like recal_static but uses adaptive_dd-table values + applies "
                 "adaptive_dd logic between recals (cushion-coupled F + "
                 "monotonic ratchet). Re-cal events reset state to fresh.")
        show_meta_recal = st.checkbox(
            "meta_recal (pick max-T strategy at each recal)", value=False,
            help="At each re-cal, looks up T_max for static / dd_decay / "
                 "adaptive_dd / hybrid and picks the strategy with the "
                 "highest. Between recals, applies that strategy's logic.")
        recal_period_months = st.slider(
            "recal period (months)", 1, 180, 60, step=1,
            help="How often re-cal strategies re-calibrate. 60 (5y) is the "
                 "knee. Monthly (1) degrades to relever-like fragility "
                 "(~30% calls). Longer is safer.")

    # GOALS — for probability/target-wealth questions
    with st.expander("Goals & targets"):
        target_wealth_M = st.number_input(
            "Target wealth ($M)", value=3.0, step=0.5, min_value=0.1,
            help="Used by the 'Probability of reaching ≥ $X by year Y' panel and "
                 "marked on the histogram with a red dashed line.")
        target_year = st.slider(
            "Target year", 5, 30, 15,
            help="Year at which to compute P(wealth ≥ target).")
        cap_enabled = st.toggle(
            "Stop levering up above a wealth cap", value=False,
            help="When ON, every strategy permanently stops adding leverage the moment "
                 "real wealth crosses the threshold below. Once latched, doesn't toggle "
                 "off if wealth dips below.")
        cap_wealth_M = st.number_input(
            "Stop-levering threshold (real $M)", value=3.0, step=0.5, min_value=0.1,
            help="Above this real-wealth threshold, no new leverage is taken. Below, "
                 "the strategy operates normally. Default matches Target wealth.")

    # SAFETY — bootstrap + stretch tuning, less often touched
    with st.expander("Safety calibration"):
        n_bootstrap = st.slider(
            "# synthetic paths", 500, 5000, 500, step=500,
            help="Block-bootstrap synthetic paths used for the 'is this leverage really "
                 "safe?' check. More = tighter call-rate measurement but slower.")
        boot_target_pct = st.slider(
            "Target call rate (%)", 0.5, 5.0, 1.0, step=0.5,
            help="Acceptable synthetic margin-call rate when picking 'bootstrap-safe' "
                 "leverage targets. Lower = more conservative.")
        block_years = st.slider(
            "Block size (years)", 1.0, 5.0, 1.0, step=0.5,
            help="Length of each random block sampled from history. 1y is default; "
                 "2y is more stressful (preserves crisis dynamics).")
        seed = st.number_input(
            "Bootstrap seed", value=42, step=1,
            help="RNG seed for the bootstrap. Same seed = same exact synthetic paths.")
        stretch_F = st.slider(
            "Drawdown stretch factor F", 1.0, 1.5, 1.1, step=0.05,
            help="F=1.0 = no stretch. F=1.2 = every historical drawdown 20% deeper. "
                 "Adds a third safety bar (T_stress) when > 1.")

    # HORIZON — rarely changed
    with st.expander("Horizon & checkpoints"):
        max_years = st.slider(
            "Max horizon (years)", 5, 50, 30,
            help="Longest projection horizon. Larger = slower, tighter on memory.")
        checkpoints_str = st.text_input(
            "Checkpoints (years, comma-separated)", "5,10,15,20,25,30",
            help="Years at which to report percentiles, leverage, and probabilities.")

    # DISPLAY
    with st.expander("Display"):
        real_dollars = st.toggle(
            "Show in real $ (today's purchasing power)", value=True,
            help="Off = nominal $ at each year along each path. Calibration & safety "
                 "panels stay in real terms regardless.")


# ---------------------------------------------------------------------------
# Cached path builders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_paths(max_days, n_bootstrap, block_years_int, seed):
    dates, px, tsy, mrate, cpi = load(with_cpi=True)
    ret_c, tsy_c, cpi_c, avail_c, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=max_days)
    ret_h, tsy_h, cpi_h, avail_h, entry_dates_h = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    rng = np.random.default_rng(int(seed))
    block_days = int(block_years_int / 100.0 * TD)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, n_bootstrap, block_days, rng)
    return dict(
        calib=(ret_c, tsy_c, cpi_c, avail_c),
        proj=(ret_h, tsy_h, cpi_h, avail_h, entry_dates_h),
        boot=(ret_b, tsy_b, cpi_b),
    )


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------

checkpoints = sorted(float(x.strip()) for x in checkpoints_str.split(",") if x.strip())
max_days = int(max_years * TD)
boot_target = boot_target_pct / 100.0
block_years_int = int(block_years * 100)   # for the cache key

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("Run / refresh", type="primary", use_container_width=True)
with col2:
    st.caption("Click to recalibrate after changing scenario or bootstrap settings.")


def selected_strategies():
    out = []
    if show_static:
        out.append(("static", dict(kind="static")))
    if show_relever:
        out.append(("relever", dict(kind="relever")))
    if show_dd:
        out.append(("dd_decay", dict(kind="dd_decay", F=dd_F, floor=1.0)))
    if show_adaptive:
        out.append(("adaptive_dd", dict(kind="adaptive_dd", F=dd_F, floor=1.0)))
    if show_wealth:
        out.append(("wealth_decay", dict(kind="wealth_decay", floor=1.0)))
    if show_hybrid:
        out.append(("hybrid", dict(kind="hybrid", F=dd_F, floor=1.0)))
    if show_adaptive_hybrid:
        out.append(("adaptive_hybrid", dict(kind="adaptive_hybrid", F=dd_F, floor=1.0)))
    if show_r_hybrid:
        out.append(("r_hybrid", dict(kind="r_hybrid", F=dd_F, floor=1.0)))
    if show_vol_hybrid:
        out.append(("vol_hybrid", dict(kind="vol_hybrid", F=dd_F, floor=1.0)))
    if show_dip_hybrid:
        out.append(("dip_hybrid", dict(kind="dip_hybrid", F=dd_F, floor=1.0)))
    if show_rate_hybrid:
        out.append(("rate_hybrid", dict(kind="rate_hybrid", F=dd_F, floor=1.0)))
    if show_recal:
        out.append(("recal_static", dict(kind="recal_static", floor=1.0)))
    if show_recal_hybrid:
        out.append(("recal_hybrid", dict(kind="recal_hybrid", F=dd_F, floor=1.0)))
    if show_recal_adaptive_dd:
        out.append(("recal_adaptive_dd", dict(kind="recal_adaptive_dd", F=dd_F, floor=1.0)))
    if show_meta_recal:
        out.append(("meta_recal", dict(kind="meta_recal", F=dd_F, floor=1.0)))
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def compute(C, S, T, S2, max_days, checkpoints_tuple, strategies_tuple,
            boot_target, stretch_F, dd_F, cap_real, wealth_X,
            vol_factor, dip_threshold, dip_bonus, rate_threshold, rate_factor,
            recal_period_months,
            _paths_key, _paths):
    """Cached calibration + projection.
    Cache keys on the leading scalars + tuples; `_paths` is opaque (underscore
    tells Streamlit not to hash). `_paths_key` is the cache identity for the
    paths dict (so cache invalidates if paths change but the underscore arg
    can stay opaque)."""
    # Reconstruct strategies list from hashable tuple
    strategies = []
    for name in strategies_tuple:
        if name == "static":
            strategies.append((name, dict(kind="static")))
        elif name == "relever":
            strategies.append((name, dict(kind="relever")))
        elif name == "dd_decay":
            strategies.append((name, dict(kind="dd_decay", F=dd_F, floor=1.0)))
        elif name == "adaptive_dd":
            strategies.append((name, dict(kind="adaptive_dd", F=dd_F, floor=1.0)))
        elif name == "wealth_decay":
            strategies.append((name, dict(kind="wealth_decay", floor=1.0)))
        elif name == "hybrid":
            strategies.append((name, dict(kind="hybrid", F=dd_F, floor=1.0)))
        elif name == "adaptive_hybrid":
            strategies.append((name, dict(kind="adaptive_hybrid", F=dd_F, floor=1.0)))
        elif name == "r_hybrid":
            strategies.append((name, dict(kind="r_hybrid", F=dd_F, floor=1.0)))
        elif name == "vol_hybrid":
            strategies.append((name, dict(kind="vol_hybrid", F=dd_F, floor=1.0)))
        elif name == "dip_hybrid":
            strategies.append((name, dict(kind="dip_hybrid", F=dd_F, floor=1.0)))
        elif name == "rate_hybrid":
            strategies.append((name, dict(kind="rate_hybrid", F=dd_F, floor=1.0)))
        elif name == "recal_static":
            strategies.append((name, dict(kind="recal_static", floor=1.0)))
        elif name == "recal_hybrid":
            strategies.append((name, dict(kind="recal_hybrid", F=dd_F, floor=1.0)))
        elif name == "recal_adaptive_dd":
            strategies.append((name, dict(kind="recal_adaptive_dd", F=dd_F, floor=1.0)))
        elif name == "meta_recal":
            strategies.append((name, dict(kind="meta_recal", F=dd_F, floor=1.0)))
    checkpoints = list(checkpoints_tuple)

    # Common kwargs for every simulate/calibration call (overlay params)
    overlay_kw = dict(
        vol_factor=vol_factor,
        dip_threshold=dip_threshold,
        dip_bonus=dip_bonus,
        rate_threshold=rate_threshold,
        rate_factor=rate_factor,
    )

    # Identify which strategies need which lookup tables.
    # meta_recal candidates are wealth-aware bases only: {static, hybrid,
    # adaptive_hybrid}. Static deleverages passively via DCA dilution;
    # the other two enforce wealth_X via their internal glide.
    recal_strategies_needed = set()
    for name in strategies_tuple:
        if name == "recal_static":
            recal_strategies_needed.add("static")
        elif name == "recal_hybrid":
            recal_strategies_needed.add("hybrid")
        elif name == "recal_adaptive_dd":
            recal_strategies_needed.add("adaptive_dd")
        elif name == "meta_recal":
            for s in ("static", "hybrid", "adaptive_hybrid"):
                recal_strategies_needed.add(s)
    needs_meta = "meta_recal" in strategies_tuple

    if recal_strategies_needed:
        e_recal_grid = np.array([
            max(C * 0.5, 100_000.0),
            C,
            C * 3.0,
            C * 10.0,
            C * 30.0,
            C * 100.0,
        ], dtype=np.float64)
        h_recal_grid_years = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        h_recal_grid_years = h_recal_grid_years[h_recal_grid_years <= max_days / TD]
        if len(h_recal_grid_years) == 0:
            h_recal_grid_years = np.array([max_days / TD])
        h_recal_grid_days = (h_recal_grid_years * TD).astype(np.int64)
        ret_c, tsy_c, cpi_c, avail_c = _paths["calib"]
        ret_b, tsy_b, cpi_b = _paths["boot"]
        # Stretched paths if stretch_F > 1
        ret_s_for_recal = stretch_returns(ret_c, stretch_F) if stretch_F > 1.0 else None
        # Each recal picks the highest target that's safe BY THE SAME STANDARD as
        # the rest of the app: hist 0% + bootstrap ≤ boot_target + stretch ≤ 0%.
        # Strategy-level call rate is a consequence of multi-event exposure (~3%
        # at boot_target=1% over a 30y/5y schedule).
        # per_strat_tables[kind] = (t_table, score_table). Score is p50
        # real wealth at the cell's well-defended T_max evaluated over
        # `recal_period_days` (myopic mode) — captures expected wealth at
        # the next recal event, matching the user's "decide for the next
        # chunk" mental model. For the last segment (cell horizon <
        # recal_period), the score correctly degrades to terminal wealth.
        score_horizon_days_local = int(recal_period_months * 21)
        per_strat_tables = {}
        for kind in recal_strategies_needed:
            per_strat_tables[kind] = compute_recal_table(
                ret_c, tsy_c, cpi_c, avail_c, S2, e_recal_grid,
                h_recal_grid_years, kind=kind, F=dd_F, wealth_X=wealth_X,
                hist_target=0.0, coarse_n=8, fine_n=8,
                ret_b=ret_b, tsy_b=tsy_b, cpi_b=cpi_b,
                boot_target=boot_target,
                ret_s=ret_s_for_recal, stretch_F=stretch_F,
                score_horizon_days=score_horizon_days_local)
        # Build 3D meta tables if needed (fixed order static / hybrid / adaptive_hybrid).
        # Wealth-aware candidates only — see recal_strategies_needed comment above.
        if needs_meta:
            meta_kinds = ["static", "hybrid", "adaptive_hybrid"]
            t_recal_tables_meta = np.stack([per_strat_tables[k][0] for k in meta_kinds])
            meta_score_tables = np.stack([per_strat_tables[k][1] for k in meta_kinds])
            meta_strategy_codes = np.array(
                [{"static": 0, "hybrid": 4, "adaptive_hybrid": 10}[k]
                 for k in meta_kinds], dtype=np.int64)
        else:
            t_recal_tables_meta = np.zeros((1, 1, 1), dtype=np.float64)
            meta_score_tables = np.zeros((1, 1, 1), dtype=np.float64)
            meta_strategy_codes = np.zeros(1, dtype=np.int64)
    else:
        e_recal_grid = np.zeros(1, dtype=np.float64)
        h_recal_grid_days = np.zeros(1, dtype=np.int64)
        per_strat_tables = {}
        t_recal_tables_meta = np.zeros((1, 1, 1), dtype=np.float64)
        meta_score_tables = np.zeros((1, 1, 1), dtype=np.float64)
        meta_strategy_codes = np.zeros(1, dtype=np.int64)
    recal_period_days = int(recal_period_months * 21)   # ~21 trading days per month

    def recal_kw_for(name):
        """Return per-strategy recal kwargs (different table per kind)."""
        kw = dict(
            recal_period_days=recal_period_days,
            e_recal_grid=e_recal_grid,
            h_recal_grid_days=h_recal_grid_days,
            t_recal_tables_meta=t_recal_tables_meta,
            meta_score_tables=meta_score_tables,
            meta_strategy_codes=meta_strategy_codes,
        )
        if name == "recal_static":
            kw["t_recal_table"] = per_strat_tables.get("static",
                (np.zeros((1, 1), dtype=np.float64),))[0]
        elif name == "recal_hybrid":
            kw["t_recal_table"] = per_strat_tables.get("hybrid",
                (np.zeros((1, 1), dtype=np.float64),))[0]
        elif name == "recal_adaptive_dd":
            kw["t_recal_table"] = per_strat_tables.get("adaptive_dd",
                (np.zeros((1, 1), dtype=np.float64),))[0]
        else:
            kw["t_recal_table"] = np.zeros((1, 1), dtype=np.float64)
        return kw

    paths = _paths
    ret_c, tsy_c, cpi_c, avail_c = paths["calib"]
    ret_h, tsy_h, cpi_h, avail_h, entry_dates_h = paths["proj"]
    ret_b, tsy_b, cpi_b = paths["boot"]

    # Stretched calibration paths (only built if F > 1)
    if stretch_F > 1.0:
        ret_s = stretch_returns(ret_c, stretch_F)
    else:
        ret_s = None

    # Map recal_X strategies to their base kind for T_init calibration.
    # The base kind's well-defended T_rec is used as T_init for the recal
    # simulation: at year 0, the recal strategy behaves like the plain
    # base kind. At each recal event, lookup-table values take over.
    BASE_KIND_FOR_RECAL = {
        "recal_static": "static",
        "recal_hybrid": "hybrid",
        "recal_adaptive_dd": "adaptive_dd",
    }

    calibrated = {}
    for name, spec in strategies:
        kind = spec["kind"]
        F = spec.get("F", 1.5)

        # For recal_static / recal_hybrid / recal_adaptive_dd: calibrate
        # using the base kind, NOT the recal trajectory. The recal lookup
        # table is well-defended per cell already; T_init equals what the
        # plain base kind would give at year 0. The years-0-to-first-recal
        # phase therefore behaves like the plain base.
        # meta_recal is handled in a dedicated block below.
        if kind in BASE_KIND_FOR_RECAL:
            calib_kind = BASE_KIND_FOR_RECAL[kind]
            calib_recal_kw = {}   # base kinds don't use the lookup table
        elif kind == "meta_recal":
            continue   # handled in the meta_recal block below
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

    # meta_recal: calibrate all wealth-aware base candidates and pick the
    # candidate with the highest expected p50 real terminal wealth (NOT
    # highest T). The winner's index in META_KINDS becomes init_strat_idx
    # so the years-before-first-recal phase applies the chosen strategy's
    # logic. dd_decay/adaptive_dd are NOT in this set because they don't
    # honor wealth_X; static deleverages passively, hybrid/adaptive_hybrid
    # via internal wealth glide.
    META_KINDS = ["static", "hybrid", "adaptive_hybrid"]   # MUST match the meta_kinds ordering above
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
            T_rec_base = min(T_h, T_bo, T_st)

            # Score = p50 real wealth on calibration paths at T_rec_base
            # over the myopic horizon (next recal event), mirroring the
            # per-cell score table at recal events. Captures "best for
            # the next chunk", not best for full 30y.
            score_horizon = min(int(recal_period_months * 21), max_days)
            real_eq_b, called_b_score, _, _ = simulate(
                ret_c, tsy_c, cpi_c, base_kind, T_rec_base,
                C, S, T, S2, score_horizon,
                avail=np.minimum(avail_c, score_horizon), F=F,
                cap_real=cap_real, wealth_X=wealth_X, **overlay_kw)
            terminal = real_eq_b[:, score_horizon]
            valid = ~(np.isnan(terminal) | called_b_score)
            score = float(np.nanpercentile(terminal[valid], 50)) if valid.any() else float("-inf")

            per_base[base_kind] = dict(T_hist=T_h, T_boot=T_bo, T_stress=T_st,
                                        T_rec=T_rec_base, score=score)

        scores = [per_base[k]["score"] for k in META_KINDS]
        winner_idx = int(np.argmax(scores))
        winner_kind = META_KINDS[winner_idx]
        winner = per_base[winner_kind]
        T_rec = winner["T_rec"]

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
            per_base=per_base,
        )

    # cpi factor per path/day (used for nominal conversion)
    cpi_factor = cpi_h / cpi_h[:, 0:1]

    def per_checkpoint_arrays(real_eq):
        """For each checkpoint, returns dict y -> dict(real, nominal, entry_idx)
        where arrays are for paths that have valid (non-NaN) data at that day."""
        out = {}
        for y in checkpoints:
            d_idx = int(y * TD)
            if d_idx > max_days:
                continue
            col = real_eq[:, d_idx]
            valid = ~np.isnan(col)
            if not valid.any():
                continue
            real_vals = col[valid]
            nom_vals = real_vals * cpi_factor[valid, d_idx]
            idx = np.where(valid)[0]
            out[y] = dict(real=real_vals, nominal=nom_vals, idx=idx)
        return out

    # Day indices for the user's checkpoints (used to capture leverage)
    cp_days = np.array(
        [int(y * TD) for y in checkpoints if int(y * TD) <= max_days],
        dtype=np.int64)

    # Unlev baseline (no recal — pure unleveraged)
    real_eq_u, called_u, _, lev_cp_u = simulate(ret_h, tsy_h, cpi_h, "static", 1.0,
                                       C, S, T, S2, max_days, avail=avail_h,
                                       checkpoint_days=cp_days, cap_real=cap_real,
                                       wealth_X=wealth_X, **overlay_kw)
    ps_u = percentiles_at(real_eq_u, checkpoints, max_days)
    proj_results = {"unlev": (1.0, ps_u)}
    safety = {}   # name -> dict of safety metrics at T_rec
    per_cp = {"unlev": per_checkpoint_arrays(real_eq_u)}
    worst = {}   # name -> dict(entry_date, real_traj, nominal_traj, called)
    # Leverage percentiles at each checkpoint, keyed by strategy
    lev_summary = {"unlev": {y: dict(p25=1.0, p50=1.0, p75=1.0, p90=1.0)
                              for y in checkpoints}}
    def collect_worst_per_year(real_eq_arr, called_arr):
        """For each checkpoint year, find the historical path with the lowest
        real wealth AT THAT YEAR among paths still alive there."""
        out = {}
        for y in checkpoints:
            d_idx = int(y * TD)
            if d_idx > max_days:
                continue
            col = real_eq_arr[:, d_idx]
            valid = ~np.isnan(col)
            if not valid.any():
                continue
            mins = np.where(valid, col, np.inf)
            wk = int(np.argmin(mins))
            out[float(y)] = dict(
                entry_date=str(entry_dates_h[wk]),
                real_traj=real_eq_arr[wk, :].copy(),
                nominal_traj=(real_eq_arr[wk, :] * cpi_factor[wk, :]).copy(),
                called=bool(called_arr[wk]),
                wealth_at_year=float(real_eq_arr[wk, d_idx]),
            )
        return out

    worst["unlev"] = collect_worst_per_year(real_eq_u, called_u)

    # No-cap comparison projections (run only if cap is active for the main run)
    proj_no_cap = {}   # name -> per_cp dict (real & nominal arrays per checkpoint)

    for name, spec in strategies:
        c = calibrated[name]
        T_target = c["T_rec"]
        kind = spec["kind"]
        F = spec.get("F", 1.5)
        init_strat_idx = c.get("init_strat_idx", 0)
        # Projection on historical (variable horizon) — uses cap_real
        real_eq, called_h, peak_lev_h, lev_at_cp = simulate(
            ret_h, tsy_h, cpi_h, kind, T_target,
            C, S, T, S2, max_days, avail=avail_h, F=F,
            checkpoint_days=cp_days, cap_real=cap_real, wealth_X=wealth_X,
            **overlay_kw, **recal_kw_for(name),
            init_strat_idx=init_strat_idx)
        ps = percentiles_at(real_eq, checkpoints, max_days)
        proj_results[name] = (T_target, ps)
        per_cp[name] = per_checkpoint_arrays(real_eq)

        # Same projection without the leverage cap (for comparison)
        if cap_real != float("inf"):
            real_eq_nc, _, _, _ = simulate(
                ret_h, tsy_h, cpi_h, kind, T_target,
                C, S, T, S2, max_days, avail=avail_h, F=F,
                cap_real=float("inf"), wealth_X=wealth_X, **overlay_kw,
                **recal_kw_for(name),
                init_strat_idx=init_strat_idx)
            proj_no_cap[name] = per_checkpoint_arrays(real_eq_nc)

        # Leverage percentiles at each checkpoint (only over surviving paths)
        lev_summary[name] = {}
        for ci, y in enumerate(checkpoints):
            if ci >= cp_days.shape[0]:
                continue
            col = lev_at_cp[:, ci]
            valid = ~np.isnan(col)
            if not valid.any():
                continue
            arr = col[valid]
            lev_summary[name][y] = dict(
                p25=float(np.percentile(arr, 25)),
                p50=float(np.percentile(arr, 50)),
                p75=float(np.percentile(arr, 75)),
                p90=float(np.percentile(arr, 90)),
            )

        # Worst path at each checkpoint year (lowest real wealth that year)
        worst[name] = collect_worst_per_year(real_eq, called_h)

        # Safety on bootstrap (full horizon synthetic paths)
        _, called_b, peak_lev_b, _ = simulate(
            ret_b, tsy_b, cpi_b, kind, T_target,
            C, S, T, S2, max_days, F=F, cap_real=cap_real, wealth_X=wealth_X,
            **overlay_kw, **recal_kw_for(name),
            init_strat_idx=init_strat_idx)

        # Peak leverage percentiles on SURVIVORS only (called paths hit >= 4.0x by definition)
        survivors_h = peak_lev_h[~called_h]
        survivors_b = peak_lev_b[~called_b]
        # Calibration paths (full horizon) for additional context
        _, called_cf, peak_lev_cf, _ = simulate(
            ret_c, tsy_c, cpi_c, kind, T_target,
            C, S, T, S2, max_days, avail=avail_c, F=F, cap_real=cap_real,
            wealth_X=wealth_X, **overlay_kw, **recal_kw_for(name),
            init_strat_idx=init_strat_idx)

        # Stress test (stretched drawdowns), if enabled
        if ret_s is not None:
            _, called_s, peak_lev_s, _ = simulate(
                ret_s, tsy_c, cpi_c, kind, T_target,
                C, S, T, S2, max_days, avail=avail_c, F=F, cap_real=cap_real,
                wealth_X=wealth_X, **overlay_kw, **recal_kw_for(name),
                init_strat_idx=init_strat_idx)
            survivors_s = peak_lev_s[~called_s]
            stress_calls = int(called_s.sum())
            n_stress = len(called_s)
            peak_stress_max = float(survivors_s.max()) if len(survivors_s) else float("nan")
            peak_stress_p99 = float(np.percentile(survivors_s, 99)) if len(survivors_s) else float("nan")
        else:
            stress_calls, n_stress = None, None
            peak_stress_max = peak_stress_p99 = None

        safety[name] = dict(
            T_target=T_target,
            n_hist=len(called_h), hist_calls=int(called_h.sum()),
            n_calib=len(called_cf), calib_calls=int(called_cf.sum()),
            n_boot=len(called_b), boot_calls=int(called_b.sum()),
            n_stress=n_stress, stress_calls=stress_calls,
            peak_lev_hist_p50=float(np.percentile(survivors_h, 50)) if len(survivors_h) else float("nan"),
            peak_lev_hist_p90=float(np.percentile(survivors_h, 90)) if len(survivors_h) else float("nan"),
            peak_lev_hist_p99=float(np.percentile(survivors_h, 99)) if len(survivors_h) else float("nan"),
            peak_lev_hist_max=float(survivors_h.max()) if len(survivors_h) else float("nan"),
            peak_lev_boot_p50=float(np.percentile(survivors_b, 50)) if len(survivors_b) else float("nan"),
            peak_lev_boot_p90=float(np.percentile(survivors_b, 90)) if len(survivors_b) else float("nan"),
            peak_lev_boot_p99=float(np.percentile(survivors_b, 99)) if len(survivors_b) else float("nan"),
            peak_lev_boot_max=float(survivors_b.max()) if len(survivors_b) else float("nan"),
            peak_stress_p99=peak_stress_p99,
            peak_stress_max=peak_stress_max,
        )

    return calibrated, proj_results, safety, per_cp, worst, lev_summary, proj_no_cap


# ---------------------------------------------------------------------------
# Trigger compute on Run
# ---------------------------------------------------------------------------

if run or "results" not in st.session_state:
    if not selected_strategies():
        st.warning("Select at least one strategy from the sidebar.")
    else:
        with st.spinner("Loading paths..."):
            paths = get_paths(max_days, n_bootstrap, block_years_int, seed)
        with st.spinner("Calibrating + projecting..."):
            t0 = time.time()
            strategies_tuple = tuple(name for name, _ in selected_strategies())
            paths_key = (max_days, n_bootstrap, block_years_int, int(seed))
            cap_real_val = float(cap_wealth_M) * 1e6 if cap_enabled else float("inf")
            wealth_X_val = float(wealth_X_M) * 1e6
            dip_threshold_val = float(dip_threshold_pct) / 100.0
            rate_threshold_val = float(rate_threshold_pct) / 100.0
            calibrated, proj_results, safety, per_cp, worst, lev_summary, proj_no_cap = compute(
                C, S, T, S2, max_days, tuple(checkpoints),
                strategies_tuple, boot_target, stretch_F, dd_F, cap_real_val,
                wealth_X_val,
                float(vol_factor), dip_threshold_val, float(dip_bonus),
                rate_threshold_val, float(rate_factor),
                int(recal_period_months),
                paths_key, paths)
            elapsed = time.time() - t0
        st.session_state["results"] = dict(
            calibrated=calibrated,
            proj_results=proj_results,
            safety=safety,
            per_cp=per_cp,
            worst=worst,
            lev_summary=lev_summary,
            proj_no_cap=proj_no_cap,
            cap_real=cap_real_val,
            cap_enabled=cap_enabled,
            cap_wealth_M=cap_wealth_M,
            checkpoints=checkpoints,
            params=dict(C=C, S=S, T=T, S2=S2,
                        max_years=max_years, n_bootstrap=n_bootstrap,
                        boot_target_pct=boot_target_pct, block_years=block_years,
                        stretch_F=stretch_F, dd_F=dd_F, wealth_X_M=wealth_X_M),
            elapsed=elapsed,
        )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x / 1e3:.0f}k"


if "results" in st.session_state:
    res = st.session_state["results"]
    p = res["params"]

    mode = "real" if real_dollars else "nominal"
    mode_label = "real $ (today's purchasing power)" if real_dollars else "nominal $"

    def get_pct(strategy_name, year):
        """Return (p10, p25, p50, p75, p90, mean, n_paths) at year for mode."""
        cp = res["per_cp"].get(strategy_name, {}).get(year)
        if cp is None:
            return None
        arr = cp[mode]
        if len(arr) == 0:
            return None
        ps = np.percentile(arr, [10, 25, 50, 75, 90])
        return (float(ps[0]), float(ps[1]), float(ps[2]),
                float(ps[3]), float(ps[4]), float(arr.mean()), int(len(arr)))

    def get_T(strategy_name):
        if strategy_name == "unlev":
            return 1.0
        c = res["calibrated"].get(strategy_name)
        return c["T_rec"] if c else 1.0

    st.success(
        f"Computed in {res['elapsed']:.1f}s — "
        f"C=\\${p['C']:,}, S=\\${p['S']:,}/yr (×{p['T']:g}y), "
        f"S2=\\${p['S2']:,}/yr, "
        f"horizon={p['max_years']:g}y. "
        f"Showing: **{mode_label}**"
    )

    # --- Calibration table ---
    st.subheader("Calibration")
    st.caption(
        "T_hist_safe = largest target with **0 historical calls** at full horizon. "
        "T_boot_safe = largest target with **≤ target%** synthetic calls. "
        "T_recommended = min of both (satisfies both safety bars)."
    )
    stretch_F_used = res["params"].get("stretch_F", 1.0)
    calib_data = []
    any_meta = any(c["spec"]["kind"] == "meta_recal"
                   for c in res["calibrated"].values())
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
            # Only meta_recal's choice is informative; other rows are blank.
            if c["spec"]["kind"] == "meta_recal":
                row["Init base"] = c.get("init_base_kind") or ""
            else:
                row["Init base"] = ""
        calib_data.append(row)
    st.dataframe(pd.DataFrame(calib_data), hide_index=True, use_container_width=True)
    if stretch_F_used > 1.0:
        st.caption(f"T_stress = largest target with **0% calls when every historical drawdown is amplified by F={stretch_F_used:.2f}**.")

    # --- Safety verification at the recommended target ---
    st.subheader("Safety @ recommended target")
    st.caption(
        "Margin-call counts and peak-leverage percentiles measured AT the T_recommended "
        "value for each strategy. Hist (variable horizon) uses every post-1932 entry to its data limit; "
        "Hist (full horizon) uses only entries with the full max-horizon of forward data; "
        "Boot uses synthetic 1y-block-bootstrap paths. Peak leverage on SURVIVORS only "
        "(called paths reach the call threshold by definition)."
    )
    safety_rows = []
    safe_target_pct = res["params"]["boot_target_pct"]
    for name, s in res["safety"].items():
        boot_pct = 100 * s["boot_calls"] / max(s["n_boot"], 1)
        row = {
            "Strategy": name,
            "T_target": f"{s['T_target']:.3f}x",
            "Hist calls (variable)": f"{s['hist_calls']} / {s['n_hist']} ({100 * s['hist_calls'] / max(s['n_hist'], 1):.2f}%)",
            "Hist calls (full)": f"{s['calib_calls']} / {s['n_calib']} ({100 * s['calib_calls'] / max(s['n_calib'], 1):.2f}%)",
            "Boot calls": f"{s['boot_calls']} / {s['n_boot']} ({boot_pct:.2f}%)",
            "Boot ≤ target?": "✅" if boot_pct <= safe_target_pct + 0.05 else "⚠️",
        }
        if s.get("n_stress") is not None:
            stress_pct = 100 * s["stress_calls"] / max(s["n_stress"], 1)
            row[f"Stress calls (F={stretch_F_used:.2f})"] = (
                f"{s['stress_calls']} / {s['n_stress']} ({stress_pct:.2f}%)"
            )
        safety_rows.append(row)
    st.dataframe(pd.DataFrame(safety_rows), hide_index=True, use_container_width=True)

    # Peak leverage table
    st.markdown("**Peak leverage observed (survivors), by strategy**")
    peak_rows = []
    for name, s in res["safety"].items():
        peak_rows.append({
            "Strategy": name,
            "T_target": f"{s['T_target']:.3f}x",
            "hist p50": f"{s['peak_lev_hist_p50']:.3f}x",
            "hist p90": f"{s['peak_lev_hist_p90']:.3f}x",
            "hist p99": f"{s['peak_lev_hist_p99']:.3f}x",
            "hist max": f"{s['peak_lev_hist_max']:.3f}x",
            "boot p50": f"{s['peak_lev_boot_p50']:.3f}x",
            "boot p90": f"{s['peak_lev_boot_p90']:.3f}x",
            "boot p99": f"{s['peak_lev_boot_p99']:.3f}x",
            "boot max": f"{s['peak_lev_boot_max']:.3f}x",
        })
    st.dataframe(pd.DataFrame(peak_rows), hide_index=True, use_container_width=True)
    st.caption(
        "Call threshold = 4.000x. Peak-leverage values well below that on survivors confirm "
        "the recommended target leaves headroom against drawdowns."
    )

    strategy_names = list(res["proj_results"].keys())

    # --- Cross-strategy median comparison ---
    st.subheader(f"Projected p50 by strategy & checkpoint ({mode_label})")
    rows = []
    for name in strategy_names:
        row = {"Strategy": name, "T": f"{get_T(name):.3f}x"}
        for y in res["checkpoints"]:
            ps = get_pct(name, y)
            row[f"{int(y)}y"] = fmt_money(ps[2]) if ps else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- Line chart of p50 across strategies (static matplotlib, no zoom) ---
    st.subheader(f"p50 trajectory ({mode_label}, in $M)")
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in strategy_names:
        ys, vals = [], []
        for y in res["checkpoints"]:
            ps = get_pct(name, y)
            if ps:
                ys.append(y)
                vals.append(ps[2] / 1e6)
        if ys:
            ax.plot(ys, vals, marker="o", label=name)
    ax.set_xlabel("Years")
    ax.set_ylabel(f"Wealth (M USD, {mode_label.split(' ')[0]})")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    # --- Cap on / cap off comparison ---
    if res.get("cap_enabled") and res.get("proj_no_cap"):
        cap_M = res.get("cap_wealth_M", 3.0)
        st.subheader(
            f"Cap-on vs cap-off comparison (stop levering at \\${cap_M:.1f}M, "
            f"{mode_label})"
        )
        compare_strat = st.selectbox(
            "Strategy to compare",
            options=[s for s in strategy_names if s != "unlev"]
                     if any(s != "unlev" for s in strategy_names) else strategy_names,
            index=0,
            key="cap_compare_strat",
        )
        if compare_strat:
            on_arrs = res["per_cp"].get(compare_strat, {})
            off_arrs = res["proj_no_cap"].get(compare_strat, {})
            ys, on_p50, off_p50, on_p10, off_p10, on_p90, off_p90 = [], [], [], [], [], [], []
            for y in res["checkpoints"]:
                a_on = on_arrs.get(y, {}).get(mode) if isinstance(on_arrs.get(y), dict) else None
                a_off = off_arrs.get(y, {}).get(mode) if isinstance(off_arrs.get(y), dict) else None
                if a_on is None or a_off is None or len(a_on) == 0 or len(a_off) == 0:
                    continue
                ys.append(y)
                on_p10.append(np.percentile(a_on, 10) / 1e6)
                on_p50.append(np.percentile(a_on, 50) / 1e6)
                on_p90.append(np.percentile(a_on, 90) / 1e6)
                off_p10.append(np.percentile(a_off, 10) / 1e6)
                off_p50.append(np.percentile(a_off, 50) / 1e6)
                off_p90.append(np.percentile(a_off, 90) / 1e6)
            if ys:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.fill_between(ys, on_p10, on_p90, alpha=0.18, color="C0",
                                label="cap-on  p10-p90")
                ax.plot(ys, on_p50, marker="o", color="C0", label="cap-on  p50")
                ax.fill_between(ys, off_p10, off_p90, alpha=0.18, color="C1",
                                label="cap-off p10-p90")
                ax.plot(ys, off_p50, marker="s", color="C1", label="cap-off p50")
                ax.axhline(cap_M, color="grey", linestyle=":", alpha=0.5,
                           label=f"cap = {cap_M:.1f}M")
                ax.set_xlabel("Years")
                ax.set_ylabel("Wealth (M USD)")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(alpha=0.3)
                st.pyplot(fig, clear_figure=True)
                st.caption(
                    f"Same strategy ({compare_strat} @ {get_T(compare_strat):.3f}x), "
                    "same starting parameters; cap-on stops levering up the moment "
                    f"real wealth crosses \\${cap_M:.1f}M, cap-off keeps levering. "
                    "Cap-off captures more upside on lucky paths but exposes you to "
                    "deeper drawdowns and more margin-call risk on unlucky ones."
                )

    # --- Probability of reaching target wealth ---
    st.subheader(
        f"Probability of reaching ≥ ${target_wealth_M:.1f}M by year {target_year}"
    )
    target_dollars = float(target_wealth_M) * 1e6
    prob_rows = []
    for name in strategy_names:
        cp = res["per_cp"].get(name, {}).get(float(target_year))
        if cp is None or len(cp[mode]) == 0:
            prob_rows.append({"Strategy": name, "T": f"{get_T(name):.3f}x",
                              "P(≥ target)": "—", "n_paths": 0})
            continue
        arr = cp[mode]
        prob = float((arr >= target_dollars).mean())
        prob_rows.append({
            "Strategy": name,
            "T": f"{get_T(name):.3f}x",
            "P(≥ target)": f"{100 * prob:.1f}%",
            "n_paths": len(arr),
        })
    st.dataframe(pd.DataFrame(prob_rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Fraction of historical paths whose {mode} wealth at year {target_year} "
        f"reached or exceeded ${target_wealth_M:.1f}M. Adjust target $ and year in the sidebar."
    )

    # --- Wealth distribution histogram at a chosen checkpoint ---
    st.subheader("Wealth distribution at a chosen checkpoint")
    hist_year = st.selectbox(
        "Checkpoint year for histogram",
        options=res["checkpoints"],
        index=min(len(res["checkpoints"]) - 1, 3),
        format_func=lambda x: f"{int(x)}y",
    )
    hist_strats = st.multiselect(
        "Strategies to overlay",
        options=strategy_names,
        default=[s for s in strategy_names if s in ("unlev", "dd_decay")],
    )
    if hist_strats:
        # Gather all strategies' clipped data first so we can use SHARED bin
        # edges across them (same bin widths → bar heights are comparable).
        clipped = []
        for name in hist_strats:
            cp = res["per_cp"].get(name, {}).get(float(hist_year))
            if cp is None or len(cp[mode]) == 0:
                continue
            arr_M = cp[mode] / 1e6
            cap = float(np.percentile(arr_M, 99))
            clipped.append((name, np.clip(arr_M, 0, cap), len(arr_M)))
        if clipped:
            lo = min(arr.min() for _, arr, _ in clipped)
            hi = max(arr.max() for _, arr, _ in clipped)
            bins = np.linspace(lo, hi, 30)
            fig, ax = plt.subplots(figsize=(8, 4))
            for name, arr, n in clipped:
                ax.hist(arr, bins=bins, alpha=0.5, label=f"{name} (n={n})")
            ax.axvline(target_wealth_M, color="red", linestyle="--",
                       label=f"target \\${target_wealth_M:.1f}M")
            # Use plain text labels — avoid '$' which triggers matplotlib mathtext.
            label_mode = "real" if real_dollars else "nominal"
            ax.set_xlabel(f"Wealth at year {int(hist_year)} ({label_mode}, M USD)")
            ax.set_ylabel("# paths")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig, clear_figure=True)
            st.caption("Histogram clipped at each strategy's p99; shared bin edges so bars across "
                       "strategies are directly comparable. Red dashed line = target wealth.")

    # --- Per-strategy detailed percentiles ---
    st.subheader(f"Detailed percentiles per strategy ({mode_label})")
    pct_labels = ["p10", "p25", "p50", "p75", "p90", "mean"]
    for name in strategy_names:
        T_t = get_T(name)
        with st.expander(f"{name} @ {T_t:.3f}x",
                         expanded=(name == "dd_decay" or name == "unlev")):
            rows = []
            for y in res["checkpoints"]:
                ps = get_pct(name, y)
                if ps is None:
                    continue
                row = {"Year": f"{int(y)}y"}
                for i, label in enumerate(pct_labels):
                    row[label] = fmt_money(ps[i])
                row["paths"] = ps[6]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- p10/p90 fan chart for one strategy ---
    cap_state = "cap on" if res.get("cap_enabled") else "cap off"
    st.subheader(f"Wealth fan (p10 / p50 / p90), {mode_label} — {cap_state}")
    strategy_to_fan = st.selectbox(
        "Choose a strategy to display the fan",
        options=strategy_names,
        index=min(len(strategy_names) - 1, 1),
    )
    if strategy_to_fan:
        ys, p10s, p50s, p90s = [], [], [], []
        for y in res["checkpoints"]:
            ps = get_pct(strategy_to_fan, y)
            if ps:
                ys.append(y)
                p10s.append(ps[0] / 1e6)
                p50s.append(ps[2] / 1e6)
                p90s.append(ps[4] / 1e6)
        if ys:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(ys, p10s, p90s, alpha=0.2, label="p10–p90")
            ax.plot(ys, p50s, marker="o", color="C0", label="p50")
            ax.plot(ys, p10s, linestyle="--", color="C0", alpha=0.7, label="p10")
            ax.plot(ys, p90s, linestyle="--", color="C0", alpha=0.7, label="p90")
            ax.set_xlabel("Years")
            ax.set_ylabel("Wealth (M USD)")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig, clear_figure=True)

    # --- Worst-path inspector ---
    st.subheader("Worst historical path at a chosen horizon")
    col_strat, col_year = st.columns(2)
    with col_strat:
        worst_name = st.selectbox(
            "Strategy",
            options=strategy_names,
            index=min(len(strategy_names) - 1, 1),
            key="worst_strat",
        )
    with col_year:
        worst_year = st.selectbox(
            "Worst-at-year (horizon)",
            options=res["checkpoints"],
            index=min(len(res["checkpoints"]) - 1, 3),
            format_func=lambda x: f"{int(x)}y",
            key="worst_year",
        )
    w_dict = res["worst"].get(worst_name, {})
    w = w_dict.get(float(worst_year))
    if w is not None:
        traj_arr = w["real_traj"] if real_dollars else w["nominal_traj"]
        valid_d = ~np.isnan(traj_arr)
        if valid_d.any():
            horizon_d = int(float(worst_year) * TD)
            last_valid_d = int(np.where(valid_d)[0].max())
            last_d = min(horizon_d, last_valid_d)   # clip to selected horizon
            xs = np.arange(last_d + 1) / TD
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(xs, traj_arr[:last_d + 1] / 1e6, color="C3")
            ax.axhline(w["wealth_at_year"] / 1e6, color="grey", linestyle=":",
                       alpha=0.4, label=f"wealth at horizon = "
                                          f"\\${w['wealth_at_year']/1e6:.2f}M")
            ax.set_xlabel("Years")
            ax.set_ylabel("Wealth (M USD)")
            ax.set_xlim(0, float(worst_year))
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        st.caption(
            f"Lowest-real-wealth path AT YEAR {int(worst_year)}: entered "
            f"**{w['entry_date']}**, real wealth at year {int(worst_year)} = "
            f"**\\${w['wealth_at_year']/1e6:.2f}M** "
            f"({'CALLED' if w['called'] else 'survived'} the strategy)."
        )
    else:
        st.info("No worst-path data for this strategy/horizon combination "
                "(maybe horizon exceeds available historical data).")

    # --- Leverage at checkpoints ---
    st.subheader("Leverage ratio at checkpoints (median across paths)")
    st.caption(
        "Static drifts down naturally as DCA dilutes leverage. Relever holds at target. "
        "dd_decay ratchets target down on observed drawdowns. "
        "wealth_decay glides target toward 1.0x as REAL equity grows from C to wealth_X "
        "(rises again on drawdowns). hybrid takes the lower of dd_decay and wealth_decay. "
        "Unlev is fixed at 1.000x."
    )
    lev_summary = res.get("lev_summary", {})
    rows = []
    for name in strategy_names:
        row = {"Strategy": name, "T_init": f"{get_T(name):.3f}x"}
        for y in res["checkpoints"]:
            d = lev_summary.get(name, {}).get(y)
            row[f"{int(y)}y"] = f"{d['p50']:.3f}x" if d else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Median leverage trajectory across strategies
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in strategy_names:
        ys_lev, vals = [], []
        for y in res["checkpoints"]:
            d = lev_summary.get(name, {}).get(y)
            if d:
                ys_lev.append(y)
                vals.append(d["p50"])
        if ys_lev:
            ax.plot(ys_lev, vals, marker="o", label=name)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5, label="unleveraged")
    ax.set_xlabel("Years")
    ax.set_ylabel("Leverage (median)")
    ax.set_ylim(bottom=0.95)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    with st.expander("Leverage detail (p25 / p50 / p75 / p90)"):
        for name in strategy_names:
            sub_rows = []
            for y in res["checkpoints"]:
                d = lev_summary.get(name, {}).get(y)
                if not d:
                    continue
                sub_rows.append({
                    "Year": f"{int(y)}y",
                    "p25": f"{d['p25']:.3f}x",
                    "p50": f"{d['p50']:.3f}x",
                    "p75": f"{d['p75']:.3f}x",
                    "p90": f"{d['p90']:.3f}x",
                })
            st.markdown(f"**{name}** @ T_init = {get_T(name):.3f}x")
            st.dataframe(pd.DataFrame(sub_rows), hide_index=True, use_container_width=True)

else:
    st.info("Click **Run / refresh** to compute.")
