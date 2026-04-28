"""
Mixed_Analysis.py — interactive 3-way (hybrid + SSO + UPRO) sweep.

For each allocation triple (h_hyb, h_sso, h_upro), calibrates the hybrid
sleeve's T_init to a well-defended target (boot ≤ 1% + hist 0%) under a
combined-collateral margin call check, then projects on historical and
bootstrap paths and reports tail metrics + percentiles.

Computation is cached on the scenario tuple — re-running the same scenario
is fast.
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load
from project_portfolio import (build_historical_paths, build_bootstrap_paths,
                                stretch_returns, simulate_3way,
                                find_safe_T_3way, find_max_safe_T_grid, TD)


st.set_page_config(page_title="Mixed Analysis", layout="wide")

# Defaults
DEFAULT_ALLOCS = [
    "100,0,0", "75,25,0", "50,50,0", "25,75,0", "0,100,0",
    "75,0,25", "50,0,50", "25,0,75", "0,0,100",
    "0,75,25", "0,50,50", "0,25,75",
    "50,25,25", "34,33,33", "25,50,25", "25,25,50",
]


def parse_alloc(s):
    """Parse 'h,s,u' (each 0-100, summing to 100) → (h_hyb, h_sso, h_upro)
    as fractions in [0, 1]."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if any(v < 0 or v > 100 for v in vals):
        return None
    if abs(sum(vals) - 100) > 0.5:
        return None
    return tuple(v / 100.0 for v in vals)


# ---------------------------------------------------------------------------
# Path caching (re-uses the main app's path builder pattern)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_paths(max_days, n_boot, block_days, seed):
    dates, px, tsy, mrate, cpi = load(with_cpi=True)
    ret_h, tsy_h, cpi_h, avail_h, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    rng = np.random.default_rng(seed)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, n_boot, block_days, rng)
    avail_b = np.full(ret_b.shape[0], max_days, dtype=np.int64)
    return ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b, avail_b


@st.cache_data(show_spinner=False, max_entries=8)
def run_sweep(C, S, T_yrs, S2, wealth_X, horizon_yr, F, boot_target,
              n_boot, block_days, seed, broker_bump_days, alloc_strings):
    """Run the 3-way sweep; returns a list of dicts (one per allocation)."""
    max_days = horizon_yr * TD
    paths = build_paths(max_days, n_boot, block_days, seed)
    ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b, avail_b = paths
    h_days = horizon_yr * TD

    # Standalone hybrid reference T (for context display)
    T_h_hist_solo = find_max_safe_T_grid(ret_h, tsy_h, cpi_h, "hybrid", 0.0,
                                          C, S, T_yrs, S2, max_days, avail=avail_h,
                                          F=F, wealth_X=wealth_X)
    T_h_boot_solo = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, "hybrid", boot_target,
                                          C, S, T_yrs, S2, max_days, F=F,
                                          wealth_X=wealth_X)
    T_solo_hybrid = float(min(T_h_hist_solo, T_h_boot_solo))

    results = []
    for s in alloc_strings:
        alloc = parse_alloc(s)
        if alloc is None:
            continue
        h, ss, u = alloc

        if h > 0:
            T_b = find_safe_T_3way(
                ret_b, tsy_b, cpi_b, h, ss, u, C, S, T_yrs, S2, max_days,
                avail_b, F, 1.0, wealth_X, broker_bump_days, boot_target)
            T_h = find_safe_T_3way(
                ret_h, tsy_h, cpi_h, h, ss, u, C, S, T_yrs, S2, max_days,
                avail_h, F, 1.0, wealth_X, broker_bump_days, 0.0)
            T = float(min(T_b, T_h))
        else:
            T = 1.0
            T_b = T_h = 1.0

        # Project hist + boot
        re_h, called_h, peak_h = simulate_3way(
            ret_h, tsy_h, cpi_h, T, h, ss, u, C, S, T_yrs, S2, max_days,
            avail_h, F, 1.0, wealth_X, broker_bump_days)
        re_b, called_b, _ = simulate_3way(
            ret_b, tsy_b, cpi_b, T, h, ss, u, C, S, T_yrs, S2, max_days,
            avail_b, F, 1.0, wealth_X, broker_bump_days)

        # Mark called paths' wealth as 0 (conservative wipeout)
        for k in range(re_h.shape[0]):
            if called_h[k]:
                re_h[k, :avail_h[k] + 1] = np.where(
                    np.isnan(re_h[k, :avail_h[k] + 1]), 0.0,
                    re_h[k, :avail_h[k] + 1])
        for k in range(re_b.shape[0]):
            if called_b[k]:
                re_b[k, :] = np.where(
                    np.isnan(re_b[k, :]), 0.0, re_b[k, :])

        # Hist percentiles (paths with at least h_days of avail)
        eligible = avail_h >= h_days
        v_h = re_h[eligible, h_days]
        v_h = v_h[~np.isnan(v_h)]
        # Boot percentiles
        v_b = re_b[:, h_days]
        v_b = v_b[~np.isnan(v_b)]

        if len(v_h) == 0 or len(v_b) == 0:
            continue

        p1h, p5h, p10h, p50h, p90h = [float(x) for x in
                                       np.percentile(v_h, [1, 5, 10, 50, 90])]
        p1b, p5b, p10b, p50b, p90b = [float(x) for x in
                                       np.percentile(v_b, [1, 5, 10, 50, 90])]

        pl_p99 = (float(np.percentile(peak_h[~called_h], 99))
                  if (~called_h).any() else float("nan"))

        results.append(dict(
            label=f"{int(h*100):3d}/{int(ss*100):3d}/{int(u*100):3d}",
            h=h, s=ss, u=u, T=T, T_b=T_b, T_h=T_h,
            peak_lev_p99=pl_p99,
            hist_call=float(called_h[eligible].mean()),
            boot_call=float(called_b.mean()),
            min_b=float(v_b.min()),
            frac_below_C_b=float((v_b < C).mean()),
            p1_h=p1h, p5_h=p5h, p10_h=p10h, p50_h=p50h, p90_h=p90h,
            p1_b=p1b, p5_b=p5b, p10_b=p10b, p50_b=p50b, p90_b=p90b,
        ))

    return results, T_solo_hybrid


def pareto_frontier(results, keys=("p10_h", "p50_h", "p90_h")):
    """Return labels of Pareto-optimal allocations on the given metric keys
    (all keys treated as 'higher is better')."""
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, r2 in enumerate(results):
            if i == j:
                continue
            ge_all = all(r2[k] >= r[k] for k in keys)
            gt_any = any(r2[k] > r[k] for k in keys)
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            pareto.append(r["label"])
    return pareto


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Mixed Strategy Analysis: Hybrid + SSO + UPRO")
st.caption("3-sleeve portfolio with shared margin collateral and "
           "DCA-only rebalancing (no selling). For each allocation, "
           "the hybrid sleeve's T_init is calibrated to a well-defended "
           "target (boot ≤ target + hist 0%) using a unified-account "
           "margin-call check.")

with st.expander("How it works", expanded=False):
    st.markdown(r"""
**Sleeves**
- **Hybrid**: SPX held with margin loan. Standard hybrid logic
  (dd_decay ratchet ∧ wealth_decay glide on TOTAL real wealth).
- **SSO**: 2x daily-reset SPX ETF.
- **UPRO**: 3x daily-reset SPX ETF.

**Combined-collateral margin-call check** (Reg-T leveraged-ETF maint):
$$\text{loan}_h \le 0.75 \cdot \text{stocks}_h + 0.50 \cdot \text{stocks}_{sso} + 0.25 \cdot \text{stocks}_{upro}$$
SSO contributes ~2/3 of equivalent SPX collateral; UPRO contributes ~1/3.

**DCA-rebalancing rule** (no selling): each contribution day, route the
day's contribution to underweight sleeves proportionally to their
deficit-vs-target. If all sleeves are at-or-above target, route at
target proportions.

**Calibration**: for each allocation with hybrid % > 0, we binary-search
T_init for the hybrid sleeve such that the bootstrap call rate at the
combined-collateral level is ≤ target (default 1%). Hybrid sleeves can
typically run at HIGHER T_init in a mixed account because the SSO/UPRO
sleeves provide additional collateral support — the simulator captures
this, including the dynamic decay of that support during drawdowns.

**Conservative wipeout**: when a margin call fires, the path is treated
as fully wiped (real_eq = 0). In real-world partial-liquidation behavior
the broker would only liquidate enough to satisfy maintenance, so
remaining SSO/UPRO holdings would survive — this model is conservative.
""")

st.subheader("Scenario inputs")

col_a, col_b = st.columns(2)
with col_a:
    C = st.number_input("Initial capital C ($)", value=160_000, step=10_000,
                         key="mix_C")
    S = st.number_input("Annual savings S ($/yr, real)", value=180_000,
                         step=10_000, key="mix_S")
    T_yrs = st.number_input("Years at higher savings (T)", value=5,
                             step=1, key="mix_Tyrs")
    S2 = st.number_input("Annual savings after T (S2, $/yr)", value=30_000,
                          step=5_000, key="mix_S2")
with col_b:
    wealth_X_M = st.number_input("Wealth target (real $M, where hybrid → 1x)",
                                  value=3.0, step=0.5, format="%.1f",
                                  key="mix_wealthX")
    horizon_yr = st.number_input("Horizon (years)", value=30, min_value=5,
                                  max_value=30, step=1, key="mix_horizon")
    F = st.number_input("dd_decay factor F", value=1.5, step=0.1,
                         min_value=0.5, max_value=3.0, key="mix_F")
    broker_bump_yrs = st.number_input(
        "Broker-rate years at start (Tsy+150 bps)", value=2, step=1,
        min_value=0, max_value=5, key="mix_bump")

st.subheader("Safety bar + bootstrap")
col_c, col_d = st.columns(2)
with col_c:
    boot_target_pct = st.number_input("Boot call target (%)", value=1.0,
                                       step=0.1, format="%.1f",
                                       key="mix_boot_target")
    n_boot = st.number_input("Bootstrap paths", value=500, step=100,
                              min_value=100, max_value=5000,
                              key="mix_n_boot")
with col_d:
    block_yrs = st.number_input("Bootstrap block (years)", value=1.0,
                                 step=0.5, key="mix_block_yrs")
    seed = st.number_input("RNG seed", value=42, step=1, key="mix_seed")

st.subheader("Allocations to evaluate (h_hyb, h_sso, h_upro — sum to 100)")
alloc_text = st.text_area(
    "One allocation per line, format 'h,s,u' summing to 100",
    value="\n".join(DEFAULT_ALLOCS),
    height=200,
    key="mix_alloc_text",
    help="Each line is a 3-way split, e.g. '50,25,25' for "
         "50% hybrid + 25% SSO + 25% UPRO. Lines that don't parse "
         "are skipped silently.")

run_btn = st.button("Run analysis", type="primary", key="mix_run")

# ---------------------------------------------------------------------------
# Compute and display
# ---------------------------------------------------------------------------

if run_btn or st.session_state.get("mix_results") is not None:
    if run_btn:
        alloc_strings = [s.strip() for s in alloc_text.strip().split("\n")
                          if s.strip()]
        wealth_X = (wealth_X_M * 1e6 if wealth_X_M > 0 else float("inf"))
        boot_target = boot_target_pct / 100.0
        block_days = int(block_yrs * TD)
        broker_bump_days = int(broker_bump_yrs * TD)

        with st.spinner(f"Running {len(alloc_strings)} allocations × "
                        f"binary-search calibration ({n_boot} boot paths). "
                        f"~5-15s per allocation..."):
            t0 = time.time()
            results, T_solo = run_sweep(
                int(C), float(S), float(T_yrs), float(S2),
                float(wealth_X), int(horizon_yr), float(F),
                float(boot_target), int(n_boot), int(block_days), int(seed),
                int(broker_bump_days), tuple(alloc_strings))
            elapsed = time.time() - t0

        st.session_state["mix_results"] = results
        st.session_state["mix_T_solo"] = T_solo
        st.session_state["mix_elapsed"] = elapsed
        st.session_state["mix_scenario"] = dict(C=int(C), wealth_X_M=wealth_X_M)

    results = st.session_state["mix_results"]
    T_solo = st.session_state["mix_T_solo"]
    elapsed = st.session_state.get("mix_elapsed", 0)
    scen = st.session_state.get("mix_scenario", dict(C=int(C), wealth_X_M=wealth_X_M))

    st.success(f"Computed {len(results)} allocations in {elapsed:.1f}s. "
               f"Reference standalone hybrid T_rec: **{T_solo:.3f}x**.")

    # Pareto frontier
    pareto = pareto_frontier(results, ("p10_h", "p50_h", "p90_h"))

    # Best by criterion
    best_p50 = max(results, key=lambda r: r["p50_h"])
    best_p10 = max(results, key=lambda r: r["p10_h"])
    best_p90 = max(results, key=lambda r: r["p90_h"])
    best_min = max(results, key=lambda r: r["min_b"])
    best_fbc = min(results, key=lambda r: r["frac_below_C_b"])

    st.subheader("Best by criterion")
    summary_rows = [
        ("Best p10 (hist) — downside protection", best_p10),
        ("Best p50 (hist) — median EV",            best_p50),
        ("Best p90 (hist) — upside",                best_p90),
        ("Best min wealth (boot, conservative) — extreme tail", best_min),
        ("Best <C% (boot) — fewest paths below initial",        best_fbc),
    ]
    summary_df = pd.DataFrame([
        dict(criterion=name, alloc=r["label"], T_hyb=f"{r['T']:.2f}x",
             p10=f"${r['p10_h']/1e6:.2f}M",
             p50=f"${r['p50_h']/1e6:.2f}M",
             p90=f"${r['p90_h']/1e6:.2f}M",
             boot_min=f"${r['min_b']/1e6:.2f}M",
             below_C=f"{r['frac_below_C_b']*100:.1f}%")
        for name, r in summary_rows
    ])
    st.dataframe(summary_df, hide_index=True)

    st.caption(f"**Pareto-optimal on (p10/p50/p90 hist):** "
               f"`{', '.join(pareto)}`")

    # Full results table
    st.subheader("Full sweep results")
    st.caption("`pkLv99` = 99th-percentile peak hybrid-sleeve leverage on "
               "surviving paths (can exceed 4× when SSO/UPRO collateral lifts "
               "the call threshold). `min_b` = literal worst-case wealth on "
               "the bootstrap path set; `<C%` = fraction of bootstrap paths "
               "ending below initial capital.")

    rows = []
    for r in results:
        rows.append(dict(
            alloc=r["label"],
            T_hyb=f"{r['T']:.3f}x",
            pkLv99=f"{r['peak_lev_p99']:.2f}x",
            hist_call=f"{r['hist_call']*100:.2f}%",
            boot_call=f"{r['boot_call']*100:.2f}%",
            min_b=f"${r['min_b']/1e6:.2f}M",
            below_C=f"{r['frac_below_C_b']*100:.1f}%",
            p10_h=f"${r['p10_h']/1e6:.2f}M",
            p50_h=f"${r['p50_h']/1e6:.2f}M",
            p90_h=f"${r['p90_h']/1e6:.2f}M",
            p10_b=f"${r['p10_b']/1e6:.2f}M",
            p50_b=f"${r['p50_b']/1e6:.2f}M",
            p90_b=f"${r['p90_b']/1e6:.2f}M",
            pareto="✓" if r["label"] in pareto else "",
        ))
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)

    # Charts
    st.subheader("Wealth distribution by allocation (historical)")
    chart_df = pd.DataFrame([
        dict(alloc=r["label"], pct="p10", wealth_M=r["p10_h"] / 1e6)
        for r in results
    ] + [
        dict(alloc=r["label"], pct="p50", wealth_M=r["p50_h"] / 1e6)
        for r in results
    ] + [
        dict(alloc=r["label"], pct="p90", wealth_M=r["p90_h"] / 1e6)
        for r in results
    ])
    pivot = chart_df.pivot(index="alloc", columns="pct",
                            values="wealth_M").reindex([r["label"] for r in results])
    st.bar_chart(pivot, height=400)

    st.subheader("Bootstrap tail: min wealth and frac below initial C")
    tail_df = pd.DataFrame([
        dict(alloc=r["label"],
             min_wealth_M=r["min_b"] / 1e6,
             frac_below_C_pct=r["frac_below_C_b"] * 100)
        for r in results
    ]).set_index("alloc")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Worst-case wealth on bootstrap (literal min)")
        st.bar_chart(tail_df[["min_wealth_M"]], height=400)
    with col2:
        st.caption(f"Fraction of bootstrap paths ending below initial "
                   f"C=${scen['C']/1e3:.0f}k (%)")
        st.bar_chart(tail_df[["frac_below_C_pct"]], height=400)
