"""
analyze_mixed_hybrid_sso.py — backtest mixed hybrid + SSO portfolios.

Design (per CLAUDE.md §5j brainstorm):
- Hybrid sleeve and SSO sleeve run as independent sub-portfolios.
- Initial capital C splits as (h·C, (1−h)·C). DCA flows S split the same.
- No selling. Rebalancing is implicit via "DCA splits stay at h" — i.e.,
  no active rebalancing, drift accepted (the no-selling constraint).
- Total wealth = hybrid_wealth + sso_wealth at each day.
- If hybrid sleeve hits margin call, hybrid_wealth → 0 from that day.
  Future DCA dollars to the hybrid sleeve are *lost* in this model
  (conservative — real user would redirect to SSO). Affects only the
  ~1% of bootstrap paths that hit the hybrid call; aggregate impact
  on p10/p50/p90 is small.
- wealth_X is sleeve-relative for hybrid: scaled by h so that hybrid
  sleeve targets unlev when ITS wealth = h · total_target_X. With both
  sleeves growing roughly proportionally, total wealth ≈ target_X at
  that point.

Sweep: h ∈ {0.0, 0.25, 0.50, 0.75, 1.00}
       × scenarios A (heavy DCA), B (light DCA), C (lump)
       × horizons {20y, 30y}.

Reports per (scenario, horizon, h):
  hybrid T_rec, hybrid call rate (hist + boot), p10/p50/p90 of TOTAL wealth,
  p10/p50/p90 of bootstrap-only (where the call risk lives).

Run with: .venv/bin/python -u analyze_mixed_hybrid_sso.py
"""

import numpy as np

from data_loader import load
from project_portfolio import (build_historical_paths, build_bootstrap_paths,
                                stretch_returns, simulate, find_max_safe_T_grid,
                                TD)


F_DEFAULT = 1.5
STRETCH = 1.1
BOOT_TARGET = 0.01
N_BOOT_PATHS = 500
BLOCK_DAYS = 252


SCENARIOS = [
    dict(name="A: heavy-DCA", C=160_000, S=180_000, T_yrs=5, S2=30_000,
         wealth_X_total=3_000_000),
    dict(name="B: light-DCA", C=1_000_000, S=30_000, T_yrs=30, S2=30_000,
         wealth_X_total=10_000_000),
    dict(name="C: lump-sum",  C=1_000_000, S=0, T_yrs=30, S2=0,
         wealth_X_total=float("inf")),
]
HORIZONS_YR = [20, 30]
ALLOCS = [0.0, 0.25, 0.50, 0.75, 1.00]   # h = fraction in hybrid


def well_defended_hybrid_T(ret_h, tsy_h, cpi_h, avail_h,
                            ret_b, tsy_b, cpi_b, ret_s,
                            C, S, T_yrs, S2, max_days, wealth_X):
    """Joint-defended hybrid T (hist 0% + boot ≤1% + stretch 0%).
    Scale-invariant in capital, so the per-dollar T is the same regardless
    of allocation. We pass scenario-level C/S since wealth_X interacts."""
    T_h = find_max_safe_T_grid(ret_h, tsy_h, cpi_h, "hybrid", 0.0,
                                C, S, T_yrs, S2, max_days, avail=avail_h,
                                F=F_DEFAULT, wealth_X=wealth_X)
    T_b = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, "hybrid", BOOT_TARGET,
                                C, S, T_yrs, S2, max_days, F=F_DEFAULT,
                                wealth_X=wealth_X)
    T_s = find_max_safe_T_grid(ret_s, tsy_h, cpi_h, "hybrid", 0.0,
                                C, S, T_yrs, S2, max_days, avail=avail_h,
                                F=F_DEFAULT, wealth_X=wealth_X)
    return float(min(T_h, T_b, T_s))


def project_one_sleeve(ret, tsy, cpi, avail_or_none, kind, T, F,
                        C_sleeve, S_sleeve, T_yrs, S2_sleeve,
                        max_days, wealth_X):
    """Run a single sleeve. Returns (real_eq[K, max_days+1], called[K])."""
    real_eq, called, _, _ = simulate(
        ret, tsy, cpi, kind, T,
        C_sleeve, S_sleeve, T_yrs, S2_sleeve, max_days,
        avail=avail_or_none, F=F, wealth_X=wealth_X)
    return real_eq, called


def combine_sleeves(real_eq_h, called_h, real_eq_s):
    """Total wealth at each day = hybrid (0 if called) + sso. NaN if both
    are NaN (i.e., past avail for that path). NaN-safe."""
    h = np.where(np.isnan(real_eq_h), 0.0, real_eq_h)
    s = np.where(np.isnan(real_eq_s), 0.0, real_eq_s)
    out = h + s
    # Mask paths where both sleeves are out-of-data (hybrid called paths
    # are still tracked because SSO continues, so we use SSO's NaN as
    # the "out of data" indicator).
    out = np.where(np.isnan(real_eq_s), np.nan, out)
    return out


PCTILES = [1, 5, 10, 50, 90]


def percentiles_at_horizon(real_eq, avail, horizon_days):
    """Return PCTILES percentiles of real wealth at horizon_days, restricted
    to paths with avail >= horizon_days."""
    eligible = avail >= horizon_days
    vals = real_eq[eligible, horizon_days]
    valid = ~np.isnan(vals)
    if not valid.any():
        return [float("nan")] * len(PCTILES), 0
    pct = np.nanpercentile(vals[valid], PCTILES)
    return [float(x) for x in pct], int(valid.sum())


def boot_percentiles(real_eq, horizon_days):
    """Bootstrap paths have full horizon by construction."""
    vals = real_eq[:, horizon_days]
    valid = ~np.isnan(vals)
    if not valid.any():
        return [float("nan")] * len(PCTILES), 0
    pct = np.nanpercentile(vals[valid], PCTILES)
    return [float(x) for x in pct], int(valid.sum())


def boot_tail_metrics(real_eq, horizon_days, C):
    """Tail-aware metrics: minimum wealth (literal worst-case path)
    and fraction of paths ending below initial capital C."""
    vals = real_eq[:, horizon_days]
    valid = ~np.isnan(vals)
    v = vals[valid]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.min()), float((v < C).mean())


def main():
    print("Loading + building paths...")
    dates, px, tsy, mrate, cpi = load(with_cpi=True)
    max_days = 30 * TD
    ret_h, tsy_h, cpi_h, avail_h, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    ret_s_stretched = stretch_returns(ret_h, STRETCH)
    rng = np.random.default_rng(42)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, N_BOOT_PATHS, BLOCK_DAYS, rng)
    avail_b = np.full(ret_b.shape[0], max_days, dtype=np.int64)
    print(f"  hist paths: {ret_h.shape[0]}, boot paths: {ret_b.shape[0]}")

    for scen in SCENARIOS:
        wxt = scen["wealth_X_total"]
        wxt_str = f"${wxt/1e6:.1f}M" if wxt != float("inf") else "none"
        print()
        print("=" * 110)
        print(f"SCENARIO {scen['name']}: C=${scen['C']/1e3:.0f}k  "
              f"S=${scen['S']/1e3:.0f}k×{scen['T_yrs']}y  "
              f"S2=${scen['S2']/1e3:.0f}k  wealth_X(total)={wxt_str}")
        print("=" * 110)

        # Calibrate hybrid T (scale-invariant, so use scenario-level params)
        T_hyb = well_defended_hybrid_T(
            ret_h, tsy_h, cpi_h, avail_h,
            ret_b, tsy_b, cpi_b, ret_s_stretched,
            scen["C"], scen["S"], scen["T_yrs"], scen["S2"], max_days,
            wxt)
        print(f"Hybrid T (well-defended): {T_hyb:.3f}x  (F={F_DEFAULT})")

        for h_yr in HORIZONS_YR:
            h_days = h_yr * TD
            print(f"\n  Horizon: {h_yr}y  — bootstrap (worst-case + percentiles, "
                  f"frac below initial C=${scen['C']/1e3:.0f}k)")
            print(f"    {'h=hybrid%':>10}  {'eff_lev':>7}  "
                  f"{'h_call_b':>9}  "
                  f"{'min':>7}  {'<C%':>5}  "
                  f"{'p1':>7}  {'p5':>7}  {'p10':>7}  "
                  f"{'p50':>7}  {'p90':>7}")
            print("    " + "-" * 84)

            for h_frac in ALLOCS:
                # Sleeve allocations
                C_h = scen["C"] * h_frac
                S_h = scen["S"] * h_frac
                S2_h = scen["S2"] * h_frac
                C_s = scen["C"] * (1.0 - h_frac)
                S_s = scen["S"] * (1.0 - h_frac)
                S2_s = scen["S2"] * (1.0 - h_frac)
                # Sleeve-relative wealth_X for hybrid: scale by h
                wxt_h = (wxt * h_frac if wxt != float("inf") and h_frac > 0
                         else float("inf"))

                # Hybrid sleeve (skip if h=0)
                if h_frac > 0:
                    real_eq_h_hist, called_h_hist = project_one_sleeve(
                        ret_h, tsy_h, cpi_h, avail_h, "hybrid", T_hyb,
                        F_DEFAULT, C_h, S_h, scen["T_yrs"], S2_h,
                        max_days, wxt_h)
                    real_eq_h_boot, called_h_boot = project_one_sleeve(
                        ret_b, tsy_b, cpi_b, None, "hybrid", T_hyb,
                        F_DEFAULT, C_h, S_h, scen["T_yrs"], S2_h,
                        max_days, wxt_h)
                else:
                    K_h = ret_h.shape[0]
                    K_b = ret_b.shape[0]
                    real_eq_h_hist = np.zeros((K_h, max_days + 1))
                    called_h_hist = np.zeros(K_h, dtype=bool)
                    real_eq_h_boot = np.zeros((K_b, max_days + 1))
                    called_h_boot = np.zeros(K_b, dtype=bool)

                # SSO sleeve (skip if h=1)
                if h_frac < 1.0:
                    real_eq_s_hist, _ = project_one_sleeve(
                        ret_h, tsy_h, cpi_h, avail_h, "sso", 2.0,
                        F_DEFAULT, C_s, S_s, scen["T_yrs"], S2_s,
                        max_days, float("inf"))   # SSO ignores wealth_X
                    real_eq_s_boot, _ = project_one_sleeve(
                        ret_b, tsy_b, cpi_b, None, "sso", 2.0,
                        F_DEFAULT, C_s, S_s, scen["T_yrs"], S2_s,
                        max_days, float("inf"))
                else:
                    real_eq_s_hist = np.full_like(real_eq_h_hist, np.nan)
                    real_eq_s_hist[:, :max_days+1] = 0.0
                    # mask past-avail
                    for k in range(real_eq_h_hist.shape[0]):
                        real_eq_s_hist[k, avail_h[k] + 1:] = np.nan
                    real_eq_s_boot = np.zeros_like(real_eq_h_boot)

                # Combine. For 100% hybrid, called paths must show as $0 (a
                # margin call wipes all wealth) — NOT as NaN, which would
                # exclude them from percentiles and hide the actual tail risk.
                # combine_sleeves handles this naturally for mixed (called
                # hybrid → hybrid contribution=0, sso contribution intact).
                if h_frac == 1.0:
                    total_hist = real_eq_h_hist.copy()
                    total_boot = real_eq_h_boot.copy()
                    # Called paths: replace NaN with 0 within avail range
                    for k in range(real_eq_h_hist.shape[0]):
                        if called_h_hist[k]:
                            total_hist[k, :avail_h[k] + 1] = np.where(
                                np.isnan(total_hist[k, :avail_h[k] + 1]),
                                0.0, total_hist[k, :avail_h[k] + 1])
                    for k in range(real_eq_h_boot.shape[0]):
                        if called_h_boot[k]:
                            total_boot[k, :] = np.where(
                                np.isnan(total_boot[k, :]),
                                0.0, total_boot[k, :])
                elif h_frac == 0.0:
                    total_hist = real_eq_s_hist
                    total_boot = real_eq_s_boot
                else:
                    total_hist = combine_sleeves(real_eq_h_hist, called_h_hist,
                                                  real_eq_s_hist)
                    total_boot = combine_sleeves(real_eq_h_boot, called_h_boot,
                                                  real_eq_s_boot)

                # Effective leverage at t=0: (h * T_hyb + (1-h) * 2.0)
                eff_lev = h_frac * T_hyb + (1 - h_frac) * 2.0

                pct_b, _ = boot_percentiles(total_boot, h_days)
                p1b, p5b, p10b, p50b, p90b = pct_b
                min_b, frac_below_C = boot_tail_metrics(
                    total_boot, h_days, scen["C"])
                cb = float(called_h_boot.mean()) if h_frac > 0 else 0.0

                label = (f"{int(h_frac*100):3d}/{int((1-h_frac)*100):3d}"
                         if 0 < h_frac < 1
                         else ("100% hyb" if h_frac == 1 else "100% SSO"))
                print(f"    {label:>10}  {eff_lev:>5.2f}x  "
                      f"{cb*100:>7.2f}%  "
                      f"${min_b/1e6:>5.2f}M  {frac_below_C*100:>4.1f}%  "
                      f"${p1b/1e6:>5.2f}M  ${p5b/1e6:>5.2f}M  ${p10b/1e6:>5.2f}M  "
                      f"${p50b/1e6:>5.2f}M  ${p90b/1e6:>5.2f}M")


if __name__ == "__main__":
    main()
