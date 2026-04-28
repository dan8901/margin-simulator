"""
analyze_etf_sweep.py — broader scenario sweep comparing SSO / UPRO against
the project's main margin strategies.

Sweep dimensions:
  Scenario:
    A  — heavy DCA (user's primary): C=160k, S=180k×5y, S2=30k, wealth_X=3M
    B  — light DCA / lump-tilted:    C=1M,   S=30k×30y,    S2=30k, wealth_X=10M
    C  — pure lump:                  C=1M,   S=0,          S2=0,   no wealth_X
  Horizon: 20y, 30y
  Strategies:
    unlev (T=1)
    static  @ well-defended (boot ≤1% + hist 0% + stretch 0%)
    hybrid  @ well-defended (same)
    SSO  (2x daily reset)
    UPRO (3x daily reset)

Reports per (scenario, horizon, strategy):
  T_rec, hist call %, boot call %, real-wealth p10/p50/p90 at end of horizon

Run with: .venv/bin/python analyze_etf_sweep.py
"""

import numpy as np

from data_loader import load
from project_portfolio import (build_historical_paths, build_bootstrap_paths,
                                stretch_returns, simulate, find_max_safe_T_grid,
                                call_rate, TD)


F_DEFAULT = 1.5
STRETCH = 1.1
BOOT_TARGET = 0.01
N_BOOT_PATHS = 500
BLOCK_DAYS = 252


def well_defended_T(ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b,
                    ret_s, kind, F, C, S, T_yrs, S2, max_days, wealth_X):
    """Joint-defended T_init: min of (hist 0%, boot ≤target, stretch 0%)."""
    overlay = dict(wealth_X=wealth_X)
    T_h = find_max_safe_T_grid(ret_h, tsy_h, cpi_h, kind, 0.0,
                                C, S, T_yrs, S2, max_days,
                                avail=avail_h, F=F, **overlay)
    T_b = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, kind, BOOT_TARGET,
                                C, S, T_yrs, S2, max_days, F=F, **overlay)
    T_s = find_max_safe_T_grid(ret_s, tsy_h, cpi_h, kind, 0.0,
                                C, S, T_yrs, S2, max_days,
                                avail=avail_h, F=F, **overlay)
    return float(min(T_h, T_b, T_s)), float(T_h), float(T_b), float(T_s)


def project(ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b,
            kind, T, F, C, S, T_yrs, S2, max_days, wealth_X, horizon_days):
    """Project on hist + boot, return:
       (hist_call_rate, boot_call_rate, real_wealth_p10/p50/p90 at horizon_days,
        also reports at min(avail, horizon_days) for hist).
    """
    overlay = dict(wealth_X=wealth_X)
    real_eq_h, called_h, _, _ = simulate(
        ret_h, tsy_h, cpi_h, kind, T,
        C, S, T_yrs, S2, max_days,
        avail=avail_h, F=F, **overlay)
    _, called_b, _, _ = simulate(
        ret_b, tsy_b, cpi_b, kind, T,
        C, S, T_yrs, S2, max_days, F=F, **overlay)

    # Wealth at horizon_days (or last avail, whichever is smaller)
    capped = np.minimum(avail_h, horizon_days)
    wealth = np.array([real_eq_h[k, capped[k]] if capped[k] >= horizon_days
                       else np.nan for k in range(real_eq_h.shape[0])])
    valid = ~np.isnan(wealth)
    if not valid.any():
        p10 = p50 = p90 = float("nan")
    else:
        p10, p50, p90 = np.nanpercentile(wealth[valid], [10, 50, 90])
    return (float(called_h.mean()), float(called_b.mean()),
            float(p10), float(p50), float(p90))


SCENARIOS = [
    dict(name="A: heavy-DCA",
         C=160_000, S=180_000, T_yrs=5, S2=30_000,
         wealth_X=3_000_000),
    dict(name="B: light-DCA",
         C=1_000_000, S=30_000, T_yrs=30, S2=30_000,
         wealth_X=10_000_000),
    dict(name="C: lump-sum",
         C=1_000_000, S=0, T_yrs=30, S2=0,
         wealth_X=float("inf")),
]
HORIZONS_YR = [20, 30]


def main():
    print("Loading data + building paths (used across all scenarios)...")
    dates, px, tsy, mrate, cpi = load(with_cpi=True)
    max_days = 30 * TD

    ret_h, tsy_h, cpi_h, avail_h, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    ret_s = stretch_returns(ret_h, STRETCH)

    rng = np.random.default_rng(42)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, N_BOOT_PATHS, BLOCK_DAYS, rng)

    print(f"  hist paths: {ret_h.shape[0]}, boot paths: {ret_b.shape[0]}")
    print()

    for scen in SCENARIOS:
        print("=" * 100)
        print(f"SCENARIO {scen['name']}: C=${scen['C']/1e3:.0f}k  "
              f"S=${scen['S']/1e3:.0f}k×{scen['T_yrs']}y  "
              f"S2=${scen['S2']/1e3:.0f}k  "
              f"wealth_X=${scen['wealth_X']/1e6:.1f}M" if scen['wealth_X'] != float('inf')
              else f"SCENARIO {scen['name']}: C=${scen['C']/1e3:.0f}k  "
                   f"S=${scen['S']/1e3:.0f}k×{scen['T_yrs']}y  "
                   f"S2=${scen['S2']/1e3:.0f}k  wealth_X=none")
        print("=" * 100)

        # Calibrate static + hybrid (well-defended, full 30y)
        print("Calibrating static and hybrid (well-defended)...")
        T_stat, T_sh, T_sb, T_ss = well_defended_T(
            ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b, ret_s,
            "static", F_DEFAULT, scen["C"], scen["S"], scen["T_yrs"],
            scen["S2"], max_days, scen["wealth_X"])
        T_hyb, T_hh, T_hb, T_hs = well_defended_T(
            ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b, ret_s,
            "hybrid", F_DEFAULT, scen["C"], scen["S"], scen["T_yrs"],
            scen["S2"], max_days, scen["wealth_X"])
        print(f"  static:  T_rec={T_stat:.3f}  (hist={T_sh:.3f}, boot={T_sb:.3f}, stretch={T_ss:.3f})")
        print(f"  hybrid:  T_rec={T_hyb:.3f}  (hist={T_hh:.3f}, boot={T_hb:.3f}, stretch={T_hs:.3f})")

        for h_yr in HORIZONS_YR:
            h_days = h_yr * TD
            print(f"\n  Horizon: {h_yr}y")
            print(f"    {'Strategy':16s}  {'T':>5}  {'hist%':>6}  {'boot%':>6}  "
                  f"{'p10 ($M)':>10}  {'p50 ($M)':>10}  {'p90 ($M)':>10}")
            print(f"    " + "-" * 84)

            specs = [
                ("unlev",  "static", 1.0,    F_DEFAULT),
                ("static", "static", T_stat, F_DEFAULT),
                ("hybrid", "hybrid", T_hyb,  F_DEFAULT),
                ("SSO 2x", "sso",    2.0,    F_DEFAULT),
                ("UPRO 3x","upro",   3.0,    F_DEFAULT),
            ]
            for label, kind, T, F in specs:
                hcall, bcall, p10, p50, p90 = project(
                    ret_h, tsy_h, cpi_h, avail_h, ret_b, tsy_b, cpi_b,
                    kind, T, F, scen["C"], scen["S"], scen["T_yrs"],
                    scen["S2"], max_days, scen["wealth_X"], h_days)
                print(f"    {label:16s}  {T:>4.2f}x  {hcall*100:>5.2f}%  "
                      f"{bcall*100:>5.2f}%  ${p10/1e6:>8.2f}  "
                      f"${p50/1e6:>8.2f}  ${p90/1e6:>8.2f}")
        print()


if __name__ == "__main__":
    main()
