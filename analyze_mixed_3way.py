"""
analyze_mixed_3way.py — comprehensive 3-sleeve analysis: hybrid + SSO + UPRO.

For each allocation (h, s, u) with h+s+u=1:
  1. Calibrate hybrid sleeve T_init to well-defended:
     min(hist 0% calls, boot ≤ 1% calls, stretch 0% calls at F=1.1)
     under combined-collateral margin call check.
  2. Project on historical paths (variable horizon via avail).
  3. Project on bootstrap paths (full 30y).
  4. Report wealth percentiles + tail metrics.

Combined collateral rule (Reg-T, leveraged ETF maintenance):
  loan_h ≤ 0.75·stocks_h + 0.50·stocks_sso + 0.25·stocks_upro
  (SPX: 25% mm, SSO 2x: 50% mm, UPRO 3x: 75% mm)

DCA rule (no selling):
  Each contribution day, route the day's contribution proportionally to
  each sleeve's *underweight deficit* vs target allocation. If all sleeves
  at-or-above target (deficits sum to 0), route at target proportions.
  This is the "tax-efficient rebalancing via DCA" the user specified.

Hybrid sleeve uses hybrid logic (dd_decay ratchet ∧ wealth_decay glide
on TOTAL real wealth) on its sleeve equity. Wealth glide deleverages
the hybrid sleeve as TOTAL wealth approaches wealth_X.

Run: .venv/bin/python -u analyze_mixed_3way.py
"""

import numpy as np

from data_loader import load
from project_portfolio import (build_historical_paths, build_bootstrap_paths,
                                find_max_safe_T_grid, simulate_3way,
                                find_safe_T_3way, stretch_returns, TD)


F_DEFAULT = 1.5
STRETCH_F = 1.1
BOOT_TARGET = 0.01
N_BOOT_PATHS = 500
BLOCK_DAYS = 252




# Allocation grid: (h_hyb, h_sso, h_upro), summing to 1
ALLOCS = [
    (1.00, 0.00, 0.00),  # 100% hybrid
    (0.75, 0.25, 0.00),
    (0.50, 0.50, 0.00),
    (0.25, 0.75, 0.00),
    (0.00, 1.00, 0.00),  # 100% SSO
    (0.75, 0.00, 0.25),
    (0.50, 0.00, 0.50),
    (0.25, 0.00, 0.75),
    (0.00, 0.00, 1.00),  # 100% UPRO
    (0.00, 0.75, 0.25),
    (0.00, 0.50, 0.50),
    (0.00, 0.25, 0.75),
    (0.50, 0.25, 0.25),
    (0.34, 0.33, 0.33),
    (0.25, 0.50, 0.25),
    (0.25, 0.25, 0.50),
]

SCENARIOS = [
    dict(name="A: heavy-DCA", C=160_000, S=180_000, T_yrs=5, S2=30_000,
         wealth_X=3_000_000),
    dict(name="B: light-DCA", C=1_000_000, S=30_000, T_yrs=30, S2=30_000,
         wealth_X=10_000_000),
    dict(name="C: lump-sum",  C=1_000_000, S=0, T_yrs=30, S2=0,
         wealth_X=float("inf")),
]
HORIZON_YR = 30
PCTILES = [10, 50, 90]


def project_metrics(ret, tsy, cpi, avail, T, alloc, C, S, T_yrs, S2,
                    max_days, h_days, F, wealth_X, broker_bump_days,
                    is_bootstrap):
    h, s, u = alloc
    real_eq, called, peak_lev = simulate_3way(
        ret, tsy, cpi, T, h, s, u, C, S, T_yrs, S2, max_days,
        avail, F, 1.0, wealth_X, broker_bump_days)
    # Mark called paths' wealth as 0 (margin call → wipe in conservative model)
    for k in range(real_eq.shape[0]):
        if called[k]:
            real_eq[k, :avail[k] + 1] = np.where(
                np.isnan(real_eq[k, :avail[k] + 1]),
                0.0, real_eq[k, :avail[k] + 1])
    # Restrict to paths with at least h_days of avail (for hist) — bootstrap
    # paths always have full horizon
    eligible = avail >= h_days
    vals = real_eq[eligible, h_days]
    valid = ~np.isnan(vals)
    v = vals[valid]
    if len(v) == 0:
        pct = [float("nan")] * len(PCTILES)
        return float("nan"), float("nan"), float("nan"), pct, float("nan"), float("nan")
    pct = [float(x) for x in np.nanpercentile(v, PCTILES)]
    min_w = float(v.min())
    frac_below_C = float((v < C).mean())
    cr = float(called[eligible].mean())
    pl = peak_lev[~called]
    pl_p99 = float(np.percentile(pl, 99)) if len(pl) > 0 else float("nan")
    return cr, min_w, frac_below_C, pct, pl_p99, float("nan")


def main():
    print("Loading + building paths...")
    dates, px, tsy, mrate, cpi = load(with_cpi=True)
    max_days = HORIZON_YR * TD
    ret_h, tsy_h, cpi_h, avail_h, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    rng = np.random.default_rng(42)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, N_BOOT_PATHS, BLOCK_DAYS, rng)
    avail_b = np.full(ret_b.shape[0], max_days, dtype=np.int64)
    ret_s = stretch_returns(ret_h, STRETCH_F) if STRETCH_F > 1.0 else None
    print(f"  hist paths: {ret_h.shape[0]}, boot paths: {ret_b.shape[0]}, "
          f"stretch_F={STRETCH_F}")
    print()

    h_days = HORIZON_YR * TD

    # Pre-compile JIT
    _ = simulate_3way(ret_h[:2], tsy_h[:2], cpi_h[:2],
                       1.5, 0.5, 0.25, 0.25,
                       100_000.0, 0.0, 5.0, 0.0, 252,
                       np.array([252, 252], dtype=np.int64),
                       1.5, 1.0, 1e18, 0)

    for scen in SCENARIOS:
        wxt = scen["wealth_X"]
        wxt_str = f"${wxt/1e6:.1f}M" if wxt != float("inf") else "none"
        print("=" * 130)
        print(f"SCENARIO {scen['name']}: C=${scen['C']/1e3:.0f}k  "
              f"S=${scen['S']/1e3:.0f}k×{scen['T_yrs']}y  S2=${scen['S2']/1e3:.0f}k  "
              f"wealth_X={wxt_str}  horizon={HORIZON_YR}y")
        print("=" * 130)
        print(f"  {'h/s/u':>11}  {'T_hyb':>6}  {'pkLv99':>7}  "
              f"{'h_call':>7}  {'b_call':>7}  "
              f"{'min_b':>6}  {'<C%_b':>5}  "
              f"{'p10_h':>7}  {'p50_h':>7}  {'p90_h':>7}  "
              f"{'p10_b':>7}  {'p50_b':>7}  {'p90_b':>7}")
        print("  " + "-" * 122)

        results = []
        for alloc in ALLOCS:
            h, s, u = alloc
            # Calibrate T (only matters if h > 0)
            if h > 0:
                T_b = find_safe_T_3way(
                    ret_b, tsy_b, cpi_b, h, s, u, scen["C"], scen["S"],
                    scen["T_yrs"], scen["S2"], max_days, avail_b,
                    F_DEFAULT, 1.0, wxt, 0, BOOT_TARGET)
                T_h = find_safe_T_3way(
                    ret_h, tsy_h, cpi_h, h, s, u, scen["C"], scen["S"],
                    scen["T_yrs"], scen["S2"], max_days, avail_h,
                    F_DEFAULT, 1.0, wxt, 0, 0.0)
                if ret_s is not None:
                    T_st = find_safe_T_3way(
                        ret_s, tsy_h, cpi_h, h, s, u, scen["C"], scen["S"],
                        scen["T_yrs"], scen["S2"], max_days, avail_h,
                        F_DEFAULT, 1.0, wxt, 0, 0.0)
                else:
                    T_st = float("inf")
                T = float(min(T_b, T_h, T_st))
            else:
                T = 1.0

            # Project on hist + boot
            cr_h, min_h, fbc_h, pct_h, pl_p99, _ = project_metrics(
                ret_h, tsy_h, cpi_h, avail_h, T, alloc, scen["C"], scen["S"],
                scen["T_yrs"], scen["S2"], max_days, h_days, F_DEFAULT,
                wxt, 0, False)
            cr_b, min_b, fbc_b, pct_b, _, _ = project_metrics(
                ret_b, tsy_b, cpi_b, avail_b, T, alloc, scen["C"], scen["S"],
                scen["T_yrs"], scen["S2"], max_days, h_days, F_DEFAULT,
                wxt, 0, True)
            p10h, p50h, p90h = pct_h
            p10b, p50b, p90b = pct_b

            label = f"{int(h*100):2d}/{int(s*100):2d}/{int(u*100):2d}"
            print(f"  {label:>11}  {T:>5.2f}x  {pl_p99:>5.2f}x  "
                  f"{cr_h*100:>5.2f}%  {cr_b*100:>5.2f}%  "
                  f"${min_b/1e6:>4.2f}M  {fbc_b*100:>4.1f}%  "
                  f"${p10h/1e6:>5.2f}M  ${p50h/1e6:>5.2f}M  ${p90h/1e6:>5.2f}M  "
                  f"${p10b/1e6:>5.2f}M  ${p50b/1e6:>5.2f}M  ${p90b/1e6:>5.2f}M")

            results.append(dict(
                alloc=alloc, T=T, peak_lev_p99=pl_p99,
                hist_call=cr_h, boot_call=cr_b,
                min_boot=min_b, frac_below_C_boot=fbc_b,
                p10_h=p10h, p50_h=p50h, p90_h=p90h,
                p10_b=p10b, p50_b=p50b, p90_b=p90b))

        # Identify Pareto-optimal allocations (better p10 AND p50 AND p90 than another)
        pareto = []
        for i, r in enumerate(results):
            dominated = False
            for j, r2 in enumerate(results):
                if i == j:
                    continue
                if (r2["p10_h"] >= r["p10_h"] and
                    r2["p50_h"] >= r["p50_h"] and
                    r2["p90_h"] >= r["p90_h"] and
                    (r2["p10_h"] > r["p10_h"] or
                     r2["p50_h"] > r["p50_h"] or
                     r2["p90_h"] > r["p90_h"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r["alloc"])
        print(f"\n  Pareto-optimal on (p10/p50/p90 hist): "
              f"{[(int(a[0]*100), int(a[1]*100), int(a[2]*100)) for a in pareto]}")

        # Best by various criteria (using historical p-values; bootstrap for tail)
        best_p50 = max(results, key=lambda r: r["p50_h"])
        best_p10 = max(results, key=lambda r: r["p10_h"])
        best_minb = max(results, key=lambda r: r["min_boot"])
        best_fbc = min(results, key=lambda r: r["frac_below_C_boot"])
        print(f"  Best p50 (hist):       "
              f"{best_p50['alloc']}  T={best_p50['T']:.2f}  ${best_p50['p50_h']/1e6:.2f}M")
        print(f"  Best p10 (hist):       "
              f"{best_p10['alloc']}  T={best_p10['T']:.2f}  ${best_p10['p10_h']/1e6:.2f}M")
        print(f"  Best min wealth (boot):"
              f"{best_minb['alloc']}  T={best_minb['T']:.2f}  ${best_minb['min_boot']/1e6:.2f}M")
        print(f"  Best <C% (boot):       "
              f"{best_fbc['alloc']}  T={best_fbc['T']:.2f}  {best_fbc['frac_below_C_boot']*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
