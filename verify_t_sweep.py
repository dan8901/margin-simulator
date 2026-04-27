"""Verify compute_t_sweep returns sensible values across strategies and T values.

Run with: .venv/bin/python -u verify_t_sweep.py

Prints per-strategy, per-T_val:
- T_rec
- p10/p50/p90 real wealth at year 30 across historical paths

Sanity expectations:
- 'unlev' T_rec is 1.0x at every T_val.
- p50 real wealth at year 30 is monotonically non-decreasing in T_val for
  unlev (more cumulative DCA -> more terminal wealth, holding everything
  else equal). For leveraged strategies, monotonicity holds approximately
  modulo small re-calibration noise; we allow 1% drops.
"""
import sys
import numpy as np

from data_loader import load
from project_portfolio import (
    build_historical_paths, build_bootstrap_paths, TD,
)
from app import compute_t_sweep

# Primary scenario - keep in sync with app.py defaults
C = 160_000.0
S = 180_000.0
S2 = 30_000.0
HORIZON_YEARS = 30
F = 1.5
WEALTH_X = 3_000_000.0
BOOT_TARGET_PCT = 1.0
STRETCH_F = 1.1
N_BOOT = 500
RECAL_PERIOD_MONTHS = 60
BROKER_BUMP_YEARS = 2
CHECKPOINTS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

max_days = HORIZON_YEARS * TD
broker_bump_days = int(BROKER_BUMP_YEARS * TD)

print(f"Scenario: C={C:,.0f}  S={S:,.0f}/yr*Tx  S2={S2:,.0f}/yr  "
      f"X={WEALTH_X:,.0f}  H={HORIZON_YEARS}y  F={F}  stretch={STRETCH_F}  "
      f"boot_target={BOOT_TARGET_PCT}%  N_boot={N_BOOT}  "
      f"recal_period={RECAL_PERIOD_MONTHS}mo  broker_bump={BROKER_BUMP_YEARS}y")

dates, px, tsy, mrate, cpi = load(with_cpi=True)
ret_c, tsy_c, cpi_c, avail_c, _ = build_historical_paths(
    dates, px, tsy, cpi, max_days, min_days=max_days)
ret_h, tsy_h, cpi_h, avail_h, entry_dates_h = build_historical_paths(
    dates, px, tsy, cpi, max_days, min_days=TD)
rng = np.random.default_rng(42)
ret_b, tsy_b, cpi_b = build_bootstrap_paths(
    dates, px, tsy, cpi, max_days, n_paths=N_BOOT,
    block_days=int(1.0 * TD), rng=rng)

paths = dict(
    calib=(ret_c, tsy_c, cpi_c, avail_c),
    proj=(ret_h, tsy_h, cpi_h, avail_h, entry_dates_h),
    boot=(ret_b, tsy_b, cpi_b),
)
paths_key = (max_days, N_BOOT, 1, 42)

# (kind, F, t_values). meta_recal is slowest so use only 2 T values for it.
TEST_STRATEGIES = [
    ("unlev", 1.5, (0, 5, 10, 15)),
    ("static", 1.5, (0, 5, 10, 15)),
    ("hybrid", 1.5, (0, 5, 10, 15)),
    ("recal_hybrid", 1.5, (0, 5, 10, 15)),
    ("meta_recal", 1.5, (0, 10)),
]

print(f"\nRunning compute_t_sweep for {len(TEST_STRATEGIES)} strategies "
      f"(this is slow; meta_recal is ~30-60s per T value):")

failed = False
for kind, F_strat, t_values in TEST_STRATEGIES:
    print(f"\n  {kind} (F={F_strat}, T_values={t_values})...", flush=True)
    out = compute_t_sweep(
        C, S, S2, max_days, CHECKPOINTS,
        kind, F_strat,
        BOOT_TARGET_PCT / 100.0, STRETCH_F, F_strat,
        float("inf"), WEALTH_X,
        # overlay defaults (no vol/dip/rate hybrids tested here)
        0.0, 0.20, 0.30, 0.05, 5.0,
        RECAL_PERIOD_MONTHS, 1.0, broker_bump_days,
        paths_key, paths,
        t_values=t_values,
    )
    p50_30y_prev = -float("inf")
    for T_val in t_values:
        r = out[T_val]
        per_cp_30 = r["per_cp"].get(30.0)
        if per_cp_30 is None:
            print(f"    T={T_val:2d}: T_rec={r['T_rec']:.3f}x  (no 30y data)")
            continue
        real = per_cp_30["real"]
        p10, p50, p90 = np.percentile(real, [10, 50, 90])
        print(f"    T={T_val:2d}: T_rec={r['T_rec']:.3f}x  "
              f"p10=${p10/1e6:5.2f}M p50=${p50/1e6:5.2f}M p90=${p90/1e6:5.2f}M  "
              f"(n={len(real)})")
        if p50 < p50_30y_prev * 0.99:
            print(f"      WARN: p50 dropped >1% from prev T (was ${p50_30y_prev/1e6:.2f}M)")
            failed = True
        p50_30y_prev = p50
    if kind == "unlev":
        for T_val in t_values:
            if out[T_val]["T_rec"] != 1.0:
                print(f"      FAIL: unlev T_rec at T={T_val} is {out[T_val]['T_rec']}, expected 1.0")
                failed = True

if failed:
    print("\nFAILED — see warnings above")
    sys.exit(1)
print("\nOK — all sanity checks passed")
