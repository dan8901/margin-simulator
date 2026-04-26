"""Verify recal_X T_init calibration matches plain base-kind T_rec.

Run with: .venv/bin/python verify_recal_tinit.py

This script computes well-defended T_rec for plain {static, hybrid,
adaptive_dd, dd_decay} using the same calibration pattern as app.py
(historical 0% calls + bootstrap <= 1% + stretched 0% calls).

Post-fix expectation:
- recal_static    T_rec in the streamlit calibration table ~ T_static here
- recal_hybrid    T_rec ~ T_hybrid
- recal_adaptive_dd T_rec ~ T_adaptive
- meta_recal      T_rec ~ max(T_static, T_hybrid, T_adaptive, T_dd_decay)
- meta_recal      init_base = argmax(...) (one of static/dd_decay/adaptive_dd/hybrid)

Pre-fix: recal_X T_rec collapses to 1.0x (binary search bottoms out).
"""
import numpy as np

from data_loader import load
from project_portfolio import (
    build_historical_paths, build_bootstrap_paths,
    find_max_safe_T_grid, stretch_returns, TD,
)

# User's primary scenario - keep in sync with the streamlit defaults
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

print(f"Scenario: C={C:,.0f}  S={S:,.0f}/yr*{T_yrs:g}y  S2={S2:,.0f}/yr  "
      f"X={WEALTH_X:,.0f}  H={HORIZON_YEARS}y  F={F}  stretch={STRETCH_F}  "
      f"boot_target={BOOT_TARGET:.0%}  N_boot={N_BOOT}")

# Load data + paths (mirrors app.py:279-281)
dates, px, tsy, mrate, cpi = load(with_cpi=True)
ret_c, tsy_c, cpi_c, avail_c, _entry_dates = build_historical_paths(
    dates, px, tsy, cpi, max_days, min_days=max_days)

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
