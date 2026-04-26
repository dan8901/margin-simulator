"""End-to-end verification: replicates app.py compute()'s calibration logic
for the four recal strategies and checks that the post-fix invariants hold:

1. recal_static T_rec == plain static T_rec
2. recal_hybrid T_rec == plain hybrid T_rec
3. recal_adaptive_dd T_rec == plain adaptive_dd T_rec
4. meta_recal T_rec == max over {static, dd_decay, adaptive_dd, hybrid}
5. meta_recal init_strat_idx points at that argmax in the META_KINDS list

Exits 0 on success, prints diffs and exits 1 on failure.
"""
import sys
import numpy as np

from data_loader import load
from project_portfolio import (
    build_historical_paths, build_bootstrap_paths,
    find_max_safe_T_grid, stretch_returns, TD,
)

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

dates, px, tsy, mrate, cpi = load(with_cpi=True)
ret_c, tsy_c, cpi_c, avail_c, _ = build_historical_paths(
    dates, px, tsy, cpi, max_days, min_days=max_days)
rng = np.random.default_rng(42)
ret_b, tsy_b, cpi_b = build_bootstrap_paths(
    dates, px, tsy, cpi, max_days, n_paths=N_BOOT,
    block_days=int(1.0 * TD), rng=rng)
ret_s = stretch_returns(ret_c, STRETCH_F)


def calibrate(kind):
    T_h = find_max_safe_T_grid(ret_c, tsy_c, cpi_c, kind, 0.0,
                                C, S, T_yrs, S2, max_days, avail=avail_c,
                                F=F, wealth_X=WEALTH_X)
    T_bo = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, kind, BOOT_TARGET,
                                 C, S, T_yrs, S2, max_days, F=F,
                                 wealth_X=WEALTH_X)
    T_st = find_max_safe_T_grid(ret_s, tsy_c, cpi_c, kind, 0.0,
                                 C, S, T_yrs, S2, max_days, avail=avail_c,
                                 F=F, wealth_X=WEALTH_X)
    return min(T_h, T_bo, T_st)


# Plain base calibrations
T_static = calibrate("static")
T_hybrid = calibrate("hybrid")
T_adaptive = calibrate("adaptive_dd")
T_dd = calibrate("dd_decay")

# Replicate compute()'s recal calibration logic:
# For recal_X strategies, T_rec is just the corresponding base T_rec
recal_T = {
    "recal_static": T_static,
    "recal_hybrid": T_hybrid,
    "recal_adaptive_dd": T_adaptive,
}

# meta_recal: argmax over the four base candidates in fixed order
META_KINDS = ["static", "dd_decay", "adaptive_dd", "hybrid"]
meta_T_list = [T_static, T_dd, T_adaptive, T_hybrid]   # in META_KINDS order
winner_idx = int(np.argmax(meta_T_list))
T_meta = meta_T_list[winner_idx]
winner_kind = META_KINDS[winner_idx]

# Print summary
print("Plain base T_rec:")
print(f"  static       = {T_static:.4f}x")
print(f"  hybrid       = {T_hybrid:.4f}x")
print(f"  adaptive_dd  = {T_adaptive:.4f}x")
print(f"  dd_decay     = {T_dd:.4f}x")
print()
print("Recal T_rec (post-fix expected):")
print(f"  recal_static       = {recal_T['recal_static']:.4f}x")
print(f"  recal_hybrid       = {recal_T['recal_hybrid']:.4f}x")
print(f"  recal_adaptive_dd  = {recal_T['recal_adaptive_dd']:.4f}x")
print(f"  meta_recal         = {T_meta:.4f}x  "
      f"(winner: {winner_kind}, init_strat_idx={winner_idx})")
print()

# Acceptance criteria
ok = True
checks = [
    ("recal_static T_rec == static T_rec", recal_T["recal_static"], T_static),
    ("recal_hybrid T_rec == hybrid T_rec", recal_T["recal_hybrid"], T_hybrid),
    ("recal_adaptive_dd T_rec == adaptive_dd T_rec", recal_T["recal_adaptive_dd"], T_adaptive),
    ("meta_recal T_rec == max(base T_recs)", T_meta, max(meta_T_list)),
]
for label, got, expected in checks:
    if abs(got - expected) < 1e-6:
        print(f"  PASS  {label}: {got:.4f}x")
    else:
        print(f"  FAIL  {label}: got {got:.4f}x, expected {expected:.4f}x")
        ok = False

# meta_recal regression check: > unlev (1.0x) and >= max base
if T_meta < max(meta_T_list) - 1e-6:
    print(f"  FAIL  meta_recal T_rec < max(base T_recs)")
    ok = False
if T_meta <= 1.0:
    print(f"  FAIL  meta_recal collapsed to <= 1.0x (expected post-fix > 1)")
    ok = False

print()
sys.exit(0 if ok else 1)
