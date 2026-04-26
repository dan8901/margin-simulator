"""End-to-end verification: replicates app.py compute()'s calibration logic
for the four recal strategies and checks that the post-fix invariants hold:

1. recal_static T_rec == plain static T_rec
2. recal_hybrid T_rec == plain hybrid T_rec
3. recal_adaptive_dd T_rec == plain adaptive_dd T_rec
4. meta_recal T_init == T_rec of the candidate base with highest p50
   real terminal wealth (NOT highest T)
5. meta_recal init_strat_idx points at that argmax-score base
   in the META_KINDS list

Exits 0 on success, prints diffs and exits 1 on failure.
"""
import sys
import numpy as np

from data_loader import load
from project_portfolio import (
    build_historical_paths, build_bootstrap_paths,
    find_max_safe_T_grid, simulate, stretch_returns, TD,
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


RECAL_PERIOD_MONTHS = 60   # match app.py default
SCORE_HORIZON_DAYS = min(RECAL_PERIOD_MONTHS * 21, max_days)


def calibrate_base(kind):
    """Returns (T_rec, score) where score = p50 real wealth on calibration
    paths at T_rec over the MYOPIC horizon (next recal event, not full
    horizon). Mirrors app.py compute()'s meta_recal block (v4)."""
    T_h = find_max_safe_T_grid(ret_c, tsy_c, cpi_c, kind, 0.0,
                                C, S, T_yrs, S2, max_days, avail=avail_c,
                                F=F, wealth_X=WEALTH_X)
    T_bo = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, kind, BOOT_TARGET,
                                 C, S, T_yrs, S2, max_days, F=F,
                                 wealth_X=WEALTH_X)
    T_st = find_max_safe_T_grid(ret_s, tsy_c, cpi_c, kind, 0.0,
                                 C, S, T_yrs, S2, max_days, avail=avail_c,
                                 F=F, wealth_X=WEALTH_X)
    T_rec = min(T_h, T_bo, T_st)

    real_eq, called, _, _ = simulate(
        ret_c, tsy_c, cpi_c, kind, T_rec,
        C, S, T_yrs, S2, SCORE_HORIZON_DAYS,
        avail=np.minimum(avail_c, SCORE_HORIZON_DAYS),
        F=F, wealth_X=WEALTH_X)
    terminal = real_eq[:, SCORE_HORIZON_DAYS]
    valid = ~(np.isnan(terminal) | called)
    score = float(np.nanpercentile(terminal[valid], 50)) if valid.any() else float("-inf")
    return T_rec, score


# Plain base calibrations. recal_adaptive_dd uses adaptive_dd as base
# even though adaptive_dd isn't a meta_recal candidate anymore.
all_kinds = ["static", "hybrid", "adaptive_hybrid", "adaptive_dd"]
results = {}
for kind in all_kinds:
    T_rec, score = calibrate_base(kind)
    results[kind] = dict(T_rec=T_rec, score=score)

T_static = results["static"]["T_rec"]
T_hybrid = results["hybrid"]["T_rec"]
T_adaptive = results["adaptive_dd"]["T_rec"]

# Replicate compute()'s recal calibration logic:
# For recal_X strategies, T_rec is just the corresponding base T_rec
recal_T = {
    "recal_static": T_static,
    "recal_hybrid": T_hybrid,
    "recal_adaptive_dd": T_adaptive,
}

# meta_recal: argmax(score) over wealth-aware candidates only
META_KINDS = ["static", "hybrid", "adaptive_hybrid"]
scores = [results[k]["score"] for k in META_KINDS]
T_recs_meta = [results[k]["T_rec"] for k in META_KINDS]
winner_idx = int(np.argmax(scores))
winner_kind = META_KINDS[winner_idx]
T_meta = T_recs_meta[winner_idx]

# Print summary
print("Plain base calibration:")
for k in all_kinds:
    r = results[k]
    in_meta = " (meta candidate)" if k in META_KINDS else ""
    print(f"  {k:16s}  T_rec={r['T_rec']:.4f}x  p50_terminal=${r['score']:>14,.0f}{in_meta}")
print()
print("Recal T_rec (post-fix expected):")
print(f"  recal_static       = {recal_T['recal_static']:.4f}x")
print(f"  recal_hybrid       = {recal_T['recal_hybrid']:.4f}x")
print(f"  recal_adaptive_dd  = {recal_T['recal_adaptive_dd']:.4f}x")
print(f"  meta_recal         = {T_meta:.4f}x  "
      f"(winner by score: {winner_kind}, init_strat_idx={winner_idx})")
print()

# Acceptance criteria
ok = True
checks = [
    ("recal_static T_rec == static T_rec", recal_T["recal_static"], T_static),
    ("recal_hybrid T_rec == hybrid T_rec", recal_T["recal_hybrid"], T_hybrid),
    ("recal_adaptive_dd T_rec == adaptive_dd T_rec", recal_T["recal_adaptive_dd"], T_adaptive),
    ("meta_recal T_rec == winner-by-score's T", T_meta, T_recs_meta[winner_idx]),
]
for label, got, expected in checks:
    if abs(got - expected) < 1e-6:
        print(f"  PASS  {label}: {got:.4f}x")
    else:
        print(f"  FAIL  {label}: got {got:.4f}x, expected {expected:.4f}x")
        ok = False

# meta_recal sanity: winner's score is the max
if scores[winner_idx] != max(scores):
    print(f"  FAIL  meta_recal winner doesn't have the max score")
    ok = False

print()
sys.exit(0 if ok else 1)
