"""
Well-defended max-safe: target satisfying BOTH safety constraints jointly.

Constraints (at 10% DCA):
  (A) Historical post-1932 dual-horizon (20y AND 30y) call rate = 0% at
      cap = 3.0x ("broker tightens to 33% maintenance during stress").
  (B) Block-bootstrap synthetic 30y call rate <= 1.0% at cap = 4.0x
      (5000 paths, 252-day blocks). Tests for path-overfitting.

For each strategy:
  - Binary search for largest T satisfying both constraints simultaneously.
  - Report well-defended T, p10/p50 IRR @ 30y, p50 end-of-horizon leverage.
  - Compare to historical (cap=4.0x) max-safe to show the haircut.

Then: block-size sensitivity for one strategy across {63, 126, 252, 504, 1260}.

The drawdown-stress (F=1.2) buffer is INTENTIONALLY EXCLUDED — bootstrap is
the empirical version of "fat tails / future drawdowns deeper than past,"
and stacking F=1.2 on top is double-counting.
"""
import numpy as np
from datetime import datetime
from scipy.optimize import brentq
from data_loader import load

dates, px_orig, tsy, mrate = load()
TRADING_DAYS = 252
DAYS_PER_MONTH = 21
M_box = np.concatenate(([1.0], np.cumprod(1 + (tsy + 0.0015)[1:] / TRADING_DAYS)))
post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])

log_ret = np.log(px_orig[1:] / px_orig[:-1])
rate_growth = M_box[1:] / M_box[:-1]


# ----- Historical entries simulator -----
def simulate_historical(strategy, T_initial, annual_dca, horizon_years, cap,
                         decay_per_year=0.0, wealth_mult_to_floor=None,
                         floor=1.0):
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px_orig)]
    N = len(idxs)

    spx = np.full(N, T_initial, dtype=float)
    loan = np.full(N, T_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)
    max_equity = np.full(N, 1.0, dtype=float)

    if strategy == 'wealth_decay' and wealth_mult_to_floor is not None \
       and T_initial > floor and wealth_mult_to_floor > 1.0:
        slope_wd = (T_initial - floor) / (wealth_mult_to_floor - 1.0)
    else:
        slope_wd = 0.0

    for k in range(1, H + 1):
        spx = spx * (px_orig[idxs + k] / px_orig[idxs + k - 1])
        loan = loan * (M_box[idxs + k] / M_box[idxs + k - 1])

        if k % DAYS_PER_MONTH == 0:
            active = ~called
            spx = np.where(active, spx + monthly, spx)
            if strategy != 'static':
                equity = spx - loan
                pos = equity > 0
                if strategy == 'monthly':
                    current_target = np.full(N, T_initial)
                elif strategy == 'time_decay':
                    years = k / TRADING_DAYS
                    current_target = np.full(
                        N, max(T_initial - decay_per_year * years, floor))
                elif strategy == 'wealth_decay':
                    max_equity = np.maximum(
                        max_equity,
                        np.where(pos & active, equity, max_equity))
                    current_target = np.maximum(
                        T_initial - slope_wd * (max_equity - 1.0), floor)
                cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
                want = (cur_lev < current_target) & active & pos
                delta_D = np.maximum(current_target * equity - spx, 0.0)
                loan = np.where(want, loan + delta_D, loan)
                spx = np.where(want, spx + delta_D, spx)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= cap))
        called |= new_calls

    equity = spx - loan
    pos = equity > 0
    end_lev = np.where(pos & ~called, spx / np.maximum(equity, 1e-12), np.nan)
    terminal = np.where(called, 0.0, spx - loan)
    return end_lev, called, terminal


# ----- Bootstrap path generator and simulator -----
def make_bootstrap_paths(N_paths, H_days, block_size=252, seed=None):
    rng = np.random.default_rng(seed)
    post_lr = post1932[1:]
    candidate_starts = np.arange(len(log_ret) - block_size + 1)
    valid_starts = candidate_starts[post_lr[candidate_starts]]
    n_blocks = int(np.ceil(H_days / block_size))
    total_len = n_blocks * block_size
    out_lr = np.empty((N_paths, total_len))
    out_rg = np.empty((N_paths, total_len))
    arange_block = np.arange(block_size)
    for b in range(n_blocks):
        starts = rng.choice(valid_starts, N_paths)
        idx = starts[:, None] + arange_block
        out_lr[:, b*block_size:(b+1)*block_size] = log_ret[idx]
        out_rg[:, b*block_size:(b+1)*block_size] = rate_growth[idx]
    return out_lr[:, :H_days], out_rg[:, :H_days]


def simulate_bootstrap(lr_p, rg_p, strategy, T_initial, annual_dca, cap=4.0,
                       decay_per_year=0.0, wealth_mult_to_floor=None,
                       floor=1.0):
    N, H = lr_p.shape
    monthly = annual_dca / 12.0

    spx = np.full(N, T_initial, dtype=float)
    loan = np.full(N, T_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)
    max_equity = np.full(N, 1.0, dtype=float)

    if strategy == 'wealth_decay' and wealth_mult_to_floor is not None \
       and T_initial > floor and wealth_mult_to_floor > 1.0:
        slope_wd = (T_initial - floor) / (wealth_mult_to_floor - 1.0)
    else:
        slope_wd = 0.0

    spx_factors = np.exp(lr_p)

    for k in range(H):
        spx = spx * spx_factors[:, k]
        loan = loan * rg_p[:, k]

        if (k + 1) % DAYS_PER_MONTH == 0:
            active = ~called
            spx = np.where(active, spx + monthly, spx)
            if strategy != 'static':
                equity = spx - loan
                pos = equity > 0
                if strategy == 'monthly':
                    current_target = np.full(N, T_initial)
                elif strategy == 'time_decay':
                    years = (k + 1) / TRADING_DAYS
                    current_target = np.full(
                        N, max(T_initial - decay_per_year * years, floor))
                elif strategy == 'wealth_decay':
                    max_equity = np.maximum(
                        max_equity,
                        np.where(pos & active, equity, max_equity))
                    current_target = np.maximum(
                        T_initial - slope_wd * (max_equity - 1.0), floor)
                cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
                want = (cur_lev < current_target) & active & pos
                delta_D = np.maximum(current_target * equity - spx, 0.0)
                loan = np.where(want, loan + delta_D, loan)
                spx = np.where(want, spx + delta_D, spx)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= cap))
        called |= new_calls

    equity = spx - loan
    pos = equity > 0
    end_lev = np.where(pos & ~called, spx / np.maximum(equity, 1e-12), np.nan)
    terminal = np.where(called, 0.0, spx - loan)
    return end_lev, called, terminal


def irr_one(terminal, H, annual_dca):
    if terminal <= 0:
        return np.nan
    m = annual_dca / 12.0
    M = int(H * 12)
    def f(r):
        if abs(r) < 1e-10:
            return 1 + m * M - terminal
        d = (1 + r) ** (1 / 12)
        return (1 + r) ** H + m * (d ** M - 1) / (d - 1) - terminal
    try:
        return brentq(f, -0.5, 1.5, xtol=1e-7)
    except ValueError:
        return np.nan


def per_path_irrs(terminal, H, dca):
    out = np.full(len(terminal), np.nan)
    for i, t in enumerate(terminal):
        if t > 0:
            out[i] = irr_one(t, H, dca)
    return out


# ============================================================================
# Joint constraint binary search
# ============================================================================
def find_well_defended(strategy_kw, annual_dca, lr_p, rg_p,
                        bootstrap_threshold=0.01, hist_cap=3.0):
    """Largest T with hist 0% calls at hist_cap (20y AND 30y, cap=3.0)
    AND bootstrap call rate <= bootstrap_threshold (cap=4.0)."""
    lo, hi = 1.005, 2.50
    for _ in range(14):
        mid = (lo + hi) / 2

        cr_h20 = simulate_historical(annual_dca=annual_dca, T_initial=mid,
                                       horizon_years=20, cap=hist_cap,
                                       **strategy_kw)[1].mean()
        cr_h30 = simulate_historical(annual_dca=annual_dca, T_initial=mid,
                                       horizon_years=30, cap=hist_cap,
                                       **strategy_kw)[1].mean()
        cr_boot = simulate_bootstrap(lr_p, rg_p, T_initial=mid,
                                       annual_dca=annual_dca, cap=4.0,
                                       **strategy_kw)[1].mean()

        ok = (cr_h20 <= 0.0) and (cr_h30 <= 0.0) and (cr_boot <= bootstrap_threshold)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


DCA = 0.10
strategies = [
    ('static',           {'strategy': 'static'}),
    ('time-decay 2pp',   {'strategy': 'time_decay', 'decay_per_year': 0.02}),
    ('wealth-decay WM=20', {'strategy': 'wealth_decay',
                            'wealth_mult_to_floor': 20.0}),
    ('monthly relever',  {'strategy': 'monthly'}),
]

print("Generating bootstrap paths (5000 × 30y, 252-day blocks)...")
N_PATHS = 5000
H_DAYS = 30 * TRADING_DAYS
lr_p, rg_p = make_bootstrap_paths(N_PATHS, H_DAYS, block_size=252, seed=42)
print("  Done.\n")

print("=" * 115)
print(f"WELL-DEFENDED max-safe targets, DCA={DCA*100:.0f}%")
print(f"  Constraint A: historical 0% calls at cap=3.0x (both 20y and 30y)")
print(f"  Constraint B: bootstrap call rate ≤ 1% at cap=4.0x ({N_PATHS} synthetic 30y paths, 252-day blocks)")
print("=" * 115)
print(f"{'Strategy':<22}  {'Hist max-safe':>14}  {'Well-defended':>14}  {'haircut':>8}  "
      f"{'p50@30y':>8}  {'p10@30y':>8}  {'endlev@30y':>11}  {'boot call%':>11}")
print("-" * 115)

# Historical (cap=4.0) max-safe at 10% DCA, from prior runs
hist_max_safe = {
    'static':            1.612,
    'time-decay 2pp':    1.591,
    'wealth-decay WM=20': 1.562,
    'monthly relever':   1.429,
}

results = {}
for name, kw in strategies:
    T_def = find_well_defended(kw, DCA, lr_p, rg_p,
                                bootstrap_threshold=0.01, hist_cap=3.0)
    T_hist = hist_max_safe[name]
    haircut = T_def / T_hist

    # Eval at well-defended target
    end_lev, _, term = simulate_historical(
        annual_dca=DCA, T_initial=T_def, horizon_years=30, cap=4.0, **kw)
    irrs = per_path_irrs(term, 30, DCA) * 100
    valid = ~np.isnan(irrs)
    surv = end_lev[~np.isnan(end_lev)]
    p10 = np.percentile(irrs[valid], 10)
    p50 = np.percentile(irrs[valid], 50)
    p50_el = np.median(surv)

    _, called_b, _ = simulate_bootstrap(lr_p, rg_p, T_initial=T_def,
                                         annual_dca=DCA, cap=4.0, **kw)
    boot_cr = called_b.mean() * 100

    print(f"{name:<22}  {T_hist:>13.3f}x  {T_def:>13.3f}x  {haircut:>7.3f}×  "
          f"{p50:>7.2f}%  {p10:>7.2f}%  {p50_el:>10.2f}x  {boot_cr:>10.2f}%")
    results[name] = T_def


# ============================================================================
# Block-size sensitivity (one strategy)
# ============================================================================
print()
print("=" * 90)
print("Block-size sensitivity for time-decay 2pp at 10% DCA, T = 1.591x (hist max-safe)")
print(f"{N_PATHS} synthetic paths per block size, cap=4.0x.")
print("=" * 90)
print(f"  {'block size':>12} {'block years':>12}  {'boot call%':>12}  {'p50 IRR':>10}")
print("-" * 90)

block_sizes = [21, 63, 126, 252, 504, 1260]  # 1mo, 3mo, 6mo, 1y, 2y, 5y
T_test = 1.591
strat_test = {'strategy': 'time_decay', 'decay_per_year': 0.02}
for bs in block_sizes:
    lr_b, rg_b = make_bootstrap_paths(N_PATHS, H_DAYS, block_size=bs, seed=42)
    _, called_b, term_b = simulate_bootstrap(lr_b, rg_b, T_initial=T_test,
                                              annual_dca=DCA, cap=4.0,
                                              **strat_test)
    cr = called_b.mean() * 100
    irrs = per_path_irrs(term_b, 30, DCA) * 100
    valid = ~np.isnan(irrs)
    p50 = np.percentile(irrs[valid], 50) if valid.sum() > 0 else np.nan
    print(f"  {bs:>11}d {bs/TRADING_DAYS:>11.2f}y  {cr:>11.2f}%  {p50:>9.2f}%")
