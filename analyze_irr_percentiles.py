"""
For requested strategies, show per-path IRR percentiles across DCA levels.

Strategies:
  (A) Static at DUAL-horizon max-safe (one-time loan, no rebalancing)
  (B) Monthly relever at DUAL-horizon max-safe
  (C) Annual relever at DUAL-horizon max-safe
  (D) Monthly relever with target decay 2pp/yr, at DUAL-horizon max-safe

DUAL-HORIZON MAX-SAFE: largest T_initial with 0% calls at BOTH 20y and 30y.
This prevents the failure mode where a target sized for 20y safety actually
has call risk at 30y (affects monthly relever in particular).

For each (strategy, DCA), simulate across all post-1932 entries,
compute IRR for each path, report percentiles (p10, p25, p50, p75, p90)
at both 20y and 30y horizons using the SAME dual-safe target.
"""
import numpy as np
from datetime import datetime
from scipy.optimize import brentq
from data_loader import load

dates, px, tsy, mrate = load()
TRADING_DAYS = 252
DAYS_PER_MONTH = 21

M_box = np.concatenate(([1.0], np.cumprod(1 + (tsy + 0.0015)[1:] / TRADING_DAYS)))
post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate(target_initial, annual_dca, horizon_years,
             rebalance_freq="none", band_lower=1.0,
             decay_per_year=0.0):
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    spx = np.full(N, target_initial, dtype=float)
    loan = np.full(N, target_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)

    reb_period = {"none": 0, "monthly": DAYS_PER_MONTH,
                  "quarterly": 3*DAYS_PER_MONTH,
                  "annual": 12*DAYS_PER_MONTH}[rebalance_freq]

    for k in range(1, H + 1):
        spx_g = px[idxs + k] / px[idxs + k - 1]
        box_g = M_box[idxs + k] / M_box[idxs + k - 1]
        spx = spx * spx_g
        loan = loan * box_g

        if k % DAYS_PER_MONTH == 0:
            spx = np.where(~called, spx + monthly, spx)

        if reb_period > 0 and k % reb_period == 0 and k > 0:
            years = k / TRADING_DAYS
            current_target = max(target_initial - decay_per_year * years, 1.0)
            equity = spx - loan
            pos = equity > 0
            cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
            want = (cur_lev < band_lower * current_target) & (~called) & pos
            delta_D = np.maximum(current_target * equity - spx, 0.0)
            loan = np.where(want, loan + delta_D, loan)
            spx = np.where(want, spx + delta_D, spx)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls

    terminal = np.where(called, 0.0, spx - loan)
    return terminal, called


def find_max_safe_dual(annual_dca, horizons=(20, 30), **kwargs):
    """Largest T_initial with 0% calls at ALL given horizons."""
    lo, hi = 1.01, 3.50
    for _ in range(16):
        mid = (lo + hi) / 2
        safe = True
        for H in horizons:
            _, c = simulate(mid, annual_dca, H, **kwargs)
            if c.mean() > 0.0:
                safe = False
                break
        if safe:
            lo = mid
        else:
            hi = mid
    return lo


def irr(terminal, H, annual_dca):
    if terminal <= 0:
        return np.nan
    m = annual_dca / 12.0
    M = int(H * 12)
    def f(r):
        if abs(r) < 1e-10:
            return 1 + m * M - terminal
        d = (1 + r) ** (1/12)
        return (1+r)**H + m * (d**M - 1) / (d - 1) - terminal
    try:
        return brentq(f, -0.99, 2.0, xtol=1e-7)
    except (ValueError, RuntimeError):
        return np.nan


def per_path_irrs(terminal, H, dca):
    out = np.full(len(terminal), np.nan)
    for i, t in enumerate(terminal):
        if t > 0:
            out[i] = irr(t, H, dca)
    return out


DCAs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

strategies = [
    ("static @ max-safe",   "find",        {"rebalance_freq": "none"}),
    ("monthly relever",     "find",        {"rebalance_freq": "monthly"}),
    ("annual  relever",     "find",        {"rebalance_freq": "annual"}),
    ("decay 2pp monthly",   "find",        {"rebalance_freq": "monthly", "decay_per_year": 0.02}),
]


# Pre-compute dual-horizon max-safe target for each (strategy, dca) once.
print("Computing dual-horizon max-safe targets (0% calls at both 20y and 30y)...")
dual_safe = {}  # (strat_name, dca) -> target
for dca in DCAs:
    for strat_name, _sizing, kwargs in strategies:
        t = find_max_safe_dual(dca, **kwargs)
        dual_safe[(strat_name, dca)] = t
        print(f"  {strat_name:<22} DCA={dca*100:>3.0f}%  T={t:.3f}x")


def run_horizon(H_YEARS):
    print("\n" + "=" * 118)
    print(f"HORIZON = {H_YEARS} YEARS.  IRR percentiles across post-1932 entries.")
    print("Targets are dual-horizon max-safe (0% calls at both 20y and 30y).")
    print("=" * 118)
    header = (f"{'Strategy':<22}  {'DCA':>4}  {'Target':>6}  "
              f"{'p10':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}  "
              f"{'mean':>7}  {'call%':>6}")
    for dca in DCAs:
        print("\n" + header)
        for strat_name, _sizing, kwargs in strategies:
            target = dual_safe[(strat_name, dca)]
            term, called = simulate(target, dca, H_YEARS, **kwargs)
            irrs = per_path_irrs(term, H_YEARS, dca)
            valid = ~np.isnan(irrs)
            if valid.sum() == 0:
                row = f"{strat_name:<22}  {dca*100:>3.0f}%  {target:>5.3f}x  (all called)"
            else:
                irrs_v = irrs[valid]
                p10 = np.percentile(irrs_v, 10) * 100
                p25 = np.percentile(irrs_v, 25) * 100
                p50 = np.percentile(irrs_v, 50) * 100
                p75 = np.percentile(irrs_v, 75) * 100
                p90 = np.percentile(irrs_v, 90) * 100
                mean = irrs_v.mean() * 100
                cr = called.mean() * 100
                row = (f"{strat_name:<22}  {dca*100:>3.0f}%  {target:>5.3f}x  "
                       f"{p10:>6.2f}%  {p25:>6.2f}%  {p50:>6.2f}%  "
                       f"{p75:>6.2f}%  {p90:>6.2f}%  {mean:>6.2f}%  {cr:>5.2f}%")
            print(row)


run_horizon(20)
run_horizon(30)
