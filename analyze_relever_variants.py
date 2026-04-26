"""
Compare multiple re-lever strategies at different DCA levels.

Variants:
  - static:          one-time loan, never rebalance
  - monthly:         relever to target at each monthly DCA
  - quarterly:       relever to target every 3 months
  - annual:          relever to target once a year
  - band-0.90:       relever monthly, only when leverage < 0.90 * target
  - band-0.80:       relever monthly, only when leverage < 0.80 * target
  - decay-2pp:       monthly relever, target decays 2pp/yr (e.g., 1.50→1.30 over 10y)
  - decay-3pp:       target decays 3pp/yr
  - cap-2x-loan:     monthly relever, but total cumulative loan capped at 2×initial

For each (strategy, DCA) we:
  (a) Binary-search for max-safe target L (zero calls at 20y horizon)
  (b) Compute median IRR at that max-safe target (both 20y and 30y)
  (c) Also report at several common targets for comparison
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
             rebalance_freq="none",  # 'none', 'monthly', 'quarterly', 'annual'
             band_lower=1.0,          # relever only if current_lev < band_lower*target
             decay_per_year=0.0,      # target decreases linearly by this per year
             cap_total_loan=None):    # max cumulative loan as fraction of initial equity
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0

    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    spx = np.full(N, target_initial, dtype=float)
    loan = np.full(N, target_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)
    cum_loan = np.full(N, target_initial - 1.0, dtype=float)

    if rebalance_freq == "monthly":
        reb_period = DAYS_PER_MONTH
    elif rebalance_freq == "quarterly":
        reb_period = 3 * DAYS_PER_MONTH
    elif rebalance_freq == "annual":
        reb_period = 12 * DAYS_PER_MONTH
    else:
        reb_period = 0  # no rebalancing

    for k in range(1, H + 1):
        spx_g = px[idxs + k] / px[idxs + k - 1]
        box_g = M_box[idxs + k] / M_box[idxs + k - 1]
        spx = spx * spx_g
        loan = loan * box_g

        if k % DAYS_PER_MONTH == 0:
            active = ~called
            spx = np.where(active, spx + monthly, spx)

        if reb_period > 0 and k % reb_period == 0 and k > 0:
            years_elapsed = k / TRADING_DAYS
            current_target = max(target_initial - decay_per_year * years_elapsed, 1.0)
            equity = spx - loan
            pos = equity > 0
            cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
            # Only relever if current lev < band_lower * current_target
            should_relever = (cur_lev < band_lower * current_target) & (~called) & pos
            # Compute delta_D = target * E - A, but only apply where should_relever
            delta_D = current_target * equity - spx
            delta_D = np.maximum(delta_D, 0.0)
            # Cap on cumulative loan
            if cap_total_loan is not None:
                room = np.maximum(cap_total_loan - cum_loan, 0.0)
                delta_D = np.minimum(delta_D, room)
            loan = np.where(should_relever, loan + delta_D, loan)
            spx = np.where(should_relever, spx + delta_D, spx)
            cum_loan = np.where(should_relever, cum_loan + delta_D, cum_loan)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls

    terminal = np.where(called, 0.0, spx - loan)
    return {"terminal": terminal, "called": called, "cum_loan": cum_loan}


def find_max_safe(annual_dca, horizon_years, **strategy_kwargs):
    lo, hi = 1.01, 3.50
    for _ in range(16):
        mid = (lo + hi) / 2
        r = simulate(mid, annual_dca, horizon_years, **strategy_kwargs)
        if r["called"].mean() <= 0.0:
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
        return brentq(f, -0.5, 1.5, xtol=1e-7)
    except ValueError:
        return np.nan


# =================================================================
# Part A: Max-safe target by (strategy, DCA) at 20y horizon
# =================================================================
strategies = [
    ("static",              {"rebalance_freq": "none"}),
    ("monthly",             {"rebalance_freq": "monthly"}),
    ("quarterly",           {"rebalance_freq": "quarterly"}),
    ("annual",              {"rebalance_freq": "annual"}),
    ("monthly band=0.90",   {"rebalance_freq": "monthly", "band_lower": 0.90}),
    ("monthly band=0.80",   {"rebalance_freq": "monthly", "band_lower": 0.80}),
    ("monthly decay 2pp",   {"rebalance_freq": "monthly", "decay_per_year": 0.02}),
    ("monthly decay 3pp",   {"rebalance_freq": "monthly", "decay_per_year": 0.03}),
    ("monthly cap=2x loan", {"rebalance_freq": "monthly", "cap_total_loan": 2.0}),
]
DCAs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

print("=" * 115)
print("Max-safe starting TARGET leverage (zero calls at 20y horizon, post-1932)")
print("=" * 115)
header = f"{'Strategy':<22}" + "".join(f"{('DCA='+str(int(d*100))+'%'):>11}" for d in DCAs)
print(header)

max_safe_table = {}
for strat_name, kwargs in strategies:
    row = [f"{strat_name:<22}"]
    for dca in DCAs:
        ms = find_max_safe(dca, 20, **kwargs)
        max_safe_table[(strat_name, dca)] = ms
        row.append(f"{ms:>9.2f}x")
    print("  ".join(row))

# =================================================================
# Part B: Median IRR at each strategy's max-safe target, 20y and 30y
# =================================================================
for horizon_years in (20, 30):
    print("\n" + "=" * 115)
    print(f"Median IRR at max-safe target (sized at 20y-safe level), horizon = {horizon_years}y")
    print("=" * 115)
    header = f"{'Strategy':<22}" + "".join(f"{('DCA='+str(int(d*100))+'%'):>11}" for d in DCAs)
    print(header)
    for strat_name, kwargs in strategies:
        row = [f"{strat_name:<22}"]
        for dca in DCAs:
            ms = max_safe_table[(strat_name, dca)]
            r = simulate(ms, dca, horizon_years, **kwargs)
            valid = r["terminal"] > 0
            if valid.sum() == 0:
                row.append("  —  ")
                continue
            med_t = np.median(r["terminal"][valid])
            ir = irr(med_t, horizon_years, dca) * 100
            # Mark if call rate non-zero at this horizon
            cr = r["called"].mean() * 100
            suffix = "" if cr < 0.01 else "!"
            row.append(f"{ir:>8.2f}%{suffix}")
        print("  ".join(row))

# =================================================================
# Part C: IRR at COMMON target 1.50x across strategies
# =================================================================
for horizon_years in (20, 30):
    print("\n" + "=" * 115)
    print(f"IRR at FIXED target = 1.50x across strategies, horizon = {horizon_years}y")
    print(f"(call rate in parentheses)")
    print("=" * 115)
    header = f"{'Strategy':<22}" + "".join(f"{('DCA='+str(int(d*100))+'%'):>16}" for d in DCAs)
    print(header)
    for strat_name, kwargs in strategies:
        row = [f"{strat_name:<22}"]
        for dca in DCAs:
            r = simulate(1.50, dca, horizon_years, **kwargs)
            valid = r["terminal"] > 0
            if valid.sum() == 0:
                row.append("  all-called  ")
                continue
            med_t = np.median(r["terminal"][valid])
            ir = irr(med_t, horizon_years, dca) * 100
            cr = r["called"].mean() * 100
            row.append(f"{ir:>6.2f}% ({cr:4.1f}%)")
        print("  ".join(row))

# =================================================================
# Part D: Comparison at 1.75x target
# =================================================================
for horizon_years in (20, 30):
    print("\n" + "=" * 115)
    print(f"IRR at FIXED target = 1.75x across strategies, horizon = {horizon_years}y")
    print(f"(call rate in parentheses)")
    print("=" * 115)
    header = f"{'Strategy':<22}" + "".join(f"{('DCA='+str(int(d*100))+'%'):>16}" for d in DCAs)
    print(header)
    for strat_name, kwargs in strategies:
        row = [f"{strat_name:<22}"]
        for dca in DCAs:
            r = simulate(1.75, dca, horizon_years, **kwargs)
            valid = r["terminal"] > 0
            if valid.sum() == 0:
                row.append("  all-called  ")
                continue
            med_t = np.median(r["terminal"][valid])
            ir = irr(med_t, horizon_years, dca) * 100
            cr = r["called"].mean() * 100
            row.append(f"{ir:>6.2f}% ({cr:4.1f}%)")
        print("  ".join(row))
