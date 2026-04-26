"""
Same as analyze_irr_percentiles.py, but with INFLATION-ADJUSTED DCA.
The user's monthly contribution grows at `annual_dca_growth` per year
(default 2.5% for inflation). All else identical.

IRR equation with growing contributions:
  terminal = (1+r)^H + m_0 * (1+r)^(H - 1/12) * (1 - x^M) / (1 - x)
  where x = ((1+g)/(1+r))^(1/12), M = 12*H, m_0 = annual_dca/12, g = growth.

For each (strategy, DCA, horizon):
  - Find max-safe target for THIS DCA-growth assumption
  - Simulate, compute per-path IRR, report percentiles.
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

DCA_GROWTH = 0.025  # 2.5% / year nominal (inflation-adjusted = real-flat)


def simulate(target_initial, annual_dca, horizon_years,
             rebalance_freq="none", band_lower=1.0,
             decay_per_year=0.0, annual_dca_growth=DCA_GROWTH):
    H = int(horizon_years * TRADING_DAYS)
    m0 = annual_dca / 12.0
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
            month_num = k // DAYS_PER_MONTH
            # Growing DCA: contribution at month_num (1-indexed)
            # grows by (1+g)^((month_num-1)/12)
            years_elapsed = (month_num - 1) / 12.0
            monthly_now = m0 * (1.0 + annual_dca_growth) ** years_elapsed
            spx = np.where(~called, spx + monthly_now, spx)

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


def find_max_safe(annual_dca, horizon_years, **kwargs):
    lo, hi = 1.01, 3.50
    for _ in range(16):
        mid = (lo + hi) / 2
        _, c = simulate(mid, annual_dca, horizon_years, **kwargs)
        if c.mean() <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def irr_growing(terminal, H, annual_dca, annual_dca_growth=DCA_GROWTH):
    """IRR given: $1 at t=0, contributions at end of each month growing
    at annual_dca_growth per year.
    Closed-form geometric series; numerically solve for r."""
    if terminal <= 0:
        return np.nan
    M = int(H * 12)
    m0 = annual_dca / 12.0
    g = annual_dca_growth
    def f(r):
        if abs(r) < 1e-10 and abs(g) < 1e-10:
            return 1 + m0 * M - terminal
        # x = ((1+g)/(1+r))^(1/12)
        ratio = (1.0 + g) / (1.0 + r)
        if ratio <= 0:
            return 1e18
        x = ratio ** (1.0 / 12.0)
        if abs(x - 1.0) < 1e-12:
            dca_fv = m0 * (1 + r) ** (H - 1/12) * M
        else:
            dca_fv = m0 * (1 + r) ** (H - 1/12) * (1 - x**M) / (1 - x)
        return (1 + r) ** H + dca_fv - terminal
    try:
        return brentq(f, -0.5, 1.5, xtol=1e-7)
    except (ValueError, RuntimeError):
        return np.nan


def per_path_irrs(terminal, H, dca):
    out = np.full(len(terminal), np.nan)
    for i, t in enumerate(terminal):
        if t > 0:
            out[i] = irr_growing(t, H, dca)
    return out


DCAs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
strategies = [
    ("static @ max-safe",   {"rebalance_freq": "none"}),
    ("monthly relever",     {"rebalance_freq": "monthly"}),
    ("annual  relever",     {"rebalance_freq": "annual"}),
    ("decay 2pp monthly",   {"rebalance_freq": "monthly", "decay_per_year": 0.02}),
]


def run_horizon(H_YEARS):
    print("\n" + "=" * 118)
    print(f"HORIZON = {H_YEARS} YEARS. DCA grows at {DCA_GROWTH*100:.1f}%/yr (inflation-adjusted).")
    print(f"IRR percentiles across post-1932 entries. Each strategy sized at its own max-safe target.")
    print("=" * 118)
    header = (f"{'Strategy':<22}  {'DCA':>4}  {'Target':>6}  "
              f"{'p10':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}  "
              f"{'mean':>7}  {'call%':>6}")
    for dca in DCAs:
        print("\n" + header)
        for strat_name, kwargs in strategies:
            target = find_max_safe(dca, H_YEARS, **kwargs)
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
