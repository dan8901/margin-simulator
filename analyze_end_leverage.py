"""
End-of-horizon LEVERAGE percentiles across strategies.

Shows how effective each strategy is at deleveraging the portfolio by the end
of the horizon. Contrast:
  - static:          passive drift-down (no rebalance)
  - time-decay 2pp:  target decays on calendar schedule
  - wealth-decay WM=20: target decays with HWM growth
  - monthly relever: pinned at constant target

For each strategy, sized at its 20y max-safe (per prior work):
  DCA=10%, post-1932 entries, report end-leverage p10/p25/p50/p75/p90 at 20y and 30y.

Surviving paths only (called paths have no meaningful end-leverage).
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


def simulate(strategy, T_initial, annual_dca, horizon_years,
             decay_per_year=0.0, wealth_mult_to_floor=None, floor=1.0):
    """
    strategy in {'static', 'monthly', 'time_decay', 'wealth_decay'}.
    Returns (end_leverage, called) arrays, one per entry.
    end_leverage is NaN for called paths.
    """
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0

    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
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
        spx = spx * (px[idxs + k] / px[idxs + k - 1])
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
                else:
                    raise ValueError(strategy)

                cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
                want = (cur_lev < current_target) & active & pos
                delta_D = np.maximum(current_target * equity - spx, 0.0)
                loan = np.where(want, loan + delta_D, loan)
                spx = np.where(want, spx + delta_D, spx)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
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


def call_rate(T_initial, annual_dca, horizon_years, **kw):
    _, called, _ = simulate(T_initial=T_initial, annual_dca=annual_dca,
                            horizon_years=horizon_years, **kw)
    return called.mean()


def find_max_safe_dual(annual_dca, **kw):
    """Find largest T_initial with 0% calls at BOTH 20y and 30y."""
    lo, hi = 1.01, 2.50
    for _ in range(12):  # ~3e-4 precision
        mid = (lo + hi) / 2
        cr20 = call_rate(mid, annual_dca, 20, **kw)
        cr30 = call_rate(mid, annual_dca, 30, **kw)
        if cr20 <= 0.0 and cr30 <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


DCA = 0.10
strategy_kwargs = [
    ('static',             {'strategy': 'static'}),
    ('time-decay 2pp',     {'strategy': 'time_decay', 'decay_per_year': 0.02}),
    ('wealth-decay WM=20', {'strategy': 'wealth_decay',
                            'wealth_mult_to_floor': 20.0}),
    ('monthly relever',    {'strategy': 'monthly'}),
]

print("Finding dual-horizon max-safe targets (0% calls at both 20y and 30y)...")
strategies = []
for name, kw in strategy_kwargs:
    T = find_max_safe_dual(DCA, **kw)
    print(f"  {name:<22}  T_initial = {T:.3f}x")
    strategies.append((name, {'T_initial': T, **kw}))
print()

for horizon in (20, 30):
    print("=" * 90)
    print(f"End-of-horizon leverage percentiles, DCA=10%, horizon={horizon}y, "
          f"post-1932 entries")
    print("=" * 90)
    header = (f"{'Strategy':<22}{'Target':>8}  "
              f"{'p10':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p90':>7}"
              f"{'mean':>8}{'calls':>8}{'N':>6}")
    print(header)
    print("-" * 90)
    for name, kw in strategies:
        T = kw['T_initial']
        end_lev, called, _ = simulate(annual_dca=DCA, horizon_years=horizon, **kw)
        surv = end_lev[~np.isnan(end_lev)]
        cr = called.mean() * 100
        if len(surv) == 0:
            print(f"{name:<22}{T:>7.2f}x  (all paths called)")
            continue
        p10, p25, p50, p75, p90 = np.percentile(surv, [10, 25, 50, 75, 90])
        mn = surv.mean()
        print(f"{name:<22}{T:>7.2f}x  "
              f"{p10:>6.2f}x{p25:>6.2f}x{p50:>6.2f}x{p75:>6.2f}x{p90:>6.2f}x"
              f"{mn:>7.2f}x{cr:>7.2f}%{len(surv):>6d}")
    print()

# Also print a transposed view just of p50 by horizon to make the story pop
print("=" * 90)
print("Median end-leverage side-by-side (DCA=10%):")
print("=" * 90)
print(f"  {'Strategy':<22}{'target':>9} {'end-lev @ 20y':>16} {'end-lev @ 30y':>16}")
for name, kw in strategies:
    T = kw['T_initial']
    out = [f"  {name:<22}{T:>8.2f}x"]
    for horizon in (20, 30):
        end_lev, _, _ = simulate(annual_dca=DCA, horizon_years=horizon, **kw)
        surv = end_lev[~np.isnan(end_lev)]
        if len(surv) == 0:
            out.append(f"{'—':>16}")
        else:
            p50 = np.median(surv)
            out.append(f"{p50:>14.2f}x  ")
    print("".join(out))

# ======================================================================
# IRR percentiles at dual-horizon max-safe targets
# ======================================================================
for horizon in (20, 30):
    print()
    print("=" * 90)
    print(f"Per-path IRR percentiles (%), DCA=10%, horizon={horizon}y, "
          f"dual-horizon max-safe targets")
    print("=" * 90)
    print(f"{'Strategy':<22}{'Target':>8}  "
          f"{'p10':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p90':>7}{'mean':>8}")
    print("-" * 90)
    for name, kw in strategies:
        T = kw['T_initial']
        _, called, terminal = simulate(annual_dca=DCA, horizon_years=horizon,
                                       **kw)
        irrs = per_path_irrs(terminal, horizon, DCA) * 100
        valid = ~np.isnan(irrs)
        if valid.sum() == 0:
            print(f"{name:<22}{T:>7.2f}x  (no valid paths)")
            continue
        p10, p25, p50, p75, p90 = np.percentile(irrs[valid], [10, 25, 50, 75, 90])
        mn = irrs[valid].mean()
        print(f"{name:<22}{T:>7.2f}x  "
              f"{p10:>6.2f}%{p25:>6.2f}%{p50:>6.2f}%{p75:>6.2f}%{p90:>6.2f}%"
              f"{mn:>7.2f}%")
