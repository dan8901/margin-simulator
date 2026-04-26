"""
Generalization buffer #1: lower the call-threshold cap.

Real brokers can tighten maintenance requirements during stress (CLAUDE.md
caveat #2). Default analysis uses 4.0x cap (Reg-T, 25% maintenance). This
script sweeps caps {3.0x, 3.5x, 4.0x} and re-finds dual-horizon max-safe.

Interpretation: "if my broker tightens to 33% maintenance during a crisis
(cap = 3.0x), what target was I safe at?"

For each strategy at 10% DCA, report:
  - Dual-horizon max-safe T_initial under each cap
  - p50 IRR at that target (20y and 30y)
  - p50 end-of-horizon leverage
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


def simulate(strategy, T_initial, annual_dca, horizon_years, cap=4.0,
             decay_per_year=0.0, wealth_mult_to_floor=None, floor=1.0):
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
        new_calls = (~called) & ((equity <= 0) | (lev >= cap))
        called |= new_calls

    equity = spx - loan
    pos = equity > 0
    end_lev = np.where(pos & ~called, spx / np.maximum(equity, 1e-12), np.nan)
    terminal = np.where(called, 0.0, spx - loan)
    return end_lev, called, terminal


def find_max_safe_dual(annual_dca, cap, **kw):
    lo, hi = 1.005, 2.50
    for _ in range(14):
        mid = (lo + hi) / 2
        cr20 = simulate(annual_dca=annual_dca, T_initial=mid,
                        horizon_years=20, cap=cap, **kw)[1].mean()
        cr30 = simulate(annual_dca=annual_dca, T_initial=mid,
                        horizon_years=30, cap=cap, **kw)[1].mean()
        if cr20 <= 0.0 and cr30 <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


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


DCA = 0.10
strategies = [
    ('static',           {'strategy': 'static'}),
    ('time-decay 2pp',   {'strategy': 'time_decay', 'decay_per_year': 0.02}),
    ('wealth-decay WM=20', {'strategy': 'wealth_decay',
                            'wealth_mult_to_floor': 20.0}),
    ('monthly relever',  {'strategy': 'monthly'}),
]
caps = [3.0, 3.5, 4.0]

print("=" * 110)
print(f"Call-threshold sweep, DCA={DCA*100:.0f}%, dual-horizon max-safe per cap")
print("=" * 110)
print(f"{'Strategy':<22}{'Cap':>5}  {'Target':>8}  "
      f"{'p50@20y':>8}{'p10@20y':>8}{'endlev@20y':>11}  "
      f"{'p50@30y':>8}{'p10@30y':>8}{'endlev@30y':>11}")
print("-" * 110)

for name, kw in strategies:
    for cap in caps:
        T = find_max_safe_dual(DCA, cap, **kw)
        p50_20, p10_20, el_20, p50_30, p10_30, el_30 = (np.nan,) * 6
        for h in (20, 30):
            end_lev, _, term = simulate(annual_dca=DCA, T_initial=T,
                                        horizon_years=h, cap=cap, **kw)
            irrs = per_path_irrs(term, h, DCA) * 100
            valid = ~np.isnan(irrs)
            surv = end_lev[~np.isnan(end_lev)]
            if h == 20:
                p50_20 = np.percentile(irrs[valid], 50)
                p10_20 = np.percentile(irrs[valid], 10)
                el_20 = np.median(surv)
            else:
                p50_30 = np.percentile(irrs[valid], 50)
                p10_30 = np.percentile(irrs[valid], 10)
                el_30 = np.median(surv)
        print(f"{name:<22}{cap:>4.1f}x  {T:>7.3f}x  "
              f"{p50_20:>7.2f}%{p10_20:>7.2f}%{el_20:>10.2f}x  "
              f"{p50_30:>7.2f}%{p10_30:>7.2f}%{el_30:>10.2f}x")
    print()
