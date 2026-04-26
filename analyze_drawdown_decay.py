"""
Drawdown-coupled decay: a new architecture.

Hypothesis (from session 3 architectural critique): time-decay couples to
calendar (a non-risk variable); wealth-decay couples to HWM (correlated with
safety but not the actual risk variable, and gameable by DCA). The
architecturally-right variable to couple to is DRAWDOWN DEPTH — the actual
input that determines how close current leverage is to the call threshold.

Mechanism:
  max_equity(t) = HWM of equity (ratchets up only)
  max_dd(t) = ratchet of (1 - equity / max_equity) — never decreases
  target(t) = max(floor, T_initial - DD_FACTOR × max_dd(t))

Properties:
  - Event-driven, not schedule-driven: target stays at T_initial through a
    quiet period, drops only when stress is actually observed
  - Asymmetric ratchet: once a drawdown is observed, target stays lowered
    even after recovery (preserves safety once gained)
  - Couples to the actual risk variable (drawdown), not a proxy
  - DD_FACTOR has a defensible interpretation: "after a 25% drawdown, target
    falls 0.25 × DD_FACTOR points of leverage"

For each DD_FACTOR ∈ {0.5, 1.0, 1.5, 2.0}, at 10% DCA:
  - Find dual-horizon max-safe T_initial (0% calls at 20y AND 30y, cap=4.0x)
  - Report historical p10/p50 IRR @ 30y, end-leverage @ 30y
  - Run block bootstrap (5000 paths, 1y blocks) and report call rate
  - Compare to static, wealth-decay WM=20, time-decay 2pp baselines

Test specifically: does drawdown-coupled beat wealth-decay on bootstrap
fragility at comparable IRR?
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


def simulate_historical(strategy, T_initial, annual_dca, horizon_years, cap=4.0,
                         decay_per_year=0.0, wealth_mult_to_floor=None,
                         dd_factor=0.0, floor=1.0):
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px_orig)]
    N = len(idxs)

    spx = np.full(N, T_initial, dtype=float)
    loan = np.full(N, T_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)
    max_equity = np.full(N, 1.0, dtype=float)
    max_dd = np.zeros(N, dtype=float)

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
            equity = spx - loan
            pos = equity > 0

            # Update HWM and max_dd ratchet
            max_equity = np.maximum(
                max_equity,
                np.where(pos & active, equity, max_equity))
            current_dd = np.where(
                pos & active, 1.0 - equity / np.maximum(max_equity, 1e-12), 0.0)
            max_dd = np.maximum(max_dd, current_dd)

            if strategy != 'static':
                if strategy == 'monthly':
                    current_target = np.full(N, T_initial)
                elif strategy == 'time_decay':
                    years = k / TRADING_DAYS
                    current_target = np.full(
                        N, max(T_initial - decay_per_year * years, floor))
                elif strategy == 'wealth_decay':
                    current_target = np.maximum(
                        T_initial - slope_wd * (max_equity - 1.0), floor)
                elif strategy == 'drawdown_decay':
                    current_target = np.maximum(
                        T_initial - dd_factor * max_dd, floor)
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
                       dd_factor=0.0, floor=1.0):
    N, H = lr_p.shape
    monthly = annual_dca / 12.0

    spx = np.full(N, T_initial, dtype=float)
    loan = np.full(N, T_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)
    max_equity = np.full(N, 1.0, dtype=float)
    max_dd = np.zeros(N, dtype=float)

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
            equity = spx - loan
            pos = equity > 0
            max_equity = np.maximum(
                max_equity,
                np.where(pos & active, equity, max_equity))
            current_dd = np.where(
                pos & active, 1.0 - equity / np.maximum(max_equity, 1e-12), 0.0)
            max_dd = np.maximum(max_dd, current_dd)

            if strategy != 'static':
                if strategy == 'monthly':
                    current_target = np.full(N, T_initial)
                elif strategy == 'time_decay':
                    years = (k + 1) / TRADING_DAYS
                    current_target = np.full(
                        N, max(T_initial - decay_per_year * years, floor))
                elif strategy == 'wealth_decay':
                    current_target = np.maximum(
                        T_initial - slope_wd * (max_equity - 1.0), floor)
                elif strategy == 'drawdown_decay':
                    current_target = np.maximum(
                        T_initial - dd_factor * max_dd, floor)
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


def find_max_safe_dual(strategy_kw, annual_dca):
    lo, hi = 1.005, 2.50
    for _ in range(14):
        mid = (lo + hi) / 2
        cr20 = simulate_historical(annual_dca=annual_dca, T_initial=mid,
                                     horizon_years=20, **strategy_kw)[1].mean()
        cr30 = simulate_historical(annual_dca=annual_dca, T_initial=mid,
                                     horizon_years=30, **strategy_kw)[1].mean()
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
N_PATHS = 5000
H_DAYS = 30 * TRADING_DAYS

print("Generating bootstrap paths (5000 × 30y, 252-day blocks)...")
lr_p, rg_p = make_bootstrap_paths(N_PATHS, H_DAYS, block_size=252, seed=42)
print("  Done.\n")

# Baselines for comparison
baselines = [
    ('static',           {'strategy': 'static'}),
    ('wealth-decay WM=20', {'strategy': 'wealth_decay',
                             'wealth_mult_to_floor': 20.0}),
    ('time-decay 2pp',   {'strategy': 'time_decay', 'decay_per_year': 0.02}),
    ('monthly relever',  {'strategy': 'monthly'}),
]

# Drawdown-decay sweep
dd_factors = [0.5, 1.0, 1.5, 2.0, 3.0]

print("=" * 110)
print(f"DRAWDOWN-COUPLED decay vs baselines, DCA={DCA*100:.0f}%, dual-horizon max-safe (cap=4.0x)")
print("=" * 110)
print(f"{'Strategy':<25}{'Param':>8}  {'Target':>8}  "
      f"{'p50@30y':>8}{'p10@30y':>8}{'endlev@30y':>11}  {'boot call%':>11}")
print("-" * 110)

# Baselines first
for name, kw in baselines:
    T = find_max_safe_dual(kw, DCA)
    end_lev, _, term = simulate_historical(
        annual_dca=DCA, T_initial=T, horizon_years=30, **kw)
    irrs = per_path_irrs(term, 30, DCA) * 100
    valid = ~np.isnan(irrs)
    surv = end_lev[~np.isnan(end_lev)]
    p10 = np.percentile(irrs[valid], 10)
    p50 = np.percentile(irrs[valid], 50)
    p50_el = np.median(surv)

    _, called_b, _ = simulate_bootstrap(lr_p, rg_p, T_initial=T,
                                          annual_dca=DCA, **kw)
    boot_cr = called_b.mean() * 100

    print(f"{name:<25}{'—':>8}  {T:>7.3f}x  "
          f"{p50:>7.2f}%{p10:>7.2f}%{p50_el:>10.2f}x  {boot_cr:>10.2f}%")

print()

for ddf in dd_factors:
    kw = {'strategy': 'drawdown_decay', 'dd_factor': ddf}
    T = find_max_safe_dual(kw, DCA)
    end_lev, _, term = simulate_historical(
        annual_dca=DCA, T_initial=T, horizon_years=30, **kw)
    irrs = per_path_irrs(term, 30, DCA) * 100
    valid = ~np.isnan(irrs)
    surv = end_lev[~np.isnan(end_lev)]
    p10 = np.percentile(irrs[valid], 10)
    p50 = np.percentile(irrs[valid], 50)
    p50_el = np.median(surv)

    _, called_b, _ = simulate_bootstrap(lr_p, rg_p, T_initial=T,
                                          annual_dca=DCA, **kw)
    boot_cr = called_b.mean() * 100

    print(f"{'drawdown-decay':<25}{'F='+str(ddf):>8}  {T:>7.3f}x  "
          f"{p50:>7.2f}%{p10:>7.2f}%{p50_el:>10.2f}x  {boot_cr:>10.2f}%")

# Joint constraint test: well-defended for drawdown-decay
print()
print("=" * 110)
print("Well-defended (cap=3.0x historical AND bootstrap≤1% at cap=4.0x), DD_FACTOR=1.0")
print("=" * 110)


def find_well_defended(strategy_kw, annual_dca, lr_p, rg_p,
                        bootstrap_threshold=0.01, hist_cap=3.0):
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


for ddf in [1.0, 1.5, 2.0]:
    kw = {'strategy': 'drawdown_decay', 'dd_factor': ddf}
    T = find_well_defended(kw, DCA, lr_p, rg_p)
    end_lev, _, term = simulate_historical(
        annual_dca=DCA, T_initial=T, horizon_years=30, cap=4.0, **kw)
    irrs = per_path_irrs(term, 30, DCA) * 100
    valid = ~np.isnan(irrs)
    surv = end_lev[~np.isnan(end_lev)]
    p10 = np.percentile(irrs[valid], 10)
    p50 = np.percentile(irrs[valid], 50)
    p50_el = np.median(surv)
    _, called_b, _ = simulate_bootstrap(lr_p, rg_p, T_initial=T,
                                          annual_dca=DCA, **kw)
    boot_cr = called_b.mean() * 100
    print(f"  DD_FACTOR={ddf}: well-defended T={T:.3f}x, p50@30y={p50:.2f}%, "
          f"p10={p10:.2f}%, end_lev={p50_el:.2f}x, bootstrap call%={boot_cr:.2f}%")
