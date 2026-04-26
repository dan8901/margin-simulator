"""
Generalization buffer #3: block bootstrap synthetic paths.

The historical entries we use (~17k post-1932) are highly overlapping —
adjacent-day entries share most of their 30-year window. Block bootstrap
treats post-1932 daily returns as the empirical distribution and resamples
year-long blocks (with replacement) to construct genuinely independent
synthetic 30y paths.

For each strategy at its (post-1932 dual-horizon) max-safe target, evaluate:
  - Call rate across N_PATHS synthetic 30y paths
  - IRR distribution (synthetic-path p10, p50, p90)

Block size = 1 year (252 days) preserves intra-year volatility clustering
without preserving the entire historical sequence. (return, rate) pairs are
resampled JOINTLY to maintain their correlation.

If a strategy has a meaningfully positive call rate on synthetic paths despite
0% on historical entries, that's evidence the historical 0% was overfitting
to the specific path-ordering rather than reflecting structural safety.
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

# Daily inputs (length = N - 1)
log_ret = np.log(px_orig[1:] / px_orig[:-1])
rate_growth = M_box[1:] / M_box[:-1]

# Indices of valid block starts (in post-1932, full block fits in series)
BLOCK = 252
post1932_for_lr = post1932[1:]  # log_ret[t] uses px[t+1] / px[t]
candidate_starts = np.arange(len(log_ret) - BLOCK + 1)
valid_starts = candidate_starts[post1932_for_lr[candidate_starts]]
print(f"Block bootstrap: {len(valid_starts)} valid year-block starts.")


def bootstrap_paths(N_paths, H_days, block_size=BLOCK, seed=None):
    """Returns (log_ret_paths, rate_growth_paths), shape (N_paths, H_days)."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(H_days / block_size))
    total_len = n_blocks * block_size
    out_lr = np.empty((N_paths, total_len))
    out_rg = np.empty((N_paths, total_len))
    arange_block = np.arange(block_size)
    for b in range(n_blocks):
        starts = rng.choice(valid_starts, N_paths)
        indices = starts[:, None] + arange_block
        out_lr[:, b*block_size:(b+1)*block_size] = log_ret[indices]
        out_rg[:, b*block_size:(b+1)*block_size] = rate_growth[indices]
    return out_lr[:, :H_days], out_rg[:, :H_days]


def simulate_paths(lr_paths, rg_paths, strategy, T_initial, annual_dca,
                   decay_per_year=0.0, wealth_mult_to_floor=None, floor=1.0):
    N, H = lr_paths.shape
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

    spx_factors = np.exp(lr_paths)  # (N, H)

    for k in range(H):
        spx = spx * spx_factors[:, k]
        loan = loan * rg_paths[:, k]

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


def irr_one(terminal, H_years, annual_dca):
    if terminal <= 0:
        return np.nan
    m = annual_dca / 12.0
    M = int(H_years * 12)
    def f(r):
        if abs(r) < 1e-10:
            return 1 + m * M - terminal
        d = (1 + r) ** (1 / 12)
        return (1 + r) ** H_years + m * (d ** M - 1) / (d - 1) - terminal
    try:
        return brentq(f, -0.5, 1.5, xtol=1e-7)
    except ValueError:
        return np.nan


def per_path_irrs(terminal, H_years, dca):
    out = np.full(len(terminal), np.nan)
    for i, t in enumerate(terminal):
        if t > 0:
            out[i] = irr_one(t, H_years, dca)
    return out


# Targets at 10% DCA, dual-horizon-safe (post-1932) — established earlier.
DCA = 0.10
strategies = [
    ('static',           1.612, {'strategy': 'static'}),
    ('time-decay 2pp',   1.591, {'strategy': 'time_decay', 'decay_per_year': 0.02}),
    ('wealth-decay WM=20', 1.562, {'strategy': 'wealth_decay',
                                    'wealth_mult_to_floor': 20.0}),
    ('monthly relever',  1.429, {'strategy': 'monthly'}),
]

N_PATHS = 5000
H_YEARS = 30
H_DAYS = H_YEARS * TRADING_DAYS

print(f"\nGenerating {N_PATHS} synthetic {H_YEARS}y paths via 1-year block bootstrap...")
np.random.seed(42)
lr_p, rg_p = bootstrap_paths(N_PATHS, H_DAYS, seed=42)
print(f"  log_ret_paths shape: {lr_p.shape}")
print(f"  Synthetic-path mean annualized log-return: "
      f"{lr_p.mean(axis=1).mean()*TRADING_DAYS*100:.2f}% "
      f"(historical post-1932: {log_ret[post1932_for_lr].mean()*TRADING_DAYS*100:.2f}%)")

print("\n" + "=" * 110)
print(f"Block-bootstrap results, DCA={DCA*100:.0f}%, "
      f"{H_YEARS}y horizon, N={N_PATHS} synthetic paths")
print("=" * 110)
print(f"{'Strategy':<22}{'Target':>8}  {'call%':>7}  "
      f"{'p10 IRR':>8}{'p50 IRR':>8}{'p90 IRR':>8}{'mean IRR':>9}  "
      f"{'p50 endlev':>11}")
print("-" * 110)

for name, T, kw in strategies:
    end_lev, called, term = simulate_paths(lr_p, rg_p, T_initial=T,
                                            annual_dca=DCA, **kw)
    cr = called.mean() * 100
    irrs = per_path_irrs(term, H_YEARS, DCA) * 100
    valid = ~np.isnan(irrs)
    surv = end_lev[~np.isnan(end_lev)]
    if valid.sum() == 0:
        print(f"{name:<22}{T:>7.3f}x  (all called)")
        continue
    p10 = np.percentile(irrs[valid], 10)
    p50 = np.percentile(irrs[valid], 50)
    p90 = np.percentile(irrs[valid], 90)
    mn = irrs[valid].mean()
    p50_el = np.median(surv) if len(surv) > 0 else np.nan
    print(f"{name:<22}{T:>7.3f}x  {cr:>6.2f}%  "
          f"{p10:>7.2f}%{p50:>7.2f}%{p90:>7.2f}%{mn:>8.2f}%  "
          f"{p50_el:>10.2f}x")

print("\nReminder: historical post-1932 call rate at these targets is 0% by")
print("construction. Synthetic call rate > 0% indicates the historical 0% was")
print("overfit to specific path orderings rather than structural safety.")
