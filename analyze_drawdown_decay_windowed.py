"""
Drawdown-decay with TIME-WINDOWED max_dd — does forgetting old drawdowns help?

Mechanism: instead of ratcheting max_dd indefinitely, use the maximum drawdown
observed over the last N years. Old drawdowns "expire" out of the window;
target can rise back up after a long recovery.

  max_dd_window(t) = max over months in [t - WINDOW_MONTHS + 1, t] of current_dd
  target(t) = max(floor, T_initial - F * max_dd_window(t))

Key behavior change vs original:
  - Target is no longer monotonically non-increasing — it can RISE as old
    drawdowns expire.
  - When target rises and current leverage < new target, the strategy
    re-levers UP. This is "re-claiming" leverage after a long recovery.

Hypothesis: short windows (2-5y) over-react then forget too fast → vulnerable
in next crisis. Long windows (10-20y) give similar safety to original (∞)
while letting strategy reclaim leverage after a quiet decade. Optimal window
might balance "remembering 2008" with "not punishing 2024 for COVID."

Sweep WINDOW ∈ {2y, 5y, 10y, 20y, ∞} × F ∈ {1.0, 1.5, 2.0}.
∞ reproduces original drawdown-decay. Find dual-horizon max-safe + bootstrap.
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


def _simulate_core(spx_factors, rate_factors, T_initial, annual_dca,
                    dd_factor, window_months, cap=4.0, floor=1.0):
    """Drawdown-decay with rolling max-of-window dd.
    window_months = None means use lifetime max (original drawdown-decay)."""
    N, H = spx_factors.shape
    monthly = annual_dca / 12.0
    n_months = H // DAYS_PER_MONTH

    spx = np.full(N, T_initial, dtype=np.float64)
    loan = np.full(N, T_initial - 1.0, dtype=np.float64)
    called = np.zeros(N, dtype=bool)
    max_eq = np.ones(N, dtype=np.float64)

    # Store monthly current_dd values; compute rolling max each step
    dd_history = np.zeros((N, n_months), dtype=np.float64)

    for m in range(n_months):
        k0 = m * DAYS_PER_MONTH
        k1 = (m + 1) * DAYS_PER_MONTH
        cum_spx = np.cumprod(spx_factors[:, k0:k1], axis=1)
        cum_rate = np.cumprod(rate_factors[:, k0:k1], axis=1)
        agg_spx_traj = spx[:, None] * cum_spx
        agg_loan_traj = loan[:, None] * cum_rate
        agg_eq_traj = agg_spx_traj - agg_loan_traj
        agg_lev_traj = np.where(agg_eq_traj > 0,
                                agg_spx_traj / np.maximum(agg_eq_traj, 1e-12),
                                np.inf)
        breached = (agg_eq_traj <= 0) | (agg_lev_traj >= cap)
        any_breach = breached.any(axis=1)
        called |= (~called) & any_breach

        spx = spx * cum_spx[:, -1]
        loan = loan * cum_rate[:, -1]

        not_called = ~called
        spx = np.where(not_called, spx + monthly, spx)

        equity = spx - loan
        pos = equity > 0
        max_eq = np.where(pos & not_called, np.maximum(max_eq, equity), max_eq)
        current_dd = np.where(
            pos & not_called, 1.0 - equity / np.maximum(max_eq, 1e-12), 0.0)
        dd_history[:, m] = current_dd

        # Rolling max over window
        if window_months is None or window_months >= m + 1:
            max_dd_window = dd_history[:, :m+1].max(axis=1)
        else:
            max_dd_window = dd_history[:, m + 1 - window_months:m+1].max(axis=1)

        current_target = np.maximum(T_initial - dd_factor * max_dd_window, floor)
        cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        want = (cur_lev < current_target) & not_called & pos
        delta_D = np.maximum(current_target * equity - spx, 0.0)
        loan = np.where(want, loan + delta_D, loan)
        spx = np.where(want, spx + delta_D, spx)

    equity = spx - loan
    pos = equity > 0
    end_lev = np.where(pos & ~called, spx / np.maximum(equity, 1e-12), np.nan)
    terminal = np.where(called, 0.0, equity)
    return end_lev, called, terminal


def simulate_historical(T_initial, annual_dca, horizon_years, dd_factor,
                          window_months, cap=4.0):
    H = int(horizon_years * TRADING_DAYS)
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px_orig)]
    N = len(idxs)
    spx_f = np.empty((N, H))
    rate_f = np.empty((N, H))
    for k in range(H):
        spx_f[:, k] = px_orig[idxs + k + 1] / px_orig[idxs + k]
        rate_f[:, k] = M_box[idxs + k + 1] / M_box[idxs + k]
    return _simulate_core(spx_f, rate_f, T_initial, annual_dca, dd_factor,
                           window_months, cap=cap)


def simulate_bootstrap(lr_p, rg_p, T_initial, annual_dca, dd_factor,
                         window_months, cap=4.0):
    spx_f = np.exp(lr_p)
    return _simulate_core(spx_f, rg_p, T_initial, annual_dca, dd_factor,
                           window_months, cap=cap)


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


def find_max_safe_dual(annual_dca, dd_factor, window_months):
    lo, hi = 1.005, 2.50
    for _ in range(13):
        mid = (lo + hi) / 2
        cr20 = simulate_historical(mid, annual_dca, 20, dd_factor,
                                     window_months)[1].mean()
        cr30 = simulate_historical(mid, annual_dca, 30, dd_factor,
                                     window_months)[1].mean()
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

# Window in months. None = lifetime (no expiry, original drawdown-decay)
windows = [(2, "2y"), (5, "5y"), (10, "10y"), (20, "20y"), (None, "∞")]
F_values = [1.0, 1.5, 2.0]

print("=" * 110)
print(f"DRAWDOWN-DECAY with TIME-WINDOWED max_dd, DCA={DCA*100:.0f}%, dual-horizon max-safe (cap=4.0x)")
print(f"WINDOW=∞ reproduces original drawdown-decay (max_dd ratchets forever).")
print("=" * 110)
print(f"{'F':>4}  {'WINDOW':>8}  {'Target':>8}  {'p50@30y':>8}{'p10@30y':>8}{'endlev@30y':>11}  {'boot call%':>11}")
print("-" * 110)

for F in F_values:
    for window_yrs, win_label in windows:
        window_months = window_yrs * 12 if window_yrs is not None else None
        T = find_max_safe_dual(DCA, F, window_months)
        end_lev, _, term = simulate_historical(T, DCA, 30, F, window_months)
        irrs = per_path_irrs(term, 30, DCA) * 100
        valid = ~np.isnan(irrs)
        surv = end_lev[~np.isnan(end_lev)]
        p10 = np.percentile(irrs[valid], 10) if valid.sum() else np.nan
        p50 = np.percentile(irrs[valid], 50) if valid.sum() else np.nan
        p50_el = np.median(surv) if len(surv) else np.nan

        _, called_b, _ = simulate_bootstrap(lr_p, rg_p, T, DCA, F, window_months)
        boot_cr = called_b.mean() * 100

        print(f"  {F:>3.1f}  {win_label:>8}  {T:>7.3f}x  "
              f"{p50:>7.2f}%{p10:>7.2f}%{p50_el:>10.2f}x  {boot_cr:>10.2f}%")
    print()
