"""
Cohort + drawdown-decay: each DCA contribution starts an independent cohort
with its own max_dd ratchet and decay schedule. Aggregate margin call is
checked at the account level (real-world).

Design:
  - Initial $1 of equity → cohort 0 at T_initial leverage
  - Each monthly DCA $m → new cohort N at T_initial leverage on ITS OWN $m
  - Each cohort independently tracks its own max_equity HWM and max_dd ratchet
  - Each month, every active cohort: target = max(floor, T_initial - F * cohort_max_dd)
    Re-lever up within cohort if cohort_leverage < cohort_target
  - Aggregate margin call: agg_assets / agg_equity ≥ cap → CALL (everything liquidated)

Why this is interesting (vs aggregate drawdown-decay):
  1. Pure drawdown signal per cohort — DCA inflows don't contaminate max_dd
     (open question #2 from session 2 quietly resolved)
  2. New money gets fresh T_initial regardless of historical crashes
     (no inherited max_dd baggage)
  3. Old money is appropriately conservative based on what it experienced
  4. Aggregate has natural glide path — no explicit calendar rule needed

Compares against aggregate drawdown-decay at same DD_FACTOR, plus static and
single-account drawdown-decay baselines.
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


# ============================================================================
# Cohort simulator (vectorized over paths × cohorts)
# ============================================================================
def _simulate_cohort_core(spx_factors, rate_factors, T_initial, annual_dca,
                           dd_factor, cap=4.0, floor=1.0):
    """Core cohort sim. Per-cohort state updated only at monthly boundaries;
    aggregate state computed daily for margin-call detection.

    Inputs are (N_paths, H_days) per-day growth-factor arrays.
    Returns (end_lev, called, terminal) per path.
    """
    N, H = spx_factors.shape
    monthly = annual_dca / 12.0
    n_months = H // DAYS_PER_MONTH
    n_cohorts = 1 + n_months

    # Per-(path, cohort) state — updated MONTHLY only
    spx = np.zeros((N, n_cohorts), dtype=np.float64)
    loan = np.zeros((N, n_cohorts), dtype=np.float64)
    max_eq = np.zeros((N, n_cohorts), dtype=np.float64)
    max_dd = np.zeros((N, n_cohorts), dtype=np.float64)
    active = np.zeros((N, n_cohorts), dtype=bool)

    # Cohort 0: $1 equity at T_initial leverage
    spx[:, 0] = T_initial
    loan[:, 0] = T_initial - 1.0
    max_eq[:, 0] = 1.0
    active[:, 0] = True

    called = np.zeros(N, dtype=bool)

    for m in range(n_months):
        # Days in this month: indices [m*21, m*21+1, ..., (m+1)*21 - 1]
        k0 = m * DAYS_PER_MONTH
        k1 = (m + 1) * DAYS_PER_MONTH
        # Cumulative growth factors over the days in this month, starting from
        # *current* monthly state at end-of-prior-month (= state at day k0-1's close,
        # which IS the value after the previous monthly rebalance plus 0 daily steps).
        # spx_factors[:, k0:k1] are the per-day factors during this month.
        cum_spx = np.cumprod(spx_factors[:, k0:k1], axis=1)  # (N, 21)
        cum_rate = np.cumprod(rate_factors[:, k0:k1], axis=1)

        # Aggregate at each intra-month day
        agg_spx0 = spx.sum(axis=1)  # (N,)
        agg_loan0 = loan.sum(axis=1)
        # Aggregate trajectory through the month
        agg_spx_traj = agg_spx0[:, None] * cum_spx  # (N, 21)
        agg_loan_traj = agg_loan0[:, None] * cum_rate
        agg_eq_traj = agg_spx_traj - agg_loan_traj
        agg_lev_traj = np.where(agg_eq_traj > 0,
                                agg_spx_traj / np.maximum(agg_eq_traj, 1e-12),
                                np.inf)
        # Did any day breach the cap?
        breached = (agg_eq_traj <= 0) | (agg_lev_traj >= cap)
        any_breach = breached.any(axis=1)
        called |= (~called) & any_breach

        # Apply full-month compounding to cohort state
        full_spx = cum_spx[:, -1:]  # (N, 1) — broadcasts over cohort dim
        full_rate = cum_rate[:, -1:]
        spx *= full_spx
        loan *= full_rate

        # End of this month: monthly events
        not_called = ~called
        # Birth new cohort (cohort index m+1, next after the initial one)
        new_idx = m + 1
        if new_idx < n_cohorts:
            spx[not_called, new_idx] = T_initial * monthly
            loan[not_called, new_idx] = (T_initial - 1.0) * monthly
            max_eq[not_called, new_idx] = monthly
            active[not_called, new_idx] = True

        # Per-cohort rebalance
        equity = spx - loan
        pos = equity > 0
        valid = pos & active
        max_eq = np.where(valid, np.maximum(max_eq, equity), max_eq)
        current_dd = np.where(
            valid, 1.0 - equity / np.maximum(max_eq, 1e-12), 0.0)
        max_dd = np.maximum(max_dd, current_dd)
        current_target = np.maximum(T_initial - dd_factor * max_dd, floor)
        cur_lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        want = (cur_lev < current_target) & valid & (~called[:, None])
        delta_D = np.maximum(current_target * equity - spx, 0.0)
        loan = np.where(want, loan + delta_D, loan)
        spx = np.where(want, spx + delta_D, spx)

    agg_assets = spx.sum(axis=1)
    agg_loan = loan.sum(axis=1)
    agg_equity = agg_assets - agg_loan
    pos_agg = agg_equity > 0
    end_lev = np.where(pos_agg & ~called,
                        agg_assets / np.maximum(agg_equity, 1e-12), np.nan)
    terminal = np.where(called, 0.0, agg_equity)
    return end_lev, called, terminal


def simulate_cohort_historical(T_initial, annual_dca, horizon_years, dd_factor,
                                cap=4.0):
    """Cohort sim across post-1932 historical entries."""
    H = int(horizon_years * TRADING_DAYS)
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px_orig)]
    N = len(idxs)
    # Daily growth factors per entry
    spx_f = np.empty((N, H))
    rate_f = np.empty((N, H))
    for k in range(H):
        spx_f[:, k] = px_orig[idxs + k + 1] / px_orig[idxs + k]
        rate_f[:, k] = M_box[idxs + k + 1] / M_box[idxs + k]
    return _simulate_cohort_core(spx_f, rate_f, T_initial, annual_dca,
                                   dd_factor, cap=cap)


def simulate_cohort_bootstrap(lr_p, rg_p, T_initial, annual_dca, dd_factor,
                                cap=4.0):
    """Cohort sim on bootstrap synthetic paths."""
    spx_f = np.exp(lr_p)
    return _simulate_cohort_core(spx_f, rg_p, T_initial, annual_dca,
                                   dd_factor, cap=cap)


# ============================================================================
# Bootstrap path generator (shared with prior scripts)
# ============================================================================
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


def find_max_safe_dual_cohort(annual_dca, dd_factor):
    lo, hi = 1.005, 2.50
    for _ in range(12):  # ~6e-4 precision; cohort sim is heavier
        mid = (lo + hi) / 2
        cr20 = simulate_cohort_historical(mid, annual_dca, 20, dd_factor)[1].mean()
        cr30 = simulate_cohort_historical(mid, annual_dca, 30, dd_factor)[1].mean()
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


# ============================================================================
# Run
# ============================================================================
DCA = 0.10
N_PATHS = 5000
H_DAYS = 30 * TRADING_DAYS

print("Generating bootstrap paths (5000 × 30y, 252-day blocks)...")
lr_p, rg_p = make_bootstrap_paths(N_PATHS, H_DAYS, block_size=252, seed=42)
print("  Done.\n")

dd_factors = [0.5, 1.0, 1.5, 2.0, 3.0]

print("=" * 110)
print(f"COHORT + drawdown-decay, DCA={DCA*100:.0f}%, dual-horizon max-safe (cap=4.0x)")
print("=" * 110)
print(f"{'Param':>10}  {'Target':>8}  {'p50@30y':>8}{'p10@30y':>8}{'endlev@30y':>11}  {'boot call%':>11}")
print("-" * 110)

for ddf in dd_factors:
    T = find_max_safe_dual_cohort(DCA, ddf)
    end_lev, _, term = simulate_cohort_historical(T, DCA, 30, ddf)
    irrs = per_path_irrs(term, 30, DCA) * 100
    valid = ~np.isnan(irrs)
    surv = end_lev[~np.isnan(end_lev)]
    p10 = np.percentile(irrs[valid], 10) if valid.sum() else np.nan
    p50 = np.percentile(irrs[valid], 50) if valid.sum() else np.nan
    p50_el = np.median(surv) if len(surv) else np.nan

    _, called_b, _ = simulate_cohort_bootstrap(lr_p, rg_p, T, DCA, ddf)
    boot_cr = called_b.mean() * 100
    print(f"  F={ddf:>4}   {T:>7.3f}x  {p50:>7.2f}%{p10:>7.2f}%{p50_el:>10.2f}x  "
          f"{boot_cr:>10.2f}%")

print()
print("Comparison row from prior session (single-account drawdown-decay):")
print(f"  F=1.0    1.553x   12.59% 11.14%      1.15x         3.68%")
print(f"  F=2.0    1.586x   12.32% 10.95%      1.07x         2.64%")
print(f"  F=3.0    1.586x   12.10% 10.86%      1.06x         2.00%")
print(f"  static   1.612x   11.76% 10.70%      1.05x         1.46%")
