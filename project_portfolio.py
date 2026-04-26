"""
project_portfolio.py

Project a portfolio's future value in REAL (today's-purchasing-power) dollars,
using historical SPX-TR paths and synthetic bootstrap paths, with leverage
strategies CALIBRATED to the user's specific savings pattern.

For each of (static, relever, dd_decay):
  1. Binary-search the largest leverage target with 0% historical margin calls
     across post-1932 monthly entries  →  T_hist_safe
  2. Compute the bootstrap call rate at that target  →  reveals overfitting
  3. Binary-search the largest target with bootstrap call rate ≤ a threshold
     (default 1%)  →  T_boot_safe (this is the recommended target)
  4. Run the projection at T_boot_safe and report percentiles in real dollars.

Bootstrap: 1-year blocks of joint (return, tsy yield, daily-CPI-ratio) tuples
sampled from the post-1932 daily series. Block size matches the project's
default analysis (1y); see analyze_block_bootstrap.py for sensitivity to
block size.

Conventions:
  - Annual savings (S, S2) are TODAY'S dollars and grow with CPI along each
    historical/synthetic path (you maintain real purchasing-power contributions).
  - Reported portfolio values are also today's dollars (deflated by entry
    or path-start CPI).
  - Margin call threshold = 4.0x leverage (Reg-T 25%).
  - Re-lever monthly: only LEVER UP, never sell to deleverage.

Strategies modeled:
  unlev         — 1.00x, no leverage. Baseline.
  static        — Set T_init at day 0; never rebalance. Drifts toward 1.0x as
                  DCA dilutes leverage.
  relever       — Monthly re-lever to T_init.
  dd_decay      — Drawdown-coupled decay. T_init at day 0; target ratchets DOWN
                  as max observed drawdown grows: target = max(floor, T_init -
                  F * max_dd_observed). F=1.5, floor=1.0.
  wealth_decay  — Current-equity wealth glide. Target linearly interpolates
                  T_init → floor as real equity grows from C → wealth_X. Uses
                  CURRENT real equity (not HWM), so target rises again if a
                  drawdown reduces equity. progress = clip((eq-C)/(wealth_X-C),
                  0, 1); target = T_init - (T_init - floor) * progress.
  hybrid        — min(dd_decay_target, wealth_decay_target). Both signals
                  independently lower target; the more conservative wins.
  r_hybrid      — Ratcheted hybrid. Same as hybrid but the wealth_progress
                  only ratchets UP (never decreases), so once target has been
                  reduced by the wealth-glide it doesn't re-rise on a
                  drawdown. Strictly-monotonic-down target.
  vol_hybrid    — Hybrid + volatility haircut. After computing the hybrid
                  target, subtract `vol_factor * realized_60d_annualized_vol`.
                  Fires earlier than dd_decay since vol leads drawdown.
  dip_hybrid    — Hybrid + buy-the-dip floor. When current drawdown crosses
                  `dip_threshold`, target is floored at `T_init + dip_bonus`
                  (i.e. allowed to go ABOVE T_init temporarily). Contrarian
                  to dd_decay's ratchet during deep drawdowns.
  rate_hybrid   — Hybrid + interest-rate haircut. After hybrid target,
                  subtract `rate_factor * max(0, tsy_3m - rate_threshold)`.
                  Deleverages when carry trade economics deteriorate.
  adaptive_dd   — Drawdown-decay with cushion-coupled F. F_eff = F * (L_now -
                  1) / (T_init - 1), clamped non-negative. Decay is aggressive
                  when current leverage is near T_init (small cushion), mild
                  when already deleveraged (big cushion). Uses an explicit
                  monotonic-down target ratchet so a transient L_now spike
                  can lower target but a later drop in L_now never raises it.
  adaptive_hybrid — adaptive_dd + wealth_decay glide combined.
                  target = min(adaptive_dd_target, wealth_decay_target).
                  Inherits adaptive_dd's cushion-coupled F + monotonic ratchet
                  on the dd component, plus the wealth_X glide enforcing
                  unlevered at wealth_X.
  recal_static  — Periodic re-calibration of T_init. Behaves like static
                  between rebalance events. Every `recal_period_days`, looks
                  up new T_max from a pre-computed lookup table T*(E_real,
                  H_remaining) and takes additional loan if new T > current L
                  (only lever up). Models "every N years, freeze, observe
                  current state, re-decide optimal target" in algorithmic form.

Usage:
  python project_portfolio.py
  python project_portfolio.py --C 160000 --S 180000 --T 5 --S2 30000
  python project_portfolio.py --bootstrap-paths 5000 --bootstrap-block-years 2
"""

import argparse

import numba
import numpy as np

from data_loader import load


TD = 252
BOX_BPS = 0.0015
# Box-spread interest creates 60/40 capital losses for the borrower; for a
# hold-forever SPX investor with no other realized gains, only ~$3k/yr of
# those losses are usable (vs ordinary income), and even the carryforward
# option value is uncertain (depends on future realization events). Setting
# this to 0 is the honest hold-forever assumption.
BOX_TAX_BENEFIT = 0.00
CALL_THRESHOLD = 4.0
REBAL_DAYS = 21
PATH_DTYPE = np.float32   # halves memory vs float64; ~7-digit precision is plenty


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Project portfolio under user-calibrated leverage strategies.")
    p.add_argument("--C", type=float, default=160_000)
    p.add_argument("--S", type=float, default=180_000,
                   help="annual savings during years 0..T (today's $)")
    p.add_argument("--T", type=float, default=5)
    p.add_argument("--S2", type=float, default=30_000,
                   help="annual savings after year T (today's $)")
    p.add_argument("--checkpoints", type=str, default="5,10,15,20,25,30")
    p.add_argument("--max-years", type=float, default=None)
    p.add_argument("--bootstrap-paths", type=int, default=2000)
    p.add_argument("--bootstrap-block-years", type=float, default=1.0)
    p.add_argument("--bootstrap-call-target", type=float, default=0.01,
                   help="acceptable call rate for the 'well-defended' target")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def build_historical_paths(dates, px, tsy, cpi, max_days, min_days=TD):
    """Return ret[K, D+1], tsy[K, D+1], cpi[K, D+1], avail[K], entry_dates[K]
    for monthly post-1932 entries with at least `min_days` of forward data.

    Each path k has valid data for days 1..avail[k]; positions past avail[k]
    are zero-padded (ret, tsy) or carry-forward (cpi). Simulation should mask
    each path past its avail using the returned avail array — at any
    checkpoint, only paths with avail >= checkpoint contribute."""
    cutoff = np.datetime64("1932-07-01")
    dates_np = np.array(dates, dtype="datetime64[D]")
    eligible = [i for i in range(len(dates))
                if dates_np[i] >= cutoff and i + min_days < len(dates)]
    eligible = eligible[::21]
    K = len(eligible)
    ret = np.zeros((K, max_days + 1), dtype=PATH_DTYPE)
    tsy_p = np.zeros((K, max_days + 1), dtype=PATH_DTYPE)
    cpi_p = np.ones((K, max_days + 1), dtype=PATH_DTYPE)
    avail = np.zeros(K, dtype=np.int64)
    entry_dates = np.zeros(K, dtype="datetime64[D]")
    for k, i in enumerate(eligible):
        a = min(max_days, len(dates) - i - 1)
        avail[k] = a
        ret[k, 1:a + 1] = px[i + 1:i + a + 1] / px[i:i + a] - 1.0
        tsy_p[k, :a + 1] = tsy[i:i + a + 1]
        cpi_p[k, :a + 1] = cpi[i:i + a + 1]
        # Forward-fill cpi past avail to avoid zeros (real_eq compute is masked anyway)
        cpi_p[k, a + 1:] = cpi_p[k, a]
        entry_dates[k] = dates_np[i]
    return ret, tsy_p, cpi_p, avail, entry_dates


def stretch_returns(ret, F):
    """Amplify drawdowns in each path's return series by factor F.

    Tracks each path's running max from day 0, computes drawdown from that
    running max, scales the drawdown by F, and reconstitutes returns.
    F=1.0 is identity; F=1.2 makes a 50% historical drawdown into a 60% one.

    At new running highs, stretched price equals original (dd=0 → no change).
    Recovery from a deeper bottom takes a larger relative gain.
    """
    if F == 1.0:
        return ret.copy()
    log_ret = np.log1p(ret)
    cum_log = np.cumsum(log_ret, axis=1)
    px = np.exp(cum_log)   # px[:, 0] = 1.0 (assuming ret[:, 0] = 0)
    running_max = np.maximum.accumulate(px, axis=1)
    dd = 1.0 - px / running_max
    px_new = running_max * (1.0 - F * dd)
    px_new = np.maximum(px_new, 1e-10)   # avoid div-by-zero on extreme F
    ret_new = np.zeros_like(ret)
    ret_new[:, 1:] = px_new[:, 1:] / px_new[:, :-1] - 1.0
    return ret_new


def build_bootstrap_paths(dates, px, tsy, cpi, max_days, n_paths, block_days, rng):
    """Generate n_paths synthetic paths via 1-block bootstrap of joint
    (daily SPX return, tsy yield, daily CPI ratio) tuples from post-1932 data."""
    cutoff = np.datetime64("1932-07-01")
    dates_np = np.array(dates, dtype="datetime64[D]")
    cutoff_idx = int(np.searchsorted(dates_np, cutoff))
    src_ret = px[cutoff_idx + 1:] / px[cutoff_idx:-1] - 1.0
    src_tsy = tsy[cutoff_idx + 1:]
    src_cpi_ratio = cpi[cutoff_idx + 1:] / cpi[cutoff_idx:-1]
    n_src = len(src_ret)
    assert n_src > block_days, "not enough source data for block bootstrap"

    ret = np.zeros((n_paths, max_days + 1), dtype=PATH_DTYPE)
    tsy_p = np.zeros((n_paths, max_days + 1), dtype=PATH_DTYPE)
    cpi_p = np.zeros((n_paths, max_days + 1), dtype=PATH_DTYPE)
    cpi_p[:, 0] = 1.0   # arbitrary starting level; only ratios matter

    for k in range(n_paths):
        d = 1
        while d <= max_days:
            start = int(rng.integers(0, n_src - block_days))
            blen = min(block_days, max_days + 1 - d)
            ret[k, d:d + blen] = src_ret[start:start + blen]
            tsy_p[k, d:d + blen] = src_tsy[start:start + blen]
            cpi_p[k, d:d + blen] = cpi_p[k, d - 1] * np.cumprod(
                src_cpi_ratio[start:start + blen])
            d += blen
    tsy_p[:, 0] = tsy_p[:, 1]
    return ret, tsy_p, cpi_p


# ---------------------------------------------------------------------------
# Pre-compute lookup table for recal_static
# ---------------------------------------------------------------------------

def compute_recal_table(ret_c, tsy_c, cpi_c, avail_c, S2, e_grid, h_grid_years,
                         kind="static", F=1.5, wealth_X=float("inf"),
                         hist_target=0.0, hi=3.0, coarse_n=10, fine_n=10,
                         ret_b=None, tsy_b=None, cpi_b=None, boot_target=0.01,
                         ret_s=None, stretch_F=1.0):
    """Pre-compute T_max(E_real, H_remaining_years) for `kind` base strategy.

    For each (E_real, H_rem) cell, takes the **well-defended** T_max =
    min(T_hist@0%, T_boot@boot_target, T_stress@0%-stretched). Same
    safety architecture as the main app's calibration.

    Args:
        ret_c, tsy_c, cpi_c, avail_c: historical (calibration) paths
        S2: real $/yr DCA from re-cal onward
        e_grid, h_grid_years: cell grids
        kind: base strategy (static / dd_decay / adaptive_dd / hybrid)
        F, wealth_X: strategy params
        hist_target: target call rate on historical paths (default 0%)
        ret_b, tsy_b, cpi_b: bootstrap paths (None = skip bootstrap defense)
        boot_target: target bootstrap call rate (default 1%)
        ret_s: stretched calibration paths (None = skip stretch defense)
        stretch_F: stretch factor (used only for description; precomputed in ret_s)

    Returns:
        t_table[i, j] = well-defended T_max for cell (e_grid[i], h_grid_years[j]).
    """
    n_e = len(e_grid)
    n_h = len(h_grid_years)
    t_table = np.full((n_e, n_h), 1.0, dtype=np.float64)

    full_max_days = ret_c.shape[1] - 1
    use_boot = ret_b is not None and tsy_b is not None and cpi_b is not None
    use_stress = ret_s is not None and stretch_F > 1.0

    for j, h_y in enumerate(h_grid_years):
        h_days = int(h_y * TD)
        if h_days > full_max_days:
            h_days = full_max_days

        # Trim historical paths
        ret_h = ret_c[:, :h_days + 1]
        tsy_h = tsy_c[:, :h_days + 1]
        cpi_h = cpi_c[:, :h_days + 1]
        avail_h = np.minimum(avail_c, h_days)
        eligible = avail_h >= h_days
        if not eligible.any():
            continue
        ret_h = ret_h[eligible]
        tsy_h = tsy_h[eligible]
        cpi_h = cpi_h[eligible]
        avail_h_e = avail_h[eligible]

        # Trim bootstrap paths
        if use_boot:
            ret_bh = ret_b[:, :h_days + 1]
            tsy_bh = tsy_b[:, :h_days + 1]
            cpi_bh = cpi_b[:, :h_days + 1]

        # Trim stretched calibration paths
        if use_stress:
            ret_sh = ret_s[:, :h_days + 1]
            tsy_sh = tsy_c[:, :h_days + 1][eligible]
            cpi_sh = cpi_c[:, :h_days + 1][eligible]
            ret_sh = ret_sh[eligible]

        for i, e0 in enumerate(e_grid):
            T_hist = find_max_safe_T_grid(
                ret_h, tsy_h, cpi_h, kind, hist_target,
                float(e0), float(S2), 1e9, float(S2),
                h_days, avail=avail_h_e, hi=hi, F=F, wealth_X=wealth_X,
                coarse_n=coarse_n, fine_n=fine_n,
            )
            T_boot = float("inf")
            if use_boot:
                T_boot = find_max_safe_T_grid(
                    ret_bh, tsy_bh, cpi_bh, kind, boot_target,
                    float(e0), float(S2), 1e9, float(S2),
                    h_days, hi=hi, F=F, wealth_X=wealth_X,
                    coarse_n=coarse_n, fine_n=fine_n,
                )
            T_stress = float("inf")
            if use_stress:
                T_stress = find_max_safe_T_grid(
                    ret_sh, tsy_sh, cpi_sh, kind, hist_target,
                    float(e0), float(S2), 1e9, float(S2),
                    h_days, avail=avail_h_e, hi=hi, F=F, wealth_X=wealth_X,
                    coarse_n=coarse_n, fine_n=fine_n,
                )
            t_table[i, j] = min(T_hist, T_boot, T_stress)
    return t_table


def compute_recal_tables_multi(ret_c, tsy_c, cpi_c, avail_c, S2, e_grid,
                                h_grid_years, kinds, F=1.5, wealth_X=float("inf"),
                                hist_target=0.0, coarse_n=8, fine_n=8,
                                ret_b=None, tsy_b=None, cpi_b=None,
                                boot_target=0.01, ret_s=None, stretch_F=1.0):
    """Compute well-defended recal lookup tables for multiple base strategies."""
    n_s = len(kinds)
    n_e = len(e_grid)
    n_h = len(h_grid_years)
    out = np.full((n_s, n_e, n_h), 1.0, dtype=np.float64)
    for s, kind in enumerate(kinds):
        out[s] = compute_recal_table(
            ret_c, tsy_c, cpi_c, avail_c, S2, e_grid, h_grid_years,
            kind=kind, F=F, wealth_X=wealth_X, hist_target=hist_target,
            coarse_n=coarse_n, fine_n=fine_n,
            ret_b=ret_b, tsy_b=tsy_b, cpi_b=cpi_b, boot_target=boot_target,
            ret_s=ret_s, stretch_F=stretch_F)
    return out


# ---------------------------------------------------------------------------
# Vectorized simulation (Numba JIT)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _realized_vol_60(ret_path, d):
    """Annualized 60-day realized vol for a path at day d. Uses returns from
    max(1, d-59)..d (inclusive). Returns 0.0 if fewer than 2 samples."""
    lo = d - 59
    if lo < 1:
        lo = 1
    n = d - lo + 1
    if n < 2:
        return 0.0
    s = 0.0
    s2 = 0.0
    for i in range(lo, d + 1):
        r = ret_path[i]
        s += r
        s2 += r * r
    mean = s / n
    var = (s2 - n * mean * mean) / (n - 1)
    if var <= 0.0:
        return 0.0
    return np.sqrt(var * 252.0)


@numba.njit(cache=True, fastmath=True)
def _simulate_core(ret, tsy, cpi, kind_code, T_init, C, S, T_yrs, S2, max_days,
                   avail, F, floor, checkpoint_days, cap_real, wealth_X,
                   vol_factor, dip_threshold, dip_bonus,
                   rate_threshold, rate_factor,
                   recal_period_days, t_recal_table, e_recal_grid,
                   h_recal_grid_days,
                   t_recal_tables_meta, meta_strategy_codes,
                   init_strat_idx):
    """JIT-compiled per-day, per-path inner loop. Avoids temporary array
    allocations entirely. kind_code: 0=static, 1=relever, 2=dd_decay,
    3=wealth_decay, 4=hybrid, 5=r_hybrid, 6=vol_hybrid, 7=dip_hybrid,
    8=rate_hybrid.

    `checkpoint_days` (sorted int64 array): the day-indices at which to
    capture the path's instantaneous leverage. Output `lev_at_cp[k, c]` is
    leverage of path k at checkpoint c, or NaN if the path was already
    called/out-of-data at that checkpoint.

    `cap_real`: real-dollar wealth threshold above which the strategy stops
    levering up. Once a path's real equity crosses cap_real, the rebalance
    step is skipped permanently for that path (cap_reached flag). Pass
    np.inf to disable.

    Path arrays (ret, tsy, cpi) are typically float32; state arrays are
    float64 for precision. Numba auto-casts on arithmetic mixing.
    """
    K = ret.shape[0]
    n_cp = checkpoint_days.shape[0]
    stocks = np.full(K, C * T_init)
    loan = np.full(K, C * (T_init - 1.0))
    hwm_eq = np.full(K, float(C))
    max_dd = np.zeros(K)
    max_w_prog = np.zeros(K)   # ratcheted wealth progress (kind=5)
    cur_tgt = np.full(K, float(T_init))   # adaptive_dd monotonic ratchet
    T_active = np.full(K, float(T_init))   # per-path T_init (recal kinds 12/13/14)
    strat_active = np.full(K, init_strat_idx, dtype=np.int64)   # meta_recal: index of selected strategy
    called = np.zeros(K, dtype=np.bool_)
    cap_reached = np.zeros(K, dtype=np.bool_)
    peak_lev = np.full(K, float(T_init))

    real_eq = np.full((K, max_days + 1), np.nan)
    lev_at_cp = np.full((K, n_cp), np.nan)
    for k in range(K):
        real_eq[k, 0] = C

    t_switch_days = T_yrs * TD
    # recal_* kinds may add loan later via lookup; treat them as always levered
    is_levered = (T_init > 1.0) or (kind_code >= 11)
    cp_idx = 0   # which checkpoint comes next

    for d in range(1, max_days + 1):
        s_real_yr = S if d < t_switch_days else S2
        # recal_* kinds 11-14: rebalance only on recal days
        # kinds 12/13/14: ALSO rebalance on regular REBAL_DAYS (between recals)
        is_recal_day = (kind_code >= 11) and (recal_period_days > 0) and (d > 0) and (d % recal_period_days == 0)
        if kind_code == 11:
            do_rebal = is_recal_day
        elif kind_code >= 12:
            do_rebal = is_recal_day or (d % REBAL_DAYS == 0)
        else:
            do_rebal = (kind_code != 0) and (d % REBAL_DAYS == 0)
        is_cp = (cp_idx < n_cp) and (d == checkpoint_days[cp_idx])

        for k in range(K):
            if called[k] or d > avail[k]:
                continue

            stocks_new = stocks[k] * (1.0 + ret[k, d])
            if is_levered:
                box_rate = (tsy[k, d] + BOX_BPS) * (1.0 - BOX_TAX_BENEFIT)
                loan_new = loan[k] * (1.0 + box_rate / TD)
            else:
                loan_new = loan[k]

            if s_real_yr > 0.0:
                stocks_new = stocks_new + s_real_yr * cpi[k, d] / cpi[k, 0] / TD

            if is_levered:
                eq_new = stocks_new - loan_new
                if eq_new <= 0.0:
                    called[k] = True
                    continue
                lev_new = stocks_new / eq_new
                if lev_new >= CALL_THRESHOLD:
                    called[k] = True
                    continue

            stocks[k] = stocks_new
            loan[k] = loan_new

            # Capture *actual* (pre-rebalance) leverage at checkpoint days. On
            # a rebalance day this is the natural drift since the previous
            # rebalance — for relever/dd_decay this is what shows you the
            # leverage decay between rebalances; post-rebalance it would just
            # equal the target.
            if is_cp:
                eq_pre = stocks[k] - loan[k]
                if eq_pre > 0.0:
                    lev_at_cp[k, cp_idx] = stocks[k] / eq_pre

            # Latch the cap once real wealth crosses it (permanent flag)
            eq_pre_rebal = stocks[k] - loan[k]
            real_eq_now = eq_pre_rebal * cpi[k, 0] / cpi[k, d]
            if real_eq_now >= cap_real:
                cap_reached[k] = True

            if do_rebal and not cap_reached[k]:
                eq = stocks[k] - loan[k]
                if kind_code == 2:   # dd_decay
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand = T_init - F * max_dd[k]
                    target_lev = floor if cand < floor else cand
                elif kind_code == 3:   # wealth_decay (current eq, real $)
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        target_lev = T_init - (T_init - floor) * prog
                    else:
                        target_lev = floor
                elif kind_code == 4:   # hybrid (min of dd_decay & wealth_decay)
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand_dd = T_init - F * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        cand_w = T_init - (T_init - floor) * prog
                    else:
                        cand_w = floor
                    target_lev = cand_dd if cand_dd < cand_w else cand_w
                elif kind_code == 5:   # r_hybrid (ratcheted wealth progress)
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand_dd = T_init - F * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        if prog > max_w_prog[k]:
                            max_w_prog[k] = prog
                        cand_w = T_init - (T_init - floor) * max_w_prog[k]
                    else:
                        cand_w = floor
                    target_lev = cand_dd if cand_dd < cand_w else cand_w
                elif kind_code == 6:   # vol_hybrid
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand_dd = T_init - F * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        cand_w = T_init - (T_init - floor) * prog
                    else:
                        cand_w = floor
                    base = cand_dd if cand_dd < cand_w else cand_w
                    vol_ann = _realized_vol_60(ret[k], d)
                    cand = base - vol_factor * vol_ann
                    target_lev = floor if cand < floor else cand
                elif kind_code == 7:   # dip_hybrid
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand_dd = T_init - F * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        cand_w = T_init - (T_init - floor) * prog
                    else:
                        cand_w = floor
                    base = cand_dd if cand_dd < cand_w else cand_w
                    cur_dd = 0.0
                    if hwm_eq[k] > 0.0:
                        cur_dd = 1.0 - eq / hwm_eq[k]
                    if cur_dd > dip_threshold:
                        dip_floor = T_init + dip_bonus
                        target_lev = base if base > dip_floor else dip_floor
                    else:
                        target_lev = base
                elif kind_code == 8:   # rate_hybrid
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand_dd = T_init - F * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        cand_w = T_init - (T_init - floor) * prog
                    else:
                        cand_w = floor
                    base = cand_dd if cand_dd < cand_w else cand_w
                    rate_excess = tsy[k, d] - rate_threshold
                    if rate_excess < 0.0:
                        rate_excess = 0.0
                    cand = base - rate_factor * rate_excess
                    target_lev = floor if cand < floor else cand
                elif kind_code == 9:   # adaptive_dd
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    if eq > 0.0:
                        L_now = stocks[k] / eq
                    else:
                        L_now = T_init
                    if T_init > 1.0:
                        F_eff = F * (L_now - 1.0) / (T_init - 1.0)
                    else:
                        F_eff = 0.0
                    if F_eff < 0.0:
                        F_eff = 0.0
                    cand = T_init - F_eff * max_dd[k]
                    if cand < floor:
                        cand = floor
                    if cand < cur_tgt[k]:
                        cur_tgt[k] = cand
                    target_lev = cur_tgt[k]
                elif kind_code == 10:   # adaptive_hybrid (adaptive_dd + wealth)
                    if eq > hwm_eq[k]:
                        hwm_eq[k] = eq
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    if eq > 0.0:
                        L_now = stocks[k] / eq
                    else:
                        L_now = T_init
                    if T_init > 1.0:
                        F_eff = F * (L_now - 1.0) / (T_init - 1.0)
                    else:
                        F_eff = 0.0
                    if F_eff < 0.0:
                        F_eff = 0.0
                    cand_dd = T_init - F_eff * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor
                    if cand_dd < cur_tgt[k]:
                        cur_tgt[k] = cand_dd
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    if wealth_X > C:
                        prog = (real_eq_d - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        cand_w = T_init - (T_init - floor) * prog
                    else:
                        cand_w = floor
                    target_lev = cur_tgt[k] if cur_tgt[k] < cand_w else cand_w
                elif kind_code == 11:   # recal_static (lookup-table re-cal)
                    real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                    h_remaining = max_days - d
                    n_e = e_recal_grid.shape[0]
                    n_h = h_recal_grid_days.shape[0]
                    e_idx = 0
                    best_de = e_recal_grid[0] - real_eq_d
                    if best_de < 0.0:
                        best_de = -best_de
                    for ee in range(1, n_e):
                        de_v = e_recal_grid[ee] - real_eq_d
                        if de_v < 0.0:
                            de_v = -de_v
                        if de_v < best_de:
                            best_de = de_v
                            e_idx = ee
                    h_idx = 0
                    best_dh = h_recal_grid_days[0] - h_remaining
                    if best_dh < 0:
                        best_dh = -best_dh
                    for hh in range(1, n_h):
                        dh_v = h_recal_grid_days[hh] - h_remaining
                        if dh_v < 0:
                            dh_v = -dh_v
                        if dh_v < best_dh:
                            best_dh = dh_v
                            h_idx = hh
                    target_lev = t_recal_table[e_idx, h_idx]
                    if target_lev < floor:
                        target_lev = floor
                elif kind_code == 12:   # recal_hybrid
                    if is_recal_day:
                        # Lookup + state reset + T_active update
                        real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                        h_remaining = max_days - d
                        n_e = e_recal_grid.shape[0]
                        n_h = h_recal_grid_days.shape[0]
                        e_idx = 0
                        best_de = e_recal_grid[0] - real_eq_d
                        if best_de < 0.0:
                            best_de = -best_de
                        for ee in range(1, n_e):
                            de_v = e_recal_grid[ee] - real_eq_d
                            if de_v < 0.0:
                                de_v = -de_v
                            if de_v < best_de:
                                best_de = de_v
                                e_idx = ee
                        h_idx = 0
                        best_dh = h_recal_grid_days[0] - h_remaining
                        if best_dh < 0:
                            best_dh = -best_dh
                        for hh in range(1, n_h):
                            dh_v = h_recal_grid_days[hh] - h_remaining
                            if dh_v < 0:
                                dh_v = -dh_v
                            if dh_v < best_dh:
                                best_dh = dh_v
                                h_idx = hh
                        new_T = t_recal_table[e_idx, h_idx]
                        if new_T < floor:
                            new_T = floor
                        T_active[k] = new_T
                        hwm_eq[k] = eq
                        max_dd[k] = 0.0
                        target_lev = new_T
                    else:
                        # Regular hybrid logic with T_active[k] as T_init
                        T_a = T_active[k]
                        if eq > hwm_eq[k]:
                            hwm_eq[k] = eq
                        if hwm_eq[k] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k]
                            if dd_now > max_dd[k]:
                                max_dd[k] = dd_now
                        cand_dd = T_a - F * max_dd[k]
                        if cand_dd < floor:
                            cand_dd = floor
                        real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            cand_w = T_a - (T_a - floor) * prog
                        else:
                            cand_w = floor
                        target_lev = cand_dd if cand_dd < cand_w else cand_w
                elif kind_code == 13:   # recal_adaptive_dd
                    if is_recal_day:
                        real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                        h_remaining = max_days - d
                        n_e = e_recal_grid.shape[0]
                        n_h = h_recal_grid_days.shape[0]
                        e_idx = 0
                        best_de = e_recal_grid[0] - real_eq_d
                        if best_de < 0.0:
                            best_de = -best_de
                        for ee in range(1, n_e):
                            de_v = e_recal_grid[ee] - real_eq_d
                            if de_v < 0.0:
                                de_v = -de_v
                            if de_v < best_de:
                                best_de = de_v
                                e_idx = ee
                        h_idx = 0
                        best_dh = h_recal_grid_days[0] - h_remaining
                        if best_dh < 0:
                            best_dh = -best_dh
                        for hh in range(1, n_h):
                            dh_v = h_recal_grid_days[hh] - h_remaining
                            if dh_v < 0:
                                dh_v = -dh_v
                            if dh_v < best_dh:
                                best_dh = dh_v
                                h_idx = hh
                        new_T = t_recal_table[e_idx, h_idx]
                        if new_T < floor:
                            new_T = floor
                        T_active[k] = new_T
                        hwm_eq[k] = eq
                        max_dd[k] = 0.0
                        cur_tgt[k] = new_T
                        target_lev = new_T
                    else:
                        T_a = T_active[k]
                        if eq > hwm_eq[k]:
                            hwm_eq[k] = eq
                        if hwm_eq[k] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k]
                            if dd_now > max_dd[k]:
                                max_dd[k] = dd_now
                        if eq > 0.0:
                            L_now = stocks[k] / eq
                        else:
                            L_now = T_a
                        if T_a > 1.0:
                            F_eff = F * (L_now - 1.0) / (T_a - 1.0)
                        else:
                            F_eff = 0.0
                        if F_eff < 0.0:
                            F_eff = 0.0
                        cand = T_a - F_eff * max_dd[k]
                        if cand < floor:
                            cand = floor
                        if cand < cur_tgt[k]:
                            cur_tgt[k] = cand
                        target_lev = cur_tgt[k]
                elif kind_code == 14:   # meta_recal: pick max-T strategy among candidates
                    if is_recal_day:
                        real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                        h_remaining = max_days - d
                        n_e = e_recal_grid.shape[0]
                        n_h = h_recal_grid_days.shape[0]
                        e_idx = 0
                        best_de = e_recal_grid[0] - real_eq_d
                        if best_de < 0.0:
                            best_de = -best_de
                        for ee in range(1, n_e):
                            de_v = e_recal_grid[ee] - real_eq_d
                            if de_v < 0.0:
                                de_v = -de_v
                            if de_v < best_de:
                                best_de = de_v
                                e_idx = ee
                        h_idx = 0
                        best_dh = h_recal_grid_days[0] - h_remaining
                        if best_dh < 0:
                            best_dh = -best_dh
                        for hh in range(1, n_h):
                            dh_v = h_recal_grid_days[hh] - h_remaining
                            if dh_v < 0:
                                dh_v = -dh_v
                            if dh_v < best_dh:
                                best_dh = dh_v
                                h_idx = hh
                        # Pick strategy with highest T_max for this cell
                        n_meta = t_recal_tables_meta.shape[0]
                        best_s = 0
                        best_T = t_recal_tables_meta[0, e_idx, h_idx]
                        for ss in range(1, n_meta):
                            t_s = t_recal_tables_meta[ss, e_idx, h_idx]
                            if t_s > best_T:
                                best_T = t_s
                                best_s = ss
                        if best_T < floor:
                            best_T = floor
                        T_active[k] = best_T
                        strat_active[k] = best_s
                        hwm_eq[k] = eq
                        max_dd[k] = 0.0
                        cur_tgt[k] = best_T
                        target_lev = best_T
                    else:
                        # Apply active strategy's logic with T_active[k]
                        s_code = meta_strategy_codes[strat_active[k]]
                        T_a = T_active[k]
                        if s_code == 0:   # static between recals: no rebalance
                            target_lev = stocks[k] / eq if eq > 0.0 else T_a
                        elif s_code == 2:   # dd_decay
                            if eq > hwm_eq[k]:
                                hwm_eq[k] = eq
                            if hwm_eq[k] > 0.0:
                                dd_now = 1.0 - eq / hwm_eq[k]
                                if dd_now > max_dd[k]:
                                    max_dd[k] = dd_now
                            cand = T_a - F * max_dd[k]
                            target_lev = floor if cand < floor else cand
                        elif s_code == 9:   # adaptive_dd
                            if eq > hwm_eq[k]:
                                hwm_eq[k] = eq
                            if hwm_eq[k] > 0.0:
                                dd_now = 1.0 - eq / hwm_eq[k]
                                if dd_now > max_dd[k]:
                                    max_dd[k] = dd_now
                            if eq > 0.0:
                                L_now = stocks[k] / eq
                            else:
                                L_now = T_a
                            if T_a > 1.0:
                                F_eff = F * (L_now - 1.0) / (T_a - 1.0)
                            else:
                                F_eff = 0.0
                            if F_eff < 0.0:
                                F_eff = 0.0
                            cand = T_a - F_eff * max_dd[k]
                            if cand < floor:
                                cand = floor
                            if cand < cur_tgt[k]:
                                cur_tgt[k] = cand
                            target_lev = cur_tgt[k]
                        elif s_code == 4:   # hybrid
                            if eq > hwm_eq[k]:
                                hwm_eq[k] = eq
                            if hwm_eq[k] > 0.0:
                                dd_now = 1.0 - eq / hwm_eq[k]
                                if dd_now > max_dd[k]:
                                    max_dd[k] = dd_now
                            cand_dd = T_a - F * max_dd[k]
                            if cand_dd < floor:
                                cand_dd = floor
                            real_eq_d = eq * cpi[k, 0] / cpi[k, d]
                            if wealth_X > C:
                                prog = (real_eq_d - C) / (wealth_X - C)
                                if prog < 0.0:
                                    prog = 0.0
                                elif prog > 1.0:
                                    prog = 1.0
                                cand_w = T_a - (T_a - floor) * prog
                            else:
                                cand_w = floor
                            target_lev = cand_dd if cand_dd < cand_w else cand_w
                        else:
                            target_lev = T_a
                else:
                    target_lev = T_init
                delta = target_lev * eq - stocks[k]
                if delta > 0.0:
                    stocks[k] += delta
                    loan[k] += delta

            eq = stocks[k] - loan[k]
            if is_levered and eq > 0.0:
                lev_now = stocks[k] / eq
                if lev_now > peak_lev[k]:
                    peak_lev[k] = lev_now
            real_eq[k, d] = eq * cpi[k, 0] / cpi[k, d]

        if is_cp:
            cp_idx += 1

    return real_eq, called, peak_lev, lev_at_cp


_KIND_CODES = {"static": 0, "relever": 1, "dd_decay": 2,
               "wealth_decay": 3, "hybrid": 4,
               "r_hybrid": 5, "vol_hybrid": 6, "dip_hybrid": 7,
               "rate_hybrid": 8, "adaptive_dd": 9, "adaptive_hybrid": 10,
               "recal_static": 11, "recal_hybrid": 12,
               "recal_adaptive_dd": 13, "meta_recal": 14}


def simulate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
             avail=None, F=1.5, floor=1.0, checkpoint_days=None,
             cap_real=float("inf"), wealth_X=float("inf"),
             vol_factor=0.0, dip_threshold=0.0, dip_bonus=0.0,
             rate_threshold=float("inf"), rate_factor=0.0,
             recal_period_days=0, t_recal_table=None,
             e_recal_grid=None, h_recal_grid_days=None,
             t_recal_tables_meta=None, meta_strategy_codes=None,
             init_strat_idx=0):
    """Wrapper around the JIT inner loop. Coerces `kind` string to int and
    sets defaults. Returns (real_eq, called, peak_lev, lev_at_cp).

    `avail[k]` (optional) is the last valid day for path k — paths past their
    avail are masked the same way as called paths. If None, every path runs
    the full max_days.

    `checkpoint_days` (optional sorted int64 array) selects days at which to
    capture instantaneous leverage. If None, no leverage tracking; lev_at_cp
    has shape (K, 0).

    `cap_real` (default inf): real-dollar wealth threshold beyond which the
    strategy stops levering up (latches once crossed). Inf disables the cap.

    `wealth_X` (default inf): real-dollar wealth target where wealth_decay /
    hybrid kinds reach `floor`. Linear interpolation between C and wealth_X.
    Ignored for kinds other than wealth_decay (3) and hybrid (4).
    """
    K = ret.shape[0]
    if avail is None:
        avail = np.full(K, max_days, dtype=np.int64)
    elif avail.dtype != np.int64:
        avail = avail.astype(np.int64)
    if checkpoint_days is None:
        checkpoint_days = np.zeros(0, dtype=np.int64)
    elif checkpoint_days.dtype != np.int64:
        checkpoint_days = checkpoint_days.astype(np.int64)
    if t_recal_table is None:
        t_recal_table = np.zeros((1, 1), dtype=np.float64)
    elif t_recal_table.dtype != np.float64:
        t_recal_table = t_recal_table.astype(np.float64)
    if e_recal_grid is None:
        e_recal_grid = np.zeros(1, dtype=np.float64)
    elif e_recal_grid.dtype != np.float64:
        e_recal_grid = e_recal_grid.astype(np.float64)
    if h_recal_grid_days is None:
        h_recal_grid_days = np.zeros(1, dtype=np.int64)
    elif h_recal_grid_days.dtype != np.int64:
        h_recal_grid_days = h_recal_grid_days.astype(np.int64)
    if t_recal_tables_meta is None:
        t_recal_tables_meta = np.zeros((1, 1, 1), dtype=np.float64)
    elif t_recal_tables_meta.dtype != np.float64:
        t_recal_tables_meta = t_recal_tables_meta.astype(np.float64)
    if meta_strategy_codes is None:
        meta_strategy_codes = np.zeros(1, dtype=np.int64)
    elif meta_strategy_codes.dtype != np.int64:
        meta_strategy_codes = meta_strategy_codes.astype(np.int64)
    return _simulate_core(ret, tsy, cpi, _KIND_CODES[kind], float(T_init),
                          float(C), float(S), float(T_yrs), float(S2),
                          int(max_days), avail, float(F), float(floor),
                          checkpoint_days, float(cap_real), float(wealth_X),
                          float(vol_factor), float(dip_threshold),
                          float(dip_bonus), float(rate_threshold),
                          float(rate_factor),
                          int(recal_period_days), t_recal_table,
                          e_recal_grid, h_recal_grid_days,
                          t_recal_tables_meta, meta_strategy_codes,
                          int(init_strat_idx))


def call_rate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
              avail=None, F=1.5, cap_real=float("inf"), wealth_X=float("inf"),
              vol_factor=0.0, dip_threshold=0.0, dip_bonus=0.0,
              rate_threshold=float("inf"), rate_factor=0.0,
              recal_period_days=0, t_recal_table=None,
              e_recal_grid=None, h_recal_grid_days=None,
              t_recal_tables_meta=None, meta_strategy_codes=None,
              init_strat_idx=0):
    _, called, _, _ = simulate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
                            avail=avail, F=F, cap_real=cap_real, wealth_X=wealth_X,
                            vol_factor=vol_factor, dip_threshold=dip_threshold,
                            dip_bonus=dip_bonus, rate_threshold=rate_threshold,
                            rate_factor=rate_factor,
                            recal_period_days=recal_period_days,
                            t_recal_table=t_recal_table,
                            e_recal_grid=e_recal_grid,
                            h_recal_grid_days=h_recal_grid_days,
                            t_recal_tables_meta=t_recal_tables_meta,
                            meta_strategy_codes=meta_strategy_codes,
                            init_strat_idx=init_strat_idx)
    return float(called.mean())


# ---------------------------------------------------------------------------
# Grid-vectorized simulation: test many T values in one sim
# ---------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def _simulate_core_grid(ret, tsy, cpi, kind_code, T_inits, C, S, T_yrs, S2,
                        max_days, avail, F, floor, cap_real, wealth_X,
                        vol_factor, dip_threshold, dip_bonus,
                        rate_threshold, rate_factor,
                        recal_period_days, t_recal_table, e_recal_grid,
                        h_recal_grid_days,
                        t_recal_tables_meta, meta_strategy_codes,
                        init_strat_idx):
    """Vectorized over T. Runs all T_inits values simultaneously, sharing the
    same per-path data. Returns `called[K, T_count]` only — skips real_eq,
    peak_lev, and leverage tracking since binary search only needs call counts.
    Memory and arithmetic both scale with T_count, but cache reuse on the
    shared ret/tsy/cpi data and SIMD on the inner T loop make this much
    faster than T_count separate single-T sims."""
    K = ret.shape[0]
    T_count = T_inits.shape[0]

    stocks = np.empty((K, T_count))
    loan = np.empty((K, T_count))
    hwm_eq = np.full((K, T_count), float(C))
    max_dd = np.zeros((K, T_count))
    max_w_prog = np.zeros((K, T_count))
    cur_tgt = np.empty((K, T_count))
    T_active = np.empty((K, T_count))
    strat_active = np.full((K, T_count), init_strat_idx, dtype=np.int64)
    called = np.zeros((K, T_count), dtype=np.bool_)
    cap_reached = np.zeros((K, T_count), dtype=np.bool_)

    for k in range(K):
        for t in range(T_count):
            T_init = T_inits[t]
            stocks[k, t] = C * T_init
            loan[k, t] = C * (T_init - 1.0)
            cur_tgt[k, t] = T_init
            T_active[k, t] = T_init

    t_switch_days = T_yrs * TD

    for d in range(1, max_days + 1):
        s_real_yr = S if d < t_switch_days else S2
        is_recal_day = (kind_code >= 11) and (recal_period_days > 0) and (d > 0) and (d % recal_period_days == 0)
        if kind_code == 11:
            do_rebal = is_recal_day
        elif kind_code >= 12:
            do_rebal = is_recal_day or (d % REBAL_DAYS == 0)
        else:
            do_rebal = (kind_code != 0) and (d % REBAL_DAYS == 0)

        for k in range(K):
            if d > avail[k]:
                continue

            ret_d = ret[k, d]
            tsy_d = tsy[k, d]
            cpi_d = cpi[k, d]
            cpi_0 = cpi[k, 0]
            box_rate_term = (tsy_d + BOX_BPS) * (1.0 - BOX_TAX_BENEFIT) / TD

            for t in range(T_count):
                if called[k, t]:
                    continue

                T_init = T_inits[t]
                is_levered = (T_init > 1.0) or (kind_code >= 11)

                stocks_new = stocks[k, t] * (1.0 + ret_d)
                if is_levered:
                    loan_new = loan[k, t] * (1.0 + box_rate_term)
                else:
                    loan_new = loan[k, t]

                if s_real_yr > 0.0:
                    stocks_new = stocks_new + s_real_yr * cpi_d / cpi_0 / TD

                if is_levered:
                    eq_new = stocks_new - loan_new
                    if eq_new <= 0.0:
                        called[k, t] = True
                        continue
                    lev_new = stocks_new / eq_new
                    if lev_new >= CALL_THRESHOLD:
                        called[k, t] = True
                        continue

                stocks[k, t] = stocks_new
                loan[k, t] = loan_new

                # Latch the cap once real wealth crosses it
                eq_pre_rebal = stocks[k, t] - loan[k, t]
                real_eq_now = eq_pre_rebal * cpi_0 / cpi_d
                if real_eq_now >= cap_real:
                    cap_reached[k, t] = True

                if do_rebal and not cap_reached[k, t]:
                    eq = stocks[k, t] - loan[k, t]
                    if kind_code == 2:   # dd_decay
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        cand = T_init - F * max_dd[k, t]
                        target_lev = floor if cand < floor else cand
                    elif kind_code == 3:   # wealth_decay (current eq, real $)
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            target_lev = T_init - (T_init - floor) * prog
                        else:
                            target_lev = floor
                    elif kind_code == 4:   # hybrid (min of dd_decay & wealth_decay)
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        cand_dd = T_init - F * max_dd[k, t]
                        if cand_dd < floor:
                            cand_dd = floor
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            cand_w = T_init - (T_init - floor) * prog
                        else:
                            cand_w = floor
                        target_lev = cand_dd if cand_dd < cand_w else cand_w
                    elif kind_code == 5:   # r_hybrid (ratcheted wealth)
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        cand_dd = T_init - F * max_dd[k, t]
                        if cand_dd < floor:
                            cand_dd = floor
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            if prog > max_w_prog[k, t]:
                                max_w_prog[k, t] = prog
                            cand_w = T_init - (T_init - floor) * max_w_prog[k, t]
                        else:
                            cand_w = floor
                        target_lev = cand_dd if cand_dd < cand_w else cand_w
                    elif kind_code == 6:   # vol_hybrid
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        cand_dd = T_init - F * max_dd[k, t]
                        if cand_dd < floor:
                            cand_dd = floor
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            cand_w = T_init - (T_init - floor) * prog
                        else:
                            cand_w = floor
                        base = cand_dd if cand_dd < cand_w else cand_w
                        vol_ann = _realized_vol_60(ret[k], d)
                        cand = base - vol_factor * vol_ann
                        target_lev = floor if cand < floor else cand
                    elif kind_code == 7:   # dip_hybrid
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        cand_dd = T_init - F * max_dd[k, t]
                        if cand_dd < floor:
                            cand_dd = floor
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            cand_w = T_init - (T_init - floor) * prog
                        else:
                            cand_w = floor
                        base = cand_dd if cand_dd < cand_w else cand_w
                        cur_dd = 0.0
                        if hwm_eq[k, t] > 0.0:
                            cur_dd = 1.0 - eq / hwm_eq[k, t]
                        if cur_dd > dip_threshold:
                            dip_floor = T_init + dip_bonus
                            target_lev = base if base > dip_floor else dip_floor
                        else:
                            target_lev = base
                    elif kind_code == 8:   # rate_hybrid
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        cand_dd = T_init - F * max_dd[k, t]
                        if cand_dd < floor:
                            cand_dd = floor
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            cand_w = T_init - (T_init - floor) * prog
                        else:
                            cand_w = floor
                        base = cand_dd if cand_dd < cand_w else cand_w
                        rate_excess = tsy_d - rate_threshold
                        if rate_excess < 0.0:
                            rate_excess = 0.0
                        cand = base - rate_factor * rate_excess
                        target_lev = floor if cand < floor else cand
                    elif kind_code == 9:   # adaptive_dd
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        if eq > 0.0:
                            L_now = stocks[k, t] / eq
                        else:
                            L_now = T_init
                        if T_init > 1.0:
                            F_eff = F * (L_now - 1.0) / (T_init - 1.0)
                        else:
                            F_eff = 0.0
                        if F_eff < 0.0:
                            F_eff = 0.0
                        cand = T_init - F_eff * max_dd[k, t]
                        if cand < floor:
                            cand = floor
                        if cand < cur_tgt[k, t]:
                            cur_tgt[k, t] = cand
                        target_lev = cur_tgt[k, t]
                    elif kind_code == 10:   # adaptive_hybrid
                        if eq > hwm_eq[k, t]:
                            hwm_eq[k, t] = eq
                        if hwm_eq[k, t] > 0.0:
                            dd_now = 1.0 - eq / hwm_eq[k, t]
                            if dd_now > max_dd[k, t]:
                                max_dd[k, t] = dd_now
                        if eq > 0.0:
                            L_now = stocks[k, t] / eq
                        else:
                            L_now = T_init
                        if T_init > 1.0:
                            F_eff = F * (L_now - 1.0) / (T_init - 1.0)
                        else:
                            F_eff = 0.0
                        if F_eff < 0.0:
                            F_eff = 0.0
                        cand_dd = T_init - F_eff * max_dd[k, t]
                        if cand_dd < floor:
                            cand_dd = floor
                        if cand_dd < cur_tgt[k, t]:
                            cur_tgt[k, t] = cand_dd
                        real_eq_d = eq * cpi_0 / cpi_d
                        if wealth_X > C:
                            prog = (real_eq_d - C) / (wealth_X - C)
                            if prog < 0.0:
                                prog = 0.0
                            elif prog > 1.0:
                                prog = 1.0
                            cand_w = T_init - (T_init - floor) * prog
                        else:
                            cand_w = floor
                        target_lev = cur_tgt[k, t] if cur_tgt[k, t] < cand_w else cand_w
                    elif kind_code == 11:   # recal_static
                        real_eq_d = eq * cpi_0 / cpi_d
                        h_remaining = max_days - d
                        n_e = e_recal_grid.shape[0]
                        n_h = h_recal_grid_days.shape[0]
                        e_idx = 0
                        best_de = e_recal_grid[0] - real_eq_d
                        if best_de < 0.0:
                            best_de = -best_de
                        for ee in range(1, n_e):
                            de_v = e_recal_grid[ee] - real_eq_d
                            if de_v < 0.0:
                                de_v = -de_v
                            if de_v < best_de:
                                best_de = de_v
                                e_idx = ee
                        h_idx = 0
                        best_dh = h_recal_grid_days[0] - h_remaining
                        if best_dh < 0:
                            best_dh = -best_dh
                        for hh in range(1, n_h):
                            dh_v = h_recal_grid_days[hh] - h_remaining
                            if dh_v < 0:
                                dh_v = -dh_v
                            if dh_v < best_dh:
                                best_dh = dh_v
                                h_idx = hh
                        target_lev = t_recal_table[e_idx, h_idx]
                        if target_lev < floor:
                            target_lev = floor
                    elif kind_code == 12:   # recal_hybrid (grid)
                        if is_recal_day:
                            real_eq_d = eq * cpi_0 / cpi_d
                            h_remaining = max_days - d
                            n_e = e_recal_grid.shape[0]
                            n_h = h_recal_grid_days.shape[0]
                            e_idx = 0
                            best_de = e_recal_grid[0] - real_eq_d
                            if best_de < 0.0:
                                best_de = -best_de
                            for ee in range(1, n_e):
                                de_v = e_recal_grid[ee] - real_eq_d
                                if de_v < 0.0:
                                    de_v = -de_v
                                if de_v < best_de:
                                    best_de = de_v
                                    e_idx = ee
                            h_idx = 0
                            best_dh = h_recal_grid_days[0] - h_remaining
                            if best_dh < 0:
                                best_dh = -best_dh
                            for hh in range(1, n_h):
                                dh_v = h_recal_grid_days[hh] - h_remaining
                                if dh_v < 0:
                                    dh_v = -dh_v
                                if dh_v < best_dh:
                                    best_dh = dh_v
                                    h_idx = hh
                            new_T = t_recal_table[e_idx, h_idx]
                            if new_T < floor:
                                new_T = floor
                            T_active[k, t] = new_T
                            hwm_eq[k, t] = eq
                            max_dd[k, t] = 0.0
                            target_lev = new_T
                        else:
                            T_a = T_active[k, t]
                            if eq > hwm_eq[k, t]:
                                hwm_eq[k, t] = eq
                            if hwm_eq[k, t] > 0.0:
                                dd_now = 1.0 - eq / hwm_eq[k, t]
                                if dd_now > max_dd[k, t]:
                                    max_dd[k, t] = dd_now
                            cand_dd = T_a - F * max_dd[k, t]
                            if cand_dd < floor:
                                cand_dd = floor
                            real_eq_d = eq * cpi_0 / cpi_d
                            if wealth_X > C:
                                prog = (real_eq_d - C) / (wealth_X - C)
                                if prog < 0.0:
                                    prog = 0.0
                                elif prog > 1.0:
                                    prog = 1.0
                                cand_w = T_a - (T_a - floor) * prog
                            else:
                                cand_w = floor
                            target_lev = cand_dd if cand_dd < cand_w else cand_w
                    elif kind_code == 13:   # recal_adaptive_dd (grid)
                        if is_recal_day:
                            real_eq_d = eq * cpi_0 / cpi_d
                            h_remaining = max_days - d
                            n_e = e_recal_grid.shape[0]
                            n_h = h_recal_grid_days.shape[0]
                            e_idx = 0
                            best_de = e_recal_grid[0] - real_eq_d
                            if best_de < 0.0:
                                best_de = -best_de
                            for ee in range(1, n_e):
                                de_v = e_recal_grid[ee] - real_eq_d
                                if de_v < 0.0:
                                    de_v = -de_v
                                if de_v < best_de:
                                    best_de = de_v
                                    e_idx = ee
                            h_idx = 0
                            best_dh = h_recal_grid_days[0] - h_remaining
                            if best_dh < 0:
                                best_dh = -best_dh
                            for hh in range(1, n_h):
                                dh_v = h_recal_grid_days[hh] - h_remaining
                                if dh_v < 0:
                                    dh_v = -dh_v
                                if dh_v < best_dh:
                                    best_dh = dh_v
                                    h_idx = hh
                            new_T = t_recal_table[e_idx, h_idx]
                            if new_T < floor:
                                new_T = floor
                            T_active[k, t] = new_T
                            hwm_eq[k, t] = eq
                            max_dd[k, t] = 0.0
                            cur_tgt[k, t] = new_T
                            target_lev = new_T
                        else:
                            T_a = T_active[k, t]
                            if eq > hwm_eq[k, t]:
                                hwm_eq[k, t] = eq
                            if hwm_eq[k, t] > 0.0:
                                dd_now = 1.0 - eq / hwm_eq[k, t]
                                if dd_now > max_dd[k, t]:
                                    max_dd[k, t] = dd_now
                            if eq > 0.0:
                                L_now = stocks[k, t] / eq
                            else:
                                L_now = T_a
                            if T_a > 1.0:
                                F_eff = F * (L_now - 1.0) / (T_a - 1.0)
                            else:
                                F_eff = 0.0
                            if F_eff < 0.0:
                                F_eff = 0.0
                            cand = T_a - F_eff * max_dd[k, t]
                            if cand < floor:
                                cand = floor
                            if cand < cur_tgt[k, t]:
                                cur_tgt[k, t] = cand
                            target_lev = cur_tgt[k, t]
                    elif kind_code == 14:   # meta_recal (grid)
                        if is_recal_day:
                            real_eq_d = eq * cpi_0 / cpi_d
                            h_remaining = max_days - d
                            n_e = e_recal_grid.shape[0]
                            n_h = h_recal_grid_days.shape[0]
                            e_idx = 0
                            best_de = e_recal_grid[0] - real_eq_d
                            if best_de < 0.0:
                                best_de = -best_de
                            for ee in range(1, n_e):
                                de_v = e_recal_grid[ee] - real_eq_d
                                if de_v < 0.0:
                                    de_v = -de_v
                                if de_v < best_de:
                                    best_de = de_v
                                    e_idx = ee
                            h_idx = 0
                            best_dh = h_recal_grid_days[0] - h_remaining
                            if best_dh < 0:
                                best_dh = -best_dh
                            for hh in range(1, n_h):
                                dh_v = h_recal_grid_days[hh] - h_remaining
                                if dh_v < 0:
                                    dh_v = -dh_v
                                if dh_v < best_dh:
                                    best_dh = dh_v
                                    h_idx = hh
                            n_meta = t_recal_tables_meta.shape[0]
                            best_s = 0
                            best_T = t_recal_tables_meta[0, e_idx, h_idx]
                            for ss in range(1, n_meta):
                                t_s = t_recal_tables_meta[ss, e_idx, h_idx]
                                if t_s > best_T:
                                    best_T = t_s
                                    best_s = ss
                            if best_T < floor:
                                best_T = floor
                            T_active[k, t] = best_T
                            strat_active[k, t] = best_s
                            hwm_eq[k, t] = eq
                            max_dd[k, t] = 0.0
                            cur_tgt[k, t] = best_T
                            target_lev = best_T
                        else:
                            s_code = meta_strategy_codes[strat_active[k, t]]
                            T_a = T_active[k, t]
                            if s_code == 0:
                                target_lev = stocks[k, t] / eq if eq > 0.0 else T_a
                            elif s_code == 2:
                                if eq > hwm_eq[k, t]:
                                    hwm_eq[k, t] = eq
                                if hwm_eq[k, t] > 0.0:
                                    dd_now = 1.0 - eq / hwm_eq[k, t]
                                    if dd_now > max_dd[k, t]:
                                        max_dd[k, t] = dd_now
                                cand = T_a - F * max_dd[k, t]
                                target_lev = floor if cand < floor else cand
                            elif s_code == 9:
                                if eq > hwm_eq[k, t]:
                                    hwm_eq[k, t] = eq
                                if hwm_eq[k, t] > 0.0:
                                    dd_now = 1.0 - eq / hwm_eq[k, t]
                                    if dd_now > max_dd[k, t]:
                                        max_dd[k, t] = dd_now
                                if eq > 0.0:
                                    L_now = stocks[k, t] / eq
                                else:
                                    L_now = T_a
                                if T_a > 1.0:
                                    F_eff = F * (L_now - 1.0) / (T_a - 1.0)
                                else:
                                    F_eff = 0.0
                                if F_eff < 0.0:
                                    F_eff = 0.0
                                cand = T_a - F_eff * max_dd[k, t]
                                if cand < floor:
                                    cand = floor
                                if cand < cur_tgt[k, t]:
                                    cur_tgt[k, t] = cand
                                target_lev = cur_tgt[k, t]
                            elif s_code == 4:
                                if eq > hwm_eq[k, t]:
                                    hwm_eq[k, t] = eq
                                if hwm_eq[k, t] > 0.0:
                                    dd_now = 1.0 - eq / hwm_eq[k, t]
                                    if dd_now > max_dd[k, t]:
                                        max_dd[k, t] = dd_now
                                cand_dd = T_a - F * max_dd[k, t]
                                if cand_dd < floor:
                                    cand_dd = floor
                                real_eq_d = eq * cpi_0 / cpi_d
                                if wealth_X > C:
                                    prog = (real_eq_d - C) / (wealth_X - C)
                                    if prog < 0.0:
                                        prog = 0.0
                                    elif prog > 1.0:
                                        prog = 1.0
                                    cand_w = T_a - (T_a - floor) * prog
                                else:
                                    cand_w = floor
                                target_lev = cand_dd if cand_dd < cand_w else cand_w
                            else:
                                target_lev = T_a
                    else:
                        target_lev = T_init
                    delta = target_lev * eq - stocks[k, t]
                    if delta > 0.0:
                        stocks[k, t] += delta
                        loan[k, t] += delta

    return called


def find_max_safe_T_grid(ret, tsy, cpi, kind, target, C, S, T_yrs, S2, max_days,
                          avail=None, F=1.5, floor=1.0,
                          lo=1.0, hi=3.0, coarse_n=12, fine_n=12,
                          cap_real=float("inf"), wealth_X=float("inf"),
                          vol_factor=0.0, dip_threshold=0.0, dip_bonus=0.0,
                          rate_threshold=float("inf"), rate_factor=0.0,
                          recal_period_days=0, t_recal_table=None,
                          e_recal_grid=None, h_recal_grid_days=None,
                          t_recal_tables_meta=None, meta_strategy_codes=None,
                          init_strat_idx=0):
    """Two-pass grid search for largest T_init with call rate ≤ target.

    Pass 1: coarse linear grid over [lo, hi] with coarse_n points.
    Pass 2: fine linear grid in the bracket [T_safe, T_unsafe] from pass 1.

    Total = 2 simulate calls testing (coarse_n + fine_n) T values. Final
    precision = (hi - lo) / coarse_n / fine_n. Defaults give ≈ 0.014x with
    coarse_n=fine_n=12 over [1.0, 3.0]."""
    if avail is None:
        avail = np.full(ret.shape[0], max_days, dtype=np.int64)
    elif avail.dtype != np.int64:
        avail = avail.astype(np.int64)
    kind_code = _KIND_CODES[kind]

    if t_recal_table is None:
        t_recal_table = np.zeros((1, 1), dtype=np.float64)
    elif t_recal_table.dtype != np.float64:
        t_recal_table = t_recal_table.astype(np.float64)
    if e_recal_grid is None:
        e_recal_grid = np.zeros(1, dtype=np.float64)
    elif e_recal_grid.dtype != np.float64:
        e_recal_grid = e_recal_grid.astype(np.float64)
    if h_recal_grid_days is None:
        h_recal_grid_days = np.zeros(1, dtype=np.int64)
    elif h_recal_grid_days.dtype != np.int64:
        h_recal_grid_days = h_recal_grid_days.astype(np.int64)
    if t_recal_tables_meta is None:
        t_recal_tables_meta = np.zeros((1, 1, 1), dtype=np.float64)
    elif t_recal_tables_meta.dtype != np.float64:
        t_recal_tables_meta = t_recal_tables_meta.astype(np.float64)
    if meta_strategy_codes is None:
        meta_strategy_codes = np.zeros(1, dtype=np.int64)
    elif meta_strategy_codes.dtype != np.int64:
        meta_strategy_codes = meta_strategy_codes.astype(np.int64)

    def _eval(T_grid):
        called = _simulate_core_grid(ret, tsy, cpi, kind_code, T_grid,
                                     float(C), float(S), float(T_yrs), float(S2),
                                     int(max_days), avail, float(F), float(floor),
                                     float(cap_real), float(wealth_X),
                                     float(vol_factor), float(dip_threshold),
                                     float(dip_bonus), float(rate_threshold),
                                     float(rate_factor),
                                     int(recal_period_days), t_recal_table,
                                     e_recal_grid, h_recal_grid_days,
                                     t_recal_tables_meta, meta_strategy_codes,
                                     int(init_strat_idx))
        return called.mean(axis=0)

    coarse = np.linspace(lo, hi, coarse_n)
    rates = _eval(coarse)
    safe_mask = rates <= target
    if not safe_mask.any():
        return float(lo)
    last_safe_idx = int(np.where(safe_mask)[0].max())
    if last_safe_idx == coarse_n - 1:
        return float(hi)

    fine_lo = float(coarse[last_safe_idx])
    fine_hi = float(coarse[last_safe_idx + 1])
    fine = np.linspace(fine_lo, fine_hi, fine_n)
    rates = _eval(fine)
    safe_mask = rates <= target
    if not safe_mask.any():
        return fine_lo
    last_safe_idx_fine = int(np.where(safe_mask)[0].max())
    return float(fine[last_safe_idx_fine])


def find_max_safe_T(ret, tsy, cpi, kind, target, C, S, T_yrs, S2, max_days,
                     avail=None, F=1.5, lo=1.0, hi=3.0, n_iters=7,
                     wealth_X=float("inf")):
    """Binary search for largest T_init with call rate ≤ target."""
    rate_hi = call_rate(ret, tsy, cpi, kind, hi, C, S, T_yrs, S2, max_days,
                         avail=avail, F=F, wealth_X=wealth_X)
    if rate_hi <= target:
        return hi
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        if call_rate(ret, tsy, cpi, kind, mid, C, S, T_yrs, S2, max_days,
                     avail=avail, F=F, wealth_X=wealth_X) <= target:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def percentiles_at(real_eq, checkpoints, max_days):
    out = {}
    for y in checkpoints:
        d_idx = int(y * TD)
        if d_idx > max_days:
            continue
        col = real_eq[:, d_idx]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        ps = np.percentile(col, [10, 25, 50, 75, 90])
        out[y] = (ps[0], ps[1], ps[2], ps[3], ps[4], col.mean(), len(col))
    return out


def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "      —"
    if abs(x) >= 1e6:
        return f"${x / 1e6:>6.2f}M"
    return f"${x / 1e3:>6.0f}k"


def run():
    args = parse_args()
    dates, px, tsy, mrate, cpi = load(with_cpi=True)

    checkpoints = sorted(float(x) for x in args.checkpoints.split(","))
    max_years = args.max_years if args.max_years else max(checkpoints)
    max_days = int(max_years * TD)

    print("=== Parameters ===")
    print(f"  C        = ${args.C:>12,.0f}")
    print(f"  S        = ${args.S:>12,.0f}/yr (years 0..{args.T:g})")
    print(f"  S2       = ${args.S2:>12,.0f}/yr (years {args.T:g}+)")
    print(f"  Horizon  = {max_years:g}y; checkpoints = {checkpoints}")
    print(f"  Output: REAL dollars (today's purchasing power)\n")

    print("Building paths...")
    # Calibration paths: full max_days only (preserves safety semantics)
    ret_c, tsy_c, cpi_c, avail_c, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=max_days)
    print(f"  historical (calibration, full horizon): {ret_c.shape[0]} paths × {max_days + 1} days")
    # Projection paths: variable horizon, ≥1y forward — uses ALL post-1932 entries
    ret_h, tsy_h, cpi_h, avail_h, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    print(f"  historical (projection, variable horizon): {ret_h.shape[0]} paths "
          f"(min {avail_h.min()}, max {avail_h.max()} days forward)")
    rng = np.random.default_rng(args.seed)
    block_days = int(args.bootstrap_block_years * TD)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, args.bootstrap_paths, block_days, rng)
    print(f"  bootstrap:  {ret_b.shape[0]} paths × {max_days + 1} days "
          f"({args.bootstrap_block_years:g}y blocks)\n")

    strategies = [
        ("static",   dict(kind="static")),
        ("relever",  dict(kind="relever")),
        ("dd_decay", dict(kind="dd_decay", F=1.5, floor=1.0)),
    ]

    target_pct = int(100 * args.bootstrap_call_target)
    print("=== Calibration (your scenario) ===")
    print(f"  T_hist_safe   = largest T with 0% historical margin calls")
    print(f"  T_boot_safe   = largest T with ≤{target_pct}% synthetic calls (block bootstrap)")
    print(f"  T_recommended = min of both (satisfies both safety bars)")
    print()
    print(f"{'Strategy':<10}  {'T_hist_safe':>12}  {'boot@hist':>11}  "
          f"{'T_boot_safe':>12}  {'T_recommended':>14}")
    print("-" * 70)

    calibrated = {}
    for name, spec in strategies:
        kind = spec["kind"]
        F = spec.get("F", 1.5)
        T_hist = find_max_safe_T(ret_c, tsy_c, cpi_c, kind, 0.0,
                                  args.C, args.S, args.T, args.S2, max_days,
                                  avail=avail_c, F=F)
        boot_at_hist = call_rate(ret_b, tsy_b, cpi_b, kind, T_hist,
                                  args.C, args.S, args.T, args.S2, max_days, F=F)
        T_boot = find_max_safe_T(ret_b, tsy_b, cpi_b, kind, args.bootstrap_call_target,
                                  args.C, args.S, args.T, args.S2, max_days, F=F,
                                  n_iters=10)
        T_rec = min(T_hist, T_boot)
        calibrated[name] = dict(spec=spec, T_hist=T_hist,
                                boot_at_hist=boot_at_hist,
                                T_boot=T_boot, T_rec=T_rec)
        print(f"{name:<10}  {T_hist:>12.3f}  {100 * boot_at_hist:>10.2f}%  "
              f"{T_boot:>12.3f}  {T_rec:>14.3f}")
    print()

    # Projection at recommended targets (= min(T_hist_safe, T_boot_safe))
    print("=== Projection at recommended targets (real $) ===")
    header = f"{'Strategy':<14}{'T':>7}  "
    for y in checkpoints:
        header += f"{int(y):>4}y      "
    print(header)
    print("-" * len(header))

    # Unlev baseline first
    real_eq_u, _, _, _ = simulate(ret_h, tsy_h, cpi_h, "static", 1.0,
                            args.C, args.S, args.T, args.S2, max_days,
                            avail=avail_h)
    ps_u = percentiles_at(real_eq_u, checkpoints, max_days)
    line = f"{'unlev':<14}{1.0:>7.3f}  "
    for y in checkpoints:
        line += f"  {fmt_money(ps_u[y][2]) if y in ps_u else '—':>9}"
    print(line)

    for name, spec in strategies:
        c = calibrated[name]
        T = c["T_rec"]
        kind = spec["kind"]
        F = spec.get("F", 1.5)
        real_eq, _, _, _ = simulate(ret_h, tsy_h, cpi_h, kind, T,
                              args.C, args.S, args.T, args.S2, max_days,
                              avail=avail_h, F=F)
        ps = percentiles_at(real_eq, checkpoints, max_days)
        line = f"{name:<14}{T:>7.3f}  "
        for y in checkpoints:
            line += f"  {fmt_money(ps[y][2]) if y in ps else '—':>9}"
        print(line)
    print()

    # Detailed percentiles for each strategy at recommended target
    print("=== Detailed percentiles at recommended targets ===")
    for name, spec in strategies:
        c = calibrated[name]
        T = c["T_rec"]
        kind = spec["kind"]
        F = spec.get("F", 1.5)
        real_eq, _, _, _ = simulate(ret_h, tsy_h, cpi_h, kind, T,
                              args.C, args.S, args.T, args.S2, max_days,
                              avail=avail_h, F=F)
        ps = percentiles_at(real_eq, checkpoints, max_days)
        print(f"--- {name} @ {T:.3f}x ---")
        print(f"{'Year':>5}  {'p10':>10} {'p25':>10} {'p50':>10} {'p75':>10} "
              f"{'p90':>10} {'mean':>10} {'paths':>6}")
        for y in checkpoints:
            if y not in ps:
                continue
            p10, p25, p50, p75, p90, mn, n = ps[y]
            print(f"{y:>4.0f}y  {fmt_money(p10):>10} {fmt_money(p25):>10} "
                  f"{fmt_money(p50):>10} {fmt_money(p75):>10} {fmt_money(p90):>10} "
                  f"{fmt_money(mn):>10} {n:>6}")
        print()


if __name__ == "__main__":
    run()
