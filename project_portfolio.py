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
  unlev      — 1.00x, no leverage. Baseline.
  static     — Set T_init at day 0; never rebalance. Drifts toward 1.0x as
               DCA dilutes leverage.
  relever    — Monthly re-lever to T_init.
  dd_decay   — Drawdown-coupled decay. T_init at day 0; target ratchets DOWN
               as max observed drawdown grows: target = max(floor, T_init -
               F * max_dd_observed). F=1.5, floor=1.0.

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
BOX_TAX_BENEFIT = 0.20
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
# Vectorized simulation (Numba JIT)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _simulate_core(ret, tsy, cpi, kind_code, T_init, C, S, T_yrs, S2, max_days,
                   avail, F, floor, checkpoint_days, cap_real):
    """JIT-compiled per-day, per-path inner loop. Avoids temporary array
    allocations entirely. kind_code: 0=static, 1=relever, 2=dd_decay.

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
    called = np.zeros(K, dtype=np.bool_)
    cap_reached = np.zeros(K, dtype=np.bool_)
    peak_lev = np.full(K, float(T_init))

    real_eq = np.full((K, max_days + 1), np.nan)
    lev_at_cp = np.full((K, n_cp), np.nan)
    for k in range(K):
        real_eq[k, 0] = C

    t_switch_days = T_yrs * TD
    is_levered = T_init > 1.0
    cp_idx = 0   # which checkpoint comes next

    for d in range(1, max_days + 1):
        s_real_yr = S if d < t_switch_days else S2
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


_KIND_CODES = {"static": 0, "relever": 1, "dd_decay": 2}


def simulate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
             avail=None, F=1.5, floor=1.0, checkpoint_days=None,
             cap_real=float("inf")):
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
    return _simulate_core(ret, tsy, cpi, _KIND_CODES[kind], float(T_init),
                          float(C), float(S), float(T_yrs), float(S2),
                          int(max_days), avail, float(F), float(floor),
                          checkpoint_days, float(cap_real))


def call_rate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
              avail=None, F=1.5, cap_real=float("inf")):
    _, called, _, _ = simulate(ret, tsy, cpi, kind, T_init, C, S, T_yrs, S2, max_days,
                            avail=avail, F=F, cap_real=cap_real)
    return float(called.mean())


# ---------------------------------------------------------------------------
# Grid-vectorized simulation: test many T values in one sim
# ---------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def _simulate_core_grid(ret, tsy, cpi, kind_code, T_inits, C, S, T_yrs, S2,
                        max_days, avail, F, floor, cap_real):
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
    called = np.zeros((K, T_count), dtype=np.bool_)
    cap_reached = np.zeros((K, T_count), dtype=np.bool_)

    for k in range(K):
        for t in range(T_count):
            T_init = T_inits[t]
            stocks[k, t] = C * T_init
            loan[k, t] = C * (T_init - 1.0)

    t_switch_days = T_yrs * TD

    for d in range(1, max_days + 1):
        s_real_yr = S if d < t_switch_days else S2
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
                is_levered = T_init > 1.0

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
                          cap_real=float("inf")):
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

    def _eval(T_grid):
        called = _simulate_core_grid(ret, tsy, cpi, kind_code, T_grid,
                                     float(C), float(S), float(T_yrs), float(S2),
                                     int(max_days), avail, float(F), float(floor),
                                     float(cap_real))
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
                     avail=None, F=1.5, lo=1.0, hi=3.0, n_iters=7):
    """Binary search for largest T_init with call rate ≤ target."""
    rate_hi = call_rate(ret, tsy, cpi, kind, hi, C, S, T_yrs, S2, max_days,
                         avail=avail, F=F)
    if rate_hi <= target:
        return hi
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        if call_rate(ret, tsy, cpi, kind, mid, C, S, T_yrs, S2, max_days,
                     avail=avail, F=F) <= target:
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
