#!/usr/bin/env -S python -u
"""
Wealth-based decay strategy.

Target leverage decays as MAX EQUITY (high-water mark) grows, not as time passes.
  target(t) = max(1.0, T_initial - slope * (max_equity(t)/initial_equity - 1))
  slope = (T_initial - 1.0) / (wealth_mult_to_floor - 1)

Key property: during drawdowns, max equity doesn't update, so target doesn't drop.
During bull runs, max equity rises, target drops → re-levers taper off naturally.

Re-lever monthly, lever-up-only (never sell to rebalance).

For each (DCA, wealth_mult_to_floor, horizon):
  Binary search for max-safe T_initial (zero calls)
  Then simulate at that target, compute per-path IRR percentiles.

Compare to time-decay and monthly relever (already established max-safe numbers).
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


def simulate_wealth_decay(T_initial, wealth_mult_to_floor, annual_dca,
                           horizon_years, floor=1.0):
    """Wealth-based decay: target declines as max-equity (HWM) rises."""
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0

    if T_initial <= floor or wealth_mult_to_floor <= 1.0:
        slope = 0.0
    else:
        slope = (T_initial - floor) / (wealth_mult_to_floor - 1.0)

    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    spx = np.full(N, T_initial, dtype=float)
    loan = np.full(N, T_initial - 1.0, dtype=float)
    called = np.zeros(N, dtype=bool)
    max_equity = np.full(N, 1.0, dtype=float)

    for k in range(1, H + 1):
        spx = spx * (px[idxs + k] / px[idxs + k - 1])
        loan = loan * (M_box[idxs + k] / M_box[idxs + k - 1])

        if k % DAYS_PER_MONTH == 0:
            active = ~called
            spx = np.where(active, spx + monthly, spx)

            equity = spx - loan
            pos = equity > 0
            # Update HWM (only on monthly check, cheap)
            max_equity = np.maximum(max_equity,
                                    np.where(pos & active, equity, max_equity))

            # Wealth-based target
            wealth_mult = max_equity
            current_target = np.maximum(T_initial - slope * (wealth_mult - 1.0),
                                        floor)
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

    terminal = np.where(called, 0.0, spx - loan)
    return terminal, called


def find_max_safe_wealth_decay(wealth_mult_to_floor, annual_dca, horizon_years):
    lo, hi = 1.01, 3.50
    for _ in range(10):  # ~0.0025x precision, plenty
        mid = (lo + hi) / 2
        _, c = simulate_wealth_decay(mid, wealth_mult_to_floor, annual_dca,
                                     horizon_years)
        if c.mean() <= 0.0:
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
        return brentq(f, -0.99, 2.0, xtol=1e-7)
    except (ValueError, RuntimeError):
        return np.nan


def per_path_irrs(terminal, H, dca):
    out = np.full(len(terminal), np.nan)
    for i, t in enumerate(terminal):
        if t > 0:
            out[i] = irr(t, H, dca)
    return out


DCAs = [0.00, 0.10, 0.30]
wealth_multiples = [3.0, 5.0, 10.0, 20.0]


def run_horizon(H_YEARS):
    print("\n" + "=" * 125)
    print(f"WEALTH-BASED DECAY, horizon = {H_YEARS}y")
    print(f"Target decays linearly from T_initial (at wealth=1x) to floor=1.0x "
          f"(at wealth=WM x initial equity).")
    print(f"Uses high-water-mark of equity — during drawdowns, target doesn't drop further.")
    print("=" * 125)
    for dca in DCAs:
        print(f"\n-- DCA = {dca*100:.0f}%/yr --")
        print(f"{'Wealth mult':>12}  {'T_initial (max-safe)':>20}  "
              f"{'p10':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}  "
              f"{'mean':>7}  {'call%':>6}")
        for wm in wealth_multiples:
            T_init = find_max_safe_wealth_decay(wm, dca, H_YEARS)
            term, called = simulate_wealth_decay(T_init, wm, dca, H_YEARS)
            irrs = per_path_irrs(term, H_YEARS, dca)
            valid = ~np.isnan(irrs)
            if valid.sum() == 0:
                print(f"{wm:>10.1f}x  {T_init:>17.3f}x  (all called)")
                continue
            v = irrs[valid]
            p10, p25, p50, p75, p90 = [np.percentile(v, q)*100
                                        for q in (10, 25, 50, 75, 90)]
            mean = v.mean() * 100
            cr = called.mean() * 100
            print(f"{wm:>10.1f}x  {T_init:>17.3f}x  "
                  f"{p10:>6.2f}%  {p25:>6.2f}%  {p50:>6.2f}%  "
                  f"{p75:>6.2f}%  {p90:>6.2f}%  {mean:>6.2f}%  {cr:>5.2f}%")


run_horizon(20)
run_horizon(30)
