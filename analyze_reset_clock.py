"""
Does "reset the time-decay clock on up years" work?

User's intuition: if the market has gone up, I have more cushion, so I can
safely reset the decay timer and lever back up to T_initial. Let me prove
this blows up using the 2000-03-23 entry (binding post-1932 worst case).

Setup:
  - Entry: 2000-03-23 (dot-com peak, 20y horizon to 2020-03)
  - T_initial: 1.59x (max-safe for pure time-decay 2pp at 10% DCA, per prior work)
  - DCA: 10%/yr
  - Financing: box rate (Tsy + 15bps)
  - Call threshold: 4.0x

Three strategies compared:
  A. Pure time-decay 2pp:    target(t) = max(1.59 - 0.02*t, 1.0)
  B. Reset on up years:      target(t) = max(1.59 - 0.02*years_since_reset, 1.0)
                             where years_since_reset resets to 0 each annual
                             anniversary where trailing-12mo SPX > 0
  C. Monthly relever:        target = 1.59x always (no decay, no reset)

Report: leverage trajectory, peak leverage, call date (if any).

Then: across all post-1932 entries, what's max-safe for the "reset on up years"
variant vs pure time-decay?
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()
TRADING_DAYS = 252
DAYS_PER_MONTH = 21
M_box = np.concatenate(([1.0], np.cumprod(1 + (tsy + 0.0015)[1:] / TRADING_DAYS)))

# Find 2000-03-23 entry
entry_date = datetime(2000, 3, 23)
entry_idx = int(np.argmin([abs((d - entry_date).days) for d in dates]))
print(f"Entry: {dates[entry_idx].date()} (idx {entry_idx})")

T_INITIAL = 1.59
DCA_ANNUAL = 0.10
HORIZON_Y = 20
DECAY = 0.02


def simulate_path(entry_idx, horizon_years, T_initial, dca_annual,
                  strategy, verbose=False):
    """
    Returns per-day arrays: (day_index, spx_asset, loan, equity, leverage,
                             target, called_at_day).
    strategy in {'time_decay', 'reset_on_up', 'monthly_relever'}
    """
    H = int(horizon_years * TRADING_DAYS)
    monthly = dca_annual / 12.0

    spx = T_initial
    loan = T_initial - 1.0
    years_since_reset = 0.0  # only used by 'reset_on_up'
    called_at = None

    log_day, log_spx, log_loan, log_eq, log_lev, log_tgt = [], [], [], [], [], []

    def current_target(k):
        if strategy == 'time_decay':
            years = k / TRADING_DAYS
            return max(T_initial - DECAY * years, 1.0)
        elif strategy == 'reset_on_up':
            return max(T_initial - DECAY * years_since_reset, 1.0)
        elif strategy == 'monthly_relever':
            return T_initial
        else:
            raise ValueError(strategy)

    for k in range(1, H + 1):
        spx_g = px[entry_idx + k] / px[entry_idx + k - 1]
        box_g = M_box[entry_idx + k] / M_box[entry_idx + k - 1]
        spx *= spx_g
        loan *= box_g

        # Monthly DCA
        if k % DAYS_PER_MONTH == 0 and called_at is None:
            spx += monthly

        # Annual anniversary: for 'reset_on_up', check if up year
        if strategy == 'reset_on_up' and k % TRADING_DAYS == 0 and k > 0:
            spx_now = px[entry_idx + k]
            spx_yr_ago = px[entry_idx + k - TRADING_DAYS]
            if spx_now > spx_yr_ago:
                years_since_reset = 0.0
            else:
                years_since_reset += 1.0
        elif strategy == 'reset_on_up':
            # continuous increment scaled to days
            pass  # we only update on anniversaries

        # Monthly rebalance to current target (only lever UP, never sell)
        if k % DAYS_PER_MONTH == 0 and called_at is None:
            tgt = current_target(k)
            equity = spx - loan
            if equity > 0:
                desired_assets = tgt * equity
                delta = max(desired_assets - spx, 0.0)
                spx += delta
                loan += delta

        equity = spx - loan
        lev = spx / equity if equity > 0 else np.inf
        if called_at is None and (equity <= 0 or lev >= 4.0):
            called_at = k

        log_day.append(k)
        log_spx.append(spx)
        log_loan.append(loan)
        log_eq.append(equity)
        log_lev.append(lev)
        log_tgt.append(current_target(k))

    return {
        'day': np.array(log_day),
        'spx': np.array(log_spx),
        'loan': np.array(log_loan),
        'equity': np.array(log_eq),
        'leverage': np.array(log_lev),
        'target': np.array(log_tgt),
        'called_at': called_at,
    }


# ======================================================================
# Part 1: trace all three strategies on 2000-03-23 entry at 1.59x
# ======================================================================
print("\n" + "=" * 90)
print(f"2000-03-23 entry, T_initial={T_INITIAL}x, DCA={DCA_ANNUAL*100:.0f}%/yr, "
      f"horizon={HORIZON_Y}y")
print("=" * 90)

results = {}
for strat in ['time_decay', 'reset_on_up', 'monthly_relever']:
    r = simulate_path(entry_idx, HORIZON_Y, T_INITIAL, DCA_ANNUAL, strat)
    results[strat] = r
    peak_lev = np.max(r['leverage'][np.isfinite(r['leverage'])])
    peak_day = int(np.argmax(np.where(np.isfinite(r['leverage']),
                                       r['leverage'], -1)))
    peak_date = dates[entry_idx + peak_day + 1]
    call_str = "no call"
    if r['called_at'] is not None:
        call_date = dates[entry_idx + r['called_at']]
        call_str = f"CALLED on {call_date.date()}"
    term = r['equity'][-1] if r['called_at'] is None else 0.0
    print(f"\n  {strat:<18}: peak lev = {peak_lev:5.2f}x on "
          f"{peak_date.date()}  |  terminal equity = {term:6.2f}  |  {call_str}")

# Show leverage trajectory at key dates (selected milestones)
print("\n" + "-" * 90)
print("Leverage trajectory (end of each calendar year):")
print("-" * 90)
print(f"  {'date':<12} {'time_decay':>12} {'reset_on_up':>12} "
      f"{'monthly_relever':>18}")
years_out = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
for yr in years_out:
    k = yr * TRADING_DAYS - 1
    if k >= len(results['time_decay']['day']):
        continue
    d = dates[entry_idx + k + 1]
    def fmt_lev(r, k):
        lev = r['leverage'][k]
        tgt = r['target'][k]
        if not np.isfinite(lev) or lev > 100:
            return "   CALLED   "
        return f"{lev:5.2f}/{tgt:4.2f}"
    print(f"  {str(d.date()):<12} "
          f"{fmt_lev(results['time_decay'], k):>12} "
          f"{fmt_lev(results['reset_on_up'], k):>12} "
          f"{fmt_lev(results['monthly_relever'], k):>18}")

# Show target trajectory to make decay behavior visible
print("\n" + "-" * 90)
print("Target leverage trajectory (year-end) — shows what each strategy 'wants':")
print("-" * 90)
print(f"  {'date':<12} {'time_decay':>12} {'reset_on_up':>12} "
      f"{'monthly_relever':>18}")
for yr in years_out:
    k = yr * TRADING_DAYS - 1
    if k >= len(results['time_decay']['day']):
        continue
    d = dates[entry_idx + k + 1]
    tgt_td = results['time_decay']['target'][k]
    tgt_ru = results['reset_on_up']['target'][k]
    tgt_mr = results['monthly_relever']['target'][k]
    # Also mark whether YoY was positive (drove the reset)
    k_now = entry_idx + k + 1
    k_yr_ago = entry_idx + k + 1 - TRADING_DAYS
    yoy = "+" if px[k_now] > px[k_yr_ago] else "-"
    print(f"  {str(d.date()):<12}  YoY={yoy}  "
          f"{tgt_td:12.3f} {tgt_ru:12.3f} {tgt_mr:18.3f}")

import sys
if len(sys.argv) <= 1 or sys.argv[1] != "--full":
    print("\n[Skipping Part 2 (full post-1932 max-safe sweep). "
          "Run with --full to include.]")
    sys.exit(0)

# ======================================================================
# Part 2: max-safe across all post-1932 entries, reset-on-up vs pure
# ======================================================================
print("\n" + "=" * 90)
print("Max-safe initial target across ALL post-1932 entries, DCA=10%, horizon=20y")
print("=" * 90)

post1932_mask = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate_all(T_initial, strategy, dca_annual=0.10, horizon_years=20):
    """Vectorized-ish: loop over entries but reuse data."""
    H = int(horizon_years * TRADING_DAYS)
    entry_idxs = np.where(post1932_mask)[0]
    entry_idxs = entry_idxs[entry_idxs + H < len(px)]
    called_count = 0
    for ei in entry_idxs:
        r = simulate_path(ei, horizon_years, T_initial, dca_annual, strategy)
        if r['called_at'] is not None:
            called_count += 1
    return called_count / len(entry_idxs), len(entry_idxs)


def find_max_safe(strategy, dca_annual=0.10, horizon_years=20):
    lo, hi = 1.01, 3.50
    for _ in range(10):  # ~0.003x precision
        mid = (lo + hi) / 2
        call_rate, _ = simulate_all(mid, strategy, dca_annual, horizon_years)
        if call_rate <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


# Compare max-safe for the two decay variants
# Note: this is SLOW because it loops over every entry. We'll still run it.
print("  Searching for max-safe (this takes a few minutes)...")
import time
for strat in ['time_decay', 'reset_on_up']:
    t0 = time.time()
    ms = find_max_safe(strat)
    elapsed = time.time() - t0
    print(f"    {strat:<18}  max-safe = {ms:.3f}x  ({elapsed:.0f}s)")

print("\nInterpretation:")
print("  If reset_on_up's max-safe is substantially LOWER than pure time_decay,")
print("  that's direct proof that the reset rule creates more tail risk.")
