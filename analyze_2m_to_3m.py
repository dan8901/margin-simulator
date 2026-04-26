"""
analyze_2m_to_3m.py

Side analysis: how long does a $2M portfolio take to reach $3M (1.5x)?

Across post-1932 monthly entries, with several DCA levels and a few leverage
targets. Target is on EQUITY ($3M of net liquidation value), not gross stocks.
"""

import numpy as np

from data_loader import load


dates, px, tsy, mrate = load()

TD = 252
V0 = 2_000_000.0
TARGET = 3_000_000.0
BOX_BPS = 0.0015
BOX_TAX_BENEFIT = 0.20
CALL_THRESHOLD = 4.0


def simulate(start_idx, leverage, dca_yr, max_years=20):
    stocks = V0 * leverage
    loan = V0 * (leverage - 1.0)
    dca_daily = dca_yr / TD

    n_days = min(int(max_years * TD), len(dates) - start_idx - 1)

    for d in range(1, n_days + 1):
        i = start_idx + d
        stocks *= px[i] / px[i - 1]
        if loan > 0:
            box_rate = (tsy[i] + BOX_BPS) * (1.0 - BOX_TAX_BENEFIT)
            loan *= 1.0 + box_rate / TD
        if dca_daily > 0:
            stocks += dca_daily

        if loan > 0:
            eq = stocks - loan
            if eq <= 0.0 or stocks / eq >= CALL_THRESHOLD:
                return float("nan"), True

        equity = stocks - loan
        if equity >= TARGET:
            return d / TD, False

    return float("nan"), False


def percentiles(arr, qs=(10, 25, 50, 75, 90)):
    if len(arr) == 0:
        return [float("nan")] * len(qs)
    return [float(x) for x in np.percentile(arr, qs)]


def run():
    cutoff = np.datetime64("1932-07-01")
    eligible = [i for i in range(len(dates))
                if np.datetime64(dates[i]) >= cutoff and i + 5 * TD < len(dates)]
    eligible = eligible[::21]   # one entry per ~month
    print(f"Monthly entries: {len(eligible)} "
          f"({dates[eligible[0]].date()} to {dates[eligible[-1]].date()})")
    print(f"Target: $2M equity -> $3M equity (1.5x)\n")

    print(f"{'DCA':<10}{'Lev':>6}  {'p10':>6}{'p25':>6}{'p50':>6}{'p75':>6}{'p90':>6}{'mean':>6}{'call%':>7}{'unrch%':>7}")
    print("-" * 76)

    for dca in (0, 30_000, 60_000, 100_000):
        for lev in (1.00, 1.25, 1.43):
            times, calls, unreach = [], 0, 0
            for i in eligible:
                t, called = simulate(i, lev, dca)
                if called:
                    calls += 1
                elif np.isnan(t):
                    unreach += 1
                else:
                    times.append(t)
            n = len(eligible)
            ps = percentiles(times)
            mn = float(np.mean(times)) if times else float("nan")
            print(f"${dca // 1000:>3}k/yr  {lev:>5.2f}x  "
                  f"{ps[0]:6.1f}{ps[1]:6.1f}{ps[2]:6.1f}{ps[3]:6.1f}{ps[4]:6.1f}{mn:6.1f}"
                  f"{100 * calls / n:6.1f}%{100 * unreach / n:6.1f}%")
        print()


if __name__ == "__main__":
    run()
