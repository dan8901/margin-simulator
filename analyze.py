"""
Max initial leverage such that portfolio leverage never exceeds 4.0,
given 100% SPX (total return) and a one-time margin loan whose
balance stays constant (interest paid separately).

L_t = L_0 (1-f) / (1 - L_0 f)    where f = drawdown-from-entry at time t
L_t = 4 when L_0 = 4 / (1 + 3 f_max),
f_max = max future drawdown from entry date.
"""
import numpy as np
from data_loader import load

dates, px, _tsy, _mr = load()
n = len(px)
print(f"Rows: {n}   range: {dates[0].date()} .. {dates[-1].date()}")

# For each entry index i, compute worst drawdown over future days j >= i:
#   f(i) = 1 - min(px[i:]) / px[i]
# Use a reverse running min.
rev_min = np.minimum.accumulate(px[::-1])[::-1]  # rev_min[i] = min(px[i:])
worst_future_low = rev_min
f_max = 1.0 - worst_future_low / px
L0_max = 4.0 / (1.0 + 3.0 * f_max)

# Worst entry date (i.e. the one that forces smallest L0)
worst_idx = int(np.argmin(L0_max))
print()
print("=== WORST POSSIBLE ENTRY DATE ===")
print(f"Entry date           : {dates[worst_idx].date()}")
print(f"Entry SPX-TR level   : {px[worst_idx]:.4f}")
trough_idx = worst_idx + int(np.argmin(px[worst_idx:]))
print(f"Trough date          : {dates[trough_idx].date()}")
print(f"Trough SPX-TR level  : {px[trough_idx]:.4f}")
print(f"Max future drawdown  : {f_max[worst_idx]*100:.2f}%")
print(f"Max initial leverage : {L0_max[worst_idx]:.4f}x  "
      f"(= loan of {(L0_max[worst_idx]-1)*100:.2f}% of equity)")

# Percentiles across entry dates
# NOTE: we drop dates too close to the end (no future) — any day with
# f_max == 0 only because the series ended has no meaningful answer.
# To be conservative, report percentiles over all dates (worst case).
print()
print("=== PERCENTILES of max-safe initial leverage across all entry dates ===")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    v = np.percentile(L0_max, q)
    print(f"  p{q:>2}: {v:.4f}x   (loan/equity = {(v-1)*100:6.2f}%)")

print()
print("=== PERCENTILES of max future drawdown from entry (reference) ===")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    v = np.percentile(f_max, q)
    print(f"  p{q:>2}: {v*100:6.2f}%")

# Top 10 worst entry dates
print()
print("=== 10 WORST ENTRY DATES ===")
order = np.argsort(L0_max)[:10]
for i in order:
    tr = i + int(np.argmin(px[i:]))
    print(f"  {dates[i].date()}  ->  trough {dates[tr].date()}  "
          f"drawdown {f_max[i]*100:6.2f}%   L0_max {L0_max[i]:.4f}x")
