"""
Detailed walk-through of an entry on 2000-03-23 (worst post-1932 entry
date) with various initial leverages, to elaborate on the "cap near 1.41x"
result.
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, _tsy, mrate = load()

# find entry index for 2000-03-23
entry_dt = datetime(2000, 3, 23)
i0 = int(np.where(dates == entry_dt)[0][0])
print(f"Entry: {dates[i0].date()}   SPX-TR level: {px[i0]:.2f}")

daily_factor = 1.0 + mrate / 252.0

# Simulate forward from i0 for a given L0
def simulate(L0, end_idx=None):
    if end_idx is None:
        end_idx = len(px)
    n = end_idx - i0
    A = np.empty(n); D = np.empty(n); E = np.empty(n); Lv = np.empty(n)
    A[0] = L0
    D[0] = L0 - 1.0
    E[0] = 1.0
    Lv[0] = L0
    for k in range(1, n):
        A[k] = A[k-1] * px[i0 + k] / px[i0 + k - 1]
        D[k] = D[k-1] * daily_factor[i0 + k]
        E[k] = A[k] - D[k]
        Lv[k] = A[k] / E[k] if E[k] > 0 else np.inf
    return A, D, E, Lv

# trajectory at L0 = 1.4119 (theoretical max)
print("\n=== Trajectory with L0 = 1.4119x (max safe for this entry) ===")
A, D, E, Lv = simulate(1.4119)

peak_lev_idx = int(np.argmax(Lv))
min_equity_idx = int(np.argmin(E))
print(f"Peak leverage        : {Lv[peak_lev_idx]:.4f}x on {dates[i0+peak_lev_idx].date()}")
print(f"Min equity           : {E[min_equity_idx]:.4f} (= {E[min_equity_idx]*100:.1f}% of start)")
print(f"  on                 : {dates[i0+min_equity_idx].date()}")

# find first date back to L <= 2.0
recov = np.where(Lv[peak_lev_idx:] <= 2.0)[0]
if len(recov):
    ridx = peak_lev_idx + recov[0]
    print(f"First back to L<=2   : {dates[i0+ridx].date()}  "
          f"({(dates[i0+ridx]-dates[i0+peak_lev_idx]).days/365.25:.2f}y after peak)")
# find first date back to L <= initial
recov2 = np.where(Lv[peak_lev_idx:] <= 1.42)[0]
if len(recov2):
    ridx = peak_lev_idx + recov2[0]
    print(f"First back to L<=1.42: {dates[i0+ridx].date()}")
# final state at end of dataset
print(f"\nAt end ({dates[-1].date()}):")
print(f"  Assets  = {A[-1]:.3f}  (SPX-TR x{px[-1]/px[i0]:.2f})")
print(f"  Loan    = {D[-1]:.3f}  (grew {(D[-1]/D[0]-1)*100:.1f}%)")
print(f"  Equity  = {E[-1]:.3f}  (vs start 1.000)")
print(f"  Leverage= {Lv[-1]:.4f}x")

# print timeline of leverage milestones
print("\nTimeline of leverage milestones (from 2000-03-23):")
def first_cross(arr, thresh, above=True):
    mask = (arr >= thresh) if above else (arr <= thresh)
    w = np.where(mask)[0]
    return w[0] if len(w) else None
for thr in [1.5, 2.0, 2.5, 3.0, 3.5, 3.9, 4.0]:
    k = first_cross(Lv, thr, above=True)
    if k is not None:
        print(f"  First L >= {thr:.1f}x : {dates[i0+k].date()}  "
              f"({(dates[i0+k]-dates[i0]).days/365.25:5.2f}y)")

# Compare different L0 choices: peak leverage, equity drawdown, time at high leverage
print("\n=== Sensitivity of outcome to initial leverage ===")
print(f"{'L0':>6}  {'loan/eq':>8}  {'peak L':>8}  "
      f"{'peak L date':>12}  {'min eq':>8}  {'days L>=3':>9}  "
      f"{'bust?':>6}")
for L0 in [1.0, 1.10, 1.20, 1.30, 1.35, 1.40, 1.4119, 1.45, 1.50, 1.60, 1.75, 2.0]:
    A, D, E, Lv = simulate(L0)
    # count days where equity > 0 and leverage >= 3
    busted = np.any(E <= 0) or np.any(~np.isfinite(Lv))
    if busted:
        peakL = np.inf
        peak_date = "—"
        mineq = E.min()
        days3 = int(np.sum(Lv >= 3.0))
    else:
        pk = int(np.argmax(Lv))
        peakL = Lv[pk]
        peak_date = str(dates[i0+pk].date())
        mineq = E.min()
        days3 = int(np.sum(Lv >= 3.0))
    print(f"{L0:>6.4f}  {(L0-1)*100:>7.2f}%  "
          f"{peakL:>7.3f}x  {peak_date:>12}  "
          f"{mineq:>7.3f}   {days3:>8}  "
          f"{'YES' if busted else '':>6}")
