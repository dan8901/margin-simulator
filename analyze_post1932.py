"""
Same analysis as analyze_with_interest.py but restricting entry dates
to various cutoffs (post-1932, post-1950).

Note: the future-drawdown computation still uses the full series going
forward from each entry — so for post-1932 entries, nothing from 1929-32
can affect the result anyway (entries are after the trough).
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, _tsy, mrate = load()

daily_factor = 1.0 + mrate / 252.0
M = np.concatenate(([1.0], np.cumprod(daily_factor[1:])))
R = px / M
min_future_R = np.minimum.accumulate(R[::-1])[::-1]

denom = 4.0 * R - 3.0 * min_future_R
L0_max = np.minimum(4.0 * R / denom, 4.0)

# baseline no-interest
rev_min_px = np.minimum.accumulate(px[::-1])[::-1]
f_max_px = 1.0 - rev_min_px / px
L0_max_no_int = np.minimum(4.0 / (1.0 + 3.0 * f_max_px), 4.0)

def report(cutoff_dt, label):
    mask = np.array([d >= cutoff_dt for d in dates])
    idxs = np.where(mask)[0]
    Lw = L0_max[idxs]
    Ln = L0_max_no_int[idxs]
    worst = idxs[int(np.argmin(Lw))]
    bi = worst + int(np.argmin(R[worst:]))
    yr = (dates[bi] - dates[worst]).days / 365.25

    print(f"\n====== ENTRY >= {label}   (N={len(idxs)} dates) ======")
    print(f"Worst entry date     : {dates[worst].date()}")
    print(f"  Worst-point date   : {dates[bi].date()}  ({yr:.2f} years later)")
    print(f"  SPX-TR drawdown    : {(1-px[bi]/px[worst])*100:.2f}%")
    print(f"  Loan compounded    : {(M[bi]/M[worst]-1)*100:.2f}%")
    print(f"  Max initial lev    : {L0_max[worst]:.4f}x  "
          f"(loan = {(L0_max[worst]-1)*100:.2f}% of equity)")
    print(f"  -- no-int baseline : {L0_max_no_int[worst]:.4f}x")
    print()
    print("  Percentiles of max-safe initial leverage across entry dates:")
    print("          w/ interest            no interest")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        a = np.percentile(Lw, q)
        b = np.percentile(Ln, q)
        print(f"    p{q:>2}:  {a:.4f}x  ({(a-1)*100:6.2f}%)     "
              f"{b:.4f}x  ({(b-1)*100:6.2f}%)")
    print()
    print("  10 worst entry dates in this window:")
    order = idxs[np.argsort(Lw)[:10]]
    for i in order:
        bi = i + int(np.argmin(R[i:]))
        yr = (dates[bi] - dates[i]).days / 365.25
        print(f"    {dates[i].date()}  ->  {dates[bi].date()}  "
              f"({yr:4.1f}y)  SPX-TR {(1-px[bi]/px[i])*100:6.2f}%  "
              f"loan_grew {(M[bi]/M[i]-1)*100:5.2f}%  L0_max {L0_max[i]:.4f}x")

report(datetime(1932, 7, 1), "1932-07-01 (post-Depression trough)")
report(datetime(1950, 1, 1), "1950-01-01 (post-war)")
