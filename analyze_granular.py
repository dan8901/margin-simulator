"""
Granular percentile report, emphasizing the lower half of the distribution.
Cutoffs: full series, post-1932, post-1950.
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, _tsy, mrate = load()

daily_factor = 1.0 + mrate / 252.0
M = np.concatenate(([1.0], np.cumprod(daily_factor[1:])))
R = px / M
min_future_R = np.minimum.accumulate(R[::-1])[::-1]

L0_max = np.minimum(4.0 * R / (4.0 * R - 3.0 * min_future_R), 4.0)

rev_min_px = np.minimum.accumulate(px[::-1])[::-1]
f_max_px = 1.0 - rev_min_px / px
L0_max_no_int = np.minimum(4.0 / (1.0 + 3.0 * f_max_px), 4.0)

# Dense percentiles, heavier on the lower half
PCTS = [0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25,
        27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50,
        55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 99.9]

def report(cutoff_dt, label):
    if cutoff_dt is None:
        mask = np.ones_like(px, dtype=bool)
    else:
        mask = np.array([d >= cutoff_dt for d in dates])
    idxs = np.where(mask)[0]
    Lw = L0_max[idxs]
    Ln = L0_max_no_int[idxs]

    print(f"\n============ {label}   (N = {len(idxs):,}) ============")
    print(f"{'pct':>6}  {'w/ int L0':>10}  {'loan/eq':>9}   "
          f"{'no-int L0':>10}  {'loan/eq':>9}")
    for q in PCTS:
        a = np.percentile(Lw, q)
        b = np.percentile(Ln, q)
        print(f"{q:>6.1f}  {a:>9.4f}x  {(a-1)*100:>7.2f}%    "
              f"{b:>9.4f}x  {(b-1)*100:>7.2f}%")

report(None,                     "FULL SERIES (1927-12-30 onward)")
report(datetime(1932, 7, 1),     "POST-1932 (after Depression trough)")
report(datetime(1950, 1, 1),     "POST-1950")
