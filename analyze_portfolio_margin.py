"""
Compare 25% maintenance (Reg-T-ish, 4.0x cap) vs 15% maintenance
(portfolio margin, 6.667x cap).

Uses the generalized closed form from analyze_safety_cushion.py:

    L_0_max(i) = L_cap * R_i / [L_cap * R_i - (L_cap - 1) * min_future_R]

with R_t = P_t / M_t, M_t the loan-growth factor (rate/252 daily compound).
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, _tsy, mrate = load()

CAP_REGT = 4.0            # 25% maintenance
CAP_PM   = 1.0 / 0.15     # 15% maintenance = 6.6667x


def max_L0_arr(px, mrate, cap):
    daily_factor = 1.0 + mrate / 252.0
    M = np.concatenate(([1.0], np.cumprod(daily_factor[1:])))
    R = px / M
    min_future_R = np.minimum.accumulate(R[::-1])[::-1]
    L0 = cap * R / (cap * R - (cap - 1.0) * min_future_R)
    return np.minimum(L0, cap), R, M


regt_L0, R, M = max_L0_arr(px, mrate, CAP_REGT)
pm_L0, _, _   = max_L0_arr(px, mrate, CAP_PM)


def report(mask, label):
    idxs = np.where(mask)[0]
    r = regt_L0[idxs]
    p = pm_L0[idxs]
    print(f"\n===== {label}  (N={len(idxs):,}) =====")

    # Worst entries under each cap
    for cap_name, arr, cap in [("4.0x (25% maint.)", r, CAP_REGT),
                               ("6.667x (15% PM)", p, CAP_PM)]:
        j = int(np.argmin(arr))
        jf = int(idxs[j])
        bi = jf + int(np.argmin(R[jf:]))
        yr = (dates[bi] - dates[jf]).days / 365.25
        print(f"  Worst under {cap_name:<18} "
              f"entry {dates[jf].date()}  ->  worst {dates[bi].date()} "
              f"({yr:.1f}y)  L0_max = {arr[j]:.4f}x  "
              f"(loan = {(arr[j]-1)*100:.2f}% equity)")

    print(f"\n  {'pct':>5}  {'4.0x cap':>20}  {'6.667x cap':>20}  "
          f"{'PM advantage':>16}")
    for q in [0.1, 1, 5, 10, 20, 25, 50, 75, 90, 95, 99]:
        a = np.percentile(r, q)
        b = np.percentile(p, q)
        ratio = b / a
        print(f"  {q:>5.1f}  {a:>6.4f}x ({(a-1)*100:>6.2f}%)  "
              f"{b:>6.4f}x ({(b-1)*100:>6.2f}%)  "
              f"{ratio:>12.2f}x more")


report(np.ones_like(px, dtype=bool), "FULL SERIES (1927-12-30 onward)")
report(np.array([d >= datetime(1932, 7, 1) for d in dates]),
       "POST-1932 (after Depression trough)")

# Specific view of the worst post-1932 entry under each cap
print("\n" + "=" * 78)
print("Detail: 2000-03-23 entry under each cap")
print("=" * 78)
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
for cap, nm in [(CAP_REGT, "4.0x (25% maint.)"), (CAP_PM, "6.667x (15% PM)")]:
    arr, _, _ = max_L0_arr(px, mrate, cap)
    L0 = float(arr[i0])
    print(f"  {nm:<18}  L0_max = {L0:.4f}x  "
          f"loan = {(L0-1)*100:.2f}% of equity")

# The "broker switches from 15% -> 25% mid-drawdown" stress test
# (the user's irony point): size the loan at the 15% cap on entry,
# then see what peak leverage would be, and at what point the 4.0x
# threshold would be hit.
print("\n" + "=" * 78)
print('Stress test: size for 15% PM, broker tightens to 25% mid-crisis')
print("=" * 78)
daily_factor = 1.0 + mrate / 252.0
for entry_str in ["2000-03-23", "1929-09-16", "1973-01-11", "2007-10-09"]:
    dt = datetime.strptime(entry_str, "%Y-%m-%d")
    ii = np.where(dates == dt)[0]
    if len(ii) == 0:
        # find nearest
        ii = [int(np.argmin([abs((d - dt).days) for d in dates]))]
    i_entry = int(ii[0])
    L0 = float(pm_L0[i_entry])  # sized for 15% PM
    # Simulate forward and find peak leverage
    n = len(px) - i_entry
    A = np.empty(n); D = np.empty(n)
    A[0] = L0; D[0] = L0 - 1.0
    for k in range(1, n):
        A[k] = A[k-1] * px[i_entry+k] / px[i_entry+k-1]
        D[k] = D[k-1] * daily_factor[i_entry+k]
    E = A - D
    Lv = np.where(E > 0, A / np.where(E > 0, E, 1), np.inf)

    # Peak leverage & first time crossing 4.0x
    if np.any(~np.isfinite(Lv)) or np.any(E <= 0):
        peakL = np.inf
        peak_date = "BUST"
    else:
        pk = int(np.argmax(Lv))
        peakL = float(Lv[pk])
        peak_date = str(dates[i_entry+pk].date())
    # First time L >= 4.0x
    above4 = np.where(Lv >= 4.0)[0]
    first4 = dates[i_entry + int(above4[0])].date() if len(above4) else None

    print(f"\n  Entry {dates[i_entry].date()}   L0 (sized for PM) = {L0:.4f}x")
    print(f"    Peak leverage reached : {peakL:.3f}x on {peak_date}")
    if first4:
        yrs = (dates[i_entry + int(above4[0])] - dates[i_entry]).days / 365.25
        print(f"    First crosses 4.0x    : {first4}  ({yrs:.2f}y in) "
              f"<- broker can call you here if they tightened")
    else:
        print(f"    Never reached 4.0x.")
