"""
Max initial leverage such that portfolio leverage never exceeds 4.0,
given 100% SPX (total return) and a one-time margin loan whose
interest compounds daily into the loan (nothing paid back).

Mechanics (entry day i, future day t):
  A_t = L0 * P_t / P_i                    (assets scale with SPX TR)
  D_t = (L0 - 1) * M_t / M_i              (loan compounds at margin rate)
  E_t = A_t - D_t
  L_t = A_t / E_t

Setting L_t = 4 and solving for L0:
  L0 = 4m / (4m - 3p),    with p = P_t/P_i, m = M_t/M_i

Let R_t = P_t / M_t, x = R_t/R_i.  Then L0* = 4 / (4 - 3x).
Min of L0* over future t binds at min R_t  =>

   L0_max(i) = 4 * R_i / (4 * R_i - 3 * min_{t>=i} R_t)

Daily loan growth factor: (1 + margin_rate_annual / 252).
"""
import numpy as np
from data_loader import load

dates, px, _tsy, mrate = load()
n = len(px)
print(f"Rows: {n}   range: {dates[0].date()} .. {dates[-1].date()}")
print(f"Margin rate range: {mrate.min()*100:.3f}% .. {mrate.max()*100:.3f}%")

# Cumulative loan-growth factor M_t, starting at M_0 = 1.
# Each day's factor applies the margin rate in effect for that day.
daily_factor = 1.0 + mrate / 252.0
M = np.cumprod(daily_factor)
# Shift so M[0] = 1 (first day has no prior accrual)
M = M / daily_factor[0] * 1.0  # first day: no accrual before it
# Simpler: redefine M[t] = product_{k=1..t} daily_factor[k], M[0] = 1
M = np.concatenate(([1.0], np.cumprod(daily_factor[1:])))

R = px / M  # "real" asset-vs-loan index

# Running reverse min of R: min_future_R[i] = min(R[i:])
min_future_R = np.minimum.accumulate(R[::-1])[::-1]

# L0_max(i) = 4 R_i / (4 R_i - 3 min_future_R_i); cap at 4.
denom = 4.0 * R - 3.0 * min_future_R
# denom > 0 always since min_future_R <= R_i (equal at t=i)
L0_max = 4.0 * R / denom
L0_max = np.minimum(L0_max, 4.0)  # enforce user-stated cap

# For reference: also compute no-interest baseline
rev_min_px = np.minimum.accumulate(px[::-1])[::-1]
f_max_px = 1.0 - rev_min_px / px
L0_max_no_int = np.minimum(4.0 / (1.0 + 3.0 * f_max_px), 4.0)

worst = int(np.argmin(L0_max))
print()
print("=== WORST POSSIBLE ENTRY DATE (interest compounding) ===")
print(f"Entry date           : {dates[worst].date()}")
# find where R bottoms out in the future from this entry
bidx = worst + int(np.argmin(R[worst:]))
years = (dates[bidx] - dates[worst]).days / 365.25
print(f"Worst day (min R_t)  : {dates[bidx].date()}   ({years:.2f} years later)")
print(f"SPX-TR peak->trough  : {(1 - px[bidx]/px[worst])*100:.2f}%")
print(f"Loan growth factor   : {M[bidx]/M[worst]:.4f}  "
      f"(loan compounded {(M[bidx]/M[worst]-1)*100:.2f}%)")
print(f"R_t / R_i            : {(R[bidx]/R[worst]):.4f}")
print(f"Max initial leverage : {L0_max[worst]:.4f}x  "
      f"(loan = {(L0_max[worst]-1)*100:.2f}% of equity)")
print(f"  -- baseline no-int : {L0_max_no_int[worst]:.4f}x")

print()
print("=== PERCENTILES of max-safe initial leverage across all entry dates ===")
print("        w/ interest          no interest")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    a = np.percentile(L0_max, q)
    b = np.percentile(L0_max_no_int, q)
    print(f"  p{q:>2}:  {a:.4f}x  ({(a-1)*100:6.2f}%)    "
          f"{b:.4f}x  ({(b-1)*100:6.2f}%)")

print()
print("=== 10 WORST ENTRY DATES (interest compounding) ===")
order = np.argsort(L0_max)[:10]
for i in order:
    bi = i + int(np.argmin(R[i:]))
    yr = (dates[bi] - dates[i]).days / 365.25
    print(f"  {dates[i].date()}  ->  worst {dates[bi].date()}  "
          f"({yr:4.1f}y)  SPX-TR {(1-px[bi]/px[i])*100:6.2f}%  "
          f"loan_grew {(M[bi]/M[i]-1)*100:5.2f}%  L0_max {L0_max[i]:.4f}x")
