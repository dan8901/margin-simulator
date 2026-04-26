"""
Two sensitivity analyses:

(A) Peak-leverage target sensitivity. Instead of a 4.0x cap, consider
    pragmatic caps of 3.0x / 3.5x / 3.9x (margin-of-safety cushions).
    Generalized closed form (interest compounding on loan):

        L_t = L_max           (binding)
        =>  L_0 = L_max / [L_max - (L_max - 1) * (R_t / R_i)]
        =>  L_0_max(i) = L_max * R_i / [L_max * R_i - (L_max - 1) * min_future_R]

    Derivation: starting from A_t = L0 * P_t/P_i, D_t = (L0-1) * M_t/M_i,
    A_t / (A_t - D_t) = L_max  gives the identity above.

(B) 2000-2009 rate sensitivity. What if the margin rate during the
    dot-com-to-GFC window had been materially higher? Recompute the
    critical L_0 on 2000-03-23 with the margin rate during that window
    set to a flat alternative level, keeping the rest of the series unchanged.
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, _tsy, mrate = load()


def max_L0(px, mrate, L_max_cap, mask=None):
    """Return array of max-safe initial leverage per entry date, given
    a peak-leverage cap `L_max_cap` (e.g. 4.0, 3.5, ...). `mask` limits
    which entry dates to include (for percentile reporting)."""
    daily_factor = 1.0 + mrate / 252.0
    M = np.concatenate(([1.0], np.cumprod(daily_factor[1:])))
    R = px / M
    min_future_R = np.minimum.accumulate(R[::-1])[::-1]
    denom = L_max_cap * R - (L_max_cap - 1.0) * min_future_R
    L0 = L_max_cap * R / denom
    return np.minimum(L0, L_max_cap)


# ===== (A) Peak-leverage target sensitivity =====
print("=" * 78)
print("(A) Initial leverage under different PEAK-LEVERAGE CAPS")
print("    (post-1932 entry dates, interest compounding)")
print("=" * 78)

post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])

caps = [3.0, 3.5, 3.9, 4.0]
rows = {cap: max_L0(px, mrate, cap)[post1932] for cap in caps}

PCTS = [0.1, 1, 5, 10, 20, 25, 50, 75, 90, 95, 99]
print(f"{'pctile':>6}  " + "  ".join(f"cap={c:>4.1f}x".rjust(14) for c in caps))
for q in PCTS:
    parts = []
    for cap in caps:
        v = np.percentile(rows[cap], q)
        parts.append(f"{v:.4f}x ({(v-1)*100:5.1f}%)")
    print(f"{q:>6.1f}  " + "  ".join(p.rjust(14) for p in parts))

# Worst entry date per cap
print("\nWorst entry date per cap (post-1932):")
for cap in caps:
    arr = rows[cap]
    idx_in_mask = int(np.argmin(arr))
    idx_full = int(np.where(post1932)[0][idx_in_mask])
    print(f"  cap={cap:.1f}x  worst entry {dates[idx_full].date()}  "
          f"L0_max = {arr[idx_in_mask]:.4f}x  "
          f"(loan = {(arr[idx_in_mask]-1)*100:.2f}% of equity)")

# 2000-03-23 specifically under each cap
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
print("\n2000-03-23 entry specifically:")
print(f"{'cap':>6}  {'L0_max':>8}  {'loan/eq':>8}  "
      f"{'cushion vs 4x':>14}")
L4 = float(max_L0(px, mrate, 4.0)[i0])
for cap in caps:
    v = float(max_L0(px, mrate, cap)[i0])
    cushion = (L4 - v) / L4 * 100.0
    print(f"{cap:>5.1f}x  {v:.4f}x  {(v-1)*100:>6.2f}%  "
          f"{cushion:>12.1f}% less")

# ===== (B) 2000-2009 rate sensitivity =====
print("\n" + "=" * 78)
print("(B) 2000-2009 MARGIN RATE sensitivity (entry 2000-03-23)")
print("=" * 78)

# Window: from entry through GFC trough + a bit
win_start = datetime(2000, 3, 23)
win_end   = datetime(2009, 12, 31)
win_mask = np.array([(win_start <= d <= win_end) for d in dates])

actual_avg = mrate[win_mask].mean()
print(f"Actual avg margin rate 2000-03-23..2009-12-31: "
      f"{actual_avg*100:.3f}%  "
      f"(min {mrate[win_mask].min()*100:.2f}%, "
      f"max {mrate[win_mask].max()*100:.2f}%)")

alt_levels = [actual_avg, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
print(f"\n{'alt rate':>9}  {'L0_max (4x cap)':>16}  "
      f"{'loan/eq':>8}  {'vs actual':>10}")
actual_L4 = float(max_L0(px, mrate, 4.0)[i0])
for alt in alt_levels:
    mr_alt = mrate.copy()
    mr_alt[win_mask] = alt
    v = float(max_L0(px, mr_alt, 4.0)[i0])
    label = f"{alt*100:.2f}%"
    if abs(alt - actual_avg) < 1e-6:
        label += "*"
    delta = (v - actual_L4) * 100
    print(f"{label:>9}  {v:>15.4f}x  {(v-1)*100:>6.2f}%  "
          f"{delta:>+9.2f}pp")
print("  (* = historical average; the row at 4-5% ~= historical replay)")

# Also: what if rates were uniformly +X bps higher across the WHOLE window?
print("\nAlternative framing: add +Δ bps to actual rates during 2000-2009:")
print(f"{'Δ bps':>7}  {'L0_max':>10}  {'loan/eq':>8}")
for d_bps in [0, 50, 100, 200, 300, 500, 1000]:
    mr_alt = mrate.copy()
    mr_alt[win_mask] = mr_alt[win_mask] + d_bps / 10000.0
    v = float(max_L0(px, mr_alt, 4.0)[i0])
    print(f"{d_bps:>+6d}  {v:>9.4f}x  {(v-1)*100:>6.2f}%")

# And: what's the worst entry date if the rate bump is global?
print("\nGlobal rate bump (applied to entire history) — "
      "worst post-1932 entry date:")
print(f"{'Δ bps':>7}  {'worst date':>12}  {'L0_max':>10}  {'loan/eq':>8}")
for d_bps in [-100, 0, 100, 200, 300, 500]:
    mr_alt = mrate + d_bps / 10000.0
    mr_alt = np.maximum(mr_alt, 0.0)  # don't go negative
    arr = max_L0(px, mr_alt, 4.0)[post1932]
    j = int(np.argmin(arr))
    jf = int(np.where(post1932)[0][j])
    print(f"{d_bps:>+6d}  {str(dates[jf].date()):>12}  "
          f"{arr[j]:>9.4f}x  {(arr[j]-1)*100:>6.2f}%")
