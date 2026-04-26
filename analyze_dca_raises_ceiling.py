"""
How much does ongoing DCA raise the max-safe starting leverage?

Strategy: one-time loan L on day 0, interest compounds, DCA $m/month
from income into additional SPX. Same account, same broker margin.

For each annual contribution level (as fraction of initial equity),
sweep the loan fraction and find the max-safe L (zero margin calls
across all post-1932 entries with 20y of future data).

20y horizon = includes the 2000-03-23 worst-case post-1932 entry.
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()
TRADING_DAYS = 252
DAYS_PER_MONTH = 21

M_box = np.concatenate(([1.0], np.cumprod(1 + (tsy + 0.0015)[1:] / TRADING_DAYS)))
post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate(loan_frac, annual_contrib, horizon_years):
    """Simulate levered lump-sum + monthly DCA. Interest compounds.
    Returns call rate, peak-leverage distribution, terminal wealth."""
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_contrib / 12.0
    L0 = 1.0 + loan_frac

    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    spx = np.full(N, L0, dtype=float)
    loan = np.full(N, loan_frac, dtype=float)
    called = np.zeros(N, dtype=bool)
    peak_L = np.full(N, L0, dtype=float)

    for k in range(1, H + 1):
        spx_g = px[idxs + k] / px[idxs + k - 1]
        box_g = M_box[idxs + k] / M_box[idxs + k - 1]
        spx = spx * spx_g
        loan = loan * box_g

        if k % DAYS_PER_MONTH == 0:
            active = ~called
            spx = np.where(active, spx + monthly, spx)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls
        peak_L = np.maximum(peak_L, np.where(pos, lev, peak_L))

    terminal = np.where(called, 0.0, spx - loan)
    unlev_bh = px[idxs + H] / px[idxs]
    return {
        "called": called, "peak_L": peak_L, "terminal": terminal,
        "unlev_bh": unlev_bh, "idxs": idxs,
    }


def find_max_safe(annual_contrib, horizon_years, tolerance=0.0):
    """Binary search for max loan_frac with call rate <= tolerance."""
    lo, hi = 0.0, 3.0
    for _ in range(18):
        mid = (lo + hi) / 2
        r = simulate(mid, annual_contrib, horizon_years)
        if r["called"].mean() <= tolerance:
            lo = mid
        else:
            hi = mid
    return lo


# -----------------------------------------------------------
# Part 1: find max-safe loan for each contribution level
# -----------------------------------------------------------
print("=" * 85)
print(f"MAX-SAFE starting loan (zero margin calls across all "
      f"post-1932 20y-horizon entries)")
print("=" * 85)
print(f"{'Annual DCA':>12}  {'max-safe L (0%)':>17}  "
      f"{'max-safe L (0.5%)':>17}  {'max-safe L (1%)':>17}")

contrib_levels = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
max_safe = {}
for c in contrib_levels:
    safe0    = find_max_safe(c, 20, 0.000)
    safe05   = find_max_safe(c, 20, 0.005)
    safe1    = find_max_safe(c, 20, 0.010)
    max_safe[c] = (safe0, safe05, safe1)
    print(f"{c*100:>11.0f}%  "
          f"{safe0*100:>14.1f}%  ({1+safe0:.2f}x)   "
          f"{safe05*100:>13.1f}%  ({1+safe05:.2f}x)  "
          f"{safe1*100:>13.1f}%  ({1+safe1:.2f}x)")

# -----------------------------------------------------------
# Part 2: for each contribution level, show CAGR uplift vs. same-DCA unlev baseline
# -----------------------------------------------------------
print("\n" + "=" * 100)
print("CAGR uplift vs. unlevered-with-same-DCA baseline, 30y horizon")
print("=" * 100)
H_YEARS = 30

loan_levels = [0.00, 0.30, 0.41, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
print(f"\n{'contrib':>8}" +
      "".join(f"  {('L='+format(L, '.2f')+'x'):>14}" for L in [1+l for l in loan_levels]))
for c in contrib_levels:
    # Compute baseline CAGR (unlev + DCA)
    r_base = simulate(0.0, c, H_YEARS)
    base_cagr = r_base["terminal"] ** (1.0 / H_YEARS) - 1.0
    row = [f"{c*100:>7.0f}%"]
    for L in loan_levels:
        r = simulate(L, c, H_YEARS)
        term = r["terminal"]
        cagr = np.where(r["called"], 0.0, term ** (1.0 / H_YEARS) - 1.0)
        delta = np.where(r["called"], -base_cagr, cagr - base_cagr) * 100
        call_rate = r["called"].mean() * 100
        flag = "!" if call_rate > 0 else " "
        row.append(f"{delta.mean():+7.2f}pp{flag}{call_rate:>3.1f}%")
    print("".join(f"  {c:>14}" for c in row[:1]) + "".join(f"  {c:>14}" for c in row[1:]))
print("\n(! marks scenarios with non-zero call rate; % shown = call rate)")

# -----------------------------------------------------------
# Part 3: the 2000-03-23 stress test at different L and contrib levels
# -----------------------------------------------------------
print("\n" + "=" * 85)
print("Peak leverage at 2000-03-23 entry (23y horizon) for various L and DCA")
print("=" * 85)
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
H_2000 = len(px) - i0 - 1
print(f"\n{'L (loan%)':>10}" + "".join(f"  {(str(int(c*100))+'%'):>7}" for c in contrib_levels))
for L in loan_levels:
    if L == 0:
        continue
    row = [f"{L*100:>9.0f}%"]
    for c in contrib_levels:
        # Single-entry sim
        spx = 1.0 + L
        loan = L
        called = False
        peak = 1.0 + L
        for k in range(1, H_2000 + 1):
            spx *= px[i0+k]/px[i0+k-1]
            loan *= M_box[i0+k]/M_box[i0+k-1]
            if k % DAYS_PER_MONTH == 0:
                spx += c / 12
            eq = spx - loan
            if eq <= 0:
                called = True
                break
            lv = spx / eq
            peak = max(peak, lv)
            if lv >= 4.0:
                called = True
                break
        row.append(f"{peak:6.2f}x" if not called else "CALLED ")
    print("".join(row))
