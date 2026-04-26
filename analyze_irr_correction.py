"""
Verify: my "CAGR" metric was computing terminal^(1/H) - 1, which double-counts
DCA contributions as growth. The correct metric is IRR (money-weighted return)
which accounts for the timing of contributions.

Quick sanity check: if SPX returns exactly 9%/yr steady and we DCA 10%/yr,
the TRUE IRR should be 9% regardless of DCA. The "CAGR" formula will report
something much higher.
"""
import numpy as np
from datetime import datetime
from scipy.optimize import brentq
from data_loader import load

dates, px, tsy, mrate = load()
TRADING_DAYS = 252
DAYS_PER_MONTH = 21

M_box = np.concatenate(([1.0], np.cumprod(1 + (tsy + 0.0015)[1:] / TRADING_DAYS)))
post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate(loan_frac, annual_dca, horizon_years):
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0
    L0 = 1.0 + loan_frac
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)
    spx = np.full(N, L0, dtype=float)
    loan = np.full(N, loan_frac, dtype=float)
    called = np.zeros(N, dtype=bool)
    for k in range(1, H + 1):
        spx *= px[idxs + k] / px[idxs + k - 1]
        loan *= M_box[idxs + k] / M_box[idxs + k - 1]
        if k % DAYS_PER_MONTH == 0:
            spx = np.where(~called, spx + monthly, spx)
        equity = spx - loan
        called |= (~called) & ((equity <= 0) | (spx / np.maximum(equity, 1e-12) >= 4.0))
    terminal = np.where(called, 0.0, spx - loan)
    return terminal, called


def irr_with_monthly_dca(terminal, H_years, annual_contrib):
    """IRR given: $1 at t=0, annual_contrib/12 at end of each month for 12*H months.
    Returns r such that NPV(cash flows) = terminal.
    $1 * (1+r)^H + (c/12) * sum_{k=1..12H} (1+r)^((12H-k)/12) = terminal
    """
    if terminal <= 0:
        return -1.0
    m = annual_contrib / 12.0
    M = int(H_years * 12)
    def f(r):
        if abs(r) < 1e-10:
            return 1 + m * M - terminal
        d = (1 + r) ** (1/12)
        return (1+r)**H_years + m * (d**M - 1) / (d - 1) - terminal
    try:
        return brentq(f, -0.5, 1.5, xtol=1e-7)
    except ValueError:
        return np.nan


def bad_cagr(terminal, H_years):
    if terminal <= 0:
        return np.nan
    return terminal ** (1.0 / H_years) - 1.0


# Spot check: SPX steady 9%, DCA 10%, 30y.
# Truth: IRR = 9%
# Expected: bad_cagr ≠ 9%
print("=" * 80)
print("Sanity check: steady 9% SPX return, 10% DCA, 30y")
print("=" * 80)
r_true = 0.09
H = 30
c = 0.10
m = c / 12
M = H * 12
d_true = (1 + r_true) ** (1/12)
terminal_truth = (1 + r_true)**H + m * (d_true**M - 1) / (d_true - 1)
print(f"  Terminal = ${terminal_truth:.3f}")
print(f"  True IRR = {r_true*100:.2f}%")
print(f"  bad_cagr = {bad_cagr(terminal_truth, H)*100:.2f}%  "
      f"(this is what I reported!)")
print(f"  Difference (phantom inflation): "
      f"{(bad_cagr(terminal_truth, H) - r_true)*100:+.2f} pp")


# Apply to actual simulation output
print("\n" + "=" * 80)
print("Corrected IRR for selected cells (30y horizon, post-1932)")
print("=" * 80)

# For each cell, compute bad_cagr p50 and true_irr p50
cells = [
    (0.00, 0.00),  # unlev, no DCA (baseline — should match exactly)
    (0.00, 0.05),
    (0.00, 0.10),
    (0.00, 0.20),
    (0.00, 0.30),
    (0.30, 0.00),
    (0.30, 0.10),
    (0.41, 0.00),
    (0.41, 0.10),
    (0.50, 0.10),
    (0.60, 0.10),
    (0.75, 0.10),
    (1.00, 0.10),
]

print(f"\n{'L':>6}  {'DCA':>5}  {'median bad_CAGR':>16}  "
      f"{'median IRR':>11}  {'Δ(phantom)':>12}")
for L, dca in cells:
    terminal, called = simulate(L, dca, 30)
    valid = terminal > 0
    if valid.sum() == 0:
        print(f"{L*100:>5.0f}%  {dca*100:>4.0f}%  all called")
        continue
    med_term = np.median(terminal[valid])
    bc = bad_cagr(med_term, 30) * 100
    ir = irr_with_monthly_dca(med_term, 30, dca) * 100
    print(f"{L*100:>5.0f}%  {dca*100:>4.0f}%  "
          f"{bc:>14.2f}%  {ir:>9.2f}%  {bc-ir:>+11.2f} pp")

# Show corrected ΔIRR uplift vs same-DCA unlev baseline
print("\n" + "=" * 80)
print("Corrected ΔIRR (leverage's real value-add)")
print("=" * 80)
print(f"{'DCA':>6}  {'L=1.30x':>10}  {'L=1.41x':>10}  {'L=1.50x':>10}  "
      f"{'L=1.60x':>10}  {'L=1.75x':>10}")

for dca in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    # Baseline: unlev + dca
    term_base, _ = simulate(0.0, dca, 30)
    med_base = np.median(term_base[term_base > 0])
    irr_base = irr_with_monthly_dca(med_base, 30, dca) * 100
    row = [f"{dca*100:>5.0f}%"]
    for L in [0.30, 0.41, 0.50, 0.60, 0.75]:
        term, called = simulate(L, dca, 30)
        valid = term > 0
        if valid.sum() == 0:
            row.append("CALLED")
            continue
        med_t = np.median(term[valid])
        ir = irr_with_monthly_dca(med_t, 30, dca) * 100
        row.append(f"{(ir - irr_base):+5.2f}pp")
    print("  ".join(f"{c:>10}" for c in row))
