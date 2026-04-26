"""
Does DCAing the loan entry help?

Compare a lump-sum 30% loan on day 0 vs. DCAing the loan in tranches
over N months. All strategies end at the same total loan size ($0.30
nominal = 30% of initial equity).

Math: each tranche delta_k taken at day t_k contributes
  ΔA_T = delta_k * px[T]/px[t_k]    (asset growth of that tranche)
  ΔD_T = delta_k * M[T]/M[t_k]      (loan growth of that tranche)
so terminal equity = px[T]/px[i] + sum_k delta_k * (px[T]/px[t_k] - M[T]/M[t_k]).

(First term is the investor's own $1 invested in SPX on day 0.)
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()

TRADING_DAYS = 252
DAYS_PER_MONTH = 21

def cum_factor(rate_annual):
    df = 1.0 + rate_annual / TRADING_DAYS
    return np.concatenate(([1.0], np.cumprod(df[1:])))

M_broker = cum_factor(mrate)
M_box    = cum_factor(tsy + 0.0015)

post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def run(tranches, M_loan, horizon_years):
    """tranches = list of (day_offset_from_entry, loan_fraction).
    Returns array of terminal equity and array of unlevered baseline."""
    H = int(horizon_years * TRADING_DAYS)
    max_off = max(t[0] for t in tranches) if tranches else 0
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    idxs = idxs[idxs + max_off < len(px)]

    base = px[idxs + H] / px[idxs]
    term = base.copy()
    for (offset, delta) in tranches:
        tk = idxs + offset
        term = term + delta * (px[idxs + H] / px[tk] - M_loan[idxs + H] / M_loan[tk])
    return base, term


def make_dca(n_months, total=0.30):
    per = total / n_months
    return [(k * DAYS_PER_MONTH, per) for k in range(n_months)]


strategies = [
    ("lump sum        30% day 0",     [(0, 0.30)]),
    ("DCA  5% / mo  × 6  (6 mo)",     make_dca(6)),
    ("DCA 2.5% / mo × 12 (1 yr)",     make_dca(12)),
    ("DCA 1.25%/mo × 24 (2 yr)",      make_dca(24)),
    ("DCA 0.83%/mo × 36 (3 yr)",      make_dca(36)),
]


def report(horizon_years, M_loan, rate_label):
    print(f"\n=== Horizon {horizon_years}y, financing {rate_label}, "
          f"post-1932, total loan = 30% of initial equity ===")
    print(f"{'strategy':<28}  {'mean ΔCAGR':>11}  {'p10 ΔCAGR':>10}  "
          f"{'p50 ΔCAGR':>10}  {'p90 ΔCAGR':>10}  {'worst ΔCAGR':>11}  "
          f"{'stdev':>7}")
    rows = {}
    for name, tranches in strategies:
        base, term = run(tranches, M_loan, horizon_years)
        dc = ((term / base) ** (1.0 / horizon_years) - 1.0) * 100
        rows[name] = dc
        print(f"{name:<28}  {dc.mean():>+10.3f}pp  "
              f"{np.percentile(dc, 10):>+9.3f}pp  "
              f"{np.percentile(dc, 50):>+9.3f}pp  "
              f"{np.percentile(dc, 90):>+9.3f}pp  "
              f"{dc.min():>+10.3f}pp  "
              f"{dc.std():>6.3f}")
    # Show "cost of DCA" = lump_sum - DCA mean uplift
    base_mean = rows[strategies[0][0]].mean()
    print(f"\n  Cost of DCA vs lump sum (mean ΔCAGR lost):")
    for name, _ in strategies[1:]:
        print(f"    {name}: {base_mean - rows[name].mean():+6.3f} pp/yr")
    # Show worst-case improvement from DCA
    base_worst = rows[strategies[0][0]].min()
    print(f"\n  Worst-case ΔCAGR improvement from DCA:")
    for name, _ in strategies[1:]:
        print(f"    {name}: {rows[name].min() - base_worst:+6.3f} pp/yr")


for h in [20, 30]:
    report(h, M_box, "box spread (3M Tsy + 15bps)")

# Detail: 2000-03-23 entry specifically — how much does DCA help
# at the literal worst case?
print("\n" + "=" * 78)
print("Detailed case: entry 2000-03-23 (dot-com peak)")
print("=" * 78)
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
for horizon_years in [10, 20, 23]:  # 23y gets to end of 2023 dataset
    H = int(horizon_years * TRADING_DAYS)
    if i0 + H >= len(px):
        H = len(px) - i0 - 1
        horizon_years = H / TRADING_DAYS
    base = float(px[i0 + H] / px[i0])
    print(f"\n  Horizon {horizon_years:.1f}y  (to {dates[i0+H].date()}, "
          f"unlev terminal = {base:.3f}x):")
    for name, tranches in strategies:
        term = base
        for (offset, delta) in tranches:
            tk = i0 + offset
            if tk >= len(px):
                continue
            term += delta * (px[i0+H]/px[tk] - M_box[i0+H]/M_box[tk])
        dc = ((term / base) ** (1.0 / horizon_years) - 1.0) * 100
        diff = (term / base - 1.0) * 100
        print(f"    {name:<28}  terminal {term:6.3f}x  "
              f"(+{diff:5.2f}% vs unlev, ΔCAGR +{dc:5.3f}pp)")
