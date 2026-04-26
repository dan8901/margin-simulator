"""
Grid of (DCA, leverage) combinations with percentile distributions.

For each (annual DCA %, initial loan %), simulate across all post-1932
entries with 30y horizon. Report:
  - call rate
  - CAGR at p10 / p50 / p90 (and min, max)
  - ΔCAGR vs same-DCA unlev baseline
  - Peak leverage distribution
"""
import numpy as np
from datetime import datetime
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
    return terminal, called, peak_L, unlev_bh


DCAs = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LEVERAGES = [0.00, 0.30, 0.41, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50]

# Compute for both 20y and 30y horizons
all_results = {}
for H_YEARS in (20, 30):
    results = {}
    for dca in DCAs:
        for L in LEVERAGES:
            term, called, peakL, unlev = simulate(L, dca, H_YEARS)
            cagr = np.where(called, np.nan, term ** (1.0 / H_YEARS) - 1.0)
            results[(dca, L)] = {
                "call_rate": called.mean() * 100,
                "cagr": cagr,
                "term": term,
                "peak_L": peakL,
                "unlev": unlev,
            }
    all_results[H_YEARS] = results


def fmt_cagr(arr, q):
    v = np.nanpercentile(arr, q)
    return f"{v*100:5.2f}%"


def print_grid(horizon, title, cell_fn, dca_levels=DCAs, lev_levels=LEVERAGES):
    results = all_results[horizon]
    print("\n" + "=" * 110)
    print(f"[{horizon}y horizon]  {title}")
    print("=" * 110)
    header = f"{'DCA':>5}  " + "  ".join(
        f"{('L=' + format(1+L, '.2f') + 'x'):>12}" for L in lev_levels)
    print(header)
    for dca in dca_levels:
        row = [f"{dca*100:>4.0f}%"]
        for L in lev_levels:
            r = results[(dca, L)]
            row.append(cell_fn(r, dca, L))
        print("  " + "  ".join(f"{c:>12}" for c in row))


def delta_fn(horizon, q):
    baselines = {dca: all_results[horizon][(dca, 0.0)]['cagr'] for dca in DCAs}
    def f(r, dca, L):
        base = baselines[dca]
        delta = r['cagr'] - base
        v = np.nanpercentile(delta, q)
        return f"{v*100:+5.2f}pp"
    return f


for H_YEARS in (20, 30):
    print(f"\n\n{'#'*110}\n# HORIZON = {H_YEARS} YEARS\n{'#'*110}")
    print_grid(H_YEARS, "CALL RATE  (% of entries that hit margin call)",
               lambda r, d, L: f"{r['call_rate']:5.2f}%")
    print_grid(H_YEARS, "MEDIAN CAGR (p50; NaN-in-percentile = called)",
               lambda r, d, L: fmt_cagr(r['cagr'], 50))
    print_grid(H_YEARS, "DOWNSIDE CAGR (p10 — bad scenarios)",
               lambda r, d, L: fmt_cagr(r['cagr'], 10))
    print_grid(H_YEARS, "UPSIDE CAGR (p90 — favorable scenarios)",
               lambda r, d, L: fmt_cagr(r['cagr'], 90))
    print_grid(H_YEARS, "ΔCAGR vs same-DCA unlev — MEDIAN",
               delta_fn(H_YEARS, 50))
    print_grid(H_YEARS, "ΔCAGR vs same-DCA unlev — p10 (bad scenarios)",
               delta_fn(H_YEARS, 10))
    print_grid(H_YEARS, "PEAK LEVERAGE p95",
               lambda r, d, L: f"{np.percentile(r['peak_L'], 95):5.2f}x")
