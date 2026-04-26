"""
Same comparison as analyze_clean_comparison.py but with a configurable
'contributions stop at year N' constraint, so we can test the
retirement scenario.

After contributions stop:
  - No more cash flow to pay interest or DCA
  - Interest compounds into loan (for strategies that had been paying it)
  - Amortized strategies: loan is already zero by year 10 in our tests,
    so no change post-retirement
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()
TRADING_DAYS = 252
DAYS_PER_MONTH = 21

M_box  = np.concatenate(([1.0], np.cumprod(1 + (tsy + 0.0015)[1:] / TRADING_DAYS)))
M_cash = np.concatenate(([1.0], np.cumprod(1 + tsy[1:] / TRADING_DAYS)))

post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate(initial_loan, amort_years, interest_paid, annual_cash,
             horizon_years, contrib_years):
    """
    contrib_years: Cash flow only happens for this many years, then 0.
    Interest compounds after contributions stop for interest-only strategies.
    """
    H = int(horizon_years * TRADING_DAYS)
    contrib_H = int(contrib_years * TRADING_DAYS)

    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    spx = np.full(N, 1.0 + initial_loan)
    loan = np.full(N, initial_loan)
    called = np.zeros(N, dtype=bool)
    peak_lev = np.full(N, 1.0 + initial_loan)

    monthly_cash = annual_cash / 12.0
    amort_months = int(amort_years * 12) if amort_years > 0 else 0

    for k in range(1, H + 1):
        spx_g = px[idxs + k] / px[idxs + k - 1]
        box_g = M_box[idxs + k] / M_box[idxs + k - 1]
        spx = spx * spx_g
        loan = loan * box_g

        # Contributions only within contrib phase
        if k <= contrib_H and k % DAYS_PER_MONTH == 0:
            month_num = k // DAYS_PER_MONTH
            if amort_months > 0:
                scheduled = initial_loan * max(0.0, 1.0 - month_num / amort_months)
            elif interest_paid:
                scheduled = initial_loan
            else:
                scheduled = None

            active = ~called
            if scheduled is not None:
                needed = np.maximum(loan - scheduled, 0.0)
                pay = np.minimum(needed, monthly_cash)
                loan = np.where(active, loan - pay, loan)
                remaining = monthly_cash - pay
            else:
                remaining = np.full(N, monthly_cash)
            dca_amt = np.maximum(remaining, 0.0)
            spx = np.where(active, spx + dca_amt, spx)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls
        peak_lev = np.maximum(peak_lev, np.where(pos, lev, peak_lev))

    terminal = np.where(called, 0.0, spx - loan)
    unlev_bh = px[idxs + H] / px[idxs]
    return terminal, called, peak_lev, unlev_bh


H_YEARS = 30
BUDGET = 0.10

def run(contrib_years, label):
    print(f"\n{'='*95}")
    print(f"Contributions for {contrib_years}y, then nothing.  "
          f"Horizon = {H_YEARS}y, $0.10/yr income, box-spread.   {label}")
    print("=" * 95)
    print(f"{'Strategy':<42}  {'call %':>6}  {'mean term':>9}  "
          f"{'median':>8}  {'p10':>7}  {'worst':>6}  "
          f"{'mean CAGR':>9}  {'peak L p95':>11}")
    scenarios = [
        ("0% loan, pure DCA",                   0.00, 0,  False),
        ("41% lump, int compounds, DCA rest",   0.41, 0,  False),
        ("50% lump, 10y full amort, DCA rest",  0.50, 10, True),
        ("60% lump, 10y full amort, DCA rest",  0.60, 10, True),
        ("70% lump, 10y full amort, DCA rest",  0.70, 10, True),
        ("50% lump, int-only, DCA rest",        0.50, 0,  True),
        ("60% lump, int-only, DCA rest",        0.60, 0,  True),
        ("70% lump, int-only, DCA rest",        0.70, 0,  True),
    ]
    for lbl, loan, amort, int_paid in scenarios:
        term, called, peakL, unlev = simulate(loan, amort, int_paid,
                                                BUDGET, H_YEARS, contrib_years)
        cagr = np.where(called, 0.0, term ** (1.0 / H_YEARS) - 1.0)
        print(f"{lbl:<42}  {called.mean()*100:>5.2f}%  "
              f"{term.mean():>8.3f}x  "
              f"{np.median(term):>7.3f}x  "
              f"{np.percentile(term, 10):>6.3f}x  "
              f"{term.min():>5.3f}x  "
              f"{(cagr*100).mean():>8.3f}%  "
              f"{np.percentile(peakL, 95):>10.2f}x")


run(30, "(contributions for full 30y — baseline)")
run(10, "(contributions for only 10y, then retirement)")
run(20, "(contributions for 20y, then retirement)")
