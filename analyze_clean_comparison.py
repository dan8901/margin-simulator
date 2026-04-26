"""
Clean apples-to-apples comparison of leverage strategies at EQUAL ANNUAL
CASH FLOW from income (10% of initial equity per year for 30 years).

Strategies:
  (0) No leverage, 100% of cash flow into SPX monthly (DCA baseline).
  (1) 41% lump, interest compounds into loan, 100% of cash flow into SPX DCA.
  (2) X% lump, 10-yr full amortization (linear principal); remainder to DCA.
  (3) X% lump, interest-only paydown (principal stays forever); remainder to DCA.

Financing: box-spread rate (3M Tsy + 15bps).
All entry dates: post-1932 with 30y of future data.
Terminal wealth = brokerage equity at year 30 (SPX - Loan); called paths = 0.
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
             horizon_years):
    """
    initial_loan      fraction of initial equity borrowed at day 0
    amort_years       years of LINEAR principal amortization (0 = no amort)
    interest_paid     True  => monthly interest paid from cash flow
                      False => interest compounds into loan
    annual_cash       annual cash flow from income (fraction of initial equity)
    horizon_years     simulation horizon
    """
    H = int(horizon_years * TRADING_DAYS)
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    # Total SPX in brokerage (initial + any DCA)
    spx = np.full(N, 1.0 + initial_loan)
    loan = np.full(N, initial_loan)
    called = np.zeros(N, dtype=bool)
    peak_lev = np.full(N, (1.0 + initial_loan))

    monthly_cash = annual_cash / 12.0
    amort_months = int(amort_years * 12) if amort_years > 0 else 0

    total_cash_used = np.zeros(N)

    for k in range(1, H + 1):
        spx_g = px[idxs + k] / px[idxs + k - 1]
        box_g = M_box[idxs + k] / M_box[idxs + k - 1]

        spx = spx * spx_g
        loan = loan * box_g

        if k % DAYS_PER_MONTH == 0:
            month_num = k // DAYS_PER_MONTH

            # Scheduled loan balance after this month's payment
            if amort_months > 0:
                scheduled = initial_loan * max(0.0, 1.0 - month_num / amort_months)
            elif interest_paid:
                # Interest-only: keep balance at initial_loan
                scheduled = initial_loan
            else:
                # No amort, interest compounds
                scheduled = None

            active = ~called
            if scheduled is not None:
                needed = np.maximum(loan - scheduled, 0.0)
                pay = np.minimum(needed, monthly_cash)
                loan = np.where(active, loan - pay, loan)
                remaining = monthly_cash - pay
                total_cash_used += np.where(active, pay, 0.0)
            else:
                remaining = np.full(N, monthly_cash)

            # DCA remainder into SPX
            dca_amt = np.maximum(remaining, 0.0)
            spx = np.where(active, spx + dca_amt, spx)
            total_cash_used += np.where(active, dca_amt, 0.0)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls
        peak_lev = np.maximum(peak_lev, np.where(pos, lev, peak_lev))

    terminal = np.where(called, 0.0, spx - loan)
    unlev_bh = px[idxs + H] / px[idxs]  # $1 day-0 SPX, no DCA
    return {
        "terminal": terminal, "unlev_bh": unlev_bh,
        "called": called, "peak_lev": peak_lev,
        "total_cash_used": total_cash_used,
        "idxs": idxs, "H": H,
    }


def run_and_print():
    H_YEARS = 30
    BUDGET = 0.10  # 10% of initial equity per year from income
    H = int(H_YEARS * TRADING_DAYS)

    # Baseline: unlev + 10%/yr DCA (no loan at all).  Reference for "just invest income"
    base = simulate(0.0, 0, False, BUDGET, H_YEARS)

    scenarios = [
        ("0% loan (DCA only, no leverage)",    0.00, 0,  False),
        ("41% lump, int compounds, DCA rest",  0.41, 0,  False),
        ("50% lump, 10y full amort, DCA rest", 0.50, 10, True),
        ("60% lump, 10y full amort, DCA rest", 0.60, 10, True),
        ("70% lump, 10y full amort, DCA rest", 0.70, 10, True),
        ("50% lump, interest-only, DCA rest",  0.50, 0,  True),
        ("60% lump, interest-only, DCA rest",  0.60, 0,  True),
        ("70% lump, interest-only, DCA rest",  0.70, 0,  True),
    ]

    print(f"\n{'Strategy':<42}  {'call %':>7}  "
          f"{'mean term':>9}  {'median':>8}  {'p10':>7}  "
          f"{'worst':>7}  {'mean CAGR':>9}  "
          f"{'vs DCA-only':>12}  {'peak L p95':>11}")

    results = {}
    for label, loan, amort, interest_paid in scenarios:
        r = simulate(loan, amort, interest_paid, BUDGET, H_YEARS)
        term = r["terminal"]
        cagr = np.where(r["called"], 0.0, term ** (1.0 / H_YEARS) - 1.0)
        base_cagr = base["terminal"] ** (1.0 / H_YEARS) - 1.0
        dc_vs_base = np.where(r["called"], -base_cagr, cagr - base_cagr) * 100
        results[label] = r
        print(f"{label:<42}  {r['called'].mean()*100:>6.2f}%  "
              f"{term.mean():>8.3f}x  "
              f"{np.median(term):>7.3f}x  "
              f"{np.percentile(term, 10):>6.3f}x  "
              f"{term.min():>6.3f}x  "
              f"{(cagr*100).mean():>8.3f}%  "
              f"{dc_vs_base.mean():>+11.2f}pp  "
              f"{np.percentile(r['peak_lev'], 95):>10.2f}x")

    # 2000-03-23 detail
    print("\n" + "=" * 80)
    print("Detail: entry 2000-03-23, 23y horizon")
    print("=" * 80)
    i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
    H_2000 = len(px) - i0 - 1
    for label, loan, amort, interest_paid in scenarios:
        # Single-entry simulation
        spx = 1.0 + loan
        ld = loan
        called = False
        peak = 1.0 + loan
        monthly_cash = BUDGET / 12
        amort_months = int(amort * 12) if amort > 0 else 0
        total_outflow = 0.0
        for k in range(1, H_2000 + 1):
            spx *= px[i0 + k] / px[i0 + k - 1]
            ld  *= M_box[i0 + k] / M_box[i0 + k - 1]
            if k % DAYS_PER_MONTH == 0:
                mn = k // DAYS_PER_MONTH
                if amort_months > 0:
                    sch = loan * max(0.0, 1.0 - mn / amort_months)
                elif interest_paid:
                    sch = loan
                else:
                    sch = None
                if sch is not None:
                    needed = max(ld - sch, 0)
                    pay = min(needed, monthly_cash)
                    ld -= pay
                    rem = monthly_cash - pay
                    total_outflow += pay
                else:
                    rem = monthly_cash
                spx += max(rem, 0)
                total_outflow += max(rem, 0)
            eq = spx - ld
            if eq <= 0:
                called = True
                break
            lv = spx / eq
            peak = max(peak, lv)
            if lv >= 4.0:
                called = True
                break
        term = 0 if called else spx - ld
        base_unlev = px[i0 + H_2000] / px[i0]
        print(f"  {label:<42}  peak L = {peak:5.2f}x  "
              f"terminal = {'CALLED' if called else f'{term:6.3f}x'}  "
              f"(unlev B&H = {base_unlev:.3f}x)")


run_and_print()
