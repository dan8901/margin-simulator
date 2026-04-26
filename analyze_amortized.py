"""
Amortize the margin loan from external income.

Strategy B (amortized levered):
  Day 0: $1 brokerage equity, borrow L, buy SPX (brokerage leverage = 1+L).
  Each month, user pays: interest accrued that month + fixed principal L/(12N).
  After N years, loan = 0. Payments come from external income (salary).

Counterfactual (Strategy A, fair comparison):
  Day 0: $1 in SPX, no loan.
  Each month for N years, user INVESTS the same $ amount into SPX.
  After N years, continue compounding existing position.

Both strategies consume the same cash flow from the user's income.
The question: for a given (L, N), does amortized leverage beat
investing those same dollars in unlevered SPX?
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


def simulate_amortized(L_initial, amort_years, horizon_days):
    """Levered-and-amortized. Returns terminal brokerage equity, peak
    leverage, called bool, total $ paid (for comparison)."""
    amort_months = int(amort_years * 12)
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + horizon_days < len(px)]
    N = len(idxs)

    spx = np.full(N, 1.0 + L_initial)
    loan = np.full(N, L_initial)
    called = np.zeros(N, dtype=bool)
    peak_lev = np.full(N, 1.0 + L_initial)
    total_paid = np.zeros(N)

    for k in range(1, horizon_days + 1):
        spx  *= px[idxs + k] / px[idxs + k - 1]
        loan *= M_box[idxs + k] / M_box[idxs + k - 1]

        # Monthly boundary: pay scheduled amount from external income
        if k % DAYS_PER_MONTH == 0:
            month_num = k // DAYS_PER_MONTH
            if month_num <= amort_months:
                scheduled = L_initial * max(0.0, 1.0 - month_num / amort_months)
                payment = loan - scheduled
                # Only apply for non-called entries
                apply = ~called
                total_paid = np.where(apply, total_paid + payment, total_paid)
                loan = np.where(apply, scheduled, loan)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls
        peak_lev = np.maximum(peak_lev, np.where(pos, lev, peak_lev))

    terminal = np.where(called, 0.0, spx - loan)
    unlev = px[idxs + horizon_days] / px[idxs]
    return {
        "terminal": terminal, "unlev": unlev,
        "called": called, "peak_lev": peak_lev,
        "total_paid": total_paid, "idxs": idxs,
    }


def simulate_dca_counterfactual(schedule_amounts, schedule_months,
                                 horizon_days, idxs):
    """Unlev + monthly DCA: $1 in SPX day 0; then at month_k, invest
    schedule_amounts[k] into SPX. No leverage, no loan.

    schedule_amounts: per-entry per-month payment array, shape (N, M)
    schedule_months: month numbers (1..M)
    """
    N = len(idxs)
    spx = np.ones(N)
    # Apply month by month
    for j, month_num in enumerate(schedule_months):
        k = month_num * DAYS_PER_MONTH
        if k >= horizon_days:
            break
        # Grow spx from previous state to day k. But we don't track intermediate
        # states. Simpler: use closed form with cumulative growth from entry.
        pass
    # Easier: closed form
    # spx_T = 1 * px[T]/px[i] + sum_k s_k * px[T]/px[t_k]
    # where s_k is the amount invested on month k, t_k = k*21 days after entry
    terminal = px[idxs + horizon_days] / px[idxs]  # initial $1
    for j, month_num in enumerate(schedule_months):
        k = month_num * DAYS_PER_MONTH
        if k >= horizon_days:
            break
        terminal = terminal + schedule_amounts[:, j] * \
                              px[idxs + horizon_days] / px[idxs + k]
    return terminal


def run_scenario(L_initial, amort_years, horizon_years):
    """Run amortized-lev vs DCA-counterfactual with matched cash flow."""
    H = int(horizon_years * TRADING_DAYS)
    res = simulate_amortized(L_initial, amort_years, H)
    idxs = res["idxs"]
    N = len(idxs)

    # Build per-month cash flow for each entry (from total_paid and schedule)
    # Simpler: reconstruct the schedule exactly.
    amort_months = int(amort_years * 12)

    # Per-entry per-month payment
    # Month m payment = scheduled_balance[m-1] * interest(m) + L/(12N)
    # But interest depends on the entry's box rate trajectory.
    # Let's compute payments month by month, same as in simulate_amortized.
    monthly_payments = np.zeros((N, amort_months))
    loan = np.full(N, L_initial)
    for m in range(1, amort_months + 1):
        # Compound loan over the month
        # Starting day = (m-1)*21, ending day = m*21
        start = (m - 1) * DAYS_PER_MONTH
        end = m * DAYS_PER_MONTH
        if end >= H:
            break
        growth = M_box[idxs + end] / M_box[idxs + start]
        loan = loan * growth
        scheduled = L_initial * max(0.0, 1.0 - m / amort_months)
        payment = loan - scheduled
        monthly_payments[:, m - 1] = payment
        loan = np.full(N, scheduled)

    dca_terminal = simulate_dca_counterfactual(
        monthly_payments,
        list(range(1, amort_months + 1)),
        H, idxs)

    levered_term = res["terminal"]
    # CAGR
    cagr_lev = np.where(res["called"], 0.0,
                        levered_term ** (1.0 / horizon_years) - 1.0)
    cagr_dca = dca_terminal ** (1.0 / horizon_years) - 1.0
    cagr_unlev = res["unlev"] ** (1.0 / horizon_years) - 1.0

    delta_vs_dca = np.where(res["called"], -cagr_dca,
                            cagr_lev - cagr_dca) * 100
    delta_vs_unlev = np.where(res["called"], -cagr_unlev,
                              cagr_lev - cagr_unlev) * 100

    total_paid = res["total_paid"].mean()
    return {
        "L": L_initial, "N": amort_years, "H": horizon_years,
        "call_rate": res["called"].mean() * 100,
        "mean_total_paid": total_paid,
        "mean_annual_paid": total_paid / amort_years,
        "mean_delta_vs_unlev": delta_vs_unlev.mean(),
        "p10_delta_vs_unlev": np.percentile(delta_vs_unlev, 10),
        "mean_delta_vs_dca": delta_vs_dca.mean(),
        "p10_delta_vs_dca": np.percentile(delta_vs_dca, 10),
        "mean_term_mult": np.where(res["called"], 0.0,
                                    levered_term / res["unlev"]).mean(),
        "mean_term_mult_vs_dca": np.where(res["called"], 0.0,
                                           levered_term / dca_terminal).mean(),
        "peak_lev_p50": np.percentile(res["peak_lev"], 50),
        "peak_lev_p95": np.percentile(res["peak_lev"], 95),
    }


# ============================================================
# Run sweep
# ============================================================
Ls = [0.30, 0.50, 0.75, 1.00, 1.50]
Ns = [5, 10, 20]
horizons = [20, 30]

for h in horizons:
    print("\n" + "=" * 100)
    print(f"Horizon = {h}y.   Strategy B: borrow L on day 0, amortize over N years, "
          f"pay from income.")
    print(f"Counterfactual A: invest same $ in unlev SPX monthly (same cash outflow).")
    print("=" * 100)
    print(f"{'L':>6}  {'N':>3}  {'call %':>7}  {'paid $/yr':>10}  "
          f"{'ΔCAGR vs unlev':>15}  {'ΔCAGR vs DCA':>14}  "
          f"{'term vs unlev':>14}  {'term vs DCA':>12}  "
          f"{'peak L p95':>11}")
    for L in Ls:
        for N in Ns:
            s = run_scenario(L, N, h)
            print(f"{L*100:>5.0f}%  {N:>3d}y  "
                  f"{s['call_rate']:>6.2f}%  "
                  f"{s['mean_annual_paid']*100:>9.2f}%  "
                  f"{s['mean_delta_vs_unlev']:>+14.2f}pp  "
                  f"{s['mean_delta_vs_dca']:>+13.2f}pp  "
                  f"{s['mean_term_mult']:>13.3f}x  "
                  f"{s['mean_term_mult_vs_dca']:>11.3f}x  "
                  f"{s['peak_lev_p95']:>10.2f}x")
