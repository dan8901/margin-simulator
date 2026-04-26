"""
Dynamic re-levering strategy: maintain a target leverage ratio by taking
additional margin loan when leverage drops below target (due to SPX
appreciation or DCA contributions). NEVER sells to rebalance down.

Compare to static one-time loan (never rebalance).

Monthly rebalancing cadence. At each monthly tick:
  1. SPX and loan grow to current values (already done daily).
  2. Apply DCA: deposit m = annual_dca/12 into SPX.
  3. If current_leverage < target: borrow ΔD = target*E - A, invest in SPX.
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


def simulate(target_lev, annual_dca, horizon_years, relever=True):
    """Static (relever=False) or monthly lever-up-only (relever=True)."""
    H = int(horizon_years * TRADING_DAYS)
    monthly = annual_dca / 12.0
    loan_frac = target_lev - 1.0

    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + H < len(px)]
    N = len(idxs)

    spx = np.full(N, target_lev, dtype=float)
    loan = np.full(N, loan_frac, dtype=float)
    called = np.zeros(N, dtype=bool)
    peak_L = np.full(N, target_lev, dtype=float)
    cum_loan_taken = np.full(N, loan_frac, dtype=float)
    n_relever_events = np.zeros(N, dtype=int)

    for k in range(1, H + 1):
        spx_g = px[idxs + k] / px[idxs + k - 1]
        box_g = M_box[idxs + k] / M_box[idxs + k - 1]
        spx = spx * spx_g
        loan = loan * box_g

        if k % DAYS_PER_MONTH == 0:
            active = ~called
            # Monthly DCA deposit
            spx = np.where(active, spx + monthly, spx)

            if relever:
                # Rebalance up: ΔD = target * E - A, only if positive
                equity = spx - loan
                pos = equity > 0
                current_lev = np.where(pos, spx / np.maximum(equity, 1e-12),
                                       np.inf)
                want_lever = (current_lev < target_lev) & active & pos
                delta_D = target_lev * equity - spx
                delta_D = np.maximum(delta_D, 0.0)
                loan = np.where(want_lever, loan + delta_D, loan)
                spx = np.where(want_lever, spx + delta_D, spx)
                cum_loan_taken = np.where(want_lever,
                                          cum_loan_taken + delta_D,
                                          cum_loan_taken)
                n_relever_events = np.where(want_lever,
                                            n_relever_events + 1,
                                            n_relever_events)

        equity = spx - loan
        pos = equity > 0
        lev = np.where(pos, spx / np.maximum(equity, 1e-12), np.inf)
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls
        peak_L = np.maximum(peak_L, np.where(pos, lev, peak_L))

    terminal = np.where(called, 0.0, spx - loan)
    return {
        "terminal": terminal, "called": called,
        "peak_L": peak_L, "cum_loan_taken": cum_loan_taken,
        "n_relever_events": n_relever_events, "idxs": idxs,
    }


def irr_with_monthly_dca(terminal, H, annual_dca):
    if terminal <= 0:
        return np.nan
    m = annual_dca / 12.0
    M = int(H * 12)
    def f(r):
        if abs(r) < 1e-10:
            return 1 + m * M - terminal
        d = (1 + r) ** (1/12)
        return (1+r)**H + m * (d**M - 1) / (d - 1) - terminal
    try:
        return brentq(f, -0.5, 1.5, xtol=1e-7)
    except ValueError:
        return np.nan


targets = [1.30, 1.41, 1.50, 1.60, 1.75, 2.00]
dcas = [0.00, 0.05, 0.10, 0.20]
H_YEARS = 30

print("=" * 115)
print(f"DYNAMIC RE-LEVER (monthly, lever-up-only) vs STATIC (one-time loan, no rebalance)")
print(f"Horizon {H_YEARS}y, post-1932, box-spread financing, target = starting leverage")
print("=" * 115)

for dca in dcas:
    print(f"\n--- DCA = {dca*100:.0f}%/yr ---")
    print(f"{'Target':>7}  {'Strategy':>10}  {'call %':>7}  "
          f"{'med term':>9}  {'med IRR':>8}  {'p10 IRR':>8}  "
          f"{'total loan':>11}  {'ΔIRR (relev−static)':>20}  "
          f"{'peak L p95':>11}")

    for T in targets:
        # Static
        rs = simulate(T, dca, H_YEARS, relever=False)
        med_t_s = np.median(rs["terminal"][rs["terminal"] > 0]) if (rs["terminal"] > 0).any() else np.nan
        irr_s_med = irr_with_monthly_dca(med_t_s, H_YEARS, dca) * 100 if not np.isnan(med_t_s) else np.nan
        p10_t_s = np.percentile(rs["terminal"], 10)
        irr_s_p10 = irr_with_monthly_dca(p10_t_s, H_YEARS, dca) * 100 if p10_t_s > 0 else np.nan

        # Dynamic
        rd = simulate(T, dca, H_YEARS, relever=True)
        med_t_d = np.median(rd["terminal"][rd["terminal"] > 0]) if (rd["terminal"] > 0).any() else np.nan
        irr_d_med = irr_with_monthly_dca(med_t_d, H_YEARS, dca) * 100 if not np.isnan(med_t_d) else np.nan
        p10_t_d = np.percentile(rd["terminal"], 10)
        irr_d_p10 = irr_with_monthly_dca(p10_t_d, H_YEARS, dca) * 100 if p10_t_d > 0 else np.nan

        mean_total_loan = rd["cum_loan_taken"].mean()

        delta_irr = irr_d_med - irr_s_med if not (np.isnan(irr_d_med) or np.isnan(irr_s_med)) else np.nan

        # Print static
        print(f"{T:>6.2f}x  {'static':>10}  "
              f"{rs['called'].mean()*100:>6.2f}%  "
              f"{med_t_s:>8.2f}x  "
              f"{irr_s_med:>7.2f}%  "
              f"{irr_s_p10:>7.2f}%  "
              f"{T-1:>10.2f}   "
              f"{'':>19}  "
              f"{np.percentile(rs['peak_L'], 95):>10.2f}x")
        # Print dynamic
        print(f"{T:>6.2f}x  {'relever':>10}  "
              f"{rd['called'].mean()*100:>6.2f}%  "
              f"{med_t_d:>8.2f}x  "
              f"{irr_d_med:>7.2f}%  "
              f"{irr_d_p10:>7.2f}%  "
              f"{mean_total_loan:>10.2f}   "
              f"{delta_irr:>+18.2f}pp  "
              f"{np.percentile(rd['peak_L'], 95):>10.2f}x")

# 2000-03-23 stress test
print("\n" + "=" * 115)
print("Stress test: 2000-03-23 entry, 23y horizon")
print("=" * 115)
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
H_2000 = len(px) - i0 - 1

def simulate_single(i, target, dca, H, relever):
    monthly = dca / 12
    spx = target
    loan = target - 1
    called = False
    peak = target
    tot_loan = loan
    n_rel = 0
    for k in range(1, H + 1):
        spx *= px[i+k]/px[i+k-1]
        loan *= M_box[i+k]/M_box[i+k-1]
        if k % DAYS_PER_MONTH == 0:
            spx += monthly
            if relever:
                eq = spx - loan
                if eq > 0:
                    cur_lev = spx / eq
                    if cur_lev < target:
                        dD = target * eq - spx
                        if dD > 0:
                            loan += dD
                            spx += dD
                            tot_loan += dD
                            n_rel += 1
        eq = spx - loan
        if eq <= 0 or spx/max(eq, 1e-12) >= 4.0:
            called = True
            break
        peak = max(peak, spx/eq)
    term = 0 if called else spx - loan
    return term, peak, called, tot_loan, n_rel

for dca in [0.00, 0.10, 0.20]:
    print(f"\n-- DCA = {dca*100:.0f}%/yr --")
    print(f"{'Target':>7}  {'Strategy':>10}  {'terminal':>10}  "
          f"{'peak L':>7}  {'total loan':>11}  {'#relever':>9}")
    for T in targets:
        ts, ps, cs, ls, ns = simulate_single(i0, T, dca, H_2000, False)
        td, pd, cd, ld, nd = simulate_single(i0, T, dca, H_2000, True)
        print(f"{T:>6.2f}x  {'static':>10}  "
              f"{ts:>9.2f}x  {ps:>6.2f}x  {ls:>10.2f}   —")
        print(f"{T:>6.2f}x  {'relever':>10}  "
              f"{td:>9.2f}x  {pd:>6.2f}x  {ld:>10.2f}   {nd:>8d}")
