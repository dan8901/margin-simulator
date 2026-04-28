"""
analyze_mixed_collateral.py — unified two-sleeve simulator that respects
shared margin collateral.

Key difference vs analyze_mixed_hybrid_sso.py: the margin-call check uses
the COMBINED collateral pool from both sleeves:

  total equity ≥ 0.25 · stocks_SPX + 0.50 · stocks_SSO
  ⇒  loan_h ≤ 0.75 · stocks_h + 0.50 · stocks_sso

This means the hybrid sleeve can safely run at HIGHER T_init in a mixed
account because the SSO sleeve provides ~2/3 of equivalent SPX collateral
(50% maintenance vs 25%). For each h_frac, we re-calibrate T_init via
binary search over a well-defended target (boot call rate ≤ 1% + hist 0%).

The catch (still modelled correctly): SSO collateral SHRINKS faster than
SPX during drawdowns (50% SPX drop ≈ 75% SSO drop via daily reset), so
the day-0 collateral capacity is partly illusory. The simulator captures
this naturally — call check uses current stocks_sso each day.

Hybrid sleeve uses hybrid logic (dd_decay ratchet ∧ wealth_decay glide)
applied to its own sleeve equity. Wealth glide is on TOTAL real wealth.

Run with: .venv/bin/python -u analyze_mixed_collateral.py
"""

import numpy as np
import numba

from data_loader import load
from project_portfolio import (build_historical_paths, build_bootstrap_paths,
                                stretch_returns, simulate, find_max_safe_T_grid,
                                TD, BOX_BPS, BOX_TAX_BENEFIT, BROKER_BPS,
                                ETF_EXPENSE_RATIO, ETF_FIN_BPS, REBAL_DAYS,
                                CALL_THRESHOLD)


F_DEFAULT = 1.5
STRETCH = 1.1
BOOT_TARGET = 0.01
N_BOOT_PATHS = 500
BLOCK_DAYS = 252


@numba.njit(cache=True, fastmath=True)
def simulate_mixed(ret, tsy, cpi, T_init, h_frac, C, S, T_yrs, S2, max_days,
                    avail, F, floor, wealth_X, broker_bump_days):
    """Two-sleeve simulation with shared collateral.

    Hybrid sleeve: stocks_h (SPX), loan_h (margin), grows at SPX returns,
                   loan compounds at box rate (or broker rate during bump).
                   Hybrid logic (dd_decay min wealth_decay) on sleeve equity.
    SSO sleeve:    stocks_sso, daily-reset 2x SPX, no loan.
    Call check:    loan_h > 0.75·stocks_h + 0.50·stocks_sso  → called.
    DCA:           split h_frac to hybrid, (1-h_frac) to SSO.
    Wealth glide:  uses TOTAL real wealth toward wealth_X.

    Returns: (total_real_eq[K, max_days+1], called[K], peak_lev_h[K])
    """
    K = ret.shape[0]
    stocks_h = np.full(K, h_frac * C * T_init)
    loan_h = np.full(K, h_frac * C * (T_init - 1.0))
    stocks_sso = np.full(K, (1.0 - h_frac) * C)
    hwm_eq = np.full(K, h_frac * C)
    max_dd = np.zeros(K)
    called = np.zeros(K, dtype=np.bool_)
    cap_reached = np.zeros(K, dtype=np.bool_)
    peak_lev = np.full(K, T_init)

    real_eq = np.full((K, max_days + 1), np.nan)
    for k in range(K):
        real_eq[k, 0] = C

    t_switch_days = T_yrs * TD

    for d in range(1, max_days + 1):
        s_real_yr = S if d < t_switch_days else S2
        do_rebal = (d % REBAL_DAYS == 0)

        for k in range(K):
            if called[k] or d > avail[k]:
                continue

            # Hybrid: SPX growth + loan compound
            stocks_h_new = stocks_h[k] * (1.0 + ret[k, d])
            if d <= broker_bump_days:
                rate = tsy[k, d] + BROKER_BPS
            else:
                rate = (tsy[k, d] + BOX_BPS) * (1.0 - BOX_TAX_BENEFIT)
            loan_h_new = loan_h[k] * (1.0 + rate / TD)

            # SSO: daily-reset 2x
            fin_rate = tsy[k, d] + ETF_FIN_BPS
            drift_sso = 2.0 * ret[k, d] - (ETF_EXPENSE_RATIO + fin_rate) / TD
            if drift_sso <= -1.0:
                stocks_sso_new = 0.0
            else:
                stocks_sso_new = stocks_sso[k] * (1.0 + drift_sso)

            # DCA: split into both sleeves
            if s_real_yr > 0.0:
                infl = cpi[k, d] / cpi[k, 0]
                stocks_h_new += s_real_yr * h_frac * infl / TD
                stocks_sso_new += s_real_yr * (1.0 - h_frac) * infl / TD

            # Combined-collateral margin call check
            max_loan = 0.75 * stocks_h_new + 0.50 * stocks_sso_new
            if loan_h_new > max_loan:
                called[k] = True
                continue
            total_eq = stocks_h_new + stocks_sso_new - loan_h_new
            if total_eq <= 0.0:
                called[k] = True
                continue

            stocks_h[k] = stocks_h_new
            loan_h[k] = loan_h_new
            stocks_sso[k] = stocks_sso_new

            total_real_eq = total_eq * cpi[k, 0] / cpi[k, d]

            # Cap on TOTAL wealth (not sleeve)
            if total_real_eq >= 1e18:   # cap disabled
                pass

            # Track peak hybrid-sleeve leverage
            if stocks_h[k] - loan_h[k] > 0:
                lev_h = stocks_h[k] / (stocks_h[k] - loan_h[k])
                if lev_h > peak_lev[k]:
                    peak_lev[k] = lev_h

            # Rebalance (hybrid sleeve only, hybrid logic)
            if do_rebal and not cap_reached[k]:
                eq_h = stocks_h[k] - loan_h[k]
                if eq_h > 0.0:
                    # dd_decay component on hybrid sleeve equity
                    if eq_h > hwm_eq[k]:
                        hwm_eq[k] = eq_h
                    if hwm_eq[k] > 0.0:
                        dd_now = 1.0 - eq_h / hwm_eq[k]
                        if dd_now > max_dd[k]:
                            max_dd[k] = dd_now
                    cand_dd = T_init - F * max_dd[k]
                    if cand_dd < floor:
                        cand_dd = floor

                    # wealth_decay component on TOTAL real wealth
                    if wealth_X > C:
                        prog = (total_real_eq - C) / (wealth_X - C)
                        if prog < 0.0:
                            prog = 0.0
                        elif prog > 1.0:
                            prog = 1.0
                        cand_w = T_init - (T_init - floor) * prog
                    else:
                        cand_w = floor

                    target_lev = cand_dd if cand_dd < cand_w else cand_w

                    # Take additional loan only if it doesn't violate combined collateral
                    delta = target_lev * eq_h - stocks_h[k]
                    if delta > 0.0:
                        new_loan = loan_h[k] + delta
                        new_max = 0.75 * (stocks_h[k] + delta) + 0.50 * stocks_sso[k]
                        if new_loan <= new_max:
                            stocks_h[k] += delta
                            loan_h[k] += delta

            real_eq[k, d] = total_real_eq

    return real_eq, called, peak_lev


def find_safe_T_mixed(ret, tsy, cpi, h_frac, C, S, T_yrs, S2, max_days,
                       avail, F, floor, wealth_X, broker_bump_days,
                       target_call_rate, lo=1.0, hi=4.0, n_iters=14):
    """Binary search for largest T_init with call_rate ≤ target."""
    best = lo
    for _ in range(n_iters):
        mid = (lo + hi) / 2.0
        _, called, _ = simulate_mixed(
            ret, tsy, cpi, mid, h_frac, C, S, T_yrs, S2, max_days,
            avail, F, floor, wealth_X, broker_bump_days)
        cr = called.mean()
        if cr <= target_call_rate:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


SCENARIOS = [
    dict(name="A: heavy-DCA", C=160_000, S=180_000, T_yrs=5, S2=30_000,
         wealth_X=3_000_000),
    dict(name="B: light-DCA", C=1_000_000, S=30_000, T_yrs=30, S2=30_000,
         wealth_X=10_000_000),
    dict(name="C: lump-sum",  C=1_000_000, S=0, T_yrs=30, S2=0,
         wealth_X=float("inf")),
]
HORIZONS_YR = [20, 30]
ALLOCS = [0.25, 0.50, 0.75]   # only mixed; pure cases done in earlier sweep
PCTILES = [1, 5, 10, 50, 90]


def percentiles_at(real_eq, avail, h_days):
    eligible = avail >= h_days
    vals = real_eq[eligible, h_days]
    valid = ~np.isnan(vals)
    if not valid.any():
        return [float("nan")] * len(PCTILES)
    return [float(x) for x in np.nanpercentile(vals[valid], PCTILES)]


def boot_metrics(real_eq, h_days, C):
    vals = real_eq[:, h_days]
    valid = ~np.isnan(vals)
    if not valid.any():
        return float("nan"), float("nan"), [float("nan")] * len(PCTILES)
    v = vals[valid]
    pct = [float(x) for x in np.nanpercentile(v, PCTILES)]
    return float(v.min()), float((v < C).mean()), pct


def main():
    print("Loading + building paths...")
    dates, px, tsy, mrate, cpi = load(with_cpi=True)
    max_days = 30 * TD
    ret_h, tsy_h, cpi_h, avail_h, _ = build_historical_paths(
        dates, px, tsy, cpi, max_days, min_days=TD)
    rng = np.random.default_rng(42)
    ret_b, tsy_b, cpi_b = build_bootstrap_paths(
        dates, px, tsy, cpi, max_days, N_BOOT_PATHS, BLOCK_DAYS, rng)
    avail_b = np.full(ret_b.shape[0], max_days, dtype=np.int64)
    print(f"  hist paths: {ret_h.shape[0]}, boot paths: {ret_b.shape[0]}")

    for scen in SCENARIOS:
        wxt = scen["wealth_X"]
        wxt_str = f"${wxt/1e6:.1f}M" if wxt != float("inf") else "none"
        print()
        print("=" * 110)
        print(f"SCENARIO {scen['name']}: C=${scen['C']/1e3:.0f}k  "
              f"S=${scen['S']/1e3:.0f}k×{scen['T_yrs']}y  S2=${scen['S2']/1e3:.0f}k  "
              f"wealth_X={wxt_str}")
        print("=" * 110)

        # Reference: standalone hybrid T_rec (well-defended)
        T_h_hist = find_max_safe_T_grid(ret_h, tsy_h, cpi_h, "hybrid", 0.0,
                                         scen["C"], scen["S"], scen["T_yrs"],
                                         scen["S2"], max_days, avail=avail_h,
                                         F=F_DEFAULT, wealth_X=wxt)
        T_h_boot = find_max_safe_T_grid(ret_b, tsy_b, cpi_b, "hybrid", BOOT_TARGET,
                                         scen["C"], scen["S"], scen["T_yrs"],
                                         scen["S2"], max_days, F=F_DEFAULT,
                                         wealth_X=wxt)
        T_hyb_solo = float(min(T_h_hist, T_h_boot))
        print(f"Reference standalone hybrid T_rec: {T_hyb_solo:.3f}x")

        for h_frac in ALLOCS:
            # Re-calibrate T_init for the mixed-collateral system at this h_frac
            T_h_b = find_safe_T_mixed(
                ret_b, tsy_b, cpi_b, h_frac, scen["C"], scen["S"],
                scen["T_yrs"], scen["S2"], max_days, avail_b, F_DEFAULT, 1.0,
                wxt, 0, BOOT_TARGET)
            T_h_h = find_safe_T_mixed(
                ret_h, tsy_h, cpi_h, h_frac, scen["C"], scen["S"],
                scen["T_yrs"], scen["S2"], max_days, avail_h, F_DEFAULT, 1.0,
                wxt, 0, 0.0)
            T_mix = min(T_h_b, T_h_h)
            print(f"  h_frac={h_frac:.2f}: mixed T_init = {T_mix:.3f}x  "
                  f"(boot={T_h_b:.3f}, hist={T_h_h:.3f})  "
                  f"vs solo hybrid {T_hyb_solo:.3f}x  "
                  f"[+{(T_mix-T_hyb_solo)/T_hyb_solo*100:.1f}%]")

            # Project at well-defended T
            real_eq_h, called_h, peak_h = simulate_mixed(
                ret_h, tsy_h, cpi_h, T_mix, h_frac,
                scen["C"], scen["S"], scen["T_yrs"], scen["S2"], max_days,
                avail_h, F_DEFAULT, 1.0, wxt, 0)
            real_eq_b, called_b, _ = simulate_mixed(
                ret_b, tsy_b, cpi_b, T_mix, h_frac,
                scen["C"], scen["S"], scen["T_yrs"], scen["S2"], max_days,
                avail_b, F_DEFAULT, 1.0, wxt, 0)

            # Mark called paths' wealth as 0 (consistent with margin-call wipeout)
            real_eq_h_filled = real_eq_h.copy()
            real_eq_b_filled = real_eq_b.copy()
            for k in range(real_eq_h.shape[0]):
                if called_h[k]:
                    real_eq_h_filled[k, :avail_h[k] + 1] = np.where(
                        np.isnan(real_eq_h_filled[k, :avail_h[k] + 1]),
                        0.0, real_eq_h_filled[k, :avail_h[k] + 1])
            for k in range(real_eq_b.shape[0]):
                if called_b[k]:
                    real_eq_b_filled[k, :] = np.where(
                        np.isnan(real_eq_b_filled[k, :]),
                        0.0, real_eq_b_filled[k, :])

            for h_yr in HORIZONS_YR:
                h_days = h_yr * TD
                ph = percentiles_at(real_eq_h_filled, avail_h, h_days)
                min_b, fbc, pb = boot_metrics(real_eq_b_filled, h_days, scen["C"])
                p1b, p5b, p10b, p50b, p90b = pb
                print(f"    {h_yr}y  hist_call={called_h.mean()*100:.2f}%  "
                      f"boot_call={called_b.mean()*100:.2f}%  "
                      f"min_b=${min_b/1e6:.2f}M  <C={fbc*100:.1f}%  "
                      f"boot p1=${p1b/1e6:.2f}M p10=${p10b/1e6:.2f}M "
                      f"p50=${p50b/1e6:.2f}M p90=${p90b/1e6:.2f}M  "
                      f"peak_lev_h_p99={np.percentile(peak_h[~called_h], 99):.2f}x")


if __name__ == "__main__":
    main()
