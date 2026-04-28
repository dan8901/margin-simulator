"""
verify_etf_sim.py — calibrate and validate the daily-leveraged-ETF model
against real SSO and UPRO total returns.

Model:
  r_etf_t = N * r_spx_t - (expense_ratio + (N-1) * financing_rate) / 252

We fetch real SSO/UPRO TR (yfinance ^SSO and ^UPRO with auto_adjust=True
to get total return), pair with SPX TR + Tsy_3M from our CSV, and search
for the financing spread that minimizes annualized-return error.

UPRO inception: 2009-06-25
SSO inception:  2006-06-21

Run with: .venv/bin/python verify_etf_sim.py
"""

import numpy as np
import yfinance as yf

from data_loader import load


EXPENSE_RATIO = 0.0091   # 0.91%/yr, per ProShares
TD = 252


def fetch_etf_tr(ticker, start_date):
    """Fetch daily total-return series for an ETF (auto_adjust=True
    folds dividends into the price). Returns dict[date -> close]."""
    df = yf.download(ticker, start=start_date, progress=False,
                     auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    out = {}
    for ts, row in df.iterrows():
        d = ts.date()
        close = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc")
                      else row["Close"])
        out[d] = close
    return out


def align_series(dates, px, tsy, etf_dict):
    """Keep only days where both SPX and ETF have data. Returns:
    (idxs into the SPX arrays, etf prices on those days)."""
    keep_idx = []
    etf_px = []
    for i, d in enumerate(dates):
        d_only = d.date() if hasattr(d, "date") else d
        if d_only in etf_dict:
            keep_idx.append(i)
            etf_px.append(etf_dict[d_only])
    return np.array(keep_idx), np.array(etf_px)


def simulate_etf(spx_ret, tsy_ann, N, expense_ratio, fin_spread_bps):
    """Apply the daily-reset ETF formula to SPX returns + Tsy yields.
    fin_spread_bps in bps over Tsy_3m. Returns simulated daily ETF returns."""
    fin_rate = tsy_ann + fin_spread_bps / 10000.0
    daily_drag = (expense_ratio + (N - 1) * fin_rate) / TD
    return N * spx_ret - daily_drag


def cumulative(ret):
    """Compound a daily return series into a cumulative-growth path
    starting at 1.0."""
    return np.cumprod(1.0 + ret)


def annualized_return(cum, n_days):
    return cum[-1] ** (TD / n_days) - 1.0


def annualized_vol(daily_ret):
    return float(np.std(daily_ret) * np.sqrt(TD))


def fit_spread(spx_ret, tsy_ann, etf_real_ret, N):
    """Grid-search the financing spread that minimizes |annualized-return
    error| between simulated and real ETF."""
    cum_real = cumulative(etf_real_ret)
    n_days = len(etf_real_ret)
    real_cagr = annualized_return(cum_real, n_days)

    best_spread = None
    best_err = float("inf")
    for spread_bps in np.arange(-100, 200, 1.0):
        sim_ret = simulate_etf(spx_ret, tsy_ann, N, EXPENSE_RATIO, spread_bps)
        cum_sim = cumulative(sim_ret)
        sim_cagr = annualized_return(cum_sim, n_days)
        err = abs(sim_cagr - real_cagr)
        if err < best_err:
            best_err = err
            best_spread = spread_bps
    return best_spread, real_cagr


def report_etf(name, ticker, N, dates, px, tsy):
    print(f"\n{'='*70}")
    print(f"{name} ({ticker}) — N={N}x daily")
    print(f"{'='*70}")
    inception = "2009-06-25" if ticker == "UPRO" else "2006-06-21"
    print(f"Fetching {ticker} from yfinance (since {inception})...")
    etf_dict = fetch_etf_tr(ticker, inception)
    print(f"  {len(etf_dict)} daily closes")

    keep_idx, etf_px = align_series(dates, px, tsy, etf_dict)
    if len(keep_idx) < 50:
        raise RuntimeError(f"Not enough overlap days: {len(keep_idx)}")

    # Compute returns aligned on the kept dates.
    spx_aligned = px[keep_idx]
    tsy_aligned = tsy[keep_idx]

    # Daily returns: r_t = px[t]/px[t-1] - 1.  Drop t=0 of each series.
    spx_ret = spx_aligned[1:] / spx_aligned[:-1] - 1.0
    etf_real_ret = etf_px[1:] / etf_px[:-1] - 1.0
    tsy_for_drag = tsy_aligned[1:]

    print(f"  Overlap: {len(spx_ret)} daily returns "
          f"({dates[keep_idx[0]].date()} to {dates[keep_idx[-1]].date()})")

    # Best-fit financing spread.
    best_spread, real_cagr = fit_spread(spx_ret, tsy_for_drag,
                                        etf_real_ret, N)
    print(f"\nBest-fit financing spread: {best_spread:.0f} bps over Tsy_3m")
    print(f"  (assumes expense_ratio = {EXPENSE_RATIO*100:.2f}%/yr)")

    # Compare at the best-fit spread + a few standard candidates.
    cum_real = cumulative(etf_real_ret)
    n_days = len(etf_real_ret)
    real_vol = annualized_vol(etf_real_ret)

    print(f"\n  {'spread':>10} | {'sim CAGR':>10} | {'real CAGR':>10} "
          f"| {'CAGR err':>10} | {'sim vol':>10} | {'real vol':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    candidates = sorted(set([0.0, 15.0, 30.0, 50.0, 100.0, best_spread]))
    for spread_bps in candidates:
        sim_ret = simulate_etf(spx_ret, tsy_for_drag, N,
                               EXPENSE_RATIO, spread_bps)
        cum_sim = cumulative(sim_ret)
        sim_cagr = annualized_return(cum_sim, n_days)
        sim_vol = annualized_vol(sim_ret)
        err = sim_cagr - real_cagr
        marker = " <-- best" if abs(spread_bps - best_spread) < 0.5 else ""
        print(f"  {spread_bps:>10.0f} | {sim_cagr*100:>9.2f}% "
              f"| {real_cagr*100:>9.2f}% | {err*100:>+9.2f}pp "
              f"| {sim_vol*100:>9.2f}% | {real_vol*100:>9.2f}%{marker}")

    # Path-level fidelity at best spread.
    sim_best = simulate_etf(spx_ret, tsy_for_drag, N,
                            EXPENSE_RATIO, best_spread)
    cum_sim = cumulative(sim_best)
    final_err = (cum_sim[-1] / cum_real[-1] - 1.0) * 100
    daily_diff = sim_best - etf_real_ret
    daily_rmse = float(np.sqrt(np.mean(daily_diff ** 2))) * 100
    print(f"\nAt best spread:")
    print(f"  Final cumulative ratio (sim/real - 1): {final_err:+.2f}%")
    print(f"  Daily return RMSE (sim - real):        {daily_rmse:.4f}%")
    print(f"  Sim ending wealth:  ${cum_sim[-1]:.2f} per $1 start")
    print(f"  Real ending wealth: ${cum_real[-1]:.2f} per $1 start")

    return best_spread


def main():
    print("Loading SPX TR + Tsy_3M from spx_margin_history.csv...")
    dates, px, tsy, _ = load()
    print(f"  {len(dates)} rows, {dates[0].date()} to {dates[-1].date()}")

    spreads = {}
    spreads["SSO"] = report_etf("SSO (2x SPX, daily reset)", "SSO", 2.0,
                                 dates, px, tsy)
    spreads["UPRO"] = report_etf("UPRO (3x SPX, daily reset)", "UPRO", 3.0,
                                  dates, px, tsy)

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for name, sp in spreads.items():
        print(f"  {name}: best-fit financing spread = {sp:.0f} bps")
    print(f"\n  Recommended ETF_FIN_BPS for the simulator: "
          f"{int(round(np.mean(list(spreads.values())))):d} bps")


if __name__ == "__main__":
    main()
