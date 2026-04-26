"""
Two-dimensional study:
  (1) Higher leverage vs ~1.30x baseline: 1.00, 1.30, 1.35, 1.41, 1.50, 1.75, 2.00
  (2) Broker margin (3M Tsy + 40bps) vs Box-spread financing (3M Tsy + 15bps)

Strategy per entry date: take one-time loan on day 1 at chosen L0,
buy SPX, hold to horizon. Loan interest compounds into loan.
If peak leverage hits 4.0x at any point => forced liquidation:
  At call moment, equity = A/4. Remaining equity compounds in cash
  (3M Tsy) until end of horizon.
If equity goes negative => bust, terminal = 0.
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()

TRADING_DAYS = 252

# Broker rate (as in dataset): 3M Tsy + 40 bps
# Box-spread rate: 3M Tsy + 15 bps (conservative assumption; live quotes
#   on liquid 1-2y SPX boxes typically land in 3M Tsy +5..+25 bps)
# Cash rate (post-call): 3M Tsy (no spread)
broker_rate = mrate
box_rate = tsy + 0.0015
cash_rate = tsy  # where money sits after a margin call

def cum_factor(rate_annual):
    df = 1.0 + rate_annual / TRADING_DAYS
    return np.concatenate(([1.0], np.cumprod(df[1:])))

M_broker = cum_factor(broker_rate)
M_box    = cum_factor(box_rate)
M_cash   = cum_factor(cash_rate)

post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate(L0, M_loan, horizon_days):
    """For each post-1932 entry i with at least horizon_days of future
    data, return (terminal_levered, terminal_unlev, called, peak_L)."""
    entries = np.where(post1932)[0]
    entries = entries[entries + horizon_days < len(px)]

    out_lev = np.empty(len(entries))
    out_unl = np.empty(len(entries))
    called = np.zeros(len(entries), dtype=bool)
    peak_L = np.empty(len(entries))

    if L0 <= 1.0:
        # No leverage — skip call machinery
        for k, i in enumerate(entries):
            end = i + horizon_days
            out_unl[k] = px[end] / px[i]
            out_lev[k] = out_unl[k]
            peak_L[k] = 1.0
        return entries, out_unl, out_lev, called, peak_L

    # Call threshold: leverage hits 4.0x when R_t/R_i = 4(L0-1)/(3 L0).
    # (Derivation: L_t = L0*(R_t/R_i) / (L0*(R_t/R_i) - (L0-1)); set L_t=4.)
    factor = 4.0 * (L0 - 1.0) / (3.0 * L0)
    R_loan = px / M_loan

    for k, i in enumerate(entries):
        end = i + horizon_days
        R_slice = R_loan[i:end + 1]
        R_i = R_slice[0]
        unlev = px[end] / px[i]
        out_unl[k] = unlev

        call_thresh = R_i * factor
        hits = np.where(R_slice <= call_thresh)[0]
        if len(hits) == 0:
            # No call — closed-form terminal
            A_T = L0 * px[end] / px[i]
            D_T = (L0 - 1.0) * M_loan[end] / M_loan[i]
            E_T = A_T - D_T
            out_lev[k] = max(E_T, 0.0)
            R_min = R_slice.min()
            denom = L0 - (L0 - 1.0) * R_i / R_min
            peak_L[k] = L0 / denom if denom > 0 else np.inf
            if denom <= 0:
                # Bust without hitting the "exactly 4.0x" call threshold
                # (shouldn't happen because 4.0 is crossed before infinity,
                # but guard anyway)
                out_lev[k] = 0.0
                called[k] = True
        else:
            # Called at first call day
            call_idx = i + int(hits[0])
            A_call = L0 * px[call_idx] / px[i]
            E_call = A_call / 4.0  # by definition of L=4 at call
            # Compound remaining equity in cash (3M Tsy) to horizon end
            out_lev[k] = E_call * M_cash[end] / M_cash[call_idx]
            called[k] = True
            peak_L[k] = 4.0

    return entries, out_unl, out_lev, called, peak_L


def summarize(L0, M_loan, rate_label, horizon_years):
    horizon_days = int(horizon_years * TRADING_DAYS)
    _, unl, lev, called, peakL = simulate(L0, M_loan, horizon_days)
    call_rate = called.mean() * 100
    # Uplift vs unlevered
    ratio = np.where(unl > 0, lev / unl, 1.0)
    # CAGR
    cagr_unl = unl ** (1.0 / horizon_years) - 1.0
    cagr_lev = np.where(lev > 0, lev ** (1.0 / horizon_years) - 1.0, -1.0)
    cagr_diff = (cagr_lev - cagr_unl) * 100  # percentage points per year
    win = (lev > unl).mean() * 100
    return {
        "L0": L0, "rate": rate_label, "H": horizon_years,
        "N": len(unl),
        "mean_mult": ratio.mean(),
        "median_mult": np.median(ratio),
        "p10_mult": np.percentile(ratio, 10),
        "p90_mult": np.percentile(ratio, 90),
        "worst_mult": ratio.min(),
        "mean_cagr_diff": cagr_diff.mean(),
        "p10_cagr_diff": np.percentile(cagr_diff, 10),
        "p90_cagr_diff": np.percentile(cagr_diff, 90),
        "call_rate": call_rate,
        "win_rate": win,
    }


leverages = [1.00, 1.30, 1.35, 1.41, 1.50, 1.75, 2.00]
rates = [("broker 3M+40", M_broker), ("box    3M+15", M_box)]
horizons = [20, 30]

for h in horizons:
    print("\n" + "=" * 90)
    print(f"HORIZON = {h} YEARS, post-1932 entries, one-time loan held to horizon")
    print("=" * 90)
    for rate_lbl, M in rates:
        print(f"\n--- financing: {rate_lbl} ---")
        print(f"{'L0':>5}  {'mean mult':>9}  {'p10 mult':>9}  "
              f"{'p90 mult':>9}  {'worst':>7}  "
              f"{'ΔCAGR mean':>11}  {'ΔCAGR p10':>10}  "
              f"{'calls':>6}  {'beat B&H':>8}")
        for L0 in leverages:
            s = summarize(L0, M, rate_lbl, h)
            print(f"{L0:>4.2f}x  "
                  f"{s['mean_mult']:>8.3f}x  "
                  f"{s['p10_mult']:>8.3f}x  "
                  f"{s['p90_mult']:>8.3f}x  "
                  f"{s['worst_mult']:>6.3f}x  "
                  f"{s['mean_cagr_diff']:>+10.2f}pp  "
                  f"{s['p10_cagr_diff']:>+9.2f}pp  "
                  f"{s['call_rate']:>5.2f}%  "
                  f"{s['win_rate']:>7.1f}%")

# Explicit broker-vs-box comparison at 1.30x and 1.41x, 30y
print("\n" + "=" * 90)
print("BROKER vs BOX at specific leverages (30-year horizon)")
print("=" * 90)
for L0 in [1.30, 1.41, 1.50, 1.75, 2.00]:
    sb = summarize(L0, M_broker, "broker", 30)
    sx = summarize(L0, M_box, "box", 30)
    print(f"\nL0 = {L0:.2f}x")
    print(f"  broker: mean terminal {sb['mean_mult']:.3f}x, "
          f"ΔCAGR mean {sb['mean_cagr_diff']:+.2f}pp, "
          f"worst {sb['worst_mult']:.3f}x, "
          f"calls {sb['call_rate']:.2f}%")
    print(f"  box:    mean terminal {sx['mean_mult']:.3f}x, "
          f"ΔCAGR mean {sx['mean_cagr_diff']:+.2f}pp, "
          f"worst {sx['worst_mult']:.3f}x, "
          f"calls {sx['call_rate']:.2f}%")
    print(f"  box - broker: +{(sx['mean_mult']-sb['mean_mult'])*100:.2f}% "
          f"terminal, +{sx['mean_cagr_diff']-sb['mean_cagr_diff']:.2f}pp CAGR")
