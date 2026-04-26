"""
"Buy the dip" with margin — strategy evaluation.

Rule: starting on day i with $1 in SPX (100% unlevered), watch running
peak. The FIRST day SPX total-return is `threshold` below its peak
(peak measured from day i onward), take a one-time margin loan equal to
`loan_frac` of then-current equity and buy more SPX. Hold to the end of
a fixed horizon. Margin interest compounds into the loan; loan is never
paid down.

For each (threshold, horizon) we compute the distribution across all
post-1932 entry dates with at least `horizon` years of future data.

Closed-form post-trigger math:
  At trigger t, equity = px[t]/px[i], loan = loan_frac * equity,
  assets = (1+loan_frac) * equity.
  At horizon end T:
    A_T = (1+loan_frac) * px[T]/px[i]
    D_T = loan_frac * px[t]/px[i] * M[T]/M[t]
    strategy_equity = A_T - D_T
  Max leverage between t and T:
    L_peak = (1+lf) / ((1+lf) - lf * max_{s>=t}(M[s]/px[s]) / (M[t]/px[t]))
        (= (1+lf) / ((1+lf) - lf * R_t / min_{s in [t,T]} R_s))
    where R_s = px[s]/M[s]
  Bust if (1+lf) - lf * R_t/R_min <= 0, i.e. R_min <= R_t * lf/(1+lf).
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, _tsy, mrate = load()

daily_factor = 1.0 + mrate / 252.0
M = np.concatenate(([1.0], np.cumprod(daily_factor[1:])))
R = px / M

# Running reverse-min of R: min_R_after[i] = min over s>=i of R[s]
min_R_after = np.minimum.accumulate(R[::-1])[::-1]

# Restrict to post-1932 entries
post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])

TRADING_DAYS = 252

def find_trigger_day(i, end, threshold):
    """First t in [i, end] where px[t]/max(px[i:t+1]) <= 1-threshold.
    Returns index into the global arrays, or None."""
    seg = px[i:end+1]
    running_peak = np.maximum.accumulate(seg)
    dd = 1 - seg / running_peak
    hits = np.where(dd >= threshold)[0]
    if len(hits) == 0:
        return None
    return i + int(hits[0])


def simulate(threshold, horizon_years, loan_frac=0.30, cap=4.0, cushion=4.0):
    """Simulate strategy across all post-1932 entry dates with at least
    horizon_years of future data. `cap` is the hard margin-call threshold
    (4.0x Reg-T). `cushion` (default same as cap) is informational — we
    report fraction of paths that would've gotten called.

    Returns dict with the stats.
    """
    H = int(horizon_years * TRADING_DAYS)
    entries = np.where(post1932)[0]
    entries = entries[entries + H < len(px)]

    results = []
    for i in entries:
        end = i + H
        t = find_trigger_day(i, end, threshold)
        buyhold = px[end] / px[i]
        if t is None:
            # never triggered
            results.append((i, None, buyhold, buyhold, 1.0 + loan_frac, False))
            continue
        # Post-trigger closed form
        eq_at_trigger = px[t] / px[i]
        A_T = (1 + loan_frac) * px[end] / px[i]
        D_T = loan_frac * eq_at_trigger * M[end] / M[t]
        strat_eq = A_T - D_T
        # Peak leverage between t and end
        R_min = float(np.min(R[t:end+1]))
        # if bust in between: L = inf
        denom = (1 + loan_frac) - loan_frac * R[t] / R_min
        if denom <= 0:
            peak_L = np.inf
            bust = True
            strat_eq = -np.inf  # treat as wipeout
        else:
            peak_L = (1 + loan_frac) / denom
            bust = False
        results.append((i, t, buyhold, strat_eq, peak_L, bust))

    return results


def summarize(results, threshold, horizon_years, loan_frac):
    n = len(results)
    trig_mask = np.array([r[1] is not None for r in results])
    n_trig = int(trig_mask.sum())
    peak_Ls = np.array([r[4] for r in results])
    busts = np.array([r[5] for r in results])
    calls = peak_Ls > 4.0
    buyhold = np.array([r[2] for r in results])
    strat = np.array([r[3] for r in results])

    # For paths that would've been called, cap the outcome at the call
    # day (liquidation at 4.0x means equity = 25% of assets at that moment,
    # which we approximate as 0 — i.e., treat called paths as "bad"
    # without modelling broker liquidation in detail).
    # For simplicity: replace called/busted paths' equity with zero.
    strat_adj = strat.copy()
    strat_adj[calls | busts] = 0.0

    # Uplift = strat / buyhold - 1
    uplift = strat_adj / buyhold - 1.0
    # Only-triggered uplift
    uplift_trig = uplift[trig_mask]

    # CAGR uplift: (strat/buyhold)^(1/H) - 1 ≈ log-diff / H
    # Not meaningful for called paths; skip.
    valid = trig_mask & ~calls & ~busts
    if valid.sum() > 0:
        cagr_diff = ((strat[valid] / buyhold[valid]) ** (1.0 / horizon_years) - 1.0) * 100
    else:
        cagr_diff = np.array([])

    print(f"\n--- threshold={threshold*100:.0f}%  horizon={horizon_years}y  "
          f"loan={loan_frac*100:.0f}%  (N={n:,}) ---")
    print(f"  triggered:     {n_trig:,} / {n:,}  ({n_trig/n*100:.1f}%)")
    print(f"  margin calls:  {int(calls.sum()):,}  "
          f"(paths where peak L > 4.0x)")
    print(f"  wipeouts:      {int(busts.sum()):,}")

    def pct(arr, q):
        if len(arr) == 0: return float("nan")
        return float(np.percentile(arr, q))

    print(f"  Uplift vs buy-and-hold (terminal equity, all paths, "
          f"called=0):")
    print(f"    mean={uplift.mean()*100:+6.2f}%   "
          f"median={pct(uplift,50)*100:+6.2f}%   "
          f"p10={pct(uplift,10)*100:+6.2f}%   p90={pct(uplift,90)*100:+6.2f}%")
    win = (uplift > 0).sum() / n * 100
    print(f"    beat-buyhold: {win:.1f}% of entry dates")
    if n_trig > 0:
        up_t = uplift[trig_mask]
        print(f"  Triggered-only paths (N={n_trig:,}):")
        print(f"    mean={up_t.mean()*100:+6.2f}%   "
              f"median={pct(up_t,50)*100:+6.2f}%   "
              f"p10={pct(up_t,10)*100:+6.2f}%   p90={pct(up_t,90)*100:+6.2f}%")
    if len(cagr_diff):
        print(f"  CAGR uplift (triggered, non-called): "
              f"mean={cagr_diff.mean():+.2f}%/yr  "
              f"median={np.median(cagr_diff):+.2f}%/yr  "
              f"p10={np.percentile(cagr_diff,10):+.2f}%/yr  "
              f"p90={np.percentile(cagr_diff,90):+.2f}%/yr")


# ============================================================
# MAIN SWEEP
# ============================================================
print("=" * 78)
print("BUY-THE-DIP WITH MARGIN (loan = 30% of equity at trigger,")
print("hold to horizon, interest compounds into loan)")
print("=" * 78)

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
horizons = [10, 20, 30]

# Store to also show comparison table
summary = {}
for h in horizons:
    for thr in thresholds:
        res = simulate(thr, h)
        summarize(res, thr, h, 0.30)
        arr_strat = np.array([r[3] for r in res])
        arr_bh = np.array([r[2] for r in res])
        trig_mask = np.array([r[1] is not None for r in res])
        peak_Ls = np.array([r[4] for r in res])
        called = peak_Ls > 4.0
        arr_strat_adj = arr_strat.copy()
        arr_strat_adj[called] = 0
        uplift = arr_strat_adj / arr_bh - 1.0
        summary[(h, thr)] = {
            "trig_rate": trig_mask.mean(),
            "mean_uplift": uplift.mean(),
            "median_uplift_triggered": (np.median(uplift[trig_mask])
                                        if trig_mask.sum() else float("nan")),
            "win_rate": (uplift > 0).mean(),
            "call_rate": called.mean(),
        }

# Comparison tables
print("\n\n" + "=" * 78)
print("COMPARISON TABLES")
print("=" * 78)

for h in horizons:
    print(f"\nHorizon = {h} years")
    print(f"{'threshold':>10}  {'trig %':>8}  {'mean uplift':>12}  "
          f"{'trig-only med':>14}  {'beat B&H':>10}  {'called %':>9}")
    for thr in thresholds:
        s = summary[(h, thr)]
        print(f"{thr*100:>9.0f}%  {s['trig_rate']*100:>7.1f}%  "
              f"{s['mean_uplift']*100:>+11.2f}%  "
              f"{s['median_uplift_triggered']*100:>+13.2f}%  "
              f"{s['win_rate']*100:>9.1f}%  "
              f"{s['call_rate']*100:>8.2f}%")
