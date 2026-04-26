"""
"Take big loan, park in T-bills, DCA into SPX over N months."

Vectorized over all post-1932 entry dates.

Starting state (day 0):
  $1 equity, borrow $L → SPX = $1, T-bills = $L, Loan = $L
  → Assets = $(1+L), Equity = $1, Leverage = 1+L

Each tranche day (k * 21 trading days, k = 1..N),
move $L/N from T-bills to SPX. After month N, fully deployed.

Leverage = (SPX + T-bills) / (SPX + T-bills - Loan)
Margin call if leverage >= 4.0 at any point.
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()

TRADING_DAYS = 252
DAYS_PER_MONTH = 21

def cum_factor(rate):
    df = 1.0 + rate / TRADING_DAYS
    return np.concatenate(([1.0], np.cumprod(df[1:])))

M_box  = cum_factor(tsy + 0.0015)
M_cash = cum_factor(tsy)

post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate_vec(loan_frac, n_months, horizon_days):
    """Vectorized simulation of the buffered-loan strategy across all
    eligible entries. Returns terminal equity, peak leverage, called."""
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + horizon_days < len(px)]
    N = len(idxs)

    if n_months == 0:
        spx = np.full(N, 1.0 + loan_frac)
        tbill = np.zeros(N)
    else:
        spx = np.ones(N)
        tbill = np.full(N, loan_frac)
    loan = np.full(N, loan_frac)
    called = np.zeros(N, dtype=bool)
    peak_L = np.full(N, 1.0 + loan_frac)

    tranche_amt = (loan_frac / n_months) if n_months > 0 else 0.0
    tranche_days = set(k * DAYS_PER_MONTH for k in range(1, n_months + 1))

    for k in range(1, horizon_days + 1):
        # Growth factors this day for each entry
        spx_g   = px[idxs + k]     / px[idxs + k - 1]
        tbill_g = M_cash[idxs + k] / M_cash[idxs + k - 1]
        loan_g  = M_box[idxs + k]  / M_box[idxs + k - 1]

        spx    = spx    * spx_g
        tbill  = tbill  * tbill_g
        loan   = loan   * loan_g

        if k in tranche_days:
            amt = np.minimum(tranche_amt, tbill)
            tbill -= amt
            spx   += amt
            # If final tranche, flush residual
            if k == n_months * DAYS_PER_MONTH and n_months > 0:
                spx += tbill
                tbill[:] = 0.0

        assets = spx + tbill
        equity = assets - loan
        # Leverage; for equity<=0 or call: mark called
        safe_eq = np.where(equity > 0, equity, 1.0)
        lev = assets / safe_eq
        new_calls = (~called) & ((equity <= 0) | (lev >= 4.0))
        called |= new_calls
        peak_L = np.maximum(peak_L, np.where(equity > 0, lev, peak_L))

    terminal = np.where(called, 0.0, spx + tbill - loan)
    return terminal, peak_L, called, idxs


def simulate_single(i, loan_frac, n_months, horizon_days):
    """Simulate for one entry index i. Returns (terminal, peak_L, called)."""
    end = min(i + horizon_days, len(px) - 1)
    H = end - i
    if n_months == 0:
        spx = 1.0 + loan_frac
        tbill = 0.0
    else:
        spx = 1.0
        tbill = loan_frac
    loan = loan_frac
    peak_L = 1.0 + loan_frac
    called_at = None

    tranche_amt = (loan_frac / n_months) if n_months > 0 else 0.0
    tranche_set = set(k * DAYS_PER_MONTH for k in range(1, n_months + 1))

    for k in range(1, H + 1):
        spx   *= px[i+k]     / px[i+k-1]
        tbill *= M_cash[i+k] / M_cash[i+k-1]
        loan  *= M_box[i+k]  / M_box[i+k-1]

        if k in tranche_set:
            amt = min(tranche_amt, tbill)
            tbill -= amt
            spx += amt
            if k == n_months * DAYS_PER_MONTH and tbill > 0:
                spx += tbill
                tbill = 0.0

        assets = spx + tbill
        equity = assets - loan
        if equity <= 0:
            called_at = k
            break
        lev = assets / equity
        peak_L = max(peak_L, lev)
        if lev >= 4.0:
            called_at = k
            break

    if called_at is not None:
        return 0.0, peak_L, True
    return spx + tbill - loan, peak_L, False


# ============================================================
# (1) Worst-case entry (2000-03-23): peak leverage
# ============================================================
print("=" * 92)
print("(1) Entry 2000-03-23: peak leverage over 23y horizon")
print("=" * 92)
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
H_2000 = len(px) - i0 - 1

loan_fracs = [0.30, 0.41, 0.50, 0.60, 0.70]
dca_periods = [0, 6, 12, 24, 36]

print(f"\n{'loan':>5}  " +
      "  ".join(f"{('lump' if n==0 else f'{n}mo DCA'):>14}" for n in dca_periods))
peak_table = {}
for lf in loan_fracs:
    row = [f"{lf*100:>4.0f}%"]
    for nm in dca_periods:
        term, pk, c = simulate_single(i0, lf, nm, H_2000)
        peak_table[(lf, nm)] = (term, pk, c)
        label = f"{pk:.2f}x" if not c else f"CALL({pk:.1f})"
        row.append(f"{label:>14}")
    print("  ".join(row))

unlev = px[i0+H_2000] / px[i0]
print(f"\nTerminal wealth (unlevered baseline = {unlev:.3f}x):")
print(f"{'loan':>5}  " +
      "  ".join(f"{('lump' if n==0 else f'{n}mo DCA'):>14}" for n in dca_periods))
for lf in loan_fracs:
    row = [f"{lf*100:>4.0f}%"]
    for nm in dca_periods:
        term, pk, c = peak_table[(lf, nm)]
        ratio = term / unlev
        label = f"{ratio:.3f}x" if not c else "CALLED"
        row.append(f"{label:>14}")
    print("  ".join(row))

# ============================================================
# (2) Aggregate, 30y horizon
# ============================================================
print("\n" + "=" * 92)
print("(2) Aggregate over all post-1932 entries, 30y horizon")
print("=" * 92)

def aggregate_and_print(horizon_years):
    print(f"\nHorizon = {horizon_years}y\n")
    print(f"{'loan':>5}  {'DCA':>7}  {'N':>6}  {'call %':>7}  "
          f"{'mean mult':>9}  {'p10 mult':>9}  {'worst':>7}  "
          f"{'ΔCAGR mean':>11}  {'ΔCAGR p10':>10}")
    H = int(horizon_years * TRADING_DAYS)
    for lf in loan_fracs:
        for nm in dca_periods:
            term, pk, c, idxs = simulate_vec(lf, nm, H)
            unlev = px[idxs + H] / px[idxs]
            mult = np.where(c, 0.0, term / unlev)
            cagr_lev = np.where(c, 0.0, term ** (1.0 / horizon_years))
            cagr_un  = unlev ** (1.0 / horizon_years)
            dc = np.where(c,
                          -cagr_un + 0.0,   # CAGR diff = -unlev CAGR
                          cagr_lev - cagr_un) * 100
            label = "lump" if nm == 0 else f"{nm}mo"
            print(f"{lf*100:>4.0f}%  {label:>7}  {len(idxs):>6,}  "
                  f"{c.mean()*100:>6.2f}%  "
                  f"{mult.mean():>8.3f}x  "
                  f"{np.percentile(mult,10):>8.3f}x  "
                  f"{mult.min():>6.3f}x  "
                  f"{dc.mean():>+10.2f}pp  "
                  f"{np.percentile(dc,10):>+9.2f}pp")
        print()

aggregate_and_print(30)

# ============================================================
# (3) Aggregate, 20y horizon (includes 2000-era)
# ============================================================
print("=" * 92)
print("(3) Aggregate over all post-1932 entries, 20y horizon (INCLUDES 2000)")
print("=" * 92)
aggregate_and_print(20)
