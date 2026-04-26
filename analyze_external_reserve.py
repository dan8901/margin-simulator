"""
"External cash reserve as emergency backstop."

Total wealth = $1 (normalized).
  Reserve R (fraction of total) held externally, earning T-bill rate.
  Brokerage equity = 1 - R, with lump-sum loan L taken on day 0
    such that total SPX exposure = S = (1 - R) + L.
  Initial brokerage leverage = S / (1 - R).

Rule: if brokerage leverage hits `trigger` (default 3.5x), post enough
from reserve to bring brokerage leverage back to `target` (default 3.0x).
If reserve is exhausted first, post what's left; any further leverage
increase can still hit 4.0x → margin call.

Compared to buy-and-hold $1 SPX:
  unlev terminal  = px[T] / px[i]
  strat terminal  = (SPX_T - Loan_T) + Reserve_T  (if never called)
"""
import numpy as np
from datetime import datetime
from data_loader import load

dates, px, tsy, mrate = load()

TRADING_DAYS = 252

def cum_factor(rate):
    df = 1.0 + rate / TRADING_DAYS
    return np.concatenate(([1.0], np.cumprod(df[1:])))

M_box  = cum_factor(tsy + 0.0015)
M_cash = cum_factor(tsy)

post1932 = np.array([d >= datetime(1932, 7, 1) for d in dates])


def simulate_vec(S, R, horizon_days, trigger=3.5, target=3.0):
    """Vectorized simulation. S = total SPX exposure on $1 wealth.
    R = external reserve fraction of total wealth."""
    idxs = np.where(post1932)[0]
    idxs = idxs[idxs + horizon_days < len(px)]
    N = len(idxs)

    E = 1.0 - R
    L0 = S - E  # initial loan
    if L0 < 0:
        raise ValueError(f"S ({S}) must be > E ({E}); need loan >= 0")

    spx = np.full(N, S, dtype=float)
    loan = np.full(N, L0, dtype=float)
    reserve = np.full(N, R, dtype=float)
    called = np.zeros(N, dtype=bool)
    ever_posted = np.zeros(N, dtype=bool)
    peak_brok_lev = np.full(N, S / E if E > 0 else np.inf)

    for k in range(1, horizon_days + 1):
        spx_g  = px[idxs + k]     / px[idxs + k - 1]
        box_g  = M_box[idxs + k]  / M_box[idxs + k - 1]
        cash_g = M_cash[idxs + k] / M_cash[idxs + k - 1]

        spx     *= spx_g
        loan    *= box_g
        reserve *= cash_g

        brok_eq = spx - loan
        pos = brok_eq > 0
        safe_eq = np.where(pos, brok_eq, 1.0)
        brok_lev = np.where(pos, spx / safe_eq, np.inf)

        # Post reserve where leverage >= trigger AND reserve available
        needs = (brok_lev >= trigger) & (reserve > 0) & (~called)
        if needs.any():
            # X = loan - SPX*(1 - 1/target) brings leverage to target
            needed_X = loan - spx * (1.0 - 1.0 / target)
            actual_X = np.minimum(np.maximum(needed_X, 0.0), reserve)
            # Only apply where needs is True
            loan    = np.where(needs, loan    - actual_X, loan)
            reserve = np.where(needs, reserve - actual_X, reserve)
            ever_posted |= needs
            # recompute after post
            brok_eq = spx - loan
            brok_lev = np.where(brok_eq > 0, spx / np.maximum(brok_eq, 1e-12),
                                np.inf)

        # Mark calls
        new_calls = (~called) & ((brok_eq <= 0) | (brok_lev >= 4.0))
        called |= new_calls
        peak_brok_lev = np.maximum(
            peak_brok_lev,
            np.where(brok_eq > 0, brok_lev, peak_brok_lev))

    terminal = np.where(called, 0.0, spx - loan + reserve)
    unlev = px[idxs + horizon_days] / px[idxs]
    return {
        "terminal": terminal, "unlev": unlev,
        "called": called, "ever_posted": ever_posted,
        "peak_brok_lev": peak_brok_lev,
    }


def report(horizon_years):
    H = int(horizon_years * TRADING_DAYS)
    print("\n" + "=" * 100)
    print(f"Horizon = {horizon_years}y, post-1932 entries, "
          f"trigger=3.5x, target=3.0x (post from reserve)")
    print("=" * 100)
    print(f"{'S':>5}  {'R':>4}  {'L0_brok':>8}  {'call %':>7}  "
          f"{'posted %':>8}  {'mean mult':>9}  {'p10 mult':>9}  "
          f"{'worst':>7}  {'ΔCAGR mean':>11}  {'ΔCAGR p10':>10}")

    scenarios = [
        # (S, R) pairs
        (1.30, 0.00),  # baseline you've been working with
        (1.41, 0.00),  # post-1932 lump-sum max
        (1.50, 0.00),  # too much without reserve
        (1.50, 0.10),
        (1.50, 0.20),
        (1.60, 0.00),
        (1.60, 0.10),
        (1.60, 0.20),
        (1.75, 0.00),
        (1.75, 0.20),
        (1.75, 0.30),
        (1.75, 0.40),
        (2.00, 0.20),
        (2.00, 0.30),
        (2.00, 0.40),
        (2.00, 0.50),
    ]
    for S, R in scenarios:
        r = simulate_vec(S, R, H)
        E = 1.0 - R
        L0_brok = S / E
        mult = np.where(r['called'], 0.0, r['terminal'] / r['unlev'])
        cagr_term = np.where(r['called'], 0.0,
                             r['terminal'] ** (1.0 / horizon_years))
        cagr_un = r['unlev'] ** (1.0 / horizon_years)
        dc = np.where(r['called'], -cagr_un, cagr_term - cagr_un) * 100
        print(f"{S:>4.2f}x  {R*100:>3.0f}%  {L0_brok:>7.2f}x  "
              f"{r['called'].mean()*100:>6.2f}%  "
              f"{r['ever_posted'].mean()*100:>7.2f}%  "
              f"{mult.mean():>8.3f}x  "
              f"{np.percentile(mult,10):>8.3f}x  "
              f"{mult.min():>6.3f}x  "
              f"{dc.mean():>+10.2f}pp  "
              f"{np.percentile(dc,10):>+9.2f}pp")


report(20)
report(30)

# Detail on 2000-03-23
print("\n" + "=" * 100)
print("Detail: 2000-03-23 entry, 23y horizon")
print("=" * 100)
i0 = int(np.where(dates == datetime(2000, 3, 23))[0][0])
H = len(px) - i0 - 1
for S, R in [(1.41, 0.00), (1.50, 0.10), (1.50, 0.20),
             (1.60, 0.10), (1.60, 0.20), (1.75, 0.20), (1.75, 0.30),
             (2.00, 0.30), (2.00, 0.40)]:
    # Single-entry simulation
    E = 1.0 - R
    L0 = S - E
    spx = S
    loan = L0
    reserve = R
    called = False
    ever_posted = False
    peak_brok_lev = S / E
    for k in range(1, H + 1):
        spx *= px[i0 + k] / px[i0 + k - 1]
        loan *= M_box[i0 + k] / M_box[i0 + k - 1]
        reserve *= M_cash[i0 + k] / M_cash[i0 + k - 1]
        brok_eq = spx - loan
        if brok_eq <= 0:
            called = True
            break
        brok_lev = spx / brok_eq
        if brok_lev >= 3.5 and reserve > 0:
            needed = loan - spx * (1 - 1/3.0)
            actual = min(max(needed, 0), reserve)
            loan -= actual
            reserve -= actual
            ever_posted = True
            brok_eq = spx - loan
            brok_lev = spx / brok_eq if brok_eq > 0 else np.inf
        peak_brok_lev = max(peak_brok_lev, brok_lev)
        if brok_lev >= 4.0:
            called = True
            break
    unlev = px[i0 + H] / px[i0]
    terminal = 0.0 if called else spx - loan + reserve
    mult = terminal / unlev if not called else 0.0
    print(f"  S={S:.2f}x R={R*100:.0f}%  "
          f"peak brokerage L={peak_brok_lev:.2f}x  "
          f"posted={'YES' if ever_posted else 'no'}  "
          f"terminal mult={mult:.3f}x  {'CALLED' if called else ''}")
