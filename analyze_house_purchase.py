"""
analyze_house_purchase.py

Thought experiment: starting with $2M in SPY (cost basis $800K), buy a $1M house.
Goal state: $2M brokerage stocks AND house owned AND zero debt.

The simulation explicitly executes the loan/mortgage payoff (selling stocks at
that day's basis ratio) the first day the goal is reachable, and reports the
post-payoff state. So every reported "reach" is a literal zero-debt state with
>= $2M of stocks remaining.

------------------------------------------------------------
Strategy set
------------------------------------------------------------
Three families:

  * Pure cash:    cash_buy   (sell stocks to net $1M; pay $136K cap-gains tax)
  * Pure box:     full_box   (borrow $1M; brokerage leverage jumps to 2.0x — UNSAFE)
                  hybrid_125 (mix sale + box; brokerage lev 1.25x — well-defended)
                  hybrid_143 (mix sale + box; brokerage lev 1.43x — historical max-safe)
  * Pure mortgage: mortgage_20    (sell $227K stocks to fund 20% down; $800K mortgage)
                   mort_box_down  (use $200K BOX for the down; $800K mortgage; NO SALE)
                   mort_400box    ($400K box + $600K mortgage; brokerage lev 1.25x)
                   mort_600box    ($600K box + $400K mortgage; brokerage lev 1.43x)

Note on box-spread leverage: opening the short box gives you $1M cash (no
leverage, since $1M loan + $1M cash collateral net out). Leverage *appears*
the moment you withdraw that cash to fund the house — at that point your
brokerage is $V of stocks against $L of loan, so leverage = V / (V-L).
"mort_box_down" is the cleanest example: $200K withdrawn, leverage = 2 / 1.8
= 1.11x, and no taxable sale.

------------------------------------------------------------
Tax model
------------------------------------------------------------
  - LTCG = 20%, federal only (state ignored)
  - Cost basis is proportional. Initial gain ratio = 60% → effective sale
    haircut is 0.20*0.60 = 12% (so $1.136M gross sale nets $1M)
  - Cost basis grows with new DCA dollars (added at cost)
  - Box spread interest realized as 60/40 capital loss; modeled as 20% tax
    benefit on the gross interest cost
  - Mortgage interest deductibility ignored (conservative; mild bias against
    mortgage strategies)

------------------------------------------------------------
DCA scenarios (external income flow into the household)
------------------------------------------------------------
  $0/yr:  no external income. Mortgage P&I funded by selling stocks each
          month (with tax friction). Forces tax-inefficient sales.
  $60k/yr: roughly equals mortgage P&I ($60.7K/yr).
          - Mortgage strategies: $60K/yr → P&I; tiny shortfall sold from stocks;
            zero residual to invest.
          - Non-mortgage strategies: full $60K/yr → stocks (DCA).

This is apples-to-apples: same external cash flow across strategies, allocated
according to each strategy's cash-flow obligations. The earlier version had a
bug where mortgage strategies double-counted DCA (added it to stocks AND
treated it as covering the mortgage); fixed in this version.
"""

import numpy as np

from data_loader import load


dates, px, tsy, mrate = load()

TD = 252
V0 = 2_000_000.0
B0 = 800_000.0
HOUSE = 1_000_000.0
TARGET = 2_000_000.0
LTCG = 0.20
GAIN_RATIO_0 = 1.0 - B0 / V0           # 0.60
NET_FRAC_0 = 1.0 - LTCG * GAIN_RATIO_0  # 0.88

BOX_BPS = 0.0015
BOX_TAX_BENEFIT = 0.20
MORT_RATE = 0.065
MORT_YEARS = 30
CALL_THRESHOLD = 4.0


def resolve(spec):
    """Return ($-stock-sold, $-box, $-mortgage) for a strategy spec."""
    if "target_lev" in spec:
        # Hybrid: pick (sale, box) so brokerage lev = target_lev and 0.88*sale + box = HOUSE
        T = spec["target_lev"]
        k = 1.0 - 1.0 / T
        sale_amt = (HOUSE - k * V0) / (NET_FRAC_0 - k)
        box_amt = HOUSE - NET_FRAC_0 * sale_amt
        return sale_amt, box_amt, 0.0
    sf = spec.get("sale", 0.0)
    bf = spec.get("box", 0.0)
    mf = spec.get("mort", 0.0)
    return sf * HOUSE / NET_FRAC_0, bf * HOUSE, mf * HOUSE


def setup(spec):
    sale_amt, box, mort = resolve(spec)
    stocks = V0 - sale_amt
    basis = B0 * stocks / V0
    if mort > 0:
        r = MORT_RATE / 12.0
        n = MORT_YEARS * 12
        pmt = mort * r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
    else:
        pmt = 0.0
    tax_paid = sale_amt - NET_FRAC_0 * sale_amt
    return dict(stocks=stocks, basis=basis, loan=box, mort=mort,
                mort_pmt=pmt, tax_paid=tax_paid)


STRATEGIES = {
    "cash_buy":      dict(sale=1.0,  box=0.0,  mort=0.0),
    "full_box":      dict(sale=0.0,  box=1.0,  mort=0.0),
    "hybrid_125":    dict(target_lev=1.25),
    "hybrid_143":    dict(target_lev=1.43),
    "mortgage_20":   dict(sale=0.20, box=0.0,  mort=0.80),
    "mort_box_down": dict(sale=0.0,  box=0.20, mort=0.80),
    "mort_400box":   dict(sale=0.0,  box=0.40, mort=0.60),
    "mort_600box":   dict(sale=0.0,  box=0.60, mort=0.40),
}


def simulate(start_idx, spec, dca_yr, max_years=25):
    """Returns (years_to_reach, called_flag, end_state_dict_or_None)."""
    s = setup(spec)
    stocks, basis = s["stocks"], s["basis"]
    loan, mort, mort_pmt = s["loan"], s["mort"], s["mort_pmt"]

    # Cash flow accounting (apples-to-apples across strategies):
    #   external income = dca_yr
    #   if mortgage:
    #     P&I (= mort_pmt × 12 / yr) paid first from external income
    #     residual external income → invested in stocks
    #     if shortfall (P&I > external income) → sold from stocks to fund P&I
    #   else: full external income → invested in stocks
    yr_pmt = mort_pmt * 12.0
    if mort > 0:
        residual_dca_yr = max(dca_yr - yr_pmt, 0.0)
        shortfall_yr = max(yr_pmt - dca_yr, 0.0)
    else:
        residual_dca_yr = dca_yr
        shortfall_yr = 0.0
    residual_dca_daily = residual_dca_yr / TD
    shortfall_monthly = shortfall_yr / 12.0

    n_days = min(int(max_years * TD), len(dates) - start_idx - 1)
    last_month = 0

    for d in range(1, n_days + 1):
        i = start_idx + d
        # SPX growth
        stocks *= px[i] / px[i - 1]
        # Box spread interest accrual (after-tax effective)
        if loan > 0:
            box_rate = (tsy[i] + BOX_BPS) * (1.0 - BOX_TAX_BENEFIT)
            loan *= 1.0 + box_rate / TD
        # Daily residual DCA into stocks (zero if mortgage consumes the whole flow)
        if residual_dca_daily > 0:
            stocks += residual_dca_daily
            basis += residual_dca_daily

        # Daily margin-call check
        if loan > 0:
            eq = stocks - loan
            if eq <= 0.0 or stocks / eq >= CALL_THRESHOLD:
                return float("nan"), True, None

        month = d // 21
        if month > last_month:
            last_month = month

            # Mortgage: monthly amortization + (if needed) sell stocks for shortfall
            if mort > 0:
                interest = mort * MORT_RATE / 12.0
                principal = mort_pmt - interest
                if shortfall_monthly > 0 and stocks > 0:
                    g = max(0.0, 1.0 - basis / stocks)
                    sell_amt = shortfall_monthly / (1.0 - LTCG * g)
                    if sell_amt >= stocks:
                        return float("nan"), True, None
                    ratio = basis / stocks
                    stocks -= sell_amt
                    basis -= sell_amt * ratio
                mort = max(mort - principal, 0.0)

            # Reach-state check: sell stocks to clear ALL debt and end with >= $2M
            g = max(0.0, 1.0 - basis / stocks) if stocks > 0 else 0.0
            tax_eff = LTCG * g
            total_debt = loan + mort
            payoff_sale = total_debt / (1.0 - tax_eff) if total_debt > 0 else 0.0
            if stocks - payoff_sale >= TARGET:
                return d / TD, False, dict(
                    final_stocks=stocks - payoff_sale,
                    payoff_sale=payoff_sale,
                    payoff_tax=payoff_sale * tax_eff,
                    debt_cleared=total_debt,
                )

    return float("nan"), False, None


def percentiles(arr, qs=(10, 25, 50, 75, 90)):
    if len(arr) == 0:
        return [float("nan")] * len(qs)
    return [float(x) for x in np.percentile(arr, qs)]


def run():
    cutoff = np.datetime64("1932-07-01")
    eligible = [i for i in range(len(dates))
                if np.datetime64(dates[i]) >= cutoff and i + 5 * TD < len(dates)]
    eligible = eligible[::21]   # one entry per ~month
    print(f"Monthly entries: {len(eligible)} "
          f"({dates[eligible[0]].date()} to {dates[eligible[-1]].date()})")

    print("\n=== Initial state immediately after purchase ===")
    print(f"{'Strategy':<16}{'stocks':>11}{'basis':>10}{'box loan':>11}"
          f"{'mortgage':>11}{'P&I/mo':>10}{'tax_now':>10}{'broker_lev':>11}")
    for name, spec in STRATEGIES.items():
        d = setup(spec)
        if d["loan"] > 0:
            lev = f"{d['stocks'] / (d['stocks'] - d['loan']):.3f}"
        else:
            lev = "1.000"
        print(f"{name:<16}{d['stocks'] / 1000:10.0f}k {d['basis'] / 1000:9.0f}k "
              f"{d['loan'] / 1000:10.0f}k {d['mort'] / 1000:10.0f}k "
              f"{d['mort_pmt']:9.0f} {d['tax_paid'] / 1000:9.1f}k {lev:>11}")

    for dca in (0, 60_000):
        print(f"\n=== DCA = ${dca // 1000}k/yr external income ===")
        print("Time (years) until paid-off state with >= $2M stocks remaining:")
        print(f"{'Strategy':<16}{'p10':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p90':>7}"
              f"{'mean':>7}{'call%':>8}{'unrch%':>8}")

        # also collect terminal stock values at p50
        terminals = {}
        for name, spec in STRATEGIES.items():
            times, calls, unreach, fs = [], 0, 0, []
            for i in eligible:
                t, called, st = simulate(i, spec, dca)
                if called:
                    calls += 1
                elif np.isnan(t):
                    unreach += 1
                else:
                    times.append(t)
                    fs.append(st["final_stocks"])
            n = len(eligible)
            ps = percentiles(times)
            mn = float(np.mean(times)) if times else float("nan")
            print(f"{name:<16}{ps[0]:7.1f}{ps[1]:7.1f}{ps[2]:7.1f}{ps[3]:7.1f}{ps[4]:7.1f}"
                  f"{mn:7.1f}{100 * calls / n:7.1f}%{100 * unreach / n:7.1f}%")
            terminals[name] = (np.percentile(fs, 50) if fs else float("nan"))

        print("\nMedian stocks remaining AFTER paying off all debt at reach:")
        for name, val in terminals.items():
            print(f"  {name:<16} ${val / 1000:.0f}k  (target was $2,000k)")


if __name__ == "__main__":
    run()
