# SPX Margin Leverage Simulator — Project Documentation

This project builds quantitative backtests of margin-leveraged SPX strategies, using daily SPX-total-return data from 1927-12-30 to 2023-04-12. The goal: understand how much margin leverage a long-term SPX investor can safely take, how much CAGR uplift they should expect, and how the answer changes with ongoing contributions, re-levering, and other variants.

The user's specific constraint set throughout the analysis:
- 100% SPX in a **taxable brokerage account**
- Plan is to **hold forever** (leverage SPX with margin, never sell SPX)
- Broker margin (not futures, not LEAPS) — though box spreads considered
- Some ongoing savings contributions ("DCA" — deposits into the same account)

All scripts are self-contained: they read from `spx_margin_history.csv` (not the original Excel file) via `data_loader.py`. Run with the project's venv at `.venv/bin/python`.

## DOCUMENTATION POLICY (read first)

**At the end of every session that produces non-trivial findings, update CLAUDE.md without being asked.** Document: new scripts, new findings (positive or negative), corrections to prior conclusions, refined recommendations, and any change in the priority ranking of strategies. The user has explicitly asked that this happen automatically rather than on request — treat it as a standing instruction, not a per-session decision. If you're unsure whether a finding is "non-trivial enough," err on the side of writing it down. The cost of an extra paragraph in CLAUDE.md is much lower than the cost of a future session re-deriving something that was already learned.

---

## Table of contents

1. [Data](#1-data)
2. [Mathematical framework](#2-mathematical-framework)
3. [Tax treatment of leverage instruments](#3-tax-treatment-of-leverage-instruments)
4. [Script inventory](#4-script-inventory)
5. [Key findings](#5-key-findings)
5b. [Decay strategy deep dive](#5b-decay-strategy-deep-dive-added-in-second-session)
5c. [End-of-horizon leverage](#5c-end-of-horizon-leverage-added-in-third-session)
5d. [Generalization vs overfitting](#5d-generalization-vs-overfitting-added-in-third-session)
5e. [Streamlit app + new strategy families](#5e-streamlit-app--new-strategy-families-added-in-fourth-session)
5f. [Re-calibration strategies + meta_recal](#5f-re-calibration-strategies--meta_recal-added-in-fifth-session)
5g. [Recal T_init calibration fix](#5g-recal-t_init-calibration-fix-added-in-sixth-session)
5h. [Early broker-rate bump](#5h-early-broker-rate-bump-added-in-seventh-session)
5i. [Calibration uses avail-bounded all-entries paths](#5i-calibration-uses-avail-bounded-all-entries-paths-added-in-eighth-session)
6. [Practical recommendations](#6-practical-recommendations)
7. [Caveats and limitations](#7-caveats-and-limitations)
8. [Important corrections made during analysis](#8-important-corrections-made-during-analysis)
9. [Open questions & future work](#9-open-questions--future-work)
10. [Notes for future sessions](#10-notes-for-future-sessions)

---

## 1. Data

### Source

Extracted once from `/Users/dnissim/Downloads/Portfolio Margin Backtest.xlsx`, sheet "Data", via an extraction script that is no longer part of the repo (user confirmed spreadsheet won't be updated, so we removed `extract_data.py`).

### File: `spx_margin_history.csv`

~24,695 rows spanning 1927-12-30 to 2026-04-24 (daily, trading days only).
Originally 23,934 rows ending 2023-04-12 (extracted from a Bloomberg-sourced
spreadsheet); extended through 2026-04-24 in 2026-04 via `extend_data.py`
using yfinance `^SP500TR` daily ratios (anchored to the existing series at
2023-04-12) plus FRED `DGS3MO` for the 3M Treasury yield. Margin rate is
derived as `tsy_3m + 40 bps` per the project convention.

Columns:
| Column | Description |
|---|---|
| `date` | YYYY-MM-DD |
| `spx_tr` | SPX total return index (including dividends reinvested) |
| `tsy_3m` | 3-Month US Treasury yield, annualized, decimal (e.g., `0.034` = 3.4%) |
| `margin_rate` | Broker margin rate = 3M Tsy + ~40 bps, annualized, decimal |
| `cpi` | CPI-U level (FRED `CPIAUCNS`, monthly, forward-filled to daily) |

Rows with missing `spx_tr` or `margin_rate` are skipped at load. All scripts access data via `data_loader.load()`:

```python
from data_loader import load
dates, px, tsy, mrate = load()
# dates: np.array of datetime
# px:    SPX total return index
# tsy:   3M Treasury yield (annualized)
# mrate: broker margin rate (annualized)

# To also get CPI (opt-in for backwards compat):
dates, px, tsy, mrate, cpi = load(with_cpi=True)
# cpi: CPI-U level (a level, NOT a rate). Divide later/earlier to get
# cumulative inflation between two dates. Sanity: 1927-12 = 17.30,
# 2023-04 = 303.36, full-series 3.05%/yr inflation.
```

**Real-vs-nominal sanity (1927-12 to 2023-04):** Nominal SPX-TR CAGR 9.92%/yr,
real CAGR 6.66%/yr, implied inflation 3.05%/yr. To convert any nominal series
to real (in dollars of date `d0`): `series_real = series * cpi[d0] / cpi`.

CPI was added in a 2026-04 session via `merge_cpi.py` (one-shot, sources from
FRED `CPIAUCNS`, ffills monthly to daily).

### Key historical drawdowns

- **1929-09-16 → 1932-06-01**: SPX-TR fell 83.79% over 2.71 years. Worst full-series entry date.
- **2000-03-23 → 2009-03-09**: SPX-TR fell 48.04% over 8.96 years. **Worst post-1932 entry date.** This is the binding constraint for almost every safety analysis.
- **1973-01-11 → 1974-10-03**: ~48% drawdown over ~1.6 years. Important secondary stress point.
- **2007-10-09 → 2009-03-09**: ~55% drawdown over ~1.4 years. GFC.

Margin rate (3M Tsy + 40 bps) range in the data: **0.40% to 17.54%** annualized. Historical high was late 1970s/early 1980s stagflation era.

---

## 2. Mathematical framework

### Leverage ratio definition

Throughout the project, **leverage ratio** = `Total Portfolio Value / Net Liquidation Value (equity)`.

- Reg-T maintenance margin = 25% → max leverage = 4.0x before call
- Portfolio margin = 15% → max leverage = 6.667x before call

We use **4.0x as the call threshold** everywhere (the conservative / Reg-T answer), because portfolio margin agreements universally reserve the broker's right to tighten mid-crisis — exactly when you'd want the lower requirement. See `analyze_portfolio_margin.py` for the PM analysis and the "broker tightens mid-crisis" stress test.

### Closed form for peak leverage (single one-time loan, no contributions)

For a lump-sum loan of `L_0 - 1` on day 0 with interest compounding at box-spread or margin rate:

```
A_t = L_0 * px[t] / px[i]                      # asset (SPX) value at day t
D_t = (L_0 - 1) * M_loan[t] / M_loan[i]        # loan value, compounded
E_t = A_t - D_t                                # equity
L_t = A_t / E_t                                # leverage at day t
```

Where `M_loan[t] = cumprod(1 + rate_annual / 252)` is the cumulative loan-growth factor.

Let `R_t = px[t] / M_loan[t]`. Then leverage at day `t` given entry `i`:

```
L_t = L_0 * (R_t / R_i) / (L_0 * (R_t / R_i) - (L_0 - 1))
```

Peak leverage over `[i, T]` occurs when `R_t` is minimized (biggest SPX drop relative to loan growth). The critical `L_0` that makes peak leverage = cap:

```
L_0_max(i, cap) = cap * R_i / (cap * R_i - (cap - 1) * min_{t>=i} R_t)
```

This closed form drives `analyze.py`, `analyze_with_interest.py`, `analyze_granular.py`, `analyze_post1932.py`, `analyze_safety_cushion.py`, `analyze_portfolio_margin.py`, and the "buy-the-dip" analysis `analyze_dip_buy.py`.

### Adding ongoing contributions (DCA)

With monthly contributions of `m` dollars (from external income into SPX), the assets grow by each contribution amount plus its compounding from the contribution date forward:

```
A_t = L_0 * px[t]/px[i] + Σ_k m * px[t]/px[t_k]
D_t = (L_0 - 1) * M[t]/M[i]
```

No closed form — must simulate day-by-day. See `analyze_dca_leverage_grid.py`, `analyze_dca_raises_ceiling.py`.

### Re-levering (dynamic target leverage)

To maintain a target leverage `T` by taking additional loan `ΔD`:

```
ΔD = T * E - A      # new loan needed to restore target
                    # Only take if ΔD > 0 (lever up only, never sell)
```

After adding `ΔD`: new assets `A + ΔD`, new loan `D + ΔD`, equity unchanged, leverage = T.

With decay: `T(year t) = T_initial - decay_per_year * t`, floored at 1.0.

### IRR vs CAGR (critical distinction)

- **CAGR** = `(terminal / initial)^(1/H) - 1`. Assumes a single investment at t=0, no contributions.
- **IRR** = the rate `r` satisfying the NPV equation. With contributions, this is the correct "rate of return" per invested dollar.

**IRR is computed by solving numerically:**

```
1 * (1+r)^H + (m/12) * ((1+r)^H - 1) / ((1+r)^(1/12) - 1) = terminal
```

where `m` = annual contribution amount. Solved via `scipy.optimize.brentq`.

**Crucial lesson learned**: using `terminal^(1/H) - 1` as "CAGR" when there are contributions over-reports the rate of return by 1-5 pp depending on DCA level, because contributions are treated as if they were growth. The IRR correction is applied in `analyze_irr_correction.py` and `analyze_irr_percentiles.py`.

**DCA does NOT change rate of return; it scales up how much is invested.** A strategy with higher DCA has higher terminal wealth but roughly the same IRR as one with zero DCA.

---

## 3. Tax treatment of leverage instruments

Important subtlety for a taxable hold-forever SPX strategy. Each instrument has different tax characteristics:

| Instrument | Rate above Tsy | Tax characteristics | Compatibility with hold-forever SPX |
|---|---|---|---|
| **Broker margin** | +40 bps (per our data) | Interest cost, deductible against investment income if itemizing. No new tax events from the loan itself. | ✅ Best fit — leaves SPX untouched |
| **Box spread (short)** | +10-30 bps | Section 1256 contract. Interest equivalent is recognized as **60/40 capital LOSS** annually for the borrower. Tax deduction at ~18-25% blended rate. After tax: often beats broker margin for retail users without investment income. | ✅ Good fit — SPX is separate; only box itself has tax events |
| **Futures (ES/MES)** | +10-20 bps | Section 1256, marked-to-market annually. Your entire SPX-direction exposure is marked. Forces annual recognition of gains/losses even without selling. | ❌ Bad fit — annual realization violates hold-forever |

**Correction to an earlier mistake in this session**: I initially claimed box spreads hurt tax-wise for hold-forever. That was WRONG. The short box-spread borrower generates tax-deductible 60/40 capital losses equal to the interest cost. For a retail investor on standard deduction with no investment income, **box spreads usually beat broker margin after tax by 100-150 bps/yr**.

However, box spreads require options trading approval and more operational sophistication. Broker margin is the simpler default.

---

## 4. Script inventory

All scripts are standalone and runnable with `.venv/bin/python script_name.py`. They share only `data_loader.py`.

### Foundation

- **`data_loader.py`** — Loads the CSV, returns numpy arrays. `load()` returns 4-tuple (dates, spx, tsy, margin); `load(with_cpi=True)` returns 5-tuple appending CPI level.

### Verification scripts

- **`verify_recal_tinit.py`** — Computes plain well-defended T_rec for `{static, hybrid, adaptive_dd, dd_decay}` on the user's primary scenario. Used as a reference for what the recal_X strategies in `app.py` should display post-fix (see §5g). Just prints; doesn't assert.
- **`verify_recal_end_to_end.py`** — Same calibration as above, plus assertions: recal_X T_rec == base T_rec, meta_recal T_rec == argmax. Exits non-zero on regression. Run after any change to the calibration logic in `app.py compute()` or `find_max_safe_T_grid`.

### Side / utility scripts

- **`project_portfolio.py`** — CLI tool: project portfolio in REAL dollars over time given current value C, savings S for T years then S2. Auto-CALIBRATES leverage targets to the user's specific scenario: for each of (static, relever, dd_decay) it binary-searches both historical max-safe (0% calls on post-1932 monthly entries) AND bootstrap-safe (≤1% synthetic calls on N 1y-block-bootstrap paths), then uses min(T_hist, T_boot) as the recommended target per strategy. Vectorized simulation across paths; runs in ~15-20 seconds with default args. Reports percentile bands at checkpoints in real dollars (CPI-deflated by entry-date or path-start CPI), with inflation-tracked DCA contributions. Strategies modeled: `unlev`, `static`, `relever` (monthly), `dd_decay` (T_init binary-searched, F=1.5, floor=1.0). Use `--bootstrap-paths` and `--bootstrap-block-years` to tune the bootstrap.

  **Notable finding for very-high-DCA scenarios** (e.g. C=160k, S=180k/yr for 5y, S2=30k/yr): static's user-calibrated max-safe is much higher than the project's 1.43x — found 2.146x historically with only 0.35% bootstrap calls. BUT the leverage is mostly *wasted*: high DCA dilutes leverage from 2.15x to ~1.1x within 5 years, so static gives only +14% real wealth at 30y. Relever (capturing leverage on every contribution) and dd_decay both give +45-46% real wealth at 30y at much lower targets (1.23x and 1.52x respectively). Relever's bootstrap fragility (16.6% calls at hist-safe of 1.47x) confirms the project's main-line finding holds in the personalized regime. dd_decay edges relever marginally on EV with structurally lower tail risk — it's the recommended choice for high-DCA personal projection.

- **`analyze_house_purchase.py`** — One-shot thought experiment: buying a $1M house from a $2M portfolio (basis $800K). Compares cash buy / box spread / mortgage / mixed strategies. Goal state = "$2M brokerage + house + zero debt." Mixed strategy `mort_400box` (40% box + 60% mortgage, no stock sale) wins by structurally separating the leverage from the volatile asset; reaches goal in median 4.6y/3.7y (zero/$60k DCA). See script docstring for full rationale.

- **`analyze_2m_to_3m.py`** — Quick benchmark: time for a $2M portfolio to reach $3M across DCA levels and leverage targets. Median ~3.2y unleveraged at $0 DCA, ~2y at 1.43x leverage with $100k DCA. Used for cross-checking other "time to grow X→Y" results.

- **`merge_cpi.py`** — One-shot enrichment script: pulls FRED CPIAUCNS (monthly, 1913+) and merges into `spx_margin_history.csv` as a `cpi` column (forward-filled to daily). Run once; reruns are idempotent.

- **`extend_data.py`** — Append yfinance `^SP500TR` + FRED `DGS3MO` to `spx_margin_history.csv` for any dates after the existing CSV's last row. yfinance level is rebased via daily ratios anchored at the existing last value, so the series stays continuous. Re-run `merge_cpi.py` afterwards to refresh CPI on the new rows. Used to extend the data from 2023-04 (original Bloomberg dump end) through 2026-04 (live).

- **`app.py`** — Streamlit UI wrapping `project_portfolio.py`. Sliders for C/S/T/S2/horizon/bootstrap-paths/block-size/target-call-rate, checkboxes per strategy. Calibration + projection runs on a "Run / refresh" button (~5-15 s depending on bootstrap path count). Path arrays are cached via `@st.cache_resource` so they only rebuild on horizon/bootstrap changes. Output: calibration table, p50 cross-strategy table, p50 line chart, per-strategy detailed percentile tables (in expanders), and a p10/p50/p90 fan chart for a chosen strategy. Run with `.venv/bin/streamlit run app.py`.

### Basic static-leverage analyses (no contributions, one-time loan held forever, interest compounds)

- **`analyze.py`** — Earliest / simplest version. No interest compounding. Computes max-safe L₀ = 4/(1+3f_max) where f_max is max future drawdown from entry. Produces full-series and percentile stats. Finds 1929-09-16 is worst entry at 1.14x max-safe.

- **`analyze_with_interest.py`** — Same but with interest compounding on loan (box-spread rate). Derives closed form `L_0_max(i) = 4*R_i / (4*R_i - 3*min_future_R)` using R = px/M_loan. Worst-case with interest at 1929-09-16: 1.12x.

- **`analyze_post1932.py`** — Filters entry dates to start from 1932-07-01 (post-Depression trough) and 1950-01-01. **Key result: post-1932 worst is 2000-03-23 at 1.41x.**

- **`analyze_granular.py`** — Dense percentile tables (p0.1, p1, p2, ..., p99.9) of max-safe L₀ across entries. Shows that post-1932 distribution is wide.

- **`analyze_2000_entry.py`** — Detailed walkthrough of the 2000-03-23 entry. Shows peak leverage reaches 3.999x on 2009-03-09 at L₀=1.41x. Sensitivity table of different L₀ values from 1.0 to 2.0, showing how fast things deteriorate.

- **`analyze_safety_cushion.py`** — Generalizes to arbitrary peak-leverage caps (3.0x, 3.5x, 3.9x, 4.0x) using `L_0_max(i, cap) = cap*R_i / (cap*R_i - (cap-1)*min_R)`. Also tests rate sensitivity: what happens to 2000-03-23 max-safe if rates had been higher during 2000-2009.

- **`analyze_portfolio_margin.py`** — 15% vs 25% maintenance requirement (6.67x vs 4.0x cap). Includes "broker tightens mid-crisis" stress test: if you size for 15% PM but broker switches to 25% requirement during a drawdown, you get called — typically days before the bottom. Recommends sizing for 25% even in PM accounts.

### Strategy variants

- **`analyze_dip_buy.py`** — "Wait for drawdown, then lever up" strategy. At various thresholds (5%, 10%, 20%, 30%, 50% drawdown), take a 30% loan. Finds:
  - Lower thresholds dominate on mean uplift (they trigger reliably).
  - No margin calls at 30% loan regardless of threshold.
  - +0.5-0.8%/yr CAGR uplift depending on threshold.
  - Modest benefit — the effect is mostly "have more compounding time."

- **`analyze_leverage_and_financing.py`** — Broker (Tsy+40) vs box-spread (Tsy+15) financing across leverages 1.0-2.0x. Shows:
  - Box saves modest ~20bps/yr at 1.50x leverage; meaningful but not transformative.
  - Cliff between 1.50x and 1.75x — above 1.50x, 20y horizon call rates jump from ~2% to ~16%.
  - Above 1.75x, mean terminal wealth stops growing because calls eat the tail.

- **`analyze_dca_loan.py`** — DCA the loan over 6/12/24/36 months instead of lump-sum. **Finds DCAing the loan is mostly statistical illusion — costs 2-15 bps/yr of mean return for only 5-15 bps of worst-case improvement.** Small benefit, often negative at long horizons.

- **`analyze_buffered_loan.py`** — Take a bigger loan (e.g., 50-70%), park excess in T-bills, DCA into SPX over N months. Finds this extends safe loan ceiling from 41% (lump) to 50% (36-month DCA buffer) — an 8 pp improvement that's useful but not transformative. +0.08 pp CAGR over 41% lump.

- **`analyze_external_reserve.py`** — Hold cash EXTERNAL to the brokerage account, post it to loan if leverage hits 3.5x. **Finds this does NOT work** — long drawdowns drain the reserve, and the mechanics of holding part of wealth in cash actually raise initial brokerage leverage, making the account more fragile. Important negative result.

- **`analyze_amortized.py`** — Pay down the loan from external income over 5/10/20 years (self-amortizing). **Finds that amortization gives only +0.25-0.40 pp IRR over DCA-of-same-dollars counterfactual.** Not worth the commitment for most. But can safely unlock higher initial loan (up to ~100% with 10y amort and low call risk).

### Apples-to-apples comparisons

- **`analyze_clean_comparison.py`** — Same cash flow (10%/yr), different deployments:
  - Pure DCA (unlev)
  - 41% lump + interest compounds + DCA everything
  - 50/60/70% lump + 10y full amortization
  - 50/60/70% lump + interest-only (keep principal forever)
  - **Surprise result**: 41% lump + interest compounds + DCA wins. Amortization loses because it redirects cash from high-return SPX to low-return loan paydown.

- **`analyze_income_stops.py`** — What if contributions stop at year 10 (retirement) and you hold for 20 more years? **Confirms that 41% lump still wins** even in retirement scenarios. Only very narrow conditions favor amortization.

### Re-levering (dynamic leverage)

- **`analyze_dca_raises_ceiling.py`** — Finds how DCA rate affects max-safe static target. With 10% DCA, max-safe rises from 1.41x → 1.61x.

- **`analyze_dca_leverage_grid.py`** — Full grid of (DCA %, leverage) with call rates, CAGR percentiles. **Originally used the buggy CAGR-with-DCA formula** — values in this file should be re-interpreted using IRR (see `analyze_irr_correction.py`).

- **`analyze_irr_correction.py`** — Diagnoses and corrects the CAGR bug. Shows that DCA adds wealth, not rate of return.

- **`analyze_relever.py`** — Initial re-lever analysis. Compares static vs monthly-to-exact-target re-lever. **Key finding**: re-lever creates a cliff — simple monthly re-lever above 1.41x target becomes dangerous because it accumulates loan during bull runs.

- **`analyze_relever_variants.py`** — Band rebalancing (only relever when leverage < threshold × target), quarterly/annual rebalance, time-decaying target, cumulative-loan cap. **Key findings**:
  - Simple monthly/quarterly/annual re-lever stuck at ~1.43-1.47x max-safe regardless of DCA.
  - Decay 2pp/yr and loan cap let DCA raise max-safe to ~1.60x.
  - Several of these strategies are "safe at 20y horizon but start showing calls at 30y" — horizon robustness matters.

- **`analyze_irr_percentiles.py`** — Clean IRR analysis. For each of (static, monthly relever, annual relever, decay-2pp at max-safe) and each DCA level in {0, 5, 10, 15, 20, 30%}, reports per-path IRR percentiles (p10, p25, p50, p75, p90, mean) at both 20y and 30y horizons. **Updated in third session to use DUAL-HORIZON max-safe sizing** (0% calls at both 20y and 30y) instead of 20y-only — the prior version's monthly-relever targets had ~9% call risk at 30y. This is a key summary script.

- **`analyze_irr_inflation.py`** — Same analysis as `analyze_irr_percentiles.py` but with DCA growing at 2.5%/yr (inflation-adjusted, maintaining real contribution value). Uses closed-form geometric-series IRR with growing contributions. **Finding**: effect is small (±0.1pp median IRR, ~1pp higher max-safe targets). Fixed-nominal DCA results are good enough for practical decisions.

- **`analyze_wealth_decay.py`** — Wealth-based decay strategy. Target leverage declines as max-equity (high-water mark) grows: `target(t) = max(floor, T_initial − slope × (max_equity(t)/initial_equity − 1))`. Key property: during drawdowns, max equity doesn't update, so target doesn't drop. Parameterized by `wealth_mult_to_floor` — the wealth multiple at which target hits the floor (1.0x). **Finding**: wealth-decay is a spectrum. WM=3x behaves like static (fast decay), WM=20x behaves like monthly relever (slow decay). At WM=15-20x, wealth-decay nearly matches time-decay 2pp's median IRR (~13.84% vs 13.89% at 10% DCA / 20y) with the philosophical property that decay pauses during bear markets. Uses 10-iter binary search for max-safe (precision ~0.003x) to keep runtime reasonable. Takes 15-25 min to run.

- **`analyze_reset_clock.py`** — Tests the user's intuition "if the year was up, can I reset the time-decay clock?" Walks through 2000-03-23 entry at T_initial=1.59x with 10%/yr DCA, comparing pure time-decay 2pp, reset-on-up-years, and monthly relever. **Result**: reset-on-up-years and monthly relever are behaviorally identical (both reset target to 1.59x during 2003–2007 bull market) and both get called on 2009-03-02 at peak leverage 5.03x. Pure time-decay survives with peak 2.88x. Conclusively shows that gating decay on market signal reverts to simple-relever blow-up mode. Optional `--full` flag runs the slower post-1932 sweep.

- **`analyze_end_leverage.py`** — End-of-horizon leverage percentiles (p10/25/50/75/90) plus per-path IRR percentiles for static / time-decay 2pp / wealth-decay WM=20 / monthly relever, all sized at their dual-horizon max-safe targets (0% calls at both 20y and 30y). Demonstrates that decay strategies actually deleverage by horizon end (median 1.16–1.27x at 20y) while monthly relever stays pinned at target (~1.47x), and that static deleverages MOST aggressively (~1.08x). The complementary metric to peak leverage and IRR.

### Generalization-buffer scripts (added in third session)

These address the overfitting concern: dual-horizon max-safe is the largest target where the worst historical path squeaks past 4.0x cap — textbook overfitting to a specific path.

- **`analyze_call_threshold.py`** — Sweeps the call-threshold cap {3.0x, 3.5x, 4.0x} and re-finds dual-horizon max-safe under each. Models "broker tightens maintenance to 33%/29%/25% during stress." IRR cost of dropping cap 4.0→3.0 is small (15-53 bps p50 IRR). Insurance against a known historical broker behavior is cheap.

- **`analyze_stress_drawdown.py`** — Multiplies every post-1932 drawdown by F ∈ {1.0, 1.1, 1.2} and re-finds dual-horizon max-safe. Models "future drawdowns deeper than past." F=1.2 (20% deeper) cuts targets by ~10%, IRR by 26-92 bps. **Caveat**: the calibration print at the top of the script is misleading (it reports drawdown vs full-history running max, which includes the 1929 peak). The simulation correctly uses post-1932 running max.

- **`analyze_block_bootstrap.py`** — Generates 5000 synthetic 30y paths via 1-year block bootstrap of post-1932 daily (return, rate) pairs. Reports synthetic call rate at each strategy's dual-horizon-safe historical target. **Most informative buffer**: directly measures path-overfitting. Historical 0% targets produce synthetic call rates of 1.46% (static), 7.84% (wealth-decay WM=20), 10.02% (time-decay 2pp), 13.74% (monthly relever). The 6.9× gap between time-decay and static is direct evidence that decay/relever's historical 0% was path-dependent luck rather than structural safety.

- **`analyze_well_defended.py`** — Joint-constraint binary search for the largest T satisfying BOTH: (a) historical 0% calls at cap=3.0x AND (b) bootstrap call rate ≤1% at cap=4.0x. Includes block-size sensitivity (1mo, 3mo, 6mo, 1y, 2y, 5y) showing call rate is non-monotonic in block size — peaks at 2y blocks (11.36%) for time-decay, drops at 5y blocks (8.50%) because long blocks regress toward historical sequence. The drawdown-stress (F=1.2) is intentionally excluded as redundant with bootstrap.

- **`analyze_drawdown_decay.py`** — New decay architecture coupling target to observed max drawdown rather than calendar (time-decay) or HWM (wealth-decay). Mechanism: `target = max(floor, T_initial - DD_FACTOR × max_dd_observed)` where max_dd ratchets up only. Strategy starts at T_initial through quiet periods, ratchets target down once a drawdown is observed, and PERMANENTLY stays lower even after recovery. Results at 10% DCA: drawdown-decay F=1.0 has bootstrap call rate 3.68% vs wealth-decay's 7.84% at nearly identical IRR (12.59% vs 12.69%) — Pareto improvement. F=2.0 reaches 2.64% bootstrap calls at 12.32% IRR. Includes well-defended joint-constraint search.

- **`analyze_cohort_drawdown_decay.py`** — Tests the cohort/vintage framework: each DCA contribution starts a separate sub-strategy with its own max_dd ratchet and target. Aggregate margin call still on aggregate. **Empirical result: the cohort approach is WORSE than single-account drawdown-decay on bootstrap (5-6% vs 2-4% calls at similar IRR) and has lower historical max-safe targets.** The "fresh meat" failure mode: post-crisis cohorts start fresh with max_dd=0 and re-lever aggressively during recoveries; when the NEXT crisis hits they're at full leverage with no protection. Single-account "permanent learning" is a feature, not contamination. Negative result; cohort approach abandoned.

- **`analyze_drawdown_decay_persistent.py`** — Tests persistence filter to address COVID-style V-shaped over-reaction: only ratchet max_dd if drawdown persists for DURATION_MIN consecutive months. Sweeps DURATION_MIN ∈ {1, 3, 6, 12} × F ∈ {1.0, 1.5, 2.0}. **Empirical result: persistence filter is a Pareto LOSS.** At every F level, the no-filter (1mo) version has the lowest bootstrap call rate. Longer durations gain 10-30 bps p50 IRR but cost 100-200 bps bootstrap fragility — about a 6:1 bad trade. Mechanism: filter delays max_dd updates during long drawdowns (e.g., 2000-09), so the strategy commits to high leverage during the dangerous early phase before deleveraging. The original instant-ratchet is the safety feature, not a flaw. Negative result; persistence filter abandoned. Tuning F alone gives equivalent IRR-vs-safety trade-offs without adding a parameter.

- **`analyze_drawdown_decay_windowed.py`** — Tests time-windowed max_dd at HISTORICAL-max-safe sizing. Short windows have higher IRR but higher bootstrap fragility. Suggests window-tuning is worse than F-tuning. **THIS CONCLUSION WAS WRONG** — see windowed_welldefended below. The historical-max-safe view doesn't normalize for safety; the well-defended view (same boot threshold) gives the corrected picture.

- **`analyze_drawdown_decay_windowed_welldefended.py`** — Same as windowed but sizes each (F, window) at well-defended target (boot ≤1% + cap=3.0x). **CORRECTED RESULT: short windows (5y) BEAT ∞ window by 15-25 bps p50 IRR at the same safety bar.** F=1.5/5y at T=1.431x: p50 IRR 12.26% @ 1.00% boot. F=1.5/∞ at T=1.461x: p50 IRR 12.06% @ 0.98% boot. **Mechanism**: short window accepts lower T_initial but re-claims leverage as old drawdowns expire from the window. Lifetime ratchet (∞) front-loads leverage and permanently de-levers after each crisis; rolling window deleverages temporarily then recovers. End-leverage confirms (5y: 1.20-1.26x; ∞: 1.05-1.09x). **5y window with F=1.5 is the new recommended Tier 2.**

---

## 5. Key findings

### The "max-safe" numbers (zero historical margin calls at BOTH 20y and 30y, post-1932)

**Sizing rule** (revised in third session): a strategy is "safe" only if it has 0% calls at *both* 20y and 30y horizons. Sizing only at 20y was previously hiding 30y call risk for monthly relever (~9% calls at 30y at the 20y-safe target).

| DCA | Static | Monthly relever | Annual relever | Decay 2pp re-lever |
|---|---|---|---|---|
| 0%  | 1.425x | 1.425x | 1.425x | 1.425x |
| 5%  | 1.576x | 1.428x | 1.428x | 1.555x |
| 10% | 1.612x | 1.429x | 1.430x | 1.591x |
| 15% | 1.649x | 1.431x | 1.431x | 1.605x |
| 20% | 1.685x | 1.431x | 1.432x | 1.612x |
| 30% | 1.756x | 1.433x | 1.433x | 1.624x |

Striking pattern: under dual-horizon sizing, **simple monthly/annual relever's max-safe is essentially flat at ~1.43x regardless of DCA.** Adding 30 pp/yr of contributions raises the relever ceiling by only ~0.8 bps. Static and decay-2pp do benefit from DCA (the equity dilution helps); pure relever does not (because re-levering immediately consumes the dilution).

### IRR uplift from leverage (30y horizon, dual-horizon-safe targets)

Median per-path IRR across ~15,000 post-1932 entries.

| DCA | Unlev | Static @ safe | Monthly relever @ safe | Decay 2pp @ safe |
|---|---|---|---|---|
| 0%  | 10.96% | 12.05% (+1.09) | 13.67% (+2.71) | 12.65% (+1.69) |
| 5%  | 10.85% | 11.97% (+1.12) | 13.43% (+2.58) | 13.13% (+2.28) |
| 10% | 10.79% | 11.76% (+0.97) | 13.28% (+2.49) | 13.16% (+2.37) |
| 20% | 10.89% | 11.52% (+0.63) | 13.16% (+2.27) | 13.03% (+2.14) |
| 30% | 10.97% | 11.46% (+0.49) | 13.08% (+2.11) | 12.93% (+1.96) |

Monthly relever still has the highest median IRR but only by 12–35 bps over decay-2pp at positive DCA — much smaller than the old (mis-sized) gap suggested. At 0% DCA where targets are identical, decay-2pp loses 102 bps because its target decays *down* from the same starting point.

### The big structural findings

1. **Without ongoing contributions, 1.425x is the post-1932 ceiling for ALL strategies.** At 0% DCA, static, monthly relever, annual relever, and decay-2pp converge to the same max-safe target — there's no mechanism for them to differ. Binding on 2000-03-23 entry.

2. **DCA contributions raise the static and decay-2pp ceiling — but NOT the simple-relever ceiling.** 10% DCA: static → 1.61x, decay-2pp → 1.59x, monthly relever → 1.43x (essentially unchanged). Static benefits because DCA dilutes leverage; decay-2pp benefits because the schedule still drives target down despite re-levering; simple relever doesn't benefit because each re-lever immediately re-consumes the dilution.

3. **Sizing at 20y horizon hides 30y call risk for re-lever strategies.** Monthly relever at the 20y-safe target of 1.45x has ~9% margin-call rate when held 30 years. Decay strategies are robust across horizons. **Always size for safety at both 20y and 30y** ("dual-horizon max-safe").

4. **Clever re-lever variants (time-decay, loan cap, band) recover most of the static ceiling** while retaining the re-lever IRR benefit. Decay 2pp/yr with 10% DCA: dual-horizon max-safe 1.591x, median IRR 13.16% (30y).

5. **Taxes favor box spreads for retail hold-forever investors** (Section 1256 loss recognition) but the financing-cost advantage is only ~100-150 bps/yr and broker margin is operationally simpler.

6. **Amortization is usually a net loss** vs. letting the loan compound while DCAing new money. Counterintuitive but the math is clear: don't divert money from high-return (SPX ~9%) to low-return (loan ~5%).

7. **External cash reserves don't save you** from long drawdowns. The reserve drains during a multi-year drawdown, which is exactly when you need it to keep working.

8. **The 2000-03-23 entry is the binding constraint** for almost every safety question post-1932. Its unusual feature: a 9-year-long drawdown that causes loan compounding to eat ~33% of the loan, not just the SPX drop itself.

9. **Portfolio margin (15% maintenance) is a fair-weather feature.** Size as if you have 25% maintenance (broker reserves the right to tighten, and typically does during crises).

10. **DCA grows wealth but not rate of return.** This is a conceptual point that was initially missed due to a CAGR-with-contributions bug. More DCA = more total dollars deployed = more terminal wealth. But rate of return per dollar is fundamentally determined by the underlying asset returns, not DCA.

11. **Static at max-safe with high DCA has LOWER median IRR than static with low DCA** (counterintuitive). At 10% DCA, static max-safe is 1.61x giving 11.76% IRR (30y). At 30% DCA, static max-safe is 1.76x giving 11.46% IRR (30y). Pushing leverage higher *decreases* IRR because the DCA stream is unlevered — more DCA means more unlevered dollars diluting the leveraged exposure. This is why re-levering (which maintains the leverage ratio as DCA flows in) outperforms static at higher DCA levels.

12. **Wealth-based decay is a spectrum parameterized by the "wealth multiple to floor."** Fast decay (WM=3x) behaves like static. Slow decay (WM=20x) behaves like constant-target relever. The sweet spot (WM=15-20x) matches time-decay 2pp's IRR with the behavioral advantage of pausing during drawdowns. Time-decay is slightly more IRR-efficient (~5-10 bps/yr) because its known schedule allows a slightly higher max-safe target.

13. **Inflation-adjusted DCA (2.5%/yr growth) is a rounding error.** Over the 20-30 year horizons tested, growing DCA by 2.5%/yr changes max-safe targets by <0.02x and median IRR by <0.1pp. Only matters if growth rate is substantially higher (5%+/yr).

---

## 5b. Decay strategy deep dive (added in second session)

The "decay strategies" family addresses a specific user preference: **wanting to end unlevered after 20-30 years** ("once my portfolio has grown, I don't need leverage"). Several mechanisms achieve this:

### The four mechanisms compared

| Strategy | "When do I deleverage?" | Mechanism |
|---|---|---|
| **Static @ max-safe** | Passively via asset growth + DCA dilution | Take big loan on day 0, never rebalance |
| **Time-decay Npp/yr** | On a calendar schedule | Target decreases linearly with elapsed time |
| **Wealth-decay WM=x** | When portfolio grows | Target decreases with max-equity HWM |
| **Monthly relever** | Never (maintain leverage) | Re-lever to constant target monthly |

### Why the user's intuition about "ending unlevered" already favors static

A static loan-held-forever strategy *naturally* deleverages over 30 years:
- Initial leverage 1.6x on $1 equity
- After 30y: SPX × DCA accumulated grew to ~$45-50; loan compounded to ~$2.6
- End leverage: $48 / $45 ≈ 1.06x — essentially unlevered

**This makes static the most "end unlevered" strategy in practice** — no active management needed. Decay strategies actually retain MORE leverage at year 30 because they keep re-levering during bull runs.

### Key mechanics the user asked about

- **"Restart decay on up years"** defeats the safety purpose. Empirically demonstrated in `analyze_reset_clock.py`: at T_initial = 1.59x with 10% DCA, the 2000-03-23 entry has pure time-decay 2pp surviving (peak lev 2.88x) vs reset-on-up-years getting CALLED on 2009-03-02 (peak lev 5.03x, identical to plain monthly relever). The 2003–2007 bull market resets the clock 5 times, pinning target at 1.59x throughout — the resulting loan accumulation cannot survive the GFC. The 14-bps cushion that pure time-decay had at March 2007 (target 1.45x vs reset's 1.59x) is the exact difference between surviving and being called 7 days before the bottom.
- **"Start decay on first down year"** is also wrong — it sells low after a crash, exactly the wrong direction.
- Time-decay is deliberately calendar-based to be ungameable by market moves.

### Wealth-decay parameterization choices

- **Floor**: 1.0x (fully unlevered) is the natural choice
- **Wealth multiple to floor**: 15-20x is the sweet spot (matches time-decay IRR)
- **T_initial**: sized at max-safe (binary-searched for zero calls)
- **Wealth measure**: uses raw equity (includes DCA contributions). Alternative: contribution-adjusted equity (only counts investment performance). Not yet simulated.
- **HWM update**: only on monthly boundaries; during drawdowns, HWM doesn't update, so target is frozen.

### Wealth-decay IRR at 10% DCA, 20y (benchmark comparison, dual-horizon-safe targets)

| Strategy | Target | p10 | p50 | mean |
|---|---|---|---|---|
| Static @ max-safe | 1.612x | 8.12% | 12.91% | 12.64% |
| Wealth-decay WM=3 | 1.611x | 8.16% | 12.99% | 12.70% |
| Wealth-decay WM=10 | 1.606x | 8.50% | 13.56% | 13.38% |
| Wealth-decay WM=20 | 1.562x | 8.49% | 13.84% | 13.70% |
| Time-decay 2pp | 1.591x | 8.40% | 13.89% | 13.96% |
| Monthly relever | 1.429x | 8.08% | 14.09% | 14.07% |

(Monthly relever target dropped from 1.451x → 1.429x under dual-horizon safety; p50 IRR dropped 13 bps. The relative ranking is unchanged but the gap to time-decay is now smaller — and time-decay leads on p10.)

---

## 5c. End-of-horizon leverage (added in third session)

How much leverage does each strategy actually retain by the end? Computed in `analyze_end_leverage.py` at dual-horizon-safe targets, 10% DCA, post-1932 entries:

### Median end-of-horizon leverage

| Strategy | Initial T | end-lev @ 20y | end-lev @ 30y |
|---|---|---|---|
| Static | 1.612x | **1.08x** | **1.05x** |
| Wealth-decay WM=20 | 1.562x | 1.19x | 1.09x |
| Time-decay 2pp | 1.591x | 1.27x | 1.16x |
| Monthly relever | 1.429x | 1.47x | 1.49x |

### Full percentile spread @ 20y

```
                       p10    p25    p50    p75    p90
static                 1.04x  1.05x  1.08x  1.14x  1.19x  <- tightest
wealth-decay WM=20     1.09x  1.11x  1.19x  1.28x  1.46x
time-decay 2pp         1.20x  1.21x  1.27x  1.41x  1.52x
monthly relever        1.43x  1.43x  1.47x  1.58x  1.71x  <- pinned at target
```

### Takeaways

- **Static has the LOWEST end leverage** (passive drift-down is the most aggressive deleveraging mechanism). It's also the tightest distribution.
- **Decay strategies do deleverage meaningfully** (median 1.16–1.27x at 20y, 1.05–1.16x at 30y) — the calendar / HWM schedule actually achieves what it advertises.
- **Monthly relever ends near its target** with no passive deleveraging. It's running close to its own ceiling for the entire horizon.
- **Wealth-decay deleverages MORE than time-decay** at year 30 (1.09x vs 1.16x median) because by year 30 most paths have triggered enough HWM growth to drop wealth-decay's target to the floor; time-decay 2pp at 1.59x doesn't reach floor until year 29.5.

### Why this matters: monthly relever's ~13 bps IRR edge isn't free

Monthly relever beats decay strategies by ~13–35 bps p50 IRR at 10–30% DCA. In exchange:
- **0 pp of passive deleveraging** vs decay's 0.30–0.40x reduction in end-leverage by year 30
- **Tightest target** (1.43x regardless of DCA) — no room to take more loan
- **Higher peak-leverage during drawdowns** (which is also why dual-horizon sizing forced the smaller target)
- **Continued exposure to broker-tightening risk** at year 30 (still levered → still callable)

For "hold-forever, end-unlevered" preferences, the IRR edge is too small to compensate.

---

## 5d. Generalization vs overfitting (added in third session)

The dual-horizon max-safe numbers in §5 are **directly overfit to historical path ordering**. Binary search finds the largest target where the *worst observed* path (2000-03-23) squeaks past 4.0x cap. A future drawdown 5% deeper or 1 year longer would blow up every number in the table.

### Three buffer mechanisms (all run, all useful)

| Buffer | What it protects against | Cost (10% DCA, p50 IRR @ 30y) |
|---|---|---|
| Cap reduction (4.0→3.0x) | Broker tightening maintenance during crisis (happened 2008, 2020) | 15-53 bps |
| Drawdown depth stress (F=1.2) | Future drawdowns deeper than historical | 26-92 bps |
| Block bootstrap (1y, ≤1% calls) | Path-ordering overfitting | 50-117 bps |

All three move targets in the same direction by similar amounts (~0.85-0.94× of historical max-safe). The **bootstrap is the most informative** because it directly *measures* the overfitting rather than parametrically guessing at it.

### Bootstrap call rates at historical-safe targets (10% DCA, 30y, 5000 paths, 1y blocks)

This is the single most damning result for the historical "max-safe" numbers:

| Strategy | Historical target | Synthetic call rate |
|---|---|---|
| Static | 1.612x | **1.46%** |
| Wealth-decay WM=20 | 1.562x | 7.84% |
| Time-decay 2pp | 1.591x | **10.02%** |
| Monthly relever | 1.429x | **13.74%** |

The historical 0% is largely path-dependent luck for the path-coupled strategies. Static — which has no path-dependent decisions — is 5-10× more robust under random reorderings.

### Block-size sensitivity (time-decay 2pp at 1.591x)

Synthetic call rate is **non-monotonic** in block size:

| Block | Call rate |
|---|---|
| 1mo | 5.82% |
| 3mo | 6.52% |
| 6mo | 8.18% |
| **1y (default)** | **10.02%** |
| 2y | 11.36% (worst) |
| 5y | 8.50% |

Smaller blocks randomize too much (break up bad-year clusters); 2-year blocks preserve crisis dynamics without preserving the specific historical recovery; 5-year blocks regress toward historical because they preserve enough sequence (e.g., the entire 2000-2005 chunk) to behave like real history. **1y is a reasonable middle ground but may understate tail risk slightly** — 2y is more stressful.

### Well-defended max-safe (joint constraints, 10% DCA)

Largest target satisfying BOTH constraints simultaneously: (a) historical 0% calls at cap=3.0x, AND (b) bootstrap call rate ≤1% at cap=4.0x.

| Strategy | Hist max-safe | Well-defended | Haircut | p50@30y | p10@30y | end-lev | Boot call% |
|---|---|---|---|---|---|---|---|
| Static | 1.612x | **1.513x** | 0.94× | 11.61% | 10.55% | 1.04x | 0.40% |
| Time-decay 2pp | 1.591x | **1.395x** | 0.88× | 12.04% | 10.91% | 1.07x | 1.00% |
| Wealth-decay WM=20 | 1.562x | **1.352x** | 0.87× | 12.00% | 10.93% | 1.07x | 1.00% |
| Monthly relever | 1.429x | **1.236x** | 0.87× | 12.11% | 10.93% | 1.26x | 1.00% |

**The IRR ranking collapses under honest constraints.** From 152 bps spread (historical) to 50 bps spread (well-defended). Monthly relever's apparent IRR edge over decay strategies disappears entirely.

**Which constraint binds:**
- Static: cap=3.0x is binding (bootstrap call rate at well-defended target is only 0.40%, well under 1%)
- All other strategies: bootstrap=1% is binding

### Architectural critique of decay strategies (NEW — important)

The user raised a deeper critique than parameter overfitting: **the architecture itself**. Findings:

**Time-decay is architecturally weakest.** It uses calendar time as the risk variable, but calendar time has no causal connection to margin call risk. Margin calls come from drawdown depth and current leverage; the schedule "lever down 2pp/yr" only works if crises happen to arrive *during* the lever-down phase. The clock runs identically whether you're up 50% or down 50% — no state feedback. The 2pp rate happens to match the binding 2000-09 crisis duration (9-year drawdown, target falls 1.59 → 1.41 over 9 years, just barely keeping peak leverage below 4.0x). This is curve-fitting to the historical timing.

**Wealth-decay is architecturally better but still flawed.** It couples to a real state variable (HWM) and pauses during drawdowns — both genuine improvements over time-decay. But:
1. **HWM is backward-looking.** After a peak-then-crash, HWM stays at the peak; target is "low" because you USED to be wealthy, not because you currently are. Architecture happens to be safe but for the wrong reason.
2. **DCA contributions count as wealth growth.** Depositing $1M jumps HWM and drops target — but depositing money doesn't make leverage structurally safer. Signal is gameable. (Open question #2: switch to contribution-adjusted wealth.)
3. **The fundamental claim is a utility argument, not a risk argument.** "I have more wealth, I need less leverage" is about preferences, not about what makes leverage safer mechanically.
4. **Bootstrap residual fragility (7.84%)** confirms the architecture only partially escapes path overfitting.

**Static is architecturally cleanest.** It has:
- Single parameter (T_initial) with closed-form risk math
- No path-dependent decisions → not exposed to bootstrap-style randomization
- Naturally deleverages via passive math (loan fixed, equity grows from market+DCA, leverage drifts toward 1.0)
- Lowest bootstrap call rate (1.46%, 5-10× better than active strategies)

**Monthly relever is architecturally worst.** It explicitly couples to the wrong thing — it pins target through bull markets, accumulating loan exactly when "no warning signal" is exactly the lack-of-warning that masks an impending crash. Bootstrap call rate 13.74% reflects this.

### Updated architecture ranking

| Rank | Strategy | Defensibility | Bootstrap calls |
|---|---|---|---|
| 1 | Static | Closed-form risk math, single parameter, no active decisions | 1.46% |
| 2 | Static + band rebalance (untested) | Static + small active component to capture some IRR upside | TBD |
| 3 | Wealth-decay | Couples to a real but backward-looking variable; gameable by DCA | 7.84% |
| 4 | Time-decay | Couples to a non-risk variable (calendar) | 10.02% |
| 5 | Monthly relever | Maximally exposed to path randomness | 13.74% |

**The honest punch line on decay strategies:** they are more complex ways to approximate static's natural deleveraging while paying for path-dependent re-lever fragility. The active management buys 4-5 bps higher target throughout — and costs 5-7 percentage points of bootstrap call rate. **The trade is bad once you account for generalization.**

### Conceptual guardrails learned this session

- **Calendar time is not a risk variable.** Any architecture that decays based on years elapsed is curve-fitting to historical crisis timing.
- **HWM is a backward-looking proxy for cushion.** Architectures coupling to HWM are partially state-aware but miss the actual risk metric (current leverage, current drawdown).
- **DCA contributions should not count as wealth growth** in any decay rule (open question #2). However: see "shared state" guardrail below — decoupling drawdown signal from DCA at the cohort level *hurts* aggregate safety, so the right resolution may be different.
- **The right risk variables to couple to** are: current leverage (already used as call check), drawdown depth from entry/recent peak, or loan-to-equity ratio (equivalent to leverage). Drawdown-decay (now tested) confirms this works.
- **0.85-0.90× haircut on historical max-safe** is a reasonable practical buffer that covers all three failure modes (broker tightens, fat tails, path overfitting).
- **Shared/global state beats per-cohort state for hold-forever leverage.** The cohort-vintage framework (each DCA contribution gets its own max_dd ratchet) was empirically WORSE on bootstrap: 5-6% calls vs 2-4% for single-account drawdown-decay at similar IRR. Reason: post-crisis cohorts start with max_dd=0 and re-lever aggressively during recovery; the next crisis catches them unprotected ("fresh meat" problem). Single-account permanent-learning across all dollars is structurally safer.
- **Instant ratchet beats smoothed/delayed ratchet.** The persistence filter (only ratchet max_dd if drawdown lasts ≥ N months) was a Pareto loss — adding it gained 10-30 bps p50 IRR but cost 100-200 bps bootstrap fragility, about 6:1 against. Mechanism: delayed updates commit to high leverage during the dangerous early months of a long drawdown. The instant-ratchet appears to be over-reactive on V-shaped events but is the actual safety feature on the multi-year drawdowns that bind max-safe sizing.
- **At the same safety bar (well-defended), 5y rolling window BEATS ∞ window on IRR by 15-25 bps p50.** Earlier conclusion that "F-tuning beats window-tuning" was based on historical-max-safe sizing and was misleading because that framing doesn't normalize for safety. At well-defended sizing (boot ≤1% + cap=3.0x), short windows accept lower T_initial but re-claim leverage after window expires; the lifetime ratchet (∞) front-loads leverage and permanently de-levers after each crisis. End-leverage confirms: 5y window ends at 1.20-1.26x while ∞ ends at 1.05-1.09x. The "recover and continue" pattern accumulates more total leveraged exposure over 30 years than "front-load and decay." **Use 5y window with F=1.0-2.0 for Tier 2 active strategy.**
- **The bootstrap reliably catches false sophistication.** Three "more sophisticated" mechanisms tested in third session (cohorts, persistence filter, short rolling windows) all looked promising on architectural grounds but Pareto-lost vs the simple version. In each case the historical-only data was at-best-neutral on the addition; bootstrap is what showed it was actually harmful. **For any future architectural variant proposal, run bootstrap before believing it.**
- **For V-shaped (COVID-style) over-reaction concerns, lower F is the right knob.** F=1.0 is the practical calibration for forward-going investors worried about brief shocks; F=2.0+ for those willing to over-react to insure against multi-year crashes. Don't add complexity (persistence filter, rolling window) — the parameter sweep on F alone covers the relevant trade-off space.

### Drawdown-decay sensitivity to V-shaped shocks (COVID concern)

User raised a valid worry: drawdown-decay with F=2.0 would over-react to a 5-week COVID-style crash, ratcheting target permanently to floor and missing the recovery. **Empirically tested.** Persistence filter (only ratchet if drawdown lasts ≥ N months) was a Pareto loss — see `analyze_drawdown_decay_persistent.py` results.

**What actually addresses the COVID concern:** **lower F** (e.g., F=1.0 instead of F=2.0). At F=1.0, a 34% COVID-style drawdown drops target by 0.34 (1.59 → 1.25), not by 0.68 (which would floor to 1.0x). Strategy still levered post-shock, retains some recovery upside. F=1.0 also has nearly identical bootstrap call rate to F=2.0 (3.68% vs 2.64%) at slightly higher IRR (12.59% vs 12.32%). For forward-going investors worried about V-shaped shocks, **F=1.0–1.5 is the right calibration**, not a more complex filter.

**Architectural lesson:** the instant-ratchet of max_dd looks like over-reaction but is actually the safety feature. Delaying the ratchet (via persistence filter or any other "smoothing") commits to high leverage during the dangerous early phase of a long drawdown. The right knob for COVID concerns is the magnitude (F), not the timing.

**One free behavioral upgrade:** use a 20-year ROLLING WINDOW for max_dd instead of lifetime ratchet. Empirically equivalent on every metric (within 10 bps) but gives the property that a 1980 drawdown won't permanently affect your 2030 strategy. Shorter windows (≤10y) hurt safety with poor IRR-per-safety ratio compared to F-tuning.

### Drawdown-coupled decay (NEW — empirically confirmed, third-session)

Architecture: `target(t) = max(floor, T_initial - DD_FACTOR × max_dd_observed)` where `max_dd` ratchets up only (uses HWM-relative drawdown).

**Properties:**
- Event-driven: target stays at T_initial through quiet periods, drops only when stress is *actually observed*
- Asymmetric ratchet: once a drawdown is observed, target stays lowered even after recovery
- Couples to the actual risk variable (drawdown depth)
- DD_FACTOR has defensible interpretation: "after a 25% drawdown, target falls 0.25 × DD_FACTOR points of leverage"

**Empirical results, 10% DCA, dual-horizon max-safe at cap=4.0x:**

| Strategy | Param | Target | p50@30y | end-lev | Bootstrap calls |
|---|---|---|---|---|---|
| Static | — | 1.612x | 11.76% | 1.05x | 1.46% |
| Wealth-decay WM=20 | — | 1.562x | 12.69% | 1.09x | 7.84% |
| Time-decay 2pp | — | 1.591x | 13.16% | 1.16x | 9.98% |
| **Drawdown-decay** | **F=0.5** | 1.503x | 12.87% | 1.30x | **5.62%** |
| **Drawdown-decay** | **F=1.0** | 1.553x | 12.59% | 1.15x | **3.68%** |
| **Drawdown-decay** | **F=1.5** | 1.586x | 12.47% | 1.10x | **3.30%** |
| **Drawdown-decay** | **F=2.0** | 1.586x | 12.32% | 1.07x | **2.64%** |
| **Drawdown-decay** | **F=3.0** | 1.586x | 12.10% | 1.06x | **2.00%** |

**Drawdown-decay strictly Pareto-dominates wealth-decay and time-decay:**
- F=1.0 vs wealth-decay: nearly same IRR (12.59% vs 12.69%) at less than half the bootstrap call rate (3.68% vs 7.84%)
- F=3.0 vs static: +34 bps IRR (12.10% vs 11.76%) at only 0.54 pp higher bootstrap calls (2.00% vs 1.46%)
- Dominates time-decay on every metric (IRR, bootstrap, end-lev)

**Well-defended (joint cap=3.0x + bootstrap≤1%) for drawdown-decay:**

| Strategy | T | p50@30y | p10@30y | end-lev | Bootstrap |
|---|---|---|---|---|---|
| Drawdown-decay F=1.0 | 1.431x | 12.11% | 10.83% | 1.09x | 1.00% |
| Drawdown-decay F=1.5 | 1.461x | 12.06% | 10.76% | 1.06x | 0.98% |
| Drawdown-decay F=2.0 | 1.481x | 12.00% | 10.73% | 1.05x | 1.00% |

Drawdown-decay F=2.0 well-defended matches wealth-decay's well-defended IRR (12.00%) at a *higher* target (1.481x vs 1.352x) and identical end-leverage (1.05x vs 1.07x). Architectural improvement is real and measurable.

**Subtle observation:** at F ≥ 1.5, max-safe targets converge at 1.586x. Beyond that DD_FACTOR, the binding constraint stops being "decay too slow during 2000-09" and starts being "max ever observed drawdown is the cap." More aggressive decay doesn't help unless T_initial also rises.

### Updated architecture ranking (after drawdown-decay)

| Rank | Strategy | Defensibility | Bootstrap calls (10% DCA, hist max-safe) |
|---|---|---|---|
| 1 | Static | Closed-form risk math, single parameter, no active decisions | 1.46% |
| 2 | **Drawdown-decay (F=2-3)** | **Couples to actual risk variable, asymmetric ratchet, defensible parameter** | **2.0-2.6%** |
| 3 | Drawdown-decay (F=0.5-1.5) | Same as above, less aggressive | 3.3-5.6% |
| 4 | Wealth-decay | Couples to backward-looking variable, gameable by DCA | 7.84% |
| 5 | Time-decay | Couples to non-risk variable (calendar) | 9.98% |
| 6 | Monthly relever | Maximally exposed to path randomness | 13.80% |

**Drawdown-decay should replace wealth-decay and time-decay in the recommendation hierarchy.** The architectural critique was correct AND the practical payoff is real.

---

## 5e. Streamlit app + new strategy families (added in fourth session)

The fourth session built the Streamlit interface (`app.py`) and added a richer
strategy taxonomy. Most of the strategies that follow operate inside the JIT
inner loop (`_simulate_core` / `_simulate_core_grid` in
`project_portfolio.py`) under integer kind codes 0-10. Each is reachable from
the Streamlit sidebar.

### New core strategies

- **`wealth_decay` (kind 3)** — current-equity wealth glide. Target linearly
  interpolates from `T_init` (when real equity = `C`) down to `floor` (when
  real equity = `wealth_X`). NOT a ratchet; symmetric in equity. Pure
  utility-glide, gameable by DCA. Bootstrap fragility intermediate.

- **`hybrid` (kind 4)** — `min(dd_decay_target, wealth_decay_target)`. Both
  signals lower target independently; more conservative wins. Pareto-dominates
  wealth_decay (dd component rescues paths where a deep drawdown hits before
  equity reaches `wealth_X`). **User-developed.** Default for the Streamlit
  app's Tier-1 recommendation when the user has an "unlever at $X" goal.

- **`r_hybrid` (kind 5)** — ratcheted hybrid. Wealth-progress also ratchets
  up (never decreases), so the wealth-decay component becomes monotonic.
  Removes "rebound and re-lever during drawdown" failure mode. In high-DCA
  scenarios where wealth grows monotonically anyway, behaves identically to
  hybrid.

- **`vol_hybrid` (kind 6)** — `hybrid_target − vol_factor × σ_60_annualized`.
  Couples target to current 60d realized volatility (σ_60 = sqrt(252) ×
  stdev of daily returns). Vol leads drawdowns by days/weeks, so this fires
  *before* dd_decay's reactive ratchet. Common hedge-fund "vol targeting"
  pattern.

- **`dip_hybrid` (kind 7)** — buy-the-dip override. When `cur_dd > dip_threshold`,
  target is FLOORED at `T_init + dip_bonus` (allowed ABOVE `T_init` during
  deep drawdowns). **Empirical TRAP**: looks good historically (real history
  has ~one drawdown per cycle) but blows up on bootstrap (clustered drawdowns
  stack the bonus on top of accumulated damage). T_rec collapses ~15-30%.

- **`rate_hybrid` (kind 8)** — `hybrid_target − rate_factor × max(0, tsy_3M − rate_threshold)`.
  Deleverages when carry-trade economics deteriorate. Default τ=5%, ρ=5.
  Historic high-rate periods are short and rare in post-1932 sample, so
  haircut rarely fires; behaves ≈ hybrid in practice.

- **`adaptive_dd` (kind 9)** — drawdown-decay with cushion-coupled F:
  `F_eff = F × (L_now − 1) / (T_init − 1)`. Decay aggressive when leverage
  is near `T_init` (small cushion), mild when already deleveraged (big
  cushion). Monotonic-down ratchet on target. Modest Pareto improvement
  over plain `dd_decay`: bootstrap call rate roughly halved (~3.8% vs ~6%
  at historical-safe T) with ~15 bps higher p50 IRR.

- **`adaptive_hybrid` (kind 10)** — adaptive_dd + wealth glide. In
  heavy-DCA scenarios, the wealth glide is the binding constraint
  (equity crosses `wealth_X` before dd ratchet fires), so behaves
  identically to plain hybrid. Differentiates from hybrid in low-DCA
  scenarios.

### Streamlit UI architecture

- **`app.py`** wraps `project_portfolio.py` with sidebar inputs (scenario,
  strategy checkboxes, safety bars, horizon, display mode) and triggers a
  cached compute on demand.
- **Path arrays** (historical + bootstrap) are cached via `@st.cache_resource`.
  **Calibration + projection** are cached via `@st.cache_data`.
- **Multi-page setup** (`pages/Documentation.py`) provides a comprehensive
  reference for end users; covers data, math, strategy catalog, calibration
  methodology, and caveats.

### Tax assumption changed: `BOX_TAX_BENEFIT = 0.0` (was 0.20)

For a strict hold-forever SPX investor with no other realized capital
gains, only ~$3,000/yr of Section-1256 60/40 capital losses are usable
(against ordinary income); the rest accumulates as carryforward with
uncertain future value. Setting tax benefit to 0 is the honest
conservative assumption for the hold-forever profile. **Effect:**
loan grows ~20% faster ⇒ T_rec drops ~0.05x for active strategies
(2000-09 path is sensitive to long compounding); strategy ranking
unchanged.

### Defaults adjusted in fourth session

- Stretch factor F default: 1.0 → **1.1** (always defend against 10%-deeper
  drawdowns)
- Bootstrap synthetic paths default: 1000 → **500** (faster compute, still
  enough resolution)
- Wealth cap toggle default: ON → **OFF** (was confusing the cap-vs-glide
  distinction)
- Removed the 4.0x call-threshold horizontal line from leverage chart
  (compressed the visible range)

---

## 5f. Re-calibration strategies + meta_recal (added in fifth session)

The fifth session added strategies that periodically *re-pick* their target
leverage based on the path's current state, modeling the natural use of the
tool: "every N years/months, observe my actual portfolio state and re-decide
my optimal leverage."

### Concept

A re-calibration strategy:
1. Pre-computes a 2D lookup table $T^{*}(E_{\text{real}}, H_{\text{remaining}})$
   over a coarse grid (6 equity × 6 horizon = 36 cells), where each cell
   value is the well-defended max-safe T for the given starting state +
   remaining horizon (well-defended = `min(T_hist@0%, T_boot@boot_target,
   T_stretch@0%-with-stretch-1.1×)` — the same architecture as the rest of
   the app's calibration).
2. At each re-cal event (every `recal_period_days`), looks up the closest
   cell, **resets path-conditional state** (max_dd=0, hwm=current_eq), sets
   `T_active = lookup_value`, and **takes additional loan to bring leverage
   to T_active**. Lever-up only — never sells (matches hold-forever
   constraint, no tax events).
3. Between re-cal events, applies the base strategy's logic (with `T_active`
   in place of `T_init`).

### New code

- **`compute_recal_table(...)`** in `project_portfolio.py` — builds the 2D
  lookup table for a given base strategy (`static`, `dd_decay`,
  `adaptive_dd`, `hybrid`). Optional bootstrap + stretch paths produce the
  well-defended T_max per cell.
- **`compute_recal_tables_multi(..., kinds=[...])`** — wraps the above to
  produce a 3D table (n_kinds × n_e × n_h) for `meta_recal`'s candidate set.
- **JIT kind codes 11-14** in `_simulate_core` and `_simulate_core_grid`:
  - **`recal_static` (11)** — static between recals; recal event lifts to
    lookup `T_max`.
  - **`recal_hybrid` (12)** — hybrid between recals; recal resets `max_dd`
    + `hwm` and lifts `T_active` to lookup.
  - **`recal_adaptive_dd` (13)** — adaptive_dd between recals; same
    reset/lift pattern.
  - **`meta_recal` (14)** — at each recal event, picks the strategy with
    `argmax(T_max[s, e_idx, h_idx])` over candidate slices. Currently
    hard-coded to candidate set {static (s_code=0), dd_decay (s_code=2),
    adaptive_dd (s_code=9), hybrid (s_code=4)}. Between recals, applies
    the chosen strategy's logic with the chosen `T_active`.

### Per-path state additions

- `T_active[k]` (per path) — current target, updated at recal events
- `strat_active[k]` (per path, int) — for meta_recal: which candidate
  slice is currently active
- `is_levered = (T_init > 1.0) or (kind_code >= 11)` — recal kinds always
  treated as levered so loan compounding fires even when initial T = 1.0x
- The is_recal_day check fires for kind ≥ 11 on `d % recal_period_days == 0`

### Streamlit integration

- **Sidebar checkboxes**: `recal_static`, `recal_hybrid`,
  `recal_adaptive_dd`, `meta_recal`.
- **Slider**: `recal period (months)`, range 1-180, default 60 (= 5 years).
  Internally: `recal_period_days = months × 21`.
- **Pre-compute**: tables are computed inside `compute()` only for the
  base strategies actually needed by the selected recal strategies (3D
  meta-table only built if meta_recal is selected). Adds ~5-20s to first
  run depending on which strategies are toggled.
- **Stretched paths**: `ret_s_for_recal = stretch_returns(ret_c, stretch_F)`
  if `stretch_F > 1`, threaded into `compute_recal_table` as `ret_s`.
- **Per-strategy table dispatch**: `recal_kw_for(name)` returns the right
  2D table (static-table for `recal_static`, hybrid-table for
  `recal_hybrid`, adaptive_dd-table for `recal_adaptive_dd`) plus the
  shared 3D + meta_codes for meta_recal.

### Empirical findings (user's scenario, C=160k S=180k×5y S2=30k X=$3M, 30y)

**Period sweep on `recal_static`** (T_init = 1.0x, no calibration):

| Period | Hist calls | Boot calls | p50@30y |
|---|---|---|---|
| 1 month | 27.4% | 33.0% | $21.17M |
| 3 months | 27.4% | 31.4% | $20.85M |
| 1 year | 27.1% | 26.2% | $19.50M |
| 5 years | **14.9%** | **15.0%** | $15.97M |
| 10 years | 7.5% | 8.6% | $14.11M |

**Monthly recal is catastrophic** (resembles relever-with-adaptive-target).
**5y is the knee**; 10y is safest but leaves IRR on the table.

**Well-defended lookup** (cell target = boot_target = 1%, stretch 1.1×) at
5y period, with proper T_init calibration:

| Strategy | T_rec | Hist | Boot | p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| static | 2.140x | 0% | 0% | $5.98M | $10.82M | $19.11M |
| hybrid | 1.661x | 0% | 0.8% | $6.51M | $12.52M | $21.63M |
| recal_static | **1.000x** | 0% | 3.2% | $6.34M | $13.31M | $27.84M |
| recal_hybrid | **1.000x** | 0% | 3.0% | $6.34M | $13.31M | $27.60M |
| recal_adaptive_dd | **1.000x** | 0% | 2.8% | $6.30M | $13.30M | $28.62M |
| meta_recal | **1.000x** | 0% | 3.2% | $6.34M | $13.31M | $27.84M |

### Critical caveats

1. **T_init calibrates to 1.0x for all recal strategies in this scenario.**
   The find_max_safe_T_grid binary search returns the lower bound (1.0x)
   when no T_init in [1.0, 3.0] satisfies ≤1% bootstrap calls. This means
   **the strategy starts unleveraged for years 0-5 (until first recal
   event)**, missing the heavy-DCA window where plain static can safely
   take 2.14x. **This is a real limitation** — the recal strategies
   under-utilize the early-horizon high-DCA period. Open question (high
   priority): should there be a separate calibration of the pre-recal-phase
   T_init?

2. **Cell-level vs strategy-level safety distinction.** Per-cell safety
   bar = `boot_target` (1% by default). The lookup picks the highest T_max
   that's safe FROM THIS CELL FORWARD assuming static-from-here behavior.
   But the actual strategy goes through ~6 recal events over 30y, and call
   probabilities compound — strategy-level rate ends up ~3% empirically.
   This is the user-accepted trade-off: each individual decision is honest
   by the same standard as the rest of the app; the aggregate is what it
   is. An earlier auto-scaling experiment (cell_target = boot_target /
   n_recal_events) was reverted as over-conservative; the user's spec is
   explicit cell-level safety.

3. **All four recal strategies converge in this scenario.** Well-defended
   T_max values are nearly identical across {static, dd_decay,
   adaptive_dd, hybrid} in the binding cells (range [1.245, 1.612]).
   `meta_recal` correctly picks the max but max ≈ static almost
   everywhere, so it acts like recal_static. Differentiation between
   recal flavors should appear in low-DCA scenarios with deeper or
   longer drawdowns — untested.

4. **State reset at each recal is intentional but architecturally
   debatable.** Resetting `max_dd = 0` and `hwm = current_eq` means each
   recal event starts the strategy "fresh" — the dd_decay ratchet from
   prior crises is forgotten. Alternative (Option B from conversation):
   preserve `max_dd` across recals so the ratchet keeps protecting after
   prior stress. Predicted to lower bootstrap fragility from ~3% to ~1%
   while sacrificing some IRR uplift on quiet paths. **Untested**.

5. **Lookup grid is coarse**: 6×6 cells with nearest-neighbor lookup.
   Equity grid is log-spaced over [0.5×C, 100×C]; horizon grid is
   [5, 10, 15, 20, 25, 30]. Could be made finer at compute cost.
   For robust cells, the existing grid is fine; for high-equity edge
   cases the lookup may be quantized.

### Why recal isn't strictly Pareto-better than plain hybrid

- Plain `hybrid` calibrates `T_init` ≈ 1.66x for the WHOLE 30y trajectory;
  takes that loan day 0 and runs.
- `recal_hybrid` calibrates `T_init` ≈ 1.0x (no day-0 loan), then year-5
  recal levers up to lookup ≈ 1.6x. Loses 5 years of leveraged growth that
  plain hybrid captures.
- p50 difference: $13.31M (recal_hybrid) vs $12.52M (hybrid) → recal wins
  by +6%. p90 difference: $27.6M vs $21.6M → recal wins +28%. Cost: bootstrap
  call rate 3.0% vs 0.8%.

The trade is: trade tail safety for fatter upside (especially p90). The
recal strategy maintains leverage near 1.5-1.6x for years 5-30 instead of
letting it drift to 1.05x like plain static does. Sustained leverage =
sustained compounding upside on lucky paths.

### Open questions for next session

1. **High priority — fix T_init calibration.** Currently bottoms out at 1.0x.
   Possible fixes:
   - Calibrate T_init only for the pre-recal phase (years 0 to first
     recal_period), NOT the full horizon
   - Require `find_max_safe_T_grid` to return the lo bound instead of 1.0
     when no T satisfies target, so we know calibration "failed" and can
     handle accordingly
   - For the heavy-DCA scenario specifically, T_init = 2.14x (static's
     calibrated value) is probably right since years 0-5 are static-mode
2. **Test on different scenarios**: low-DCA (e.g., C=$1M S=$30k/yr), longer
   horizon, different `wealth_X`. Should differentiate the recal flavors
   from each other.
3. **Option B (preserve dd ratchet across recals)** — empirically test
   whether keeping `max_dd` permanent across recals reduces bootstrap call
   rate without sacrificing too much IRR.
4. **Adaptive recal_period** — longer after observed crashes, shorter
   during calm. Couples recal frequency to a state variable.
5. **Joint-defended sizing**: instead of per-cell ≤1%, find the T_init
   such that the *strategy as a whole* has ≤1% bootstrap calls. Likely
   produces lower IRR than current approach but matches the rest-of-app
   convention more cleanly.
6. **Document recal strategies in `pages/Documentation.py`** — the
   strategy catalog there does NOT yet include the 4 recal kinds. Worth
   adding entries in the same format as existing strategies. (Done in
   this session.)
7. **Checkpoint capture** currently captures pre-rebalance leverage; for
   recal days this misses the lift. Consider capturing both, or
   post-rebalance for recal kinds.

### How to use re-calibration in practice

Less algorithmic, more operational: every 2-3 years (or on major life
events / wealth milestones / observed drawdowns), re-run the simulator
with current portfolio state and decide a fresh strategy. The
`recal_static` / `recal_hybrid` strategies *simulate* this workflow
algorithmically, but the real workflow is for the user to do it
manually with up-to-date data.

---

## 5g. Recal T_init calibration fix (added in sixth session)

Open question #1 from §5f resolved. Spec at
`docs/superpowers/specs/2026-04-26-recal-tinit-fix-design.md`,
implementation plan at
`docs/superpowers/plans/2026-04-26-recal-tinit-fix.md`.

### Problem

Pre-fix, `find_max_safe_T_grid(kind="recal_static", ...)` ran the binary
search over the *full 30y trajectory including all recal events*. The
recal events themselves produce ~3% bootstrap calls at well-defended
per-cell T_max (1% per cell, ~6 events ⇒ multi-event compounding),
regardless of T_init. The binary search therefore bottomed out at the
lower bound (1.0x). Result: `recal_X` strategies ran unleveraged for
years 0 → first recal, missing the heavy-DCA early window where plain
`static` calibrates to ~2.0x.

### Fix

For each recal strategy, calibrate T_init using the **base kind**, not
the recal trajectory:

| Recal strategy | Base kind for T_init |
|---|---|
| `recal_static` | `static` |
| `recal_hybrid` | `hybrid` |
| `recal_adaptive_dd` | `adaptive_dd` |
| `meta_recal` | `argmax(p50 real terminal wealth)` over `{static, dd_decay, adaptive_dd, hybrid}` (see "Meta_recal pick rule" below) |

A new `init_strat_idx` int parameter was threaded through `_simulate_core` /
`_simulate_core_grid` and the Python wrappers (`simulate`, `call_rate`,
`find_max_safe_T_grid`). For `meta_recal`, this initializes `strat_active[k]`
to the winning base kind's index in `META_KINDS = ["static", "dd_decay",
"adaptive_dd", "hybrid"]` so the years-0-to-first-recal phase applies the
chosen strategy's between-recal update rule.

### Meta_recal pick rule: p50 real terminal wealth over wealth-aware bases

`meta_recal` evolved through three iterations during this session:

**v1: argmax(T_rec) over {static, dd_decay, adaptive_dd, hybrid}.**
Picked static almost always (highest T) but wasted leverage to DCA
dilution — at T=2.025x on $160k with $180k/yr DCA, leverage drifts to
~1.12x in 5 years and stays low. Effectively no different from plain
static.

**v2: argmax(p50 real terminal wealth) over the same four candidates.**
`compute_recal_table` now returns `(t_table, score_table)` where
`score_table[i, j]` = p50 real terminal wealth on calibration paths at
the cell's well-defended T_max. `compute_recal_tables_multi` returns
`(t_3d, score_3d)`. The JIT meta_recal block uses `meta_score_tables`
(passed alongside `t_recal_tables_meta`) for the pick. The same metric
drives meta_recal's year-0 T_init calibration in `app.py compute()`.

v2 picked `adaptive_dd` @ 1.496x for the user's scenario, giving
~$13.19M p50 (vs $10.67M for static). But `adaptive_dd` doesn't honor
`wealth_X` — leverage stayed at ~1.3x even at year 30 when median
wealth was already $14M, well past the user's $3M unlever target.

**v3: argmax(p50 terminal wealth) over wealth-aware candidates
{static, hybrid, adaptive_hybrid}.** Drops `dd_decay` and `adaptive_dd`
from the candidate set because they ignore `wealth_X`. The three
remaining candidates all naturally drift to floor at the wealth target
(`static` via DCA dilution, `hybrid` and `adaptive_hybrid` via internal
glide). Cell scores are self-consistent (same wealth glide assumption
for both score computation and runtime behavior).

Empirical comparison, user's primary scenario:

| Base kind | T_rec | p50 terminal | In meta candidates? |
|---|---|---|---|
| static | 2.025x | $10.67M | yes (v3) |
| hybrid | 1.661x | **$12.52M** | yes (v3) — winner |
| adaptive_hybrid | 1.661x | $12.40M | yes (v3) |
| adaptive_dd | 1.496x | $13.19M | dropped (no wealth_X) |
| dd_decay | 1.479x | $13.02M | dropped (no wealth_X) |

v3 meta_recal picks `hybrid` @ 1.661x. Note that `adaptive_hybrid` and
`hybrid` have identical T_rec — at T=1.66x the wealth glide already
deleverages aggressively, so adaptive_dd's cushion-aware F adds nothing.
Plain `hybrid` is slightly more efficient ($12.52M vs $12.40M).

**Trade-off accepted in moving v2 → v3:** ~5% lower p50 terminal wealth
(meta_recal $12.52M instead of $13.19M) in exchange for honoring the
user's $3M unlever target. For a hold-forever investor with a wealth
goal, this is the right trade.

**Implication for the recal mechanism:** with the wealth-aware candidate
set, recal lifts at year 5/10/15/20/25 still re-lever to the cell's
T_max — but the cell's T_max is computed for hybrid/adaptive_hybrid (NOT
static/dd_decay/adaptive_dd), so the wealth glide is baked into the
table values. As the user's wealth grows, the cell T_max naturally
declines (because the cell scores assume wealth_X kicks in), so the
recal lifts get smaller over time and approach the floor near $3M.

**v4 (current): myopic score — p50 wealth at NEXT RECAL EVENT, not full
horizon.** v3's full-horizon score commits to a 30y EV view at every
recal, which doesn't match the user's actual mental model of "decide
what's best for the next 5y, re-evaluate later." v4 changes the score
horizon from `h_days` (cell remaining) to `min(recal_period_days,
h_days)`. Last-segment cells (where remaining < recal_period) still
score at terminal — degrades correctly.

Implementation: `compute_recal_table` accepts `score_horizon_days`
(None = full horizon, int = myopic). `app.py` passes
`score_horizon_days = recal_period_days`. The meta_recal T_init score
calc in app.py mirrors (uses `min(recal_period_days, max_days)`).

Empirical impact for the user's primary scenario (5y recal period):

| Base | T_rec | v3 score (p50@30y) | v4 score (p50@5y) |
|---|---|---|---|
| static | 2.025x | $10.67M | $1.43M |
| hybrid | 1.661x | $12.52M (winner) | $1.46M (winner) |
| adaptive_hybrid | 1.661x | $12.40M | $1.45M |

Margins are tighter under v4 (~2% spread vs ~17% at full horizon). This
means meta_recal is MORE sensitive to small EV differences and may switch
base strategies between recal events as cell state evolves. That's the
design intent — locally optimal picks per segment.

Year-0 winner is still `hybrid` for the user's scenario. The picks at
later years (e.g., year 5/10/15) are now driven by 5y-window EV
expectations from those cells, not full-remaining-horizon EV. Empirically
should: pick higher-T strategies when there's heavy DCA still flowing
and a long way to wealth_X (front-loaded EV), and shift toward more
conservative picks as wealth approaches wealth_X (where EV plateaus).

Trade-off vs v3: locally optimal ≠ globally optimal. A myopic pick can
"burn through" a strategy's edge in one segment and leave a worse state
for future segments. Empirical question whether this matters in practice;
TBD by running both modes and comparing 30y wealth distributions.

Notes for future work:
- This is an EV pick, not a tail-aware pick. Could expose percentile
  as a parameter (p25 or p10 for risk-averse, p50 for neutral, mean for
  aggressive).
- Currently meta_recal can still SWITCH base strategies between recals.
  Worth visualizing in the UI which base is active in which segment.
- Could expose `score_horizon_mode` ∈ {myopic, sliding_window, full}
  as a sidebar option for users who want to compare.

### Verified empirical impact (user's primary scenario)

`C=160k, S=180k×5y, S2=30k, X=$3M, 30y, F=1.5, stretch=1.1, boot=1%`:

| Strategy | Pre-fix T_rec | Post-fix T_rec |
|---|---|---|
| static (plain) | 2.025x | 2.025x (unchanged) |
| hybrid (plain) | 1.661x | 1.661x (unchanged) |
| adaptive_dd (plain) | 1.496x | 1.496x (unchanged) |
| dd_decay (plain) | 1.479x | 1.479x (unchanged) |
| **recal_static** | **1.000x** | **2.025x** |
| **recal_hybrid** | **1.000x** | **1.661x** |
| **recal_adaptive_dd** | **1.000x** | **1.496x** |
| **meta_recal** | **1.000x** | **2.025x (init: static)** |

Recal strategies now Pareto-dominate plain strategies: identical year-0
behavior, plus re-leveraging at year 5 / year 10 / etc. via the lookup
table when state is favorable. Strategy-level bootstrap call rate stays
~3% (per-cell ≤1% × ~6 recal events), accepted as the cost of
multi-event exposure (see §5f for the cell-vs-strategy distinction).

### What's still open

The five other open questions from §5f are unchanged:

- Option B: preserve `max_dd` ratchet across recal events
- `recal_period_months < T_yrs` lookup-table-DCA mismatch
- Adaptive `recal_period`
- Joint-defended sizing (strategy-level ≤1% boot)
- Test on different scenarios (low-DCA, longer horizon)

### Verification

`verify_recal_tinit.py` and `verify_recal_end_to_end.py` reproduce the
post-fix calibration values without going through Streamlit. The end-to-end
script asserts the four invariants and exits non-zero on regression — useful
as a regression test for any future changes to the calibration logic.

### Process notes for next session

- The user prefers **option B** when faced with an entangled-history git
  situation (multiple uncommitted changes from a prior session getting
  swept into a focused commit). Soft-reset, split into logical units,
  re-commit. Don't bundle unrelated work into a misnamed commit.

---

## 5h. Early broker-rate bump (added in seventh session)

Models the operational reality that box-spread financing (Tsy + 15 bps,
the project's default) requires options-trading approval and operational
setup that typically isn't in place on day 0. Most retail investors will
spend the first months-to-years on plain broker margin (Tsy + 150 bps,
no Section 1256 tax benefit) before the box infrastructure is live.

### Mechanism

- New constant `BROKER_BPS = 0.0150` in `project_portfolio.py`.
- New parameter `broker_bump_days` (int, default 0) threaded through
  `_simulate_core`, `_simulate_core_grid`, `simulate`, `call_rate`, and
  `find_max_safe_T_grid`. When `d <= broker_bump_days` the daily loan
  growth uses `tsy + BROKER_BPS` (no tax benefit); otherwise the existing
  `(tsy + BOX_BPS) * (1 - BOX_TAX_BENEFIT)` rule.
- `compute_recal_table` / `compute_recal_tables_multi` deliberately do
  NOT take the parameter — their internal sub-simulations always run with
  the default 0. This is correct: each cell represents "from year-X
  forward with H_remaining left," which in real time corresponds to year
  5+ where the bump is already over. Applying the bump inside the cell
  calibration would double-count it.
- New sidebar slider in `app.py` ("Financing" expander, "Broker-rate
  years (Tsy+150 bps)", 0–5y, default 2) wires through `compute()` as a
  cache key and lands in `overlay_kw` for every top-level call. The
  meta_recal year-0 score sim correctly inherits the bump (it represents
  years 0..recal_period, the actual early phase).

### Empirical impact (user's primary scenario, hybrid, F=1.5, stretch=1.1, boot=1%)

| bump | T_rec (T_h, T_b, T_s) | p50 @ 30y | vs unlev |
|---|---|---|---|
| 0y (idealized) | 1.661x (1.959, 1.661, 1.826) | $12.52M | +32.2% |
| 2y (default) | 1.661x (1.942, 1.661, 1.810) | $12.46M | +31.7% |
| 5y (no box ever) | 1.661x (1.942, 1.661, 1.793) | $12.38M | +30.8% |

Same scenario, plain static (T_rec ≈ 2.025x): bump=0y → $10.67M, bump=2y
→ $10.66M, bump=5y → $10.66M (essentially flat).

### Key findings

1. **T_rec barely shifts under the bump** because the binding constraint
   (bootstrap fragility from 2000-09–style multi-year drawdowns) doesn't
   fire in years 0–2 / 0–5. Only T_h and T_s drift down ~15 bps with bump
   = 2y; T_b (binding for hybrid) is unchanged. Static's T_rec is
   completely unchanged at 3 decimal places.
2. **Cost of a 2-year bump ≈ 0.5% of median terminal real wealth**
   (~$60K on a $12.5M projection). 5y ≈ 1.1%. The cost compounds
   linearly in bump-years to first order — modest per year because the
   135 bps differential applies only to the loan principal, which is
   small relative to the equity stack at year 0 and even smaller after
   DCA dilution.
3. **Leverage is comfortably worth it even at broker rates.** Hybrid
   bump=5y still beats unlev by +30.8%. The "should I wait for box-spread
   approval before levering?" question has a clean answer: no — the
   option-value of starting compounding sooner dominates.
4. **The dominant cost of waiting for box is opportunity cost, not
   interest cost.** A 2-year delay in initiating leverage costs more
   than 2 years of paying broker rate, because the early years are when
   DCA-driven leverage dilution is fastest — every month un-levered in
   the heavy-DCA window is a month where unlevered DCA dollars are
   stacking up that you can't lever up later.

### Practical guidance update

Tier 1 / Tier 2 recommendations in §6 are unchanged. The only addition:
**don't delay starting leverage to wait for box-spread setup.** Start
on broker margin from day 0; switch to box once the operational pieces
(options approval, mechanics) are in place. The cost of waiting >>
the cost of paying broker rate during the transition.

### Note for future sessions

- The bump is a top-level UI parameter, NOT a default in
  `project_portfolio.py`'s CLI. CLI users still get bump=0 unless they
  pass `broker_bump_days` explicitly. CLI args could be added if/when
  needed — currently this is a Streamlit-only UX feature.
- `verify_recal_tinit.py` and `verify_recal_end_to_end.py` continue to
  pass without modification (they don't pass `broker_bump_days`, so they
  exercise the default 0 path — exactly the regression test we want).
- If the user ever asks "what if my broker rate is different from
  Tsy+150" — currently this is a hardcoded constant. Refactoring to a
  parameter (settable from the UI) is straightforward but un-done.
- The bump only fires for `d <= broker_bump_days`. Day 0 is a degenerate
  case (no compounding happens on d=0 anyway since the inner loop starts
  at d=1), so the boundary is consistent.

---

## 5i. Calibration uses avail-bounded all-entries paths (added in eighth session)

Fix to a horizon-sensitivity artifact in the Streamlit calibration. User
observed that lowering the horizon slider from 30y to 15y *lowered* T_rec
for hybrid (and identically for static, dd_decay) — counterintuitive.

### Diagnosis

`app.py` builds two historical path sets:
- `ret_c` = `build_historical_paths(..., min_days=max_days)` — only
  entries with the FULL nominal horizon of forward data
- `ret_h` = `build_historical_paths(..., min_days=TD)` — all post-1932
  entries, with each path's available-day count tracked in `avail_h`

Pre-fix, the historical (T_hist) and stretched-historical (T_stress)
safety bars used `ret_c`. So at 30y horizon, only entries from
1932-07 → 1996-04 (~763 paths) were eligible — entries from 1996-04
onward were silently excluded because they don't yet have 30 years of
forward data. At 15y horizon, entries from 1932-07 → 2011-04 (~943
paths) were eligible, including 2008-08 entries that hit the GFC in
months 0–7.

Result: the 15y "max-safe" stress bar caught the 2008-08 stress
(stretched ~55% drawdown ≈ binding leverage in year 1) at T = 1.942x.
The 30y bar didn't see 2008-08 at all and reported T = 2.025x. The
30y number wasn't "less conservative because longer horizon" — it was
artificially over-confident because it couldn't test recent stresses.

### Fix

`app.py compute()` switched the historical and stretch
`find_max_safe_T_grid` calls to use `ret_h, tsy_h, cpi_h, avail_h`
(plus `ret_s = stretch_returns(ret_h, stretch_F)`). The simulate
engine already supported per-path availability via `if d > avail[k]:
continue` in `_simulate_core` line 484 — paths are simulated only
through their available days, then masked. Capability was already
there; calibration just wasn't using it.

Bootstrap (T_boot) is unchanged — it generates synthetic horizon-length
paths and was never affected by this issue.

The terminal-wealth distributions (per-strategy percentile tables, fan
charts) continue to use `ret_h` for projection (already did, at
`app.py:603`+ display loop), so those numbers don't change either —
this fix only affects the calibration leg.

`ret_c` is still kept and used for two specific purposes that need
full-avail entries for fair comparison:
- meta_recal score computation (`app.py compute()` ~line 631) — p50
  real terminal wealth at the next recal event, fair across candidates
- recal lookup table builders (`compute_recal_table`,
  `compute_recal_tables_multi`) — cell scoring uses
  `eligible = avail_h >= h_days` filtering

### Empirical impact (user's primary scenario)

`C=160k, S=180k×5y, S2=30k, X=$3M, F=1.5, stretch=1.1, boot_target=1%, broker_bump=2y`:

| Horizon | hybrid pre | hybrid post | binding (pre) | binding (post) |
|---|---|---|---|---|
| 30y | 2.025x | 1.942x | 1937-07 stretch | 2008-08 stretch |
| 20y | 2.025x | 1.942x | 1937-07 stretch | 2008-08 stretch |
| 15y | 1.942x | 1.942x | 2008-08 stretch | 2008-08 stretch |
| 10y | 1.942x | 1.942x | 2008-08 stretch | 2008-08 stretch |

Static and dd_decay also collapse to 1.942x at every horizon. They have
identical T_rec because the binding event (2008-08) hits in the first
year — before any deleveraging mechanism diverges from static behavior
(dd_decay's ratchet hasn't fired, wealth glide barely moved from C).

The fix DROPPED the published 30y T_rec from 2.025x → 1.942x for the
user's primary scenario. This is the honest answer; the prior 2.025x
was over-confident.

### Caveats / nuance

1. **Asymmetric inclusion.** Avail-bounded entries can only LOWER T_rec,
   never raise it: a recent entry with 6y of avail can fail in those 6y
   (binding the bar), but if it survives those 6y we don't know whether
   it would survive years 7–30. So this fix is one-way conservative for
   recent entries. Acceptable trade.
2. **No min_days floor needed.** Considered a `min_days = max_days // 6`
   floor (5y at 30y, 2.5y at 15y) to keep too-recent entries from
   contributing single-year noise. Decided against — short-avail paths
   that don't bind don't hurt anything; the avail mechanism handles it
   correctly.
3. **Strategy convergence at user's scenario is real, not a bug.**
   Hybrid/static/dd_decay all show T_rec = 1.942x because the binding
   event is too early for them to differ. Differentiation would appear
   in low-DCA scenarios where wealth_X / dd_decay ratchet have time to
   act before the binding event.
4. **`verify_recal_tinit.py` and `verify_recal_end_to_end.py` still pass
   without modification** because they call `find_max_safe_T_grid`
   directly with `ret_c`-style paths (`min_days=max_days`). They check a
   different invariant (recal == base T) which is unaffected. If you
   ever want to verify the post-fix calibration matches the streamlit
   output, the verify scripts would need to be updated to use `min_days=TD`
   like the app does.
5. **What you'd need to undo this:** revert four `ret_h → ret_c` swaps
   in `app.py compute()` and the `stretch_returns(ret_h, ...)` line.
   Plus the docstring on those few lines.

### Standing rule

The historical / stretch safety bars should always test every post-1932
entry that has any forward data, with simulation bounded by per-path
avail. T_rec is a property of "what the data could falsify" — not "what
the data with 30y of forward data could falsify."

---

## 6. Practical recommendations

For the user's stated situation (taxable, hold-forever, SPX only, broker margin, with ongoing savings contributions):

### Tier 1 — simplest, most architecturally defensible (PRIMARY RECOMMENDATION)

**Static loan ~50% (target 1.51x) on broker margin, interest compounds, DCA all your new income into more SPX. Use the well-defended target, not the historical max-safe.**
- Expected IRR uplift over pure DCA: ~0.8-1.0 pp/yr (p50 IRR 11.61% at 30y vs ~10.8% unlev)
- Synthetic-path call rate (bootstrap): **0.40%** — by far the most robust strategy
- Zero historical margin calls at cap=3.0x and cap=4.0x
- No active management — completely passive
- Naturally deleverages to ~1.05x by year 30 via market growth + DCA dilution
- Box-spread financing saves another 100-150 bps/yr after tax

This is the most architecturally clean strategy: single parameter, closed-form risk math, no path-dependent decisions, lowest fragility under bootstrap. **For nearly all users, this should be the default.**

### Tier 2 — moderate yield boost via drawdown-coupled rebalancing (SECOND CHOICE)

**Drawdown-decay F=1.5–2.0 at well-defended target 1.46–1.48x.**
- +1.2 pp/yr IRR uplift over pure DCA (p50 IRR 12.0% at 30y)
- Synthetic call rate ~1% (well-defended) — bootstrap-binding constraint
- Architecturally cleanest active strategy: target stays at T_initial through quiet periods, ratchets DOWN once a drawdown is observed, never goes back up. Couples to drawdown depth (the actual risk variable that triggers margin calls).
- Requires monthly rebalancing and tracking running drawdown
- Pareto-dominates wealth-decay and time-decay (better bootstrap robustness at similar/better IRR)
- Ends nicely deleveraged (~1.05x by year 30) due to ratchet

### Tier 3 — historical max-safe (USE WITH CAUTION)

**Any strategy at the historical dual-horizon max-safe target.** Provides ~13% p50 IRR but at meaningful synthetic call rates (1.5-14% depending on strategy). The "extra" 100-130 bps over Tier 2 is largely artifact of overfitting to historical path ordering. **Only justified if you genuinely believe future paths will closely resemble post-1932 history**, which is a strong assumption.

### Things to avoid

- **Futures-based leverage** in a taxable hold-forever account (annual Section 1256 mark-to-market)
- **Simple re-lever above 1.50x target** (call rates explode)
- **External cash reserve as backstop** (doesn't work for long drawdowns)
- **Aggressive amortization** (suboptimal EV vs DCA)
- **Over-complicating with DCA timing tweaks** (the expected benefit is ~2-15 bps/yr)

---

## 7. Caveats and limitations

1. **Daily closes only.** Intraday breaches of the 4.0x threshold aren't captured. Real margin calls can trigger mid-day on volatile days.

2. **Broker-right-to-tighten not modeled.** During stress, brokers can raise house maintenance requirements above the 25% we use. This happened in 2008 and 2020. An analysis at 3.0x "effective" call threshold would be more conservative.

3. **Execution costs ignored.** Monthly re-levering assumes zero spreads, zero commissions. Realistic cost: ~1-5 bps/yr.

4. **Post-1932 only.** The 1929-1932 era is excluded because it's unrepresentative of modern market structure. If you believe such an event is possible, the max-safe ceiling drops to ~1.12-1.14x.

5. **Rate regime variation.** The dataset includes 1970s stagflation (margin rates above 17%). Our box-spread cheaper-by-25bps assumption is a simplification; actual box rates fluctuate more.

6. **Survivorship bias in the US market.** The SPX backtest uses a market that (in hindsight) was one of the best-performing in the 20th century. Forward returns may be lower.

7. **30-year sample bias.** The 30-year horizon sample excludes post-1993 entries, so it misses any entry that doesn't have 30 years of realized future data yet. For honest stress testing, use the 20-year horizon.

8. **Tax rates and deduction rules may change.** Interest deductibility, section 1256 treatment, and LTCG rates all have legal dependencies that may shift.

9. **No behavioral modeling.** The analysis assumes perfect execution of the strategy. In practice, market panic during drawdowns leads to rule-breaking.

10. **Single-asset (SPX).** No modeling of factor tilts, international diversification, or different asset classes. These were discussed but not quantified.

---

## 8. Important corrections made during analysis

### Sign error in margin-call detection (analyze_leverage_and_financing.py)

An early version had the call threshold inverted:

```python
# WRONG — had this initially
factor = 3.0 * L0 / (4.0 * (L0 - 1.0))

# CORRECT
factor = 4.0 * (L0 - 1.0) / (3.0 * L0)
```

The call condition is `R_t / R_i <= 4*(L_0 - 1) / (3*L_0)`. The inverted version reported 100% call rates for all leverages, which was clearly wrong. Fixed.

### CAGR-with-contributions inflation

Initially reported "CAGR" for strategies with DCA as `terminal^(1/H) - 1`, treating the $1 initial equity as if it alone grew to the terminal value. With DCA contributions, this double-counts the contributions as growth.

**Sanity check that revealed the bug**: if SPX returns exactly 9%/yr steady and DCA is 10%/yr, true IRR = 9% (by definition, nothing earns more than 9%). But the formula reported 11.67% CAGR for a 30-year horizon. The 2.67 pp gap is the phantom inflation.

**Fix**: use IRR, solved numerically via `scipy.optimize.brentq` on `1*(1+r)^H + (m/12)*((1+r)^H - 1)/((1+r)^(1/12) - 1) = terminal`. See `analyze_irr_correction.py`.

Consequence: **DCA does not change rate of return, only wealth**. Earlier statements like "DCA is a bigger driver than leverage" were wrong comparisons.

### Tax treatment of box spreads

Initially claimed box spreads hurt tax-wise for hold-forever. **That was wrong**. Short box spreads generate deductible 60/40 capital losses equal to the implicit interest cost. For retail users on standard deduction, box spreads typically beat broker margin by 100-150 bps after tax.

### Amortization ceiling recommendation

Initially suggested "50-70% 10y amort" as the best choice if income stops at year 10. **That was wrong.** Rigorous simulation (in `analyze_income_stops.py`) showed 41% lump with interest compounding still wins even when income stops at year 10, because SPX return > box rate means the compounding-loan-but-more-exposure position continues to beat the debt-free-but-less-exposed position.

### Claim that "decay strategy ends unlevered so it's safest"

**That was wrong.** Static actually ends the LEAST levered after 30 years (natural asset-growth dilution). Decay strategies re-lever during bull runs, so they retain MORE leverage at year 30. The user's preference of "ending unlevered" is satisfied most strongly by static, not by decay. Decay strategies give up that property in exchange for higher initial leverage capture.

### Claim that cohort/vintage decay would Pareto-dominate single-account decay

**That was wrong.** I predicted (in third session) that giving each DCA contribution its own max_dd ratchet would produce a Pareto improvement: pure investment-performance signal per cohort, no DCA contamination, fresh leverage on new money. Empirically it's worse: cohort+drawdown-decay has bootstrap call rate 5-6% vs 2-4% for single-account at similar IRR. Reason: the "fresh meat" problem — post-crisis cohorts start at max_dd=0, re-lever aggressively during recovery, and have no protection when the next crisis arrives. Single-account permanent-learning across all dollars is structurally safer for sequence-of-crises risk. The intuitive appeal of "new money should get fresh leverage" misses that "fresh leverage" = "no learned safety" = vulnerable to upcoming crashes.

### Claim that static max-safe should always beat lower-leverage static in IRR

**That was wrong.** Pushing static to max-safe with high DCA *decreases* median IRR. Example at 30y: static at 10% DCA / 1.612x gives 11.76% IRR; static at 30% DCA / 1.756x gives 11.46% IRR. Unlevered DCA dollars dilute the leveraged exposure faster than the higher starting leverage captures.

### Sizing strategies at the 20y horizon only (now corrected)

**That was wrong for re-lever strategies.** The original max-safe table sized monthly relever at 1.45x using only 20y entries. At 30y, that target produces a 9.27% margin-call rate — paths that survived 20 years got hit by *additional* drawdowns in years 20–30 (e.g., a path entered in 1980 catches stagflation + 1987 + 2000 + 2008). The fix is "dual-horizon max-safe": a target with 0% calls at BOTH 20y and 30y. This dropped monthly relever's safe target from 1.45x to ~1.43x and revealed that **simple relever's safe target is essentially flat at ~1.43x across all DCA levels** — DCA does not raise the relever ceiling under honest stress testing. Static and decay-2pp are unaffected (they were already safe at both horizons). See `analyze_irr_percentiles.py` (third-session update) and `analyze_end_leverage.py`.

---

## 9. Open questions & future work

Topics discussed but not yet fully simulated. Ranked roughly by promise:

### High-promise directions building on current work

1. **Hybrid time + wealth decay.** Target = `min(time_decay(t), wealth_decay(max_wealth))` or `max(...)` — combines time-decay's scheduled protection with wealth-decay's drawdown pause. Likely gives slightly better safety/IRR than either alone.

2. **Contribution-adjusted wealth measure.** Current wealth-decay counts DCA contributions as "wealth growth." An alternative: measure wealth as `equity / (1 + cumulative_contributions)` — pure investment performance. More accurately matches "my investment has grown" intuition. Would shift the wealth multiple interpretation (e.g., WM=5x means "my INVESTMENT has quintupled," not "my account has quintupled partly via deposits").

3. **Wealth-decay + loan cap combined.** Use wealth-decay for the target schedule but impose a hard cap on cumulative loan (e.g., 2× initial). Dual safety rail.

4. **Band rebalancing on wealth-decay.** Only re-lever when leverage drops below `band × current_target` (e.g., 90%). Reduces re-lever frequency; might safely permit higher initial targets.

5. **Quarterly wealth-decay.** Less frequent rebalance. May have different safety/return profile.

### Previously listed (still open)

6. **Options-based downside protection.** Long-dated OTM SPX puts as insurance for higher leverage (e.g., 1.75-2.0x). Would require options pricing data we don't have.

7. **Collar strategies.** Buy put + sell call to finance. Caps upside but enables safer high leverage.

8. **Factor tilts.** Small-cap, value, momentum, international. Historical 1-4%/yr risk premia. Different category of bet than leverage.

9. **Withdrawal phase.** All analyses assume either "no withdrawals" or "contributions stop." What if you're withdrawing for living expenses during retirement?

10. **Tax-loss harvesting with leverage.** Realize SPX losses during drawdowns to offset leveraged gains, then repurchase? Wash-sale rules apply.

11. **Different box-spread rates in stress.** Historical box rates widened during 2008 and 2020. We've assumed static Tsy + 15bps.

12. **Multi-asset diversification with leverage.** 60/40 portfolio with 1.5x leverage could have better risk-adjusted returns than 100% SPX at 1.4x leverage. Requires bond return data.

13. **"Constant percentage re-lever."** Instead of target leverage `T`, target "loan as % of equity" at `p`. Behaves differently in bull markets.

### Simulated in second session (no longer open)

- ✅ Inflation-adjusted DCA growth (2.5%/yr) → in `analyze_irr_inflation.py`
- ✅ Wealth-based decay with HWM → in `analyze_wealth_decay.py`
- ✅ Static @ max-safe per DCA (vs hardcoded 1.41x) → updated `analyze_irr_percentiles.py`

### Simulated in third session (no longer open)

- ✅ "Reset time-decay clock on up years" empirically tested and shown to fail → in `analyze_reset_clock.py`
- ✅ End-of-horizon leverage percentiles → in `analyze_end_leverage.py`
- ✅ Dual-horizon (20y AND 30y) max-safe sizing → updated `analyze_irr_percentiles.py`
- ✅ Call-threshold buffer (cap = 3.0x) → in `analyze_call_threshold.py`
- ✅ Drawdown depth stress (F = 1.1, 1.2) → in `analyze_stress_drawdown.py`
- ✅ Block bootstrap synthetic paths → in `analyze_block_bootstrap.py`
- ✅ Joint well-defended sizing (bootstrap + cap) → in `analyze_well_defended.py`

### High-priority new directions from third session

1. **Static + band rebalance.** Static is architecturally cleanest but gives up some IRR. A small active component — only re-lever when leverage drops X% below initial target — might capture some of the IRR upside without inheriting full bootstrap fragility. Untested.

2. ✅ **Drawdown-coupled decay** → empirically confirmed in `analyze_drawdown_decay.py` (third session). Pareto-dominates wealth-decay and time-decay.

3. **Contribution-adjusted wealth-decay** (still open from session 2). Use `equity / (1 + cumulative_contributions)` instead of raw equity. Removes the gameability where DCA mechanically lowers target. Note: drawdown-decay sidesteps this issue entirely (drawdown depth doesn't depend on absolute equity), so this may be lower priority now.

4. **2-year block bootstrap as primary safety bar.** 1y blocks (default) may understate tail risk; 2y blocks produce 11.36% calls vs 10.02% at 1y for time-decay. Worth re-running well-defended sizing with 2y blocks for a more conservative target.

5. ✅ **Cohort/vintage model** → tested in `analyze_cohort_drawdown_decay.py` (third session). Empirically WORSE than single-account due to "fresh meat" problem (post-crisis cohorts re-lever aggressively, get caught by next crisis). Negative result, abandoned.

---

## 10. Notes for future sessions

### Running the scripts

```bash
# From the project directory
/Users/dnissim/projects/margin_simulator/.venv/bin/python analyze_irr_percentiles.py
```

The venv is at `.venv/` with numpy, pandas, openpyxl, and scipy installed. Some scripts take 1-5 minutes for full grids.

### Most important scripts for someone picking up the project

1. **`data_loader.py`** — entry point for data
2. **`analyze_drawdown_decay.py`** — drawdown-coupled decay strategy, the architecturally-cleanest active approach. Pareto-dominates wealth-decay and time-decay.
3. **`analyze_well_defended.py`** — joint-constraint sizing (cap=3.0x + bootstrap≤1%) for the four main strategies. **The most honest numbers in the project.** Includes block-size sensitivity.
4. **`analyze_block_bootstrap.py`** — synthetic-path call rates at historical-safe targets. Single most damning result for "0% calls = safe" claim.
4. **`analyze_irr_percentiles.py`** — clean IRR summary at DUAL-HORIZON max-safe × DCA × horizon
5. **`analyze_end_leverage.py`** — end-of-horizon leverage percentiles + IRR. Quick to run.
6. **`analyze_call_threshold.py`** — broker-tightens stress (cap = 3.0/3.5/4.0)
7. **`analyze_stress_drawdown.py`** — drawdown depth multiplier F = 1.0/1.1/1.2 (note calibration print is misleading, see script docs)
8. **`analyze_wealth_decay.py`** — wealth-based HWM decay strategy; sweeps WM parameter
9. **`analyze_reset_clock.py`** — proof that "reset time-decay clock on up years" fails on the 2000-03-23 entry
10. **`analyze_irr_inflation.py`** — same as `analyze_irr_percentiles.py` but with DCA growing at 2.5%/yr
11. **`analyze_relever_variants.py`** — full comparison of re-lever strategies (monthly/quarterly/annual, bands, caps, decay)
12. **`analyze_clean_comparison.py`** — apples-to-apples of static vs amortization with matched cash flow
13. **`analyze_2000_entry.py`** — reference for the binding worst-case scenario
14. **`analyze_dca_leverage_grid.py`** — grid across DCA × leverage (caveat: uses old CAGR metric not IRR, interpret accordingly — magnitudes inflated by 1-5 pp at positive DCA)

### Key numeric anchors to memorize

- **SPX-TR median IRR post-1932**: ~10.9%
- **Box-spread rate** = 3M Tsy + ~15 bps
- **Broker margin rate** = 3M Tsy + ~40 bps (per our data, likely conservative)
- **Post-1932 dual-horizon max-safe at 0% DCA: 1.425x** for ALL strategies (static, monthly relever, annual relever, decay-2pp). Binding on 2000-03-23 entry, 20y horizon.
- **Full-series static max-safe** at 0% DCA: **1.14x** (binding at 1929-09-16)
- **Portfolio-margin (15%) cap**: 6.67x, but only safe at ~1.49x due to worst-case drawdown
- **Monthly relever dual-horizon max-safe is essentially flat at 1.43x across all DCA levels.**
- **Static and decay-2pp ceilings rise with DCA** (10% DCA → static 1.61x, decay-2pp 1.59x)
- **Bootstrap call rate at historical max-safe (10% DCA, 1y blocks): static 1.46%, drawdown-decay F=2.0 2.64%, wealth-decay 7.84%, time-decay 9.98%, monthly relever 13.80%.** Single most damning figure for "historical 0% calls" claim. Drawdown-decay is the architecturally-cleanest active strategy.
- **Well-defended targets (10% DCA, joint cap=3.0x + bootstrap≤1%)**: static 1.51x, wealth-decay 1.35x, time-decay 1.40x, monthly relever 1.24x. Rough rule-of-thumb: 0.85-0.90× of historical max-safe.
- **Well-defended IRR (10% DCA, p50 30y)**: 11.6% (static) to 12.1% (others) — all four strategies converge to within ~50 bps. Historical IRR ranking is largely artifact of overfitting.
- **Typical IRR uplift from static @ safe vs unlev**: +0.5-1.1 pp over 30y (depends on DCA)
- **Typical IRR uplift from monthly relever @ 1.43x vs unlev**: +2-3 pp over 30y (historical, before bootstrap correction)
- **Typical IRR uplift from decay-2pp @ safe vs unlev**: +1.7-2.4 pp over 30y (historical, before bootstrap correction)

### If asked about "can I use more than 1.43x"

Always clarify the safety basis being used:

**Historical dual-horizon max-safe** (0% calls on actual post-1932 paths at 4.0x cap):
1. **No DCA**: 1.425x for ALL strategies — no, you cannot exceed this.
2. **10% DCA + STATIC**: up to 1.61x. **30% DCA + STATIC**: up to 1.76x. (Static benefits from DCA dilution.)
3. **DCA + simple relever**: stays flat at 1.43x regardless of DCA — re-levering consumes the dilution.
4. **DCA + DECAY-2pp**: up to 1.59x at 10% DCA, 1.62x at 30% DCA.
5. **DCA + WEALTH-DECAY WM=20**: 1.56x at 10% DCA.

**Well-defended targets** (joint historical-cap=3.0x + bootstrap≤1%):
- 10% DCA: static 1.51x, decay-2pp 1.40x, wealth-decay 1.35x, monthly relever 1.24x.
- These are the "honest safety" numbers that don't rely on historical path-ordering luck.

**Above the historical max-safe**: requires either accepting tail-risk, options hedging, or very high DCA. Not recommended without explicit acknowledgement of overfitting.

### Performance tips for running simulations

- Always use `python -u` for long-running scripts; the default buffered output hides progress.
- Binary search for max-safe: 10 iterations gives ~0.003x precision, sufficient for presentation; 16 iterations gives ~0.00004x but doubles runtime. 10 is usually enough.
- For grids over (DCA × leverage × strategy), budget 15-25 min. Use `run_in_background: true`.
- IRR computation via `scipy.optimize.brentq` is the slow part (100us per call × 15k paths = 1.5s per cell). Only compute IRR percentiles for the cells you need; intermediate simulations can report terminal wealth directly.

### Common confusions to preempt

- **"DCA is a bigger driver than leverage"** — this was a WRONG conclusion based on the CAGR-with-contributions bug. DCA grows WEALTH (more total dollars invested) but not RATE OF RETURN. Leverage grows rate of return per dollar.
- **"Static ends unlevered so it's dangerous"** — backwards. Static naturally deleverages via growth + DCA dilution and is the LEAST levered at year 30 among our strategies. Decay strategies retain MORE leverage at year 30 via re-levering.
- **"Max-safe leverage should always increase with DCA"** — true for static and decay-2pp, but NOT for simple monthly/annual re-lever (max-safe stays flat at ~1.43x regardless of DCA under dual-horizon sizing).
- **"I should restart decay during up years"** — defeats the safety purpose of decay. Empirically demonstrated to fail on the 2000-03-23 entry. Don't do this.
- **"If it's safe at 20y, it's safer at 30y"** — false for re-lever strategies. Monthly relever sized at 20y-safe (1.45x) has 9% calls at 30y. Always size for safety at BOTH horizons.
- **"0% historical calls means it's safe"** — false. Bootstrap shows 1.5-14% synthetic call rates at the historical 0% targets. Historical 0% reflects path-dependent luck, not structural safety. Use well-defended sizing (~0.85-0.90× of historical max-safe) for honest safety claims.
- **"Time-decay is just a different flavor of static"** — architecturally false. Time-decay couples to calendar (a non-risk variable) and bootstrap-fails 7× more than static. It's a curve fit to historical crisis timing.
- **"Wealth-decay = better static"** — partially true (state-coupled) but flawed: HWM is backward-looking, gameable by DCA, and the underlying claim is utility (preference) not risk (mechanics). Bootstrap call rate 7.84% vs static's 1.46% reflects residual fragility.
- **"Active management beats passive for leveraged SPX"** — false at honest sizing. Active strategies' apparent IRR edge is largely overfitting; under bootstrap or well-defended constraints, all four strategies converge to ~12% p50 IRR (10% DCA, 30y). Static is the architectural-physics answer.

### Common user pitfalls to watch for

- Confusing CAGR and IRR when contributions are present.
- Treating leveraged returns as if they're linear in leverage (they're not — volatility drag kicks in).
- Assuming brokers will let you run portfolio margin (15%) safely — they can tighten mid-crisis.
- Forgetting that the "nice" re-lever strategies work because of decay/caps, not re-lever itself.
- Overweighting the average/median IRR without looking at p10 (downside).

### Style conventions

- `L_0` = initial total leverage ratio (1 + loan/equity). So `L_0 = 1.41x` means loan of 41% of initial equity.
- `loan_frac` = fraction of initial equity borrowed. `L_0 = 1 + loan_frac`.
- DCA specified as fraction of INITIAL equity per year (e.g., `0.10` = 10%/yr of starting equity).
- Call threshold = 4.0x (conservative Reg-T assumption).
- Horizon always in trading years (252 days/year).
- Monthly events happen every 21 trading days.

---

## Appendix: Closed-form reference

### Peak leverage (no DCA, one-time loan)

```
L_peak(i, T) = L_0 / (L_0 - (L_0 - 1) * R_i / min_{t in [i,T]} R_t)
```

where `R_t = px[t] / M_loan[t]` and `M_loan[t] = cumprod(1 + rate/252)`.

### Max L_0 for given call cap

```
L_0_max(i, cap) = cap * R_i / (cap * R_i - (cap - 1) * min_future_R)
```

### Setting L_peak = cap (e.g., 4.0)

```
min R_t / R_i = (cap - 1) * L_0 / ((cap * (L_0 - 1)))   # equivalently:
L_peak = cap when R_t / R_i = 4*(L_0 - 1) / (3 * L_0)  (for cap = 4)
```

### IRR equation (monthly DCA)

Solve for `r`:

```
1 * (1+r)^H + (m/12) * ( (1+r)^H - 1 ) / ( (1+r)^(1/12) - 1 ) = terminal
```

where `m` = annual DCA amount, `H` = horizon in years, `terminal` = end-of-horizon wealth.

Use `scipy.optimize.brentq(f, -0.5, 1.5)`.

### Re-lever (add loan to restore target)

```
ΔD = max(target_leverage * equity - assets, 0)
```

### Time-decaying target

```
target(year_t) = max(target_initial - decay_rate * year_t, 1.0)
```
