"""Documentation page for the margin simulator.

Comprehensive in-depth reference covering data, math framework, every
strategy, calibration methodology, and caveats. Pure markdown — no
computation, no parameters.
"""

import streamlit as st

st.set_page_config(page_title="Documentation", layout="wide")
st.title("Strategy Documentation")
st.caption(
    "Reference for the leveraged-SPX hold-forever margin simulator: data, "
    "math framework, every strategy, calibration, and caveats."
)

# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------

st.markdown("""
**Contents**

1. [Scope & assumptions](#1-scope-assumptions)
2. [Data](#2-data)
3. [Mathematical framework](#3-mathematical-framework)
4. [Sizing & safety bars](#4-sizing-safety-bars)
5. [Bootstrap methodology](#5-bootstrap-methodology)
6. [Drawdown stretch](#6-drawdown-stretch)
7. [Strategy catalog](#7-strategy-catalog)
8. [Wealth cap mechanism](#8-wealth-cap-mechanism)
9. [Architectural ranking](#9-architectural-ranking-key-findings)
10. [Caveats & limitations](#10-caveats-limitations)
11. [Glossary](#11-glossary)
""")

st.divider()

# ---------------------------------------------------------------------------
# 1. Scope & assumptions
# ---------------------------------------------------------------------------

st.header("1. Scope & assumptions")
st.markdown("""
This simulator answers a narrow question: **how much margin leverage can a
long-term SPX investor safely take, what's the expected return uplift, and
how does the answer change with ongoing contributions, re-levering, and
different decay rules?**

The user profile assumed throughout:

- 100% SPX in a **taxable brokerage account**
- Plan to **hold forever** — leverage by borrowing against SPX, never sell SPX
- **Broker margin** financing (not futures, not LEAPS) — though box-spread
  rates are used as a tax-efficient proxy
- Some ongoing **savings contributions ("DCA")** flowing into the same account
- All projections in **REAL dollars** (today's purchasing power), so DCA
  amounts grow with CPI to maintain constant purchasing-power contributions

The margin call threshold is **4.0x leverage** (Reg-T 25% maintenance — the
conservative answer). Portfolio margin would in theory allow 6.67x but
brokers reserve the right to tighten mid-crisis, so we always size to 4.0x.
""")

st.divider()

# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------

st.header("2. Data")
st.markdown("""
**File:** `spx_margin_history.csv` — daily, 1927-12-30 to 2026-04-24
(~24,700 rows).

| Column | Description |
|---|---|
| `date` | YYYY-MM-DD trading days only |
| `spx_tr` | SPX total return index (dividends reinvested) |
| `tsy_3m` | 3-Month US Treasury yield (annualized, decimal) |
| `margin_rate` | Broker margin = 3M Tsy + ~40 bps |
| `cpi` | CPI-U level (FRED CPIAUCNS, monthly, ffilled to daily) |

**Sources:**
- 1927-2023: Bloomberg-sourced spreadsheet (`Portfolio Margin Backtest.xlsx`)
- 2023-2026: yfinance `^SP500TR` daily ratios (anchored to existing series at
  2023-04-12) + FRED `DGS3MO` for 3M Treasury
- CPI throughout: FRED `CPIAUCNS`

**Sanity anchors (1927-12 to 2023-04):**
- Nominal SPX-TR CAGR: 9.92%/yr
- Real CAGR: 6.66%/yr
- Implied inflation: 3.05%/yr
- Margin rate range: 0.40% to 17.54% (late 1970s/early 1980s stagflation peak)

**Key historical drawdowns** (post-1932):
- **2000-03-23 → 2009-03-09**: −48.04% over 8.96 years. Worst post-1932 entry
  date — binds nearly every safety analysis.
- **1973-01-11 → 1974-10-03**: ~−48% over ~1.6 years
- **2007-10-09 → 2009-03-09**: ~−55% over ~1.4 years (GFC)

The **1929-1932 Great Depression drawdown** (−83.79% over 2.71 years) is
deliberately excluded from calibration as unrepresentative of modern market
structure.

**Financing rate used in simulation:** box-spread rate (3M Tsy + 15 bps) on
the loan principal — **no tax benefit assumed**. Box-spread interest does
create 60/40 capital-loss recognition under Section 1256, but for a strict
hold-forever SPX investor with no other realized capital gains, only the
\\$3,000/yr deductible-against-ordinary portion has near-term value (the rest
is a carryforward whose realization depends on uncertain future events).
Setting the effective tax benefit to zero is the honest conservative
assumption for the hold-forever profile. If your situation has substantial
realized gains (e.g., from selling other positions), the real after-tax
cost is lower than what the simulator models.
""")

st.divider()

# ---------------------------------------------------------------------------
# 3. Mathematical framework
# ---------------------------------------------------------------------------

st.header("3. Mathematical framework")

st.subheader("Notation & symbols")
st.markdown(r"""
The same symbols recur across every strategy. Defined once here, referenced
inline below.

**Time & path variables** (state at trading day $t$ on a single simulated path):

| Symbol | Meaning |
|---|---|
| $t$ | Trading day index (0 = start of simulation, 252 trading days = 1 year) |
| $px_t$ | SPX total-return index level at day $t$ |
| $r_t$ | Daily SPX return = $px_t / px_{t-1} - 1$ |
| $A_t$ | Total asset value (current SPX position market value) |
| $D_t$ | Outstanding loan balance |
| $E_t = A_t - D_t$ | Equity (net liquidation value) |
| $L_t = A_t / E_t$ | Current leverage ratio |
| $M_t$ | Cumulative loan growth factor — see below |
| $\text{HWM}_t = \max_{s \leq t} E_s$ | High-water mark (peak equity so far) |
| $\text{cur\_dd}_t = 1 - E_t / \text{HWM}_t$ | Current drawdown from HWM |
| $\text{maxdd}_t = \max_{s \leq t} \text{cur\_dd}_s$ | Max drawdown ever seen (ratchet) |

**Strategy parameters** (set once at simulation start):

| Symbol | Meaning |
|---|---|
| $T_{init}$ | Initial leverage target chosen on day 0 |
| $T_t$ | Target leverage at day $t$ (most strategies: $T_0 = T_{init}$, then evolves) |
| floor | Minimum allowed target leverage. We always use 1.0x (fully unlevered). |
| $C$ | Starting equity (entered as "Current portfolio C" in the sidebar) |
| $X$ | Wealth target (entered as `wealth_X target` in the sidebar) |
| $F$ | Drawdown-decay factor (`dd_decay F` slider) |

**Loan growth factor $M_t$:**
""")
st.latex(r"M_t = \prod_{s=1}^{t} \left(1 + \frac{\text{rate}_s}{252}\right)")
st.markdown(r"""
where $\text{rate}_s$ is the annualized box-spread rate (3M Tsy + 15 bps)
on day $s$, divided by 252 to get the daily rate. So $D_t = D_0 \cdot M_t$
for an initial loan of $D_0$. This factor compounds the loan day-by-day.

**Combined index $R_t = px_t / M_t$:** SPX index level divided by loan
growth factor. When $R_t$ falls, your assets are losing ground vs your loan
— that's when leverage rises and margin calls become possible. The
*minimum* of $R_t$ over the future is the binding stress point for any
loan held from day 0.

**Rebalance schedule:** active strategies (everything except `unlev` and
`static`) check their rule every **21 trading days** (~once per month).
Between rebalance days, leverage drifts naturally.
""")

st.subheader("Leverage ratio definition")
st.markdown("""
The most important quantity in the simulator:
""")
st.latex(r"L_t = \frac{A_t}{E_t} = \frac{A_t}{A_t - D_t}")
st.markdown(r"""
A 1.5x leverage means \$1 of equity holds \$1.50 of SPX assets, financed by
\$0.50 of loan. Margin call fires when $L_t \geq 4.0$ (the 25% Reg-T
maintenance threshold) — at which point the broker forces a sale.

Two ways leverage rises:
1. **SPX falls:** $A_t$ shrinks but $D_t$ doesn't ⇒ $L_t$ rises.
2. **Loan compounds:** $D_t$ grows daily at the box-spread rate even when
   SPX is flat ⇒ $L_t$ slowly drifts up.

Two ways leverage falls:
1. **SPX rises:** $A_t$ grows faster than $D_t$ ⇒ $L_t$ drops.
2. **DCA contributions:** new dollars enter as $A_t$ but not $D_t$ ⇒ $L_t$
   drops (this is "passive deleveraging").
""")

st.subheader("Peak leverage (one-time loan, no DCA)")
st.markdown(r"""
This subsection derives the closed form used to size `static` strategies in
the no-DCA case. With DCA, no closed form exists and the simulator runs
day-by-day instead.

Setup: at day $i$ (entry), take a loan of $(L_0 - 1) \cdot E_0$. From day
$i$ forward, the SPX position and loan evolve as:
""")
st.latex(r"""
\begin{aligned}
A_t &= L_0 \cdot E_0 \cdot \frac{px_t}{px_i} \\
D_t &= (L_0 - 1) \cdot E_0 \cdot \frac{M_t}{M_i} \\
L_t &= \frac{A_t}{A_t - D_t}
\end{aligned}
""")
st.markdown(r"""
where $L_0 = T_{init}$ is the initial leverage and $M_t / M_i$ is the
loan-growth factor between entry day $i$ and day $t$.

Substituting and simplifying with $R_t \equiv px_t / M_t$:
""")
st.latex(r"L_t = \frac{L_0}{L_0 - (L_0 - 1) \cdot R_i / R_t}")
st.markdown(r"""
**Intuition:** $L_t$ is rising when $R_t$ is falling. Peak leverage along
the path $[i, T]$ is achieved at the day $t^*$ where $R_t$ is minimized —
the worst combined "SPX vs loan" point in the future.

Setting $L_{t^*} = \text{cap}$ (the call threshold, e.g., 4.0) and solving
for the *maximum safe* $L_0$:
""")
st.latex(r"L_0^{\max}(i, \text{cap}) = \frac{\text{cap} \cdot R_i}{\text{cap} \cdot R_i - (\text{cap} - 1) \cdot \min_{t \geq i} R_t}")
st.markdown(r"""
With cap = 4.0, this simplifies to:
""")
st.latex(r"L_0^{\max} = \frac{4 R_i}{4 R_i - 3 \min_{t \geq i} R_t}")
st.markdown(r"""
**What this means in plain language:** the most leverage you can safely take
on day $i$ is determined by the worst future point along your path,
*relative to your entry*. For the binding 2000-03-23 entry, the post-1932
worst min-$R_t$ to entry-$R_t$ ratio gives $L_0^{\max} \approx 1.42$x at 0%
DCA over the full 30y horizon.
""")

st.subheader("With ongoing contributions (DCA)")
st.markdown(r"""
Throughout this app, contributions are entered as annual amounts in
**today's dollars** ("real" — corrected for inflation). Internally each
contribution is paid as $1/252$ of the annual amount per trading day, and
its dollar value is *grown by CPI* along each historical path so the
contribution maintains constant purchasing power.

Let $m$ be the per-trading-day contribution amount (real $). Assets at day
$t$ are then:
""")
st.latex(r"""
A_t = T_{init} \cdot E_0 \cdot \frac{px_t}{px_0}
+ \sum_{s=1}^{t} m_s \cdot \frac{px_t}{px_s}
""")
st.markdown(r"""
where $m_s$ is the contribution paid on day $s$ (zero on non-contribution
days, or always nonzero for daily DCA — the simulator uses daily for
smoothness). The loan $D_t$ does NOT grow with contributions; only with
interest compounding.

**No closed form** — the path-dependent contributions force day-by-day
simulation.

**Important consequence:** DCA grows $A_t$ without growing $D_t$, so
$L_t$ falls over time. This is "passive deleveraging." For `static` it's
the *only* mechanism reducing leverage — by year 30 the heavy-DCA scenarios
in this app drift `static` from $T_{init} = 2.14$x down to $L_{30} \approx 1.02$x.
""")

st.subheader("Re-levering")
st.markdown(r"""
On a rebalance day, if the current leverage $L_t$ is below the target
$T_t$, the strategy takes **additional loan** $\Delta D$ to restore the
target. The amount needed:
""")
st.latex(r"\Delta D_t = \max(0,\, T_t \cdot E_t - A_t)")
st.markdown(r"""
After rebalancing: new assets $A_t + \Delta D_t$, new loan $D_t + \Delta D_t$,
equity unchanged, leverage now equals $T_t$.

**Rule: only LEVER UP, never sell to deleverage.** If $L_t > T_t$ (e.g.,
after a drawdown spiked leverage above target), the strategy *waits* for
SPX recovery / DCA dilution to bring leverage back down. It never sells
SPX to reduce $L_t$.

**Why?** Two reasons:
1. The hold-forever constraint forbids selling SPX.
2. Selling levered SPX during a drawdown locks in losses precisely at the
   worst moment. Patience is the right behavior for a long-horizon investor.

This rule means leverage can rise above $T_t$ during drawdowns. Whether
that becomes a margin call depends on the size of the drawdown and how much
cushion was reserved in choosing $T_{init}$ in the first place.
""")

st.subheader("CAGR vs IRR (critical distinction)")
st.markdown("""
With DCA contributions, the naive `(terminal/initial)^(1/H) - 1` formula
**over-reports** rate of return — it treats contributions as if they were
growth.

The correct measure is **IRR**: solve for $r$ such that:
""")
st.latex(r"E_0 (1+r)^H + \frac{m}{12} \cdot \frac{(1+r)^H - 1}{(1+r)^{1/12} - 1} = E_H")
st.markdown("""
where $H$ is the horizon in years, $m$ is annual DCA, $E_H$ is terminal
wealth. Solved numerically via `scipy.optimize.brentq`.

**Key insight:** DCA grows *wealth* (more dollars deployed), not *rate of
return*. Two strategies with identical IRR but different DCA produce very
different terminal wealth — but reporting it as "CAGR" inflates the apparent
return on the lower-DCA path by 1-5 pp.
""")

st.divider()

# ---------------------------------------------------------------------------
# 4. Sizing & safety bars
# ---------------------------------------------------------------------------

st.header("4. Sizing & safety bars")
st.markdown("""
Every strategy is sized at the largest $T_{init}$ satisfying multiple
safety constraints simultaneously. The simulator computes:

| Bar | Definition | Purpose |
|---|---|---|
| **T_hist_safe** | Largest $T$ with **0% margin calls** on the post-1932 historical paths at full horizon | Protects against repeated history |
| **T_boot_safe** | Largest $T$ with bootstrap call rate ≤ target % (default 1%) | Protects against path-ordering overfitting |
| **T_stress** | Largest $T$ with 0% calls when historical drawdowns are stretched by factor F | Protects against fatter-tail futures |
| **T_recommended** | $\\min(T_{hist}, T_{boot}, T_{stress})$ | Joint safety bar |

The simulator runs binary search (12-coarse + 12-fine grid passes,
~0.014x precision) to find each $T$.

**Why all three?** Historical 0% calls = path-dependent luck. The post-1932
worst path is 2000-03-23, and the binary search finds the largest $T$ that
*just barely squeaks past 4.0x* on that path — textbook overfitting. The
bootstrap and stretch bars are independent generalization buffers:
- **Bootstrap** measures path-ordering overfitting directly
- **Stretch** parametrizes "drawdowns deeper than historical"

Typical rule of thumb: **well-defended T ≈ 0.85-0.90 × historical max-safe T**.
""")

st.divider()

# ---------------------------------------------------------------------------
# 5. Bootstrap methodology
# ---------------------------------------------------------------------------

st.header("5. Bootstrap methodology")
st.markdown("""
Synthetic paths are generated via **block bootstrap** of joint daily tuples
$(\\text{return}, \\text{tsy yield}, \\text{cpi ratio})$ from post-1932 source
data.

**Algorithm:**
1. Define source array of length N = number of post-1932 daily samples
2. To build a synthetic path of D days:
   - Randomly sample a starting index $s \\in [0, N - \\text{block\\_days})$
   - Copy block of `block_days` consecutive tuples
   - Repeat until D days are filled
3. Generate `n_paths` independent synthetic paths

**Block size sensitivity** (key result from project):

| Block | Synthetic call rate (time-decay 2pp at 1.591x) |
|---|---|
| 1 month | 5.82% |
| 3 months | 6.52% |
| 6 months | 8.18% |
| **1 year (default)** | **10.02%** |
| 2 years | 11.36% |
| 5 years | 8.50% |

The relationship is **non-monotonic**: short blocks randomize too much
(break up bad-year clusters), 5-year blocks regress toward historical
sequence. **2-year blocks are the most stressful**; 1-year is a reasonable
middle ground.

**Why bootstrap matters:** Historical "0% calls at full horizon" includes
exactly **one** ~9-year drawdown (2000-09). Bootstrap paths can have multiple
clustered drawdowns by chance, exactly the scenario where decay/relever
strategies blow up. The synthetic call rate at the historical-safe target
quantifies how lucky the historical sequence was.
""")

st.divider()

# ---------------------------------------------------------------------------
# 6. Drawdown stretch
# ---------------------------------------------------------------------------

st.header("6. Drawdown stretch")
st.markdown("""
The drawdown-stretch transformation amplifies every historical path's
drawdowns by factor F:

1. For each path, compute running max from day 0 onwards
2. Compute drawdown = $1 - px_t / \\text{running\\_max}_t$
3. Scale drawdown: $dd' = F \\cdot dd$
4. Reconstitute path: $px'_t = \\text{running\\_max}_t \\cdot (1 - dd')$

**Properties:**
- F = 1.0 is identity
- F = 1.1 makes a 50% historical drawdown into a 55% one
- At new running highs (dd = 0), the stretched path equals the original
- Recovery from a deeper bottom takes a larger relative gain

This models "future drawdowns may be deeper than historical." With the
default F = 1.1 you're sized for 10%-deeper-than-history drawdowns. F = 1.2
is more aggressive (~20% deeper); the project's analysis shows F = 1.2 cuts
typical max-safe targets by ~10%.
""")

st.divider()

# ---------------------------------------------------------------------------
# 7. Strategy catalog
# ---------------------------------------------------------------------------

st.header("7. Strategy catalog")

st.markdown(r"""
This catalog defines every strategy available in the simulator. Each entry
follows the same template:

- **Definition** — what the strategy does, in plain English
- **Formula** — the precise rebalance-time rule (uses notation from §3)
- **Worked example** — concrete numbers for a representative case
- **Properties / Intuition** — why it behaves the way it does
- **Parameters** — the adjustable knobs

All formulas evaluate at rebalance days only (every 21 trading days = ~once
a month) for active strategies. Between rebalances, the strategy is
"asleep": only the natural compounding of $A_t$ and $D_t$ happens, no
adjustment. Intra-rebalance margin calls *can* still trigger if leverage
crosses 4.0x at any time.

**Quick reference table** (parameter symbols defined in §3 notation):

| Strategy | Mechanism | Active params | Key property |
|---|---|---|---|
| `unlev` | none | — | Baseline (no leverage ever) |
| `static` | one-time loan, no rebalance | — | Passive deleveraging via DCA dilution |
| `relever` | monthly re-lever to $T_{init}$ | — | Maintains $T_{init}$; pinned at target |
| `dd_decay` | drawdown ratchet | $F$ | $T_t = T_{init} - F \cdot \text{maxdd}_t$ |
| `adaptive_dd` | cushion-coupled $F$ | $F$ | $F_{\text{eff}}$ scales with current cushion |
| `wealth_decay` | wealth glide (current eq) | $X$ | $T_t = T_{init} \to$ floor as $E_t \to X$ |
| `hybrid` | $\min(T^{dd}, T^{w})$ | $F$, $X$ | Combined risk + utility |
| `r_hybrid` | hybrid + ratcheted wealth-prog | $F$, $X$ | Wealth-progress only goes up |
| `vol_hybrid` | hybrid − vol haircut | $F$, $X$, $\nu$ | Deleverages on vol spikes |
| `dip_hybrid` | hybrid + buy-the-dip floor | $F$, $X$, dip_threshold, dip_bonus | Adds leverage during deep drawdowns |
| `rate_hybrid` | hybrid − rate haircut | $F$, $X$, $\tau$, $\rho$ | Deleverages when rates are high |
| `adaptive_hybrid` | adaptive_dd + wealth | $F$, $X$ | Cushion-coupled dd + wealth glide |
| `recal_static` | static + periodic table lookup | recal_period | At each re-cal: lever to $T^{*}$(equity, horizon) |
| `recal_hybrid` | hybrid + periodic re-cal | recal_period, $F$, $X$ | Hybrid logic between re-cals |
| `recal_adaptive_dd` | adaptive_dd + periodic re-cal | recal_period, $F$ | Adaptive_dd logic between re-cals |
| `meta_recal` | re-cal + pick-max strategy | recal_period, $F$, $X$ | Pick max-$T^{*}$ strategy at each re-cal |

**Naming convention:** strategies prefixed `hybrid_` or suffixed `_hybrid`
all share the wealth-glide component, so the `wealth_X` slider applies to
all of them. The base strategies (`static`, `relever`, `dd_decay`,
`adaptive_dd`) ignore `wealth_X`.
""")

st.divider()

# --- unlev ---
st.subheader("`unlev` — baseline")
st.markdown("""
**Definition:** $T = 1.0x$, no leverage at any point.

**Use:** Reference benchmark for IRR uplift comparisons. Pure DCA into SPX
with no margin debt.

**Parameters:** none.
""")

# --- static ---
st.subheader("`static` — one-time loan, no rebalance")
st.markdown(r"""
**Definition:** On day 0, take a loan of size $D_0 = (T_{init} - 1) \cdot E_0$.
After that, **no active rule fires** — never rebalance, never sell, never
take more loan. Loan compounds at the box-spread rate; assets grow with
SPX returns plus DCA contributions.

**Formula:** No rebalance, but $A_t$ and $D_t$ evolve daily:
""")
st.latex(r"A_t = T_{init} \cdot E_0 \cdot \frac{px_t}{px_0} + \sum_{s=1}^{t} m_s \cdot \frac{px_t}{px_s}")
st.latex(r"D_t = (T_{init} - 1) \cdot E_0 \cdot M_t")
st.latex(r"L_t = \frac{A_t}{A_t - D_t}")

st.markdown(r"""
$L_t$ here is just the consequence of asset growth, loan compounding, and
DCA dilution — not the result of any active decision.

**Worked behavior** (heavy-DCA scenario from the sidebar defaults):

| Time | Approx. equity | Approx. loan | Leverage |
|---|---|---|---|
| Day 0 | \$160k | \$184k | 2.14x |
| Year 5 | \$1.0M (5× via DCA + ~50% SPX) | ~\$200k | ~1.30x |
| Year 30 | ~\$10M | ~\$340k | ~1.02x (≈unlevered) |

DCA dilution dominates over the loan compounding, so leverage drifts down
to nearly 1.0x without any rebalance ever firing.

**Properties:**
- **Single parameter** ($T_{init}$). Closed-form risk math (peak leverage
  formula in §3 applies directly).
- **No path-dependent decisions** — fully passive.
- **Lowest bootstrap fragility** of any strategy (typically 0.3-1.5% calls
  at historical-safe target vs 4-14% for active strategies). The reason is
  exactly that there are no path-dependent decisions to overfit.
- **End-of-horizon leverage drifts to ~1.05x** by year 30 in heavy-DCA
  scenarios. Static naturally winds down without an explicit rule.

**Architecturally cleanest.** Every other strategy is an attempt to capture
extra IRR or enforce a goal via active management. `static` is the
physics-of-leverage answer: take the loan, let it run.

**Parameters:** $T_{init}$ (the leverage target at day 0). Sliders for
`wealth_X`, $F$, etc. are ignored.
""")

# --- relever ---
st.subheader("`relever` — monthly re-lever to T_init")
st.markdown(r"""
**Definition:** Every rebalance day (21 trading days), if current leverage
$L_t < T_{init}$, take additional loan $\Delta D$ to bring leverage back
up to $T_{init}$. The target is constant — always equal to $T_{init}$.

**Formula** (applied each rebalance day):
""")
st.latex(r"T_t = T_{init} \quad \text{(target is constant)}")
st.latex(r"\Delta D_t = \max\!\left(0,\; T_{init} \cdot E_t - A_t\right)")

st.markdown(r"""
After applying $\Delta D_t$: new $A_t \leftarrow A_t + \Delta D_t$, new
$D_t \leftarrow D_t + \Delta D_t$, equity unchanged, $L_t = T_{init}$.

**Worked example** ($T_{init} = 1.4$x, monthly DCA contribution has just
diluted leverage below target):

| Quantity | Pre-rebalance | After loan added | Post-rebalance |
|---|---|---|---|
| Equity ($E_t$) | 100 | (unchanged) | 100 |
| Assets ($A_t$) | 138 | +$\Delta D = 2$ | 140 |
| Loan ($D_t$) | 38 | +$\Delta D = 2$ | 40 |
| Leverage ($L_t$) | 1.38x | — | 1.40x ✓ |

The loan increment $\Delta D = T_{init} \cdot E_t - A_t = 1.4 \times 100 - 138 = 2$
is exactly what's needed to restore target leverage at unchanged equity.

**Properties:**
- **Pinned at target.** Leverage stays near $T_{init}$ throughout the
  horizon — never drifts down via DCA dilution because each rebalance
  re-captures the dilution as more loan.
- **Highest historical IRR uplift** at given $T_{init}$ (+2-3 pp/yr over
  `unlev` at 0% DCA, more with higher DCA).
- **Highest path-overfitting.** Bootstrap call rate at historical-safe
  $T_{init}$ is ~13-14% — by far the worst tail behavior of any strategy
  in the simulator.
- **No passive deleveraging.** End-of-horizon leverage stays near
  $T_{init}$, meaning you're running near your ceiling for 30 years.

**Architecturally weakest among the active strategies.** The maintenance
rule re-levers most aggressively during bull runs (which often precede
crashes); it's exactly counter to risk management. It only "works"
historically because the post-1932 sample mostly avoids back-to-back deep
crises — bootstrap exposes the bet.

**Parameters:** $T_{init}$.
""")

# --- dd_decay ---
st.subheader("`dd_decay` — drawdown-coupled decay")
st.markdown(r"""
**Definition:** Target ratchets DOWN as the maximum-drawdown-ever-seen
grows. Once lowered, target never re-rises (asymmetric ratchet — a "scar"
that doesn't heal).

**Formula:**
""")
st.latex(r"T_t = \max\!\left(\text{floor},\; T_{init} - F \cdot \text{maxdd}_t\right)")
st.markdown(r"""
The drawdown ratchet $\text{maxdd}_t$ is the largest cur_dd observed up
to time $t$:
""")
st.latex(r"\text{HWM}_t = \max_{s \leq t} E_s \quad \text{(high-water mark of equity)}")
st.latex(r"\text{cur\_dd}_t = 1 - E_t / \text{HWM}_t \quad \text{(current drawdown)}")
st.latex(r"\text{maxdd}_t = \max\!\left(\text{maxdd}_{t-1},\; \text{cur\_dd}_t\right) \quad \text{(ratchet up only)}")

st.markdown(r"""
So $\text{maxdd}_t$ records "the worst hit you've ever taken" and only ever
grows. After a 30% drawdown, $\text{maxdd}_t$ stays at 0.30 even after
equity recovers and HWM exceeds its previous level.

**Worked example** (F = 1.5, $T_{init} = 1.6$x):

| Time | Event | maxdd | Target $T$ |
|---|---|---|---|
| Day 0 | Start | 0.00 | 1.60x |
| Year 5 | SPX drops 30% from peak | 0.30 | 1.60 − 1.5 × 0.30 = 1.15x |
| Year 10 | SPX recovers, new HWM | 0.30 (stuck) | 1.15x (stuck) |
| Year 15 | Another 40% drawdown | 0.40 | 1.60 − 0.60 = 1.00x (floor) |

The "stuck" entries are the ratchet behavior: maxdd never decreases, so
target never re-rises after a drawdown — even after full recovery.

**Properties:**
- **Event-driven.** Through quiet periods, target stays at $T_{init}$ —
  no decay occurs until a real drawdown is observed.
- **Asymmetric.** Once a drawdown lowers target, it stays lowered even after
  full recovery. "Permanent learning."
- **Couples to the real risk variable.** Margin calls happen because of
  drawdowns; this strategy responds directly to drawdown depth.
- **Default F = 1.5.** A 20% drawdown ⇒ target falls 0.30 (e.g., 1.6x → 1.3x).

**Parameters:**
- $F \in [0.5, 3.0]$ — decay factor. Higher F = more aggressive deleveraging.
- floor = 1.0x (fully unlevered).
""")

# --- adaptive_dd ---
st.subheader("`adaptive_dd` — cushion-coupled F")
st.markdown(r"""
**Definition:** Same drawdown-coupling as `dd_decay`, but $F$ is no longer
constant — it adapts each rebalance day based on **how much cushion you
currently have**.

**The "cushion" idea.** At any moment, the gap between your *current*
leverage $L_{\text{now}}$ and $T_{init}$ measures how far you've drifted
*below* your starting target. If $L_{\text{now}} = T_{init}$ you have zero
cushion — every additional bit of drawdown brings you closer to call. If
$L_{\text{now}} = 1.0$x you're effectively unlevered — no decay needed
since you're already safe.

**Formula:**
""")
st.latex(r"L_{\text{now}} = A_t / E_t \quad \text{(current leverage at rebalance day)}")
st.latex(r"F_{\text{eff}} = F \cdot \frac{L_{\text{now}} - 1}{T_{init} - 1}")
st.latex(r"\text{candidate}_t = \max\!\left(\text{floor},\; T_{init} - F_{\text{eff}} \cdot \text{maxdd}_t\right)")
st.latex(r"T_t = \min\!\left(T_{t-1},\; \text{candidate}_t\right) \quad \text{(monotonic-down ratchet)}")

st.markdown(r"""
The $T_t = \min(T_{t-1}, \ldots)$ step is critical: $F_{\text{eff}}$ varies
over time, so without explicit ratcheting the target could re-rise after
deleveraging. The `min` enforces "target only goes down, never up."

**Worked example** (F = 1.5, $T_{init} = 1.6$x):

| Scenario | $L_{\text{now}}$ | $F_{\text{eff}}$ | maxdd just observed | New target candidate |
|---|---|---|---|---|
| Day 0 (start) | 1.60x | 1.50 (full) | 0.00 | $T_0 = 1.6$x |
| After 20% drawdown (no prior) | ≈2.00x | 1.5 × (2.0−1) / (1.6−1) = 2.50 | 0.20 | 1.6 − 2.5 × 0.20 = 1.10x |
| After years of DCA dilution | 1.05x | 1.5 × 0.05 / 0.6 = 0.125 | 0.20 (new) | 1.6 − 0.125 × 0.20 = 1.575x → ratchet keeps prior 1.10x |

Read the rows top-to-bottom: in the high-leverage state right after a
drawdown, $F_{\text{eff}}$ is *amplified* (2.5×) — fast deleverage when
danger is near. After passive DCA dilution has already brought you down to
1.05x, $F_{\text{eff}}$ is tiny (0.125) — even a fresh 20% drawdown barely
moves the candidate, and the monotonic ratchet keeps the existing tighter
target. You don't waste decay budget when you're already safe.

**Empirical result on your scenario:** adaptive_dd has ~halved bootstrap
call rate vs vanilla dd_decay at historical-safe T (~3.8% vs ~6.0%) with
~15 bps higher p50 IRR. Modest but real Pareto improvement.

**Parameters:**
- $F$ — base decay factor (same slider as dd_decay).
- floor = 1.0x.
""")

# --- wealth_decay ---
st.subheader("`wealth_decay` — current-equity wealth glide")
st.markdown(r"""
**Definition:** Target linearly glides from $T_{init}$ (when equity equals
the starting amount $C$) down to floor (when equity reaches the wealth
target $X$). Below $C$: target stays at $T_{init}$. Above $X$: target stays
at floor (1.0x).

The signal driving the glide is *current real equity* — recomputed every
rebalance day, NOT a high-water mark.

**Formula:**
""")
st.latex(r"E_t^{\text{real}} = E_t \cdot \frac{\text{cpi}_0}{\text{cpi}_t} \quad \text{(equity in today's dollars)}")
st.latex(r"\text{prog}_t = \text{clip}\!\left(\frac{E_t^{\text{real}} - C}{X - C},\; 0,\; 1\right)")
st.latex(r"T_t = T_{init} - (T_{init} - \text{floor}) \cdot \text{prog}_t")

st.markdown(r"""
where `clip(x, 0, 1)` clamps $x$ to the range $[0, 1]$. So $\text{prog}_t$
is "fraction of the way from C to X" — 0 at start, 1 at the target wealth,
linear in between.

**Worked example** ($T_{init} = 1.6$x, floor = 1.0x, $C$ = \$200k, $X$ = \$3M):

| Real equity $E_t^{\text{real}}$ | progress | Target $T_t$ |
|---|---|---|
| \$200k (= $C$) | 0.00 | 1.60x |
| \$1.6M | 0.50 | 1.30x |
| \$3M (= $X$) | 1.00 | 1.00x (floor) |
| \$5M (above $X$) | 1.00 (clipped) | 1.00x (still floor) |

**Properties:**
- **Symmetric (NOT a ratchet).** Uses current $E_t$, recomputed each
  rebalance. If a drawdown drops equity from \$2.5M back to \$2.0M,
  progress falls and target rises back up. This is intentional —
  it's a glide path, not a safety mechanism.
- **Utility-glide, not safety architecture.** "I want to be unlevered when
  I reach my wealth target $X$" is a *preference* about end-state
  leverage, not a *risk-management* response to stress.
- **Gameable by DCA.** Depositing money raises $E_t$ → drops target. But
  depositing money doesn't make leverage *structurally* safer (loan-to-asset
  ratio is unchanged). The signal is partially about wealth-from-investing
  and partially about wealth-from-depositing — only the first one is a
  meaningful glide signal.

**Bootstrap fragility:** intermediate. Better than monthly relever (target
sometimes drops), worse than static (the symmetric rebound during drawdowns
hurts). The hybrid (next entry) addresses this.

**Parameters:**
- $X$ — wealth at which target hits floor (single sidebar slider, shared
  across all `wealth_X`-using strategies).
- floor = 1.0x.

**Shared parameter:** the simulator uses one `wealth_X` slider that applies
to every strategy that has a wealth-glide component (`wealth_decay`,
`hybrid`, `r_hybrid`, `vol_hybrid`, `dip_hybrid`, `rate_hybrid`,
`adaptive_hybrid`). Adjusting it affects all of them simultaneously.
""")

# --- hybrid ---
st.subheader("`hybrid` — dd_decay + wealth_decay")
st.markdown(r"""
**Definition:** Combine `dd_decay` and `wealth_decay` by taking whichever
is more conservative each rebalance day. Two independent signals pull
target down; the lower one wins.

**Formula:**
""")
st.latex(r"T^{dd}_t = \max\!\left(\text{floor},\; T_{init} - F \cdot \text{maxdd}_t\right) \quad \text{(from dd\_decay)}")
st.latex(r"T^{w}_t = T_{init} - (T_{init} - \text{floor}) \cdot \text{prog}_t \quad \text{(from wealth\_decay)}")
st.latex(r"T_t = \min\!\left(T^{dd}_t,\; T^{w}_t\right)")

st.markdown(r"""
where $T^{dd}_t$ is the dd_decay target and $T^{w}_t$ is the wealth_decay
target, each computed exactly as in the standalone strategies.

**Behavior in different regimes:**
- **Quiet bull market, equity below X:** $T^{dd} = T_{init}$ (no drawdowns
  yet), $T^{w}$ glides down. Target follows wealth glide.
- **Drawdown before equity reaches X:** $T^{dd}$ ratchets down hard
  (drawdown signal), $T^{w}$ may even rise (equity dropped, prog dropped).
  Target follows dd ratchet — this is the safety rescue.
- **Bull market past X with no major drawdowns:** Both $T^{dd}$ near
  $T_{init}$ and $T^{w} = $ floor. Target follows wealth at floor.
- **Drawdown after equity has crossed X:** dd may bind first if its ratchet
  produces a value below floor (clipped to floor). Either way, target stays
  at floor.

**Properties:**
- **dd_decay handles risk events** (asymmetric drawdown ratchet).
- **wealth_decay enforces the unlever-at-$X$ goal** (smooth glide).
- **Pareto-dominates wealth_decay alone** — the dd component rescues paths
  where a deep drawdown hits before equity reaches $X$, with no IRR cost on
  the median path.
- **Captures both safety and utility goals** with a clean min-composition.

**This strategy was user-developed** — proposed in conversation as the
right way to compose drawdown safety with the "unlever at the wealth
target" preference.

**Parameters:**
- $F$ — dd_decay factor.
- $X$ — wealth_X, target wealth for glide.
- floor = 1.0x.
""")

# --- r_hybrid ---
st.subheader("`r_hybrid` — ratcheted hybrid")
st.markdown(r"""
**Definition:** Same as `hybrid`, but the wealth-progress is also a ratchet
— once it climbs to a value, it never decreases. The wealth-target therefore
only goes down, never back up.

**Formula:**
""")
st.latex(r"\text{prog}^{\text{ratchet}}_t = \max\!\left(\text{prog}^{\text{ratchet}}_{t-1},\; \text{prog}_t\right)")
st.latex(r"T^{w}_t = T_{init} - (T_{init} - \text{floor}) \cdot \text{prog}^{\text{ratchet}}_t")
st.latex(r"T_t = \min\!\left(T^{dd}_t,\; T^{w}_t\right)")

st.markdown(r"""
where $\text{prog}_t$ is the same wealth-progress as in `wealth_decay`, and
$\text{prog}^{\text{ratchet}}_t$ is its running maximum.

**Difference from `hybrid`:** in `hybrid`, the wealth component recomputes
$\text{prog}_t$ each rebalance from current equity — so a drawdown that
drops equity also drops $\text{prog}_t$, lifting $T^{w}_t$ back up. In
`r_hybrid`, $\text{prog}^{\text{ratchet}}_t$ holds onto the previous high
even if equity dips, so $T^{w}_t$ stays at its lowest-ever level.

**Properties:**
- **Strictly monotonic-down target** (both components are now ratchets,
  and `min` of two non-increasing functions is non-increasing).
- **Removes the "wealth-rebound during drawdown raises target" failure
  mode** that symmetric `wealth_decay` can have.
- In high-DCA scenarios where wealth grows nearly monotonically anyway,
  `r_hybrid` behaves *identically* to `hybrid` — the ratchet is rarely
  needed because $\text{prog}_t$ rarely decreases. The mechanism shows
  effect mostly in low-DCA scenarios with deep drawdowns.

**Parameters:**
- $F$, $X$. Same as hybrid.
""")

# --- vol_hybrid ---
st.subheader("`vol_hybrid` — hybrid + volatility haircut")
st.markdown(r"""
**Definition:** Compute the standard `hybrid` target, then subtract an
additional haircut proportional to recent SPX realized volatility.

**Formula:**
""")
st.latex(r"\sigma_{60,t} = \sqrt{252} \cdot \text{stdev}\!\left(r_{t-59},\, r_{t-58},\, \ldots,\, r_t\right)")
st.latex(r"T_t = \max\!\left(\text{floor},\; T^{hybrid}_t - \nu \cdot \sigma_{60,t}\right)")

st.markdown(r"""
where:
- $r_t$ is the daily SPX return on day $t$ (defined in §3 notation)
- $\sigma_{60,t}$ is the **annualized** realized standard deviation of the
  last 60 daily returns (multiplied by $\sqrt{252}$ to convert daily to
  annual volatility)
- $T^{hybrid}_t$ is the target produced by the standard `hybrid` strategy
- $\nu$ is the `vol_factor` parameter

**Worked example** (hybrid_target = 1.5x, $\nu = 1.0$):

| Regime | Annualized 60d vol $\sigma_{60}$ | Haircut $\nu \cdot \sigma_{60}$ | Target $T_t$ |
|---|---|---|---|
| Calm | 12% | 0.12x | 1.5 − 0.12 = 1.38x |
| Normal | 18% | 0.18x | 1.5 − 0.18 = 1.32x |
| Stressed (March 2020) | 50% | 0.50x | max(1.0, 1.0) = **1.00x** (floor) |

**Intuition:** Volatility spikes typically *lead* drawdowns by days to
weeks (vol-of-vol literature; "vol smile"). Coupling target to current
volatility deleverages *before* drawdown materializes — earlier than
dd_decay's reactive ratchet. Common in hedge funds as "volatility targeting."

**Empirical result on your scenario:** vol_hybrid raises T_rec from 1.71x
to 1.84x while keeping bootstrap call rate at 1%, with peak p99 leverage
tightening from 2.53x to 2.40x. Modest Pareto improvement.

**Parameters:**
- $F$, $X$ from hybrid.
- $\nu$ (`vol_factor`) — multiplier on annualized 60d vol. With $\nu = 1$,
  a 30% annualized-vol regime drops target by 0.30x.
""")

# --- dip_hybrid ---
st.subheader("`dip_hybrid` — hybrid + buy-the-dip floor")
st.markdown(r"""
**Definition:** Use the standard `hybrid` target except during deep
drawdowns, where the target is *floored* at $T_{init} + \text{dip\_bonus}$.
This means leverage is allowed to go *above* $T_{init}$ when SPX has
fallen far from its high — a contrarian bet on mean reversion.

**Formula:**
""")
st.latex(r"\text{cur\_dd}_t = 1 - E_t / \text{HWM}_t \quad \text{(current drawdown from HWM)}")
st.latex(r"""
T_t = \begin{cases}
\max\!\left(T^{hybrid}_t,\; T_{init} + \text{dip\_bonus}\right) & \text{if } \text{cur\_dd}_t > \text{dip\_threshold} \\
T^{hybrid}_t & \text{otherwise}
\end{cases}
""")

st.markdown(r"""
**Worked example** ($T_{init} = 1.5$x, dip_threshold = 0.30, dip_bonus = 0.20):

| Path state | Current dd | Triggered? | Target $T_t$ |
|---|---|---|---|
| SPX −10% from HWM | 0.10 | No | $T^{hybrid} \approx 1.5$x (dd_decay barely fired) |
| SPX −35% from HWM | 0.35 | **Yes** | $\max(T^{hybrid},\, 1.5 + 0.2) = 1.70$x ⚠ |
| SPX recovers to −20% from HWM | 0.20 | No | $T^{hybrid}$ (now ~1.0x — the dd ratchet has fired) |

The SPX −35% row is where the dip-bonus dominates: hybrid would have
ratcheted target down to ~1.0x by then, but dip_hybrid OVERRIDES and lifts
target to 1.70x — the contrarian bet on mean reversion.

**Intuition:** "Buy more leverage at the bottom — bet that mean-reversion
outpaces further-down." Contrarian to `dd_decay`'s ratchet during deep
drawdowns.

**Empirical caution — this strategy is a TRAP:** It looks good historically
(real history mostly has ONE deep drawdown per cycle, so the bonus captures
recoveries) but **fails badly on bootstrap**. Bootstrap paths can have
multiple sequential deep drawdowns, in which case the bonus stacks leverage
ON TOP of accumulated damage. T_rec collapses dramatically vs hybrid
(e.g., from 1.71x to 1.21x on your scenario). Bootstrap call rate at the
historical-safe target jumps to ~17%. Use with extreme caution.

**Parameters:**
- $F$, $X$ from hybrid.
- `dip_threshold` (e.g., 0.30 = 30% drawdown from HWM) — minimum drawdown
  that triggers the override.
- `dip_bonus` (e.g., 0.20) — extra leverage above $T_{init}$ to set as floor
  during the dip.
""")

# --- rate_hybrid ---
st.subheader("`rate_hybrid` — hybrid + interest-rate haircut")
st.markdown(r"""
**Definition:** Compute the standard `hybrid` target, then subtract a
haircut whenever the 3-Month Treasury yield exceeds a threshold. Larger
yield excess ⇒ larger haircut.

**Formula:**
""")
st.latex(r"y_{3M,t} = \text{annualized 3M Tsy yield on day } t \text{ (decimal, e.g. } 0.045 = 4.5\%\text{)}")
st.latex(r"\text{excess}_t = \max\!\left(0,\; y_{3M,t} - \tau\right)")
st.latex(r"T_t = \max\!\left(\text{floor},\; T^{hybrid}_t - \rho \cdot \text{excess}_t\right)")

st.markdown(r"""
where:
- $\tau$ is `rate_threshold` (yield above which haircut applies)
- $\rho$ is `rate_factor` (multiplier on the excess)

**Worked example** (hybrid_target = 1.5x, $\tau$ = 5%, $\rho$ = 5):

| 3M Tsy yield | excess = max(0, $y_{3M} − \tau$) | Haircut $\rho \cdot \text{excess}$ | Target $T_t$ |
|---|---|---|---|
| 4% | 0 | 0 | 1.50x (no haircut) |
| 6% | 0.01 | 0.05x | 1.45x |
| 8% | 0.03 | 0.15x | 1.35x |
| 14% (1981 stagflation) | 0.09 | 0.45x | max(1.0, 1.05) = 1.05x |

**Intuition:** When real rates are high, the leverage carry trade is less
profitable (box-spread financing cost rises while SPX-rate spread
compresses). High-rate regimes (1980-82) historically coincided with bad
equity markets — the haircut deleverages preemptively.

**Empirical caution on your scenario:** historical high-rate periods are
short and rare in the post-1932 sample (mostly 1979-1985). At the default
$\tau = 5\%$, the haircut rarely fires, and `rate_hybrid` behaves
≈ `hybrid` in practice. Could be tuned more aggressively (lower threshold,
or use real rate instead of nominal) to fire more often.

**Parameters:**
- $F$, $X$ from hybrid.
- $\tau$ (`rate_threshold`) — nominal 3M Tsy yield above which haircut
  applies. Default 5%.
- $\rho$ (`rate_factor`) — multiplier on yield excess. Default 5 (so 1pp
  of excess yield ⇒ 0.05x haircut on target).
""")

# --- adaptive_hybrid ---
st.subheader("`adaptive_hybrid` — adaptive_dd + wealth_decay")
st.markdown(r"""
**Definition:** Combine the cushion-coupled $F_{\text{eff}}$ mechanism of
`adaptive_dd` with the wealth-glide of `wealth_decay`. The dd component
uses the adaptive monotonic ratchet; the wealth component is the symmetric
glide. Final target is the lower of the two.

**Formula:**
""")
st.latex(r"L_{\text{now}} = A_t / E_t \quad \text{(current leverage at rebalance)}")
st.latex(r"F_{\text{eff}} = F \cdot \frac{L_{\text{now}} - 1}{T_{init} - 1}")
st.latex(r"\text{candidate}^{dd}_t = \max\!\left(\text{floor},\; T_{init} - F_{\text{eff}} \cdot \text{maxdd}_t\right)")
st.latex(r"T^{adapt}_t = \min\!\left(T^{adapt}_{t-1},\; \text{candidate}^{dd}_t\right) \quad \text{(monotonic-down ratchet)}")
st.latex(r"T^{w}_t = T_{init} - (T_{init} - \text{floor}) \cdot \text{prog}_t \quad \text{(symmetric wealth glide)}")
st.latex(r"T_t = \min\!\left(T^{adapt}_t,\; T^{w}_t\right)")

st.markdown(r"""
where $T^{adapt}_t$ is the adaptive_dd target and $T^{adapt}_{t-1}$ is its
value at the previous rebalance day (used for the monotonic ratchet — see
`adaptive_dd` entry for full discussion).

**Properties:**
- Inherits `adaptive_dd`'s IRR-per-safety efficiency on the dd component:
  decay only when there's actual cushion to spend.
- Inherits `hybrid`'s "unlever at $X$" guarantee from the wealth glide.
- Should Pareto-stack both improvements over plain `hybrid`.

**Empirical result on your scenario:** adaptive_hybrid is essentially
identical to `hybrid` — because in heavy-DCA scenarios, the wealth glide
component is *almost always* the binding constraint (your $E^{\text{real}}$
crosses $X$ within ~5-7 years, well before any major drawdown could even
fire the dd ratchet). The adaptive F mechanism only matters in regimes
where dd is the binding side; for low-DCA users with deep drawdowns hitting
before equity reaches $X$, the difference would be material. On your
scenario the two strategies converge.

**Parameters:**
- $F$, $X$. Same as hybrid.
- floor = 1.0x.
""")

# ---------------------------------------------------------------------------
# Re-calibration strategy family
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Re-calibration strategy family — periodic re-pick of $T_{init}$")
st.markdown(r"""
The four strategies below model the natural use of this tool: "every N
years/months, observe my actual portfolio state and re-decide my optimal
leverage." They share a common structure:

1. **Pre-compute a lookup table** $T^{*}(E_{\text{real}}, H_{\text{remaining}})$
   over a coarse 6×6 grid (equity × remaining horizon). Each cell holds the
   well-defended max-safe T for the given starting state and remaining
   horizon — using the SAME safety architecture as the rest of the app
   (historical 0% calls + bootstrap ≤ `boot_target` + drawdown stretch ≤ 0%).

2. **At each re-cal event** (every `recal_period_months`):
   - Look up the closest cell to current real equity + remaining horizon.
   - Reset path-conditional state ($\text{max\_dd} = 0$, $\text{HWM} = E_t$).
   - Set $T_{\text{active}} = $ lookup value.
   - **Take additional loan** to bring leverage to $T_{\text{active}}$.
     (Lever-up only — never sells, no tax events.)

3. **Between re-cal events**, apply the base strategy's logic with
   $T_{\text{active}}$ in place of $T_{init}$.

**Why this differs from `relever`:** target adapts to current state at each
event, gradually decaying as DCA-fraction shrinks (lookup values decrease
with growing equity). Recal is "sparse adaptive relever to a state-dependent
target," not "monthly relever to a fixed target."

**Caveat: cell-level vs strategy-level safety.** The per-cell lookup
guarantees ≤`boot_target` calls *from this cell forward*. But the strategy
goes through ~6 re-cal events over 30y, and call probabilities compound —
strategy-level boot rate ends up ~3× cell rate empirically. Each individual
decision is honest by app-standard safety; the aggregate is what it is.
""")

# --- recal_static ---
st.subheader("`recal_static` — static + periodic table lookup")
st.markdown(r"""
**Definition:** Behaves like `static` (no rebalances) between re-cal events.
At each re-cal day, looks up $T^{*}(E_t^{\text{real}}, H_{\text{rem}})$ from
the static-strategy lookup table and takes additional loan to bring
leverage to that target (only lever up).

**Formula** (re-cal events on days where $d \bmod \text{recal\_period\_days} = 0$):
""")
st.latex(r"T_{\text{active}, t} = T^{*}_{\text{static}}\!(E_t^{\text{real}},\, H - t)")
st.latex(r"\Delta D_t = \max(0,\; T_{\text{active}, t} \cdot E_t - A_t)")

st.markdown(r"""
Between re-cal events, no action — natural drift only.

**Properties:**
- **Sustained leverage**, vs plain `static` which decays toward 1.0x. Recal
  events keep leverage near the lookup value (typically 1.2-1.6x in
  high-DCA scenarios for years 5+).
- **Trade tail-risk for upside.** Bootstrap call rate ~3% (vs static's
  0.3%); p50 +5-20% over plain static; p90 substantially higher.
- **No path-conditional safety** — each re-cal looks only at current equity
  and remaining horizon, not the path's drawdown history.

**Parameters:**
- `recal_period_months` (slider) — time between re-cal events.
- $T_{init}$ — initial leverage (years 0 to first re-cal). Often calibrates
  to 1.0x because the strategy already has ≥`boot_target` boot calls from
  recal events alone.
""")

# --- recal_hybrid ---
st.subheader("`recal_hybrid` — hybrid + periodic re-cal")
st.markdown(r"""
**Definition:** Like `hybrid` but with periodic state-resetting re-cal
events. At each re-cal day, looks up $T^{*}$ from the hybrid-strategy
lookup table (computed for the user's `wealth_X`), resets state, sets
$T_{\text{active}}$ = lookup value, takes loan to reach it. Between events,
applies hybrid logic ($\min$ of dd-target and wealth-target) using
$T_{\text{active}}$.

**Formula** (re-cal day):
""")
st.latex(r"T_{\text{active}, t} = T^{*}_{\text{hybrid}}\!(E_t^{\text{real}},\, H - t),\quad \text{maxdd}_t \leftarrow 0,\quad \text{HWM}_t \leftarrow E_t")

st.markdown(r"""
Between re-cal events, target follows hybrid logic with $T_{\text{active}}$:
""")
st.latex(r"T_t = \min\!\big(T_{\text{active}, t} - F \cdot \text{maxdd}_t,\; T_{\text{active}, t} - (T_{\text{active}, t} - \text{floor}) \cdot \text{prog}_t\big)")

st.markdown(r"""
**Properties:**
- The wealth glide and dd ratchet *do* operate between re-cals, but reset at
  every re-cal event (so accumulated drawdown experience is forgotten every
  N years).
- Slightly different from `recal_static`: between re-cals, the wealth-glide
  component still drops target as equity grows past $X$, so the strategy
  can deleverage between re-cals if equity grows fast enough.

**Parameters:**
- `recal_period_months`, $F$, $X$.
""")

# --- recal_adaptive_dd ---
st.subheader("`recal_adaptive_dd` — adaptive_dd + periodic re-cal")
st.markdown(r"""
**Definition:** Like `adaptive_dd` but with periodic state-reset and
$T_{\text{active}}$ refresh. The cushion-coupled $F_{\text{eff}}$ logic
operates between re-cals; at each re-cal, max_dd resets to 0 and
$T_{\text{active}}$ jumps to the lookup value (from the adaptive_dd
lookup table).

**Formula** (re-cal day): same reset pattern as `recal_hybrid`, plus
$\text{cur\_tgt}_t \leftarrow T_{\text{active}, t}$.

Between re-cals, applies adaptive_dd's monotonic-down ratchet using
$T_{\text{active}}$:
""")
st.latex(r"F_{\text{eff}} = F \cdot (L_{\text{now}} - 1) / (T_{\text{active}} - 1)")
st.latex(r"\text{cur\_tgt}_t \leftarrow \min(\text{cur\_tgt}_{t-1},\; \max(\text{floor},\; T_{\text{active}} - F_{\text{eff}} \cdot \text{maxdd}_t))")

st.markdown(r"""
**Parameters:** `recal_period_months`, $F$.

**Note on convergence:** in scenarios where the well-defended $T^{*}$ values
are similar across base strategies (high-DCA, equity rapidly past `wealth_X`),
this behaves nearly identically to `recal_static` and `recal_hybrid`.
Differentiation appears in low-DCA scenarios with deeper drawdowns.
""")

# --- meta_recal ---
st.subheader("`meta_recal` — re-cal + pick-max-T strategy")
st.markdown(r"""
**Definition:** At each re-cal event, looks up $T^{*}$ across multiple
candidate base strategies and picks the one with the **highest** $T^{*}$
for the current cell. Between re-cals, applies the chosen strategy's
logic. The candidate set is hard-coded to {`static` (s_code=0), `dd_decay`
(s_code=2), `adaptive_dd` (s_code=9), `hybrid` (s_code=4)}.

**Formula** (re-cal day):
""")
st.latex(r"s^{*}_t = \arg\max_s\, T^{*}_s\!(E_t^{\text{real}},\, H - t)")
st.latex(r"T_{\text{active}, t} = T^{*}_{s^{*}_t}\!(E_t^{\text{real}},\, H - t)")
st.markdown(r"""
where $s$ ranges over the 4 candidate strategies. State is reset to fresh,
$\text{strat\_active}[k] = s^{*}_t$ records the chosen strategy for the
between-recals phase, $T_{\text{active}}$ is the picked value.

Between re-cal events, the simulation dispatches based on
$\text{strat\_active}[k]$: applies static / dd_decay / adaptive_dd / hybrid
logic accordingly.

**Properties:**
- Picks the **highest target leverage that is well-defended**. By
  construction, ≥ each individual recal_X strategy's target.
- In practice, well-defended $T^{*}$ values are nearly equal across the 4
  candidates in most cells (the safety bar is the binding constraint, not
  the strategy choice). meta_recal often picks `static` everywhere and
  effectively becomes `recal_static`.

**Parameters:** `recal_period_months`, $F$, $X$.

**Open extension:** could include more candidate strategies (e.g.,
`adaptive_hybrid`, `vol_hybrid`) or expand to "pick max expected wealth"
instead of max $T^{*}$ (would require forward-simulation per candidate
at each re-cal — expensive but a better criterion).
""")

st.divider()

# ---------------------------------------------------------------------------
# 8. Wealth cap mechanism
# ---------------------------------------------------------------------------

st.header("8. Wealth cap mechanism")
st.markdown("""
**Independent of strategy.** When real wealth crosses the user-set threshold
`cap_wealth_M`, the strategy *permanently latches* to "no more new leverage."
Re-lever / decay rules continue but never *add* loan; once the cap is
reached on a path, the rebalance step is skipped for the rest of the
horizon.

**Difference from wealth_decay:**
- `wealth_decay` is a smooth glide — target gradually drops toward 1.0x
  as wealth grows, and starts taking effect well before the threshold.
- The wealth cap is a hard one-way switch — strategy operates fully
  until the threshold, then immediately stops levering.

These are complementary: you can run e.g. `hybrid` (smooth glide via
wealth_X) AND have a wealth cap as a backstop. The cap is a behavioral
backstop ("if I'm rich enough, definitely stop adding leverage").
""")

st.divider()

# ---------------------------------------------------------------------------
# 9. Architectural ranking
# ---------------------------------------------------------------------------

st.header("9. Architectural ranking & key findings")
st.markdown("""
From the project's third-session analysis (the bootstrap revealed which
strategies are genuinely robust vs path-overfit). Bootstrap call rates at
each strategy's *historical max-safe target*:

| Rank | Strategy | Bootstrap calls (10% DCA, 30y) | Defensibility |
|---|---|---|---|
| 1 | `static` | 1.46% | Closed-form risk math, single parameter, no active decisions |
| 2 | `adaptive_dd` (F=2-3) | ~2-3% | Couples to actual risk variable, cushion-coupled, monotonic ratchet |
| 3 | `dd_decay` (F=2-3) | 2-2.6% | Couples to actual risk variable, asymmetric ratchet |
| 4 | `dd_decay` (F=0.5-1.5) | 3.3-5.6% | Same as above, less aggressive |
| 5 | `wealth_decay` | 7.84% | Couples to backward-looking variable, gameable by DCA |
| 6 | (deprecated time-decay) | ~10% | Couples to non-risk variable (calendar) |
| 7 | `relever` | 13.74% | Maximally exposed to path randomness |

**Re-calibration strategies** (added in 5th session): the recal_* family
operates differently — they re-pick $T_{init}$ at each event using a
well-defended lookup. Per-event safety = `boot_target` (1%); strategy-level
boot rate ~3% (multi-event compounding). Ranking-wise they sit alongside
hybrid/adaptive on architectural defensibility but trade tail-safety for
sustained leverage and higher upside.

**Key insights:**

1. **Calendar time is not a risk variable.** Any architecture that decays
   based on years elapsed is curve-fitting to historical crisis timing.
   That's why time-decay (deprecated, not in current simulator) is
   architecturally weakest among decay schemes.

2. **HWM is a backward-looking proxy for cushion.** Architectures coupling
   to HWM (wealth_decay) are partially state-aware but miss the actual risk
   metric (current leverage, current drawdown).

3. **Drawdown depth IS the risk variable.** dd_decay and adaptive_dd couple
   to it directly. The architectural critique was correct AND the practical
   payoff is real.

4. **Active management generally doesn't beat passive at honest sizing.**
   Under bootstrap or well-defended constraints, all strategies converge to
   ~12% p50 IRR (10% DCA, 30y). Static is the architectural-physics answer;
   active strategies' apparent IRR edge is largely overfitting.

5. **The hybrid + adaptive variants are the exception** — they capture both
   safety and utility (the unlever-at-$X$ goal), and adaptive_dd genuinely
   improves bootstrap fragility over plain dd_decay.

6. **Buy-the-dip is a trap.** dip_hybrid looks good historically but blows
   up on bootstrap. The historical 0% calls reflects the lucky fact that
   real history mostly has *one* drawdown per cycle; bootstrap exposes
   that bet.
""")

st.divider()

# ---------------------------------------------------------------------------
# 10. Caveats & limitations
# ---------------------------------------------------------------------------

st.header("10. Caveats & limitations")
st.markdown("""
1. **Daily closes only.** Intraday breaches of the 4.0x threshold aren't
   captured. Real margin calls can trigger mid-day on volatile days.

2. **Broker-right-to-tighten not modeled.** During stress, brokers can
   raise house maintenance requirements above the 25% Reg-T baseline.
   This happened in 2008 and 2020. Setting the call threshold to 3.0x
   instead of 4.0x is a way to model this; the simulator currently uses
   4.0x throughout.

3. **Execution costs ignored.** Monthly re-levering assumes zero spreads,
   zero commissions. Realistic cost: ~1-5 bps/yr.

4. **Post-1932 only.** The 1929-1932 era is excluded as unrepresentative
   of modern market structure. If you believe such an event is possible,
   the max-safe ceiling drops to ~1.12-1.14x.

5. **Survivorship bias in the US market.** The SPX backtest uses a market
   that (in hindsight) was one of the best-performing in the 20th century.
   Forward returns may be lower.

6. **30-year sample bias.** The 30-year horizon sample excludes post-1993
   entries (insufficient forward data), so it under-represents recent
   regimes. The 20-year horizon is broader but still excludes very recent
   entries.

7. **Tax model simplified.** Currently assumes **0% effective tax saving**
   on box-spread interest — the conservative hold-forever case where the
   60/40 capital losses generated by the loan have no immediate value
   (no capital gains to offset). If you have realizations from other
   accounts, your real after-tax cost is lower; the simulator does not
   model this. State taxes, AMT, and bracket interactions also unmodeled.

8. **No behavioral modeling.** The analysis assumes perfect execution of
   the strategy. In practice, market panic during drawdowns leads to
   rule-breaking — exactly when discipline matters most.

9. **Single asset (SPX).** No factor tilts, international diversification,
   bonds, or different asset classes. These were discussed in the
   project's notes but not quantified here.

10. **Forward returns unknowable.** Past performance is not predictive.
    Use this tool to think about *relative* properties of strategies,
    not as a literal forecast of your wealth.
""")

st.divider()

# ---------------------------------------------------------------------------
# 11. Glossary
# ---------------------------------------------------------------------------

st.header("11. Glossary")
st.markdown(r"""
- **Box spread**: A four-leg options position that synthetically borrows or
  lends. Short box spreads ≈ borrow at ~Tsy + 10-30 bps, with the implicit
  interest treated as Section-1256 60/40 capital loss for taxes.
- **Bootstrap**: Method of generating synthetic paths by sampling blocks
  from historical data. Tests path-ordering robustness.
- **CAGR**: Compound annual growth rate. With contributions, the naive
  formula over-reports return; use IRR instead.
- **Call threshold**: Leverage level above which the broker forces sale.
  Reg-T = 4.0x (25% maintenance). Portfolio margin = 6.67x (15%) — but
  brokers reserve right to tighten.
- **Cushion**: Distance from current leverage to call threshold. More
  cushion = more headroom for drawdowns.
- **DCA**: Dollar-cost averaging. Here used loosely as "ongoing
  contributions" (deposits into the same SPX account).
- **Drawdown**: Loss from peak. $dd_t = 1 - E_t / \text{HWM}_t$.
- **Floor**: Minimum target leverage. We use 1.0x (fully unlevered) for
  every strategy.
- **HWM**: High-water mark. Peak equity reached so far on a path.
- **IRR**: Internal rate of return. With contributions, the correct
  measure of "rate of return per dollar."
- **Max-safe T**: Largest $T_{init}$ satisfying a safety constraint
  (0% historical calls, ≤1% bootstrap calls, etc).
- **Recal event**: A scheduled point where a `recal_*` strategy re-picks
  its target $T_{\text{active}}$ from a precomputed lookup table based on
  current equity + remaining horizon. Resets path-conditional state.
- **Lookup table $T^{*}$**: Precomputed grid (equity × remaining-horizon)
  of well-defended max-safe $T$ values. Used by recal strategies at each
  event to set the new target.
- **Reg-T**: SEC Regulation T. Sets minimum margin requirements; 25%
  maintenance ⇒ 4.0x leverage cap.
- **Section 1256**: US tax code section that taxes certain options/futures
  contracts as 60/40 long/short capital gains/losses (regardless of
  holding period).
- **SPX-TR**: S&P 500 total return index (price + reinvested dividends).
- **T_init**: Initial leverage target chosen on day 0.
- **Wealth_X**: Wealth threshold (real dollars) at which `wealth_decay` /
  `hybrid` / variants reach the floor leverage of 1.0x.
""")

st.divider()
st.caption(
    "Source code: [github.com/dan8901/margin-simulator]"
    "(https://github.com/dan8901/margin-simulator). "
    "Detailed lab notebooks (incl. negative-result experiments and design "
    "dead-ends) live in `CLAUDE.md` in the repo."
)
