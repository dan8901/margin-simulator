"""Hybrid Strategy — Portfolio Advisor.

Operational tool for applying the hybrid strategy to a real portfolio.
Each visit: enter total account value + loan balance. Advisor reconstructs
HWM and max_dd ratchets (replaying VTI between visits via yfinance),
computes the hybrid target leverage, and tells you what action to take
(borrow $X to lever up, or hold — strategy never sells).

State persisted in `advisor_state.json` in the project root (gitignored).
"""

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import streamlit as st

from data_loader import load

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "advisor_state.json"

st.set_page_config(page_title="Hybrid Advisor", layout="wide")
st.title("Hybrid Strategy — Portfolio Advisor")
st.caption(
    "Tell me your account value + loan balance. I'll tell you the target "
    "leverage and what to do. State (HWM, max_dd ratchet) is tracked "
    "automatically; VTI total-return history is replayed between visits "
    "to catch interim drawdowns."
)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return None


def save_state(s):
    STATE_PATH.write_text(json.dumps(s, indent=2))


# ---------------------------------------------------------------------------
# Market-data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60)
def fetch_vti(start_iso, end_iso):
    """Fetch VTI dividend+split-adjusted daily closes (total return) between
    start and end (inclusive). Returns sorted list of (date, adj_close).
    Network call; cached 1 hour. Uses 'Adj Close' so dividend reinvestment
    is reflected — required for total-return tracking."""
    import yfinance as yf
    from datetime import timedelta
    start_d = date.fromisoformat(start_iso)
    end_d = date.fromisoformat(end_iso)
    df = yf.Ticker("VTI").history(
        start=start_d.isoformat(),
        end=(end_d + timedelta(days=1)).isoformat(),
        auto_adjust=True,   # auto_adjust=True puts dividend+split-adjusted prices in 'Close'
    )
    out = [(ts.date(), float(row["Close"])) for ts, row in df.iterrows()]
    out.sort()
    return out


@st.cache_data(ttl=24 * 60 * 60)
def fetch_cpi_from_fred():
    """Pull CPIAUCNS (monthly CPI-U) live from FRED's public CSV endpoint.
    No API key. Returns sorted list of (date, level). Cached 24h.
    Each observation_date is the first of the month; CPI applies to that
    month, forward-filled for any date in/after that month."""
    import urllib.request
    import csv as csv_mod
    from io import StringIO
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode("utf-8")
    out = []
    reader = csv_mod.DictReader(StringIO(text))
    for row in reader:
        v = row.get("CPIAUCNS", "")
        if not v or v == ".":
            continue
        d = datetime.strptime(row["observation_date"], "%Y-%m-%d").date()
        out.append((d, float(v)))
    out.sort()
    return out


@st.cache_data
def get_cpi_series_csv():
    """Fallback: CPI from project CSV (forward-filled daily by merge_cpi.py)."""
    dates_arr, _, _, _, cpi_arr = load(with_cpi=True)
    out = []
    for d, c in zip(dates_arr, cpi_arr):
        if np.isnan(c):
            continue
        out.append((d.date() if isinstance(d, datetime) else d, float(c)))
    return out


def _lookup_le(series, target_date):
    """Binary-ish search: largest series[i][0] <= target_date. None if all rows
    are after target_date."""
    lo, hi = 0, len(series) - 1
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= target_date:
            ans = series[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def cpi_at(target_date):
    """CPI at-or-before target_date, sourced live from FRED (with CSV
    fallback). Returns the level. CPI is monthly so any date within or
    after a published month gets that month's level."""
    try:
        series = fetch_cpi_from_fred()
    except Exception:
        series = get_cpi_series_csv()
    if not series:
        return None
    return _lookup_le(series, target_date)


# ---------------------------------------------------------------------------
# Hybrid target formula (mirrors project_portfolio._simulate_core kind 4)
# ---------------------------------------------------------------------------

def compute_hybrid_target(T_init, F, floor, max_dd, real_eq, C, wealth_X,
                           wealth_glide_exp):
    """Returns (target_lev, cand_dd, cand_w, prog)."""
    cand_dd = max(floor, T_init - F * max_dd)
    if wealth_X > C:
        prog = (real_eq - C) / (wealth_X - C)
        prog = max(0.0, min(1.0, prog))
        cand_w = T_init - (T_init - floor) * (prog ** wealth_glide_exp)
    else:
        cand_w = floor
        prog = 1.0
    target_lev = min(cand_dd, cand_w)
    return target_lev, cand_dd, cand_w, prog


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

state = load_state()

# ----- First-time setup -----
if state is None:
    st.info(
        "First-time setup. Enter your strategy params (copy from the main "
        "calibration page after running it for your scenario) and your "
        "current portfolio."
    )
    with st.form("setup"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Strategy params")
            T_init = st.number_input(
                "T_init (calibrated target leverage)",
                value=1.661, step=0.001, format="%.3f",
                help="From the main calibration page (T_rec for hybrid).",
            )
            wealth_X_M = st.number_input(
                "wealth_X — your unlever target ($M, real)",
                value=3.0, step=0.5, min_value=0.1,
                help="Real-dollar wealth above which target = 1.0x.",
            )
            F = st.number_input(
                "F (dd_decay factor)", value=1.5, step=0.1,
                help="target_dd = max(floor, T_init - F × max_dd_ratchet).",
            )
            floor = st.number_input(
                "floor", value=1.0, step=0.1,
                help="Minimum target leverage (1.0 = unlevered).",
            )
            wealth_glide_exp = st.number_input(
                "wealth_glide exponent",
                value=10.0, step=1.0,
                help="Shape of wealth glide. Default 10 (heavily back-loaded).",
            )
        with c2:
            st.subheader("Portfolio today")
            start_date = st.date_input(
                "Strategy start date",
                value=date.today(),
                help="The first day you applied this strategy. Anchors HWM "
                     "and CPI for real-equity calculations.",
            )
            C = st.number_input(
                "Initial real equity C ($)",
                value=160_000, step=10_000, min_value=0,
                help="Your equity (account value − loan) on the start date, "
                     "in real today-dollars. Anchors the wealth glide.",
            )
            current_value = st.number_input(
                "Total market value of VTI ($)",
                value=C, step=10_000, min_value=0,
                help="Gross long market value of your VTI position (the assets "
                     "side, NOT net liq). Equity = this − loan.",
            )
            current_loan = st.number_input(
                "Current loan balance ($)",
                value=0, step=10_000, min_value=0,
            )
        ok = st.form_submit_button("Save setup")
    if ok:
        eq0 = float(current_value) - float(current_loan)
        state = dict(
            start_date=start_date.isoformat(),
            C=float(C),
            T_init=float(T_init),
            wealth_X=float(wealth_X_M) * 1_000_000,
            F=float(F),
            floor=float(floor),
            wealth_glide_exp=float(wealth_glide_exp),
            hwm_eq=max(eq0, float(C)),
            max_dd_ratchet=0.0,
            last_visit_date=start_date.isoformat(),
            last_visit_value=float(current_value),
            last_visit_loan=float(current_loan),
        )
        save_state(state)
        st.success("Setup saved. Reload page to start using the advisor.")
        st.rerun()
    st.stop()


# ----- Regular check-in -----
st.subheader("Today's check-in")
c1, c2 = st.columns(2)
with c1:
    current_value = st.number_input(
        "Total market value of VTI ($)",
        value=float(state["last_visit_value"]),
        step=10_000.0, min_value=0.0,
        help="Gross long market value of your VTI position (assets, NOT "
             "net liq). On most broker dashboards this is 'Long Market "
             "Value' or 'Total Market Value of Securities'.",
    )
with c2:
    current_loan = st.number_input(
        "Current loan balance ($)",
        value=float(state["last_visit_loan"]),
        step=10_000.0, min_value=0.0,
    )

go = st.button("Compute target leverage", type="primary")

if go:
    today = date.today()
    last_visit = date.fromisoformat(state["last_visit_date"])
    last_value = float(state["last_visit_value"])
    last_loan = float(state["last_visit_loan"])
    hwm = float(state["hwm_eq"])
    max_dd = float(state["max_dd_ratchet"])

    # ----- Replay VTI between last visit and today -----
    interim_lo = None  # for diagnostics
    interim_hi = None
    n_interim_days = 0
    if today > last_visit:
        try:
            sp = fetch_vti(last_visit.isoformat(), today.isoformat())
        except Exception as e:
            st.warning(f"yfinance fetch failed ({e}); skipping interim replay.")
            sp = []
        if len(sp) >= 2:
            anchor_close = sp[0][1]
            n_interim_days = len(sp) - 1
            for d, close in sp[1:]:
                # Scale assets by VTI total-return ratio; hold loan flat
                # (intra-window loan growth is immaterial — current loan
                # in the snapshot is exact).
                ratio = close / anchor_close
                assets_d = last_value * ratio
                eq_d = assets_d - last_loan
                if eq_d <= 0:
                    continue
                if eq_d > hwm:
                    hwm = eq_d
                if hwm > 0:
                    dd = 1.0 - eq_d / hwm
                    if dd > max_dd:
                        max_dd = dd
                interim_lo = eq_d if interim_lo is None else min(interim_lo, eq_d)
                interim_hi = eq_d if interim_hi is None else max(interim_hi, eq_d)

    # ----- Snapshot today -----
    current_eq = current_value - current_loan
    if current_eq <= 0:
        st.error(
            f"Equity is non-positive (${current_eq:,.0f}). The account is "
            "already in margin-call territory. Cannot compute a target."
        )
        st.stop()

    if current_eq > hwm:
        hwm = current_eq
    if hwm > 0:
        dd_now = 1.0 - current_eq / hwm
        if dd_now > max_dd:
            max_dd = dd_now

    current_lev = current_value / current_eq

    # ----- Compute hybrid target -----
    cpi_start = cpi_at(date.fromisoformat(state["start_date"]))
    cpi_today = cpi_at(today)
    if cpi_start is None or cpi_today is None:
        st.error("CPI lookup failed (FRED unreachable + CSV empty).")
        st.stop()

    real_eq_today = current_eq * cpi_start / cpi_today

    target_lev, cand_dd, cand_w, prog = compute_hybrid_target(
        T_init=state["T_init"],
        F=state["F"],
        floor=state["floor"],
        max_dd=max_dd,
        real_eq=real_eq_today,
        C=state["C"],
        wealth_X=state["wealth_X"],
        wealth_glide_exp=state["wealth_glide_exp"],
    )

    target_assets = target_lev * current_eq
    delta_loan = target_assets - current_value

    # ----- Display -----
    st.divider()
    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Target leverage", f"{target_lev:.3f}x")
    m2.metric(
        "Current leverage",
        f"{current_lev:.3f}x",
        delta=f"{(current_lev - target_lev):+.3f}x vs target",
        delta_color="off",
    )
    m3.metric("Real equity (start-date $)", f"${real_eq_today:,.0f}")

    if delta_loan > 100:
        st.success(
            f"### Action: Borrow ${delta_loan:,.0f} and buy more VTI.\n\n"
            f"This brings your leverage from {current_lev:.3f}x up to "
            f"{target_lev:.3f}x."
        )
    elif delta_loan < -100:
        st.info(
            f"### Action: Hold.\n\n"
            f"You're {-delta_loan:,.0f} dollars above target — but the "
            f"hybrid strategy never sells (hold-forever rule). Wait for "
            f"asset growth or DCA to dilute leverage back to target."
        )
    else:
        st.info("### Action: Hold. You're at target.")

    with st.expander("Diagnostics — math behind the result"):
        st.markdown(f"""
**Current state:**
- account value = ${current_value:,.0f}
- loan = ${current_loan:,.0f}
- equity = ${current_eq:,.0f}
- nominal HWM = ${hwm:,.0f}
- max_dd ratchet = {max_dd:.2%}
- real equity (in start-date $) = ${real_eq_today:,.0f}

**dd_decay component:**
- cand_dd = max(floor, T_init − F × max_dd_ratchet)
- = max({state['floor']:.2f}, {state['T_init']:.3f} − {state['F']:.2f} × {max_dd:.4f})
- = **{cand_dd:.3f}x**

**wealth_decay component:**
- progress = (real_eq − C) / (wealth_X − C)
- = ({real_eq_today:,.0f} − {state['C']:,.0f}) / ({state['wealth_X']:,.0f} − {state['C']:,.0f})
- = {prog:.4f}
- cand_w = T_init − (T_init − floor) × progress^{state['wealth_glide_exp']:.0f}
- = **{cand_w:.3f}x**

**Target:**
- target = min(cand_dd, cand_w) = **{target_lev:.3f}x**

**Action:**
- target_assets = target × equity = {target_lev:.3f} × ${current_eq:,.0f} = ${target_assets:,.0f}
- delta_loan = target_assets − current_value = ${delta_loan:,.0f}
""")
        if n_interim_days > 0:
            interim_dd = (1.0 - interim_lo / max(interim_hi, 1)) if interim_lo else 0
            st.markdown(f"""
**Interim replay (between {last_visit} and {today}):**
- {n_interim_days} trading days replayed via SPX-TR
- low equity in window ≈ ${interim_lo:,.0f}, high ≈ ${interim_hi:,.0f}
- interim drawdown range ≈ {interim_dd:.2%}
""")

    # ----- Persist state -----
    state["hwm_eq"] = hwm
    state["max_dd_ratchet"] = max_dd
    state["last_visit_date"] = today.isoformat()
    state["last_visit_value"] = current_value
    state["last_visit_loan"] = current_loan
    save_state(state)
    st.caption(f"State saved (last visit: {today.isoformat()}).")


# ----- Advanced: state inspection / edit / reset -----
REQUIRED_KEYS = {
    "start_date", "C", "T_init", "wealth_X", "F", "floor", "wealth_glide_exp",
    "hwm_eq", "max_dd_ratchet", "last_visit_date", "last_visit_value",
    "last_visit_loan",
}

with st.expander("State (edit / reset)"):
    st.caption(
        "Edit the JSON below and click **Save** to persist. Useful if you "
        "need to fix HWM, max_dd_ratchet, or any setup value without going "
        "through reset + re-setup."
    )
    edited = st.text_area(
        "advisor_state.json",
        value=json.dumps(state, indent=2),
        height=320,
        key="state_editor",
    )
    st.download_button(
        "⬇ Download state JSON (backup)",
        data=json.dumps(state, indent=2),
        file_name="advisor_state.json",
        mime="application/json",
        help="Saves your current state to your Downloads folder. Useful "
             "before Streamlit Cloud recycles the container, or as a "
             "snapshot you can paste back in via the editor.",
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Save edits", type="primary"):
        try:
            new_state = json.loads(edited)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            missing = REQUIRED_KEYS - set(new_state.keys())
            if missing:
                st.error(f"Missing required keys: {sorted(missing)}")
            else:
                save_state(new_state)
                st.success("State saved. Page will reload.")
                st.rerun()
    if c2.button("Reload from disk"):
        st.rerun()
    if c3.button("Reset all state (start over)"):
        STATE_PATH.unlink(missing_ok=True)
        st.rerun()
