"""
extend_data.py — append SPX-TR + 3M Tsy data after the existing CSV's last date.

Sources:
  - SPX total return index: yfinance ^SP500TR (ratios chained onto last value)
  - 3M Treasury yield:      FRED DGS3MO (already downloaded to /tmp/dgs3mo.csv)
  - CPI:                    re-run merge_cpi.py separately to refresh

The yfinance level base differs from the existing series, so we use daily
RATIOS (Close[d] / Close[d-1]) and multiply onto the last existing spx_tr,
preserving series continuity.

Trading-day calendar: take the union of yfinance and DGS3MO dates after the
cutoff; both should be NYSE trading days. Forward-fill DGS3MO across any
missing days (rare; DGS3MO is daily but sometimes lags one day).
"""

import csv
from datetime import datetime, date, timedelta
from pathlib import Path

import yfinance as yf

HIST_PATH = Path(__file__).parent / "spx_margin_history.csv"
DGS3MO_PATH = Path("/tmp/dgs3mo.csv")
MARGIN_BPS = 0.004   # broker margin = 3M Tsy + 40 bps (project convention)


def load_existing():
    rows = []
    with open(HIST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_dgs3mo():
    """Return dict[date -> float (decimal)]."""
    out = {}
    with open(DGS3MO_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r["DGS3MO"]:
                continue
            d = datetime.strptime(r["observation_date"], "%Y-%m-%d").date()
            out[d] = float(r["DGS3MO"]) / 100.0
    return out


def fetch_sp500tr(start_date, end_date):
    """Return list of (date, close) tuples sorted by date."""
    df = yf.Ticker("^SP500TR").history(
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    out = []
    for ts, row in df.iterrows():
        d = ts.date()
        out.append((d, float(row["Close"])))
    out.sort()
    return out


def main():
    existing = load_existing()
    fields = list(existing[0].keys())   # date, spx_tr, tsy_3m, margin_rate, cpi
    last_row = existing[-1]
    last_date = datetime.strptime(last_row["date"], "%Y-%m-%d").date()
    last_spx_tr = float(last_row["spx_tr"])
    print(f"Existing CSV ends {last_date} with spx_tr={last_spx_tr:.4f}")

    today = date.today()
    sp = fetch_sp500tr(last_date, today)
    print(f"Fetched ^SP500TR: {len(sp)} rows, "
          f"{sp[0][0] if sp else None} to {sp[-1][0] if sp else None}")

    tsy = load_dgs3mo()
    print(f"Loaded DGS3MO: {len(tsy)} rows, last = {max(tsy)}")

    # Find the yfinance row that matches the existing last_date — use it as anchor
    sp_dict = dict(sp)
    if last_date not in sp_dict:
        print(f"WARNING: yfinance has no row for {last_date}; "
              f"using nearest <= as anchor")
        # find closest <= last_date
        candidates = [d for d in sp_dict if d <= last_date]
        if not candidates:
            raise SystemExit("No yfinance overlap with existing series")
        anchor_date = max(candidates)
    else:
        anchor_date = last_date
    anchor_close = sp_dict[anchor_date]
    print(f"yfinance anchor: {anchor_date} close={anchor_close:.2f}")
    print(f"Scale factor (existing/yf at anchor): {last_spx_tr / anchor_close:.6f}")

    # Build new rows: only dates strictly after last_date, where we have
    # both sp and tsy data. Forward-fill tsy if needed.
    new_rows = []
    last_tsy = float(last_row["tsy_3m"]) if last_row["tsy_3m"] else None
    prev_close = anchor_close
    prev_spx_tr = last_spx_tr
    for d, close in sp:
        if d <= last_date:
            prev_close = close
            continue
        # daily ratio
        ratio = close / prev_close
        spx_tr = prev_spx_tr * ratio
        prev_close = close
        prev_spx_tr = spx_tr
        # tsy
        if d in tsy:
            t = tsy[d]
            last_tsy = t
        else:
            t = last_tsy
        if t is None:
            continue
        margin = t + MARGIN_BPS
        new_rows.append({
            "date": d.isoformat(),
            "spx_tr": f"{spx_tr:.6f}",
            "tsy_3m": f"{t:.6f}",
            "margin_rate": f"{margin:.6f}",
            "cpi": "",   # will be filled by merge_cpi.py
        })

    print(f"Appending {len(new_rows)} new rows.")
    if new_rows:
        print(f"  First: {new_rows[0]}")
        print(f"  Last:  {new_rows[-1]}")

    with open(HIST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + new_rows)
    print(f"Wrote {HIST_PATH.name}, total rows: {len(existing) + len(new_rows)}")


if __name__ == "__main__":
    main()
