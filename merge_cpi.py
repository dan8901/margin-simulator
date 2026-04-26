"""
merge_cpi.py — one-shot enrichment of spx_margin_history.csv with monthly CPI-U.

Reads /tmp/cpi.csv (downloaded from FRED, series CPIAUCNS) and merges it into
spx_margin_history.csv as a new `cpi` column. Each daily row gets the CPI value
of the most recent monthly observation on or before that date (forward-fill).

Run once after downloading CPI; the existing CSV is overwritten.
"""

import csv
from datetime import datetime
from pathlib import Path

CPI_PATH = Path("/tmp/cpi.csv")
HIST_PATH = Path(__file__).parent / "spx_margin_history.csv"


def load_cpi():
    """Returns sorted list of (date, cpi) tuples."""
    out = []
    with open(CPI_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["CPIAUCNS"]:
                continue   # skip occasional missing months (e.g. delayed releases)
            d = datetime.strptime(row["observation_date"], "%Y-%m-%d").date()
            out.append((d, float(row["CPIAUCNS"])))
    out.sort()
    return out


def main():
    cpi = load_cpi()
    cpi_dates = [d for d, _ in cpi]
    cpi_vals = [v for _, v in cpi]

    rows = []
    with open(HIST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if "cpi" not in fields:
            fields = fields + ["cpi"]
        for row in reader:
            rows.append(row)

    # Two-pointer ffill since both lists are sorted
    j = 0
    for row in rows:
        d = datetime.strptime(row["date"], "%Y-%m-%d").date()
        while j + 1 < len(cpi_dates) and cpi_dates[j + 1] <= d:
            j += 1
        if cpi_dates[j] <= d:
            row["cpi"] = f"{cpi_vals[j]:.4f}"
        else:
            row["cpi"] = ""

    with open(HIST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged CPI into {HIST_PATH.name}")
    print(f"  CPI coverage: {cpi_dates[0]} to {cpi_dates[-1]}")
    print(f"  History coverage: {rows[0]['date']} to {rows[-1]['date']}")
    print(f"  First row: {rows[0]}")
    print(f"  Last row:  {rows[-1]}")


if __name__ == "__main__":
    main()
