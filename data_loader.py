"""
Shared loader for the historical SPX-TR + margin-rate + CPI series.
Reads spx_margin_history.csv (produced by extract_data.py + merge_cpi.py) so
that analysis scripts have no dependency on the original Excel file.
"""
import csv
import os
from datetime import datetime
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "spx_margin_history.csv")


def load(with_cpi=False):
    """Return (dates, spx_tr, tsy_3m, margin_rate[, cpi]).

    `with_cpi=True` appends a fifth array of CPI-U (CPIAUCNS, monthly,
    forward-filled to daily). `cpi` is a level index, not a rate; divide a
    later value by an earlier one to get cumulative inflation between dates.

    Rows with missing spx_tr or margin_rate are skipped (matching the
    behaviour of the original Excel-based scripts).
    """
    dates, spx, tsy, mr, cpi = [], [], [], [], []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["spx_tr"] or not row["margin_rate"]:
                continue
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d"))
            spx.append(float(row["spx_tr"]))
            tsy.append(float(row["tsy_3m"]) if row["tsy_3m"] else float("nan"))
            mr.append(float(row["margin_rate"]))
            cpi.append(float(row.get("cpi") or "nan"))
    out = (np.array(dates),
           np.array(spx, dtype=float),
           np.array(tsy, dtype=float),
           np.array(mr, dtype=float))
    if with_cpi:
        out = out + (np.array(cpi, dtype=float),)
    return out
