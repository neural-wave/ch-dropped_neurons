"""Write the two submission CSVs in the exact AIL format.

`;` separator, UTF-8, TRUE/FALSE, dates YYYY-MM-DD, kWp with 2 decimals, empty date/kWp for non-PV customers, one row per
customer of the test file (file order). The extension file is always written (header only when there is none).
Inputs: customer ids, PV predictions DataFrame[customer_id, pv_installation_date, pv_capacity_kwp] (PV customers only),
extensions DataFrame[customer_id, extension_date, pv_capacity_after_extension_kwp]. Outputs: predictions.csv and
pv_extensions_predictions.csv in out_dir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEP = ";"
MAIN_COLS = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXT_COLS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text("".join(SEP.join(r) + "\n" for r in [header] + rows), encoding="utf-8")


def write_outputs(out_dir: str | Path, ids: list[str], pv: pd.DataFrame, ext: pd.DataFrame) -> None:
    """Write predictions.csv and pv_extensions_predictions.csv into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {str(r.customer_id): r for r in pv.itertuples()}
    rows = []
    for c in ids:
        r = by_id.get(c)
        rows.append([c, "FALSE", "", ""] if r is None else
                    [c, "TRUE", pd.Timestamp(r.pv_installation_date).strftime("%Y-%m-%d"), f"{float(r.pv_capacity_kwp):.2f}"])
    keep = set(ids)
    erows = []
    for r in ext.itertuples():
        d = pd.to_datetime(r.extension_date, errors="coerce")
        k = pd.to_numeric(r.pv_capacity_after_extension_kwp, errors="coerce")
        if str(r.customer_id) in keep and not pd.isna(d):
            erows.append([str(r.customer_id), d.strftime("%Y-%m-%d"), f"{k:.2f}" if np.isfinite(k) and k > 0 else ""])
    _write(out / "predictions.csv", MAIN_COLS, rows)
    _write(out / "pv_extensions_predictions.csv", EXT_COLS, erows)
