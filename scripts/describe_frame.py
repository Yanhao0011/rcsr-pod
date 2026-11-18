#!/usr/bin/env python3
"""KABCO outcome distribution of the de-duplicated analysis frame by source (manuscript Table 2).

Reads outputs/v3/kabco_person_level_v3.csv (written by run_v3_frame.py) and writes
outputs/v3/frame_kabco_by_source.csv. Paths are relative to RCPO_ROOT (default: the repository root).
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("RCPO_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "outputs" / "v3"
LEVELS = ["No apparent injury (O)", "Possible injury (C)", "Minor injury (B)", "Serious injury (A)", "Fatal injury (K)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=str(OUT / "kabco_person_level_v3.csv"))
    ap.add_argument("--out", default=str(OUT / "frame_kabco_by_source.csv"))
    a = ap.parse_args()
    fr = pd.read_csv(a.frame, usecols=["record_origin", "kabco_ordinal"], low_memory=False)
    tab = pd.crosstab(fr["kabco_ordinal"].astype(int), fr["record_origin"].astype(str).str.upper())
    tab = tab.reindex(index=range(5), columns=["FARS", "CRSS"], fill_value=0)
    tab.columns.name = None
    tab["Total"] = tab["FARS"] + tab["CRSS"]
    tab.index = LEVELS
    tab.loc["Total"] = tab.sum()
    tab["Share"] = tab["Total"] / tab.loc["Total", "Total"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index_label="KABCO level")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
