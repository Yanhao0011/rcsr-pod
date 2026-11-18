"""Build person-level KABCO modeling dataset from fused FARS-CRSS source.

Reads outputs/kabco_person_level.csv (from build_fars_crss_fusion.py),
applies data-validity cleaning, collapses rare categorical levels, and
emits modeling-ready samples for the infrastructure and pavement pathways.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from frame_cleaning import clean_special_codes, coerce_numeric, fill_missing


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
FUSED_PATH = OUTPUT_DIR / "kabco_person_level.csv"

INFRASTRUCTURE_FEATURES = [
    "RUR_URB",
    "REL_ROAD",
    "ROUTE",
    "FUNC_SYS",
    "TYP_INT",
    "WRK_ZONE",
    "LGT_COND",
    "WEATHER",
    "HARM_EV",
    "MAN_COLL",
    "mode_acc_type",
    "mode_impact1",
    "mode_vtrafway",
    "mode_vtrafcon",
]

PAVEMENT_FEATURES = [
    "mode_vpavetyp",
    "mode_vsurcond",
    "veh_vspd_lim",
    "veh_trav_sp",
    "veh_vnum_lan",
    "mode_valign",
    "mode_vprofile",
]

CONTROL_FEATURES = [
    "AGE",
    "SEX",
    "PER_TYP",
    "REST_USE",
    "HELM_USE",
    "DRINKING",
    "veh_dr_drink",
    "veh_speedrel",
    "veh_rollover",
    "vehicle_count",
    "veh_deaths",
    "YEAR",
    "MONTH",
    "HOUR",
    "LONGITUD",
    "LATITUDE",
]

ALL_FEATURES = INFRASTRUCTURE_FEATURES + PAVEMENT_FEATURES + CONTROL_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare modeling-ready KABCO datasets.")
    parser.add_argument("--fused-path", type=Path, default=FUSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rare-min-count", type=int, default=100)
    parser.add_argument("--max-rows-per-pathway", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def collapse_rare_categories(frame: pd.DataFrame, categorical_cols: list[str], min_count: int) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical_cols:
        if column not in out.columns:
            continue
        series = out[column].astype("string").fillna("MISSING")
        counts = series.value_counts()
        keep = counts[counts >= min_count].index
        out[column] = np.where(series.isin(keep), series, "OTHER")
    return out


def assemble_pathway_sample(
    frame: pd.DataFrame,
    pathway_features: list[str],
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    # CRSS design columns, needed by the design-weighted sensitivity analysis; do not drop WEIGHT/PSU/STRATUM here.
    design_cols = ["WEIGHT", "PSU", "STRATUM"]
    cols_needed = ["kabco_ordinal", "severity_label", "severe_or_fatal", "fatal_vs_rest", "record_origin", "source", *design_cols, *pathway_features, *CONTROL_FEATURES]
    cols_present = [c for c in cols_needed if c in frame.columns]
    missing_design = [c for c in design_cols if c not in frame.columns]
    if missing_design:
        raise RuntimeError(
            f"Design columns missing from fused frame: {missing_design}. "
            "Re-run build_fars_crss_fusion.py with the corrected KEEP_COLS."
        )
    sub = frame.loc[:, cols_present].dropna(subset=["kabco_ordinal"]).copy()

    if len(sub) <= max_rows:
        return sub.reset_index(drop=True)

    per_class = max(50, max_rows // sub["kabco_ordinal"].nunique())
    parts = []
    for level, group in sub.groupby("kabco_ordinal"):
        n = min(len(group), per_class)
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.fused_path.exists():
        raise FileNotFoundError(
            f"Fused dataset not found: {args.fused_path}. Run build_fars_crss_fusion.py first."
        )

    frame = pd.read_csv(args.fused_path, low_memory=False)
    print(f"Loaded fused dataset: {len(frame):,} person-level records.")

    available_features = [c for c in ALL_FEATURES if c in frame.columns]
    categorical_cols = [c for c in available_features if c not in {"AGE", "veh_trav_sp", "veh_vspd_lim", "veh_vnum_lan", "veh_dr_drink", "vehicle_count", "veh_deaths", "LONGITUD", "LATITUDE", "YEAR", "MONTH", "HOUR"}]

    frame = coerce_numeric(frame, [c for c in available_features if c not in categorical_cols])
    frame = clean_special_codes(frame, available_features)
    frame = collapse_rare_categories(frame, categorical_cols, args.rare_min_count)
    frame = fill_missing(frame, available_features)

    full_path = args.output_dir / "kabco_person_level_clean.csv"
    frame.to_csv(full_path, index=False)
    print(f"Cleaned dataset written: {full_path}")

    for pathway_name, pathway_features in (("infrastructure", INFRASTRUCTURE_FEATURES), ("pavement", PAVEMENT_FEATURES)):
        sample = assemble_pathway_sample(frame, pathway_features, args.max_rows_per_pathway, args.random_state)
        out_path = args.output_dir / f"kabco_{pathway_name}_sample.csv"
        sample.to_csv(out_path, index=False)
        print(f"{pathway_name} sample: {len(sample):,} rows -> {out_path}")

    dist = frame.groupby("severity_label", dropna=False).size().reset_index(name="count")
    dist["share"] = dist["count"] / dist["count"].sum()
    dist.to_csv(args.output_dir / "kabco_severity_distribution_clean.csv", index=False)
    print("KABCO severity distribution after cleaning:")
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
