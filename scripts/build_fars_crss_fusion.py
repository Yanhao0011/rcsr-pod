"""FARS-CRSS fusion pipeline (literature 1, Yasmin et al., 2015).

Builds a unified person-level dataset spanning the full KABCO severity ladder
by replacing CRSS fatal observations with the more detailed FARS records.

Inputs (expected paths):
    data/FARS{YEAR}NationalCSV.zip       -- download from NHTSA (https://www.nhtsa.gov/file-downloads)
    data/CRSS{YEAR}NationalCSV.zip       -- download from NHTSA (https://www.nhtsa.gov/file-downloads)

If CRSS zips are absent, the script emits a person-level FARS-only dataset
with the 5-level KABCO outcome (non-fatal persons inside fatal crashes are
naturally present in FARS), and prints a clear warning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from severity_outcomes import (
    build_fatal_vs_rest,
    build_severe_or_fatal,
    map_inj_sev_to_ordinal,
    ordinal_to_label,
    survival_time_class,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# CRSS design columns, required by the design-weighted sensitivity analysis
# (CRSS design weights); do not remove.
CRSS_DESIGN_COLS = ["WEIGHT", "PSU", "PSU_VAR", "PSUSTRAT", "STRATUM", "STRATUMNAME"]

PERSON_KEEP_COLS = [
    "ST_CASE",
    "VEH_NO",
    "PER_NO",
    "AGE",
    "SEX",
    "PER_TYP",
    "INJ_SEV",
    "REST_USE",
    "HELM_USE",
    "DRINKING",
    "DRUGS",
    "DOA",
    "DEATH_TM",
    "DEATH_DA",
    "DEATH_MO",
    "DEATH_YR",
] + CRSS_DESIGN_COLS

ACCIDENT_KEEP_COLS = [
    "ST_CASE",
    "STATE",
    "YEAR",
    "MONTH",
    "DAY_WEEK",
    "HOUR",
    "MINUTE",
    "LATITUDE",
    "LONGITUD",
    "ROUTE",
    "RUR_URB",
    "FUNC_SYS",
    "HARM_EV",
    "MAN_COLL",
    "TYP_INT",
    "REL_ROAD",
    "WRK_ZONE",
    "LGT_COND",
    "WEATHER",
    "FATALS",
] + CRSS_DESIGN_COLS

VEHICLE_KEEP_COLS = [
    "ST_CASE",
    "VEH_NO",
    "BODY_TYP",
    "TRAV_SP",
    "VSPD_LIM",
    "VPAVETYP",
    "VSURCOND",
    "VTRAFWAY",
    "VNUM_LAN",
    "VALIGN",
    "VPROFILE",
    "VTRAFCON",
    "ACC_TYPE",
    "IMPACT1",
    "DEATHS",
    "DR_DRINK",
    "SPEEDREL",
    "ROLLOVER",
]


@dataclass
class FusionConfig:
    years: list[int]
    fars_root: Path
    crss_root: Path
    output_dir: Path
    fars_only: bool
    require_crss: bool


def parse_args() -> FusionConfig:
    parser = argparse.ArgumentParser(description="Fuse FARS and CRSS person-level data into a KABCO severity dataset.")
    parser.add_argument("--years", nargs="*", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--fars-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--crss-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fars-only", action="store_true", help="Skip CRSS pooling; produce FARS-only KABCO dataset.")
    parser.add_argument("--require-crss", action="store_true", help="Error out if CRSS zips are missing.")
    args = parser.parse_args()
    return FusionConfig(
        years=args.years,
        fars_root=args.fars_root,
        crss_root=args.crss_root,
        output_dir=args.output_dir,
        fars_only=args.fars_only,
        require_crss=args.require_crss,
    )


def find_zip(root: Path, source: str, year: int) -> Path | None:
    candidates = [
        root / f"{source}{year}NationalCSV.zip",
        root / f"{source}{year}.zip",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


CRSS_TO_FARS_RENAME = {
    "CASENUM": "ST_CASE",
}


def read_table_from_zip(zip_path: Path, filename: str, keep: list[str]) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        target = next(
            (name for name in archive.namelist() if name.lower().endswith(f"/{filename.lower()}") or name.lower() == filename.lower()),
            None,
        )
        if target is None:
            return pd.DataFrame(columns=keep)
        with archive.open(target) as handle:
            frame = pd.read_csv(handle, low_memory=False)
    frame.columns = [column.upper() for column in frame.columns]
    frame = frame.rename(columns=CRSS_TO_FARS_RENAME)
    available = [column for column in keep if column in frame.columns]
    return frame.loc[:, available].copy()


def load_year_tables(zip_path: Path, year: int) -> dict[str, pd.DataFrame]:
    accident = read_table_from_zip(zip_path, "accident.csv", ACCIDENT_KEEP_COLS)
    if "YEAR" not in accident.columns:
        accident["YEAR"] = year
    vehicle = read_table_from_zip(zip_path, "vehicle.csv", VEHICLE_KEEP_COLS)
    person = read_table_from_zip(zip_path, "person.csv", PERSON_KEEP_COLS)
    return {"accident": accident, "vehicle": vehicle, "person": person}


def collapse_vehicle_to_crash(vehicle: pd.DataFrame) -> pd.DataFrame:
    if vehicle.empty or "ST_CASE" not in vehicle.columns:
        return pd.DataFrame(columns=["ST_CASE"])
    numeric_cols = ["TRAV_SP", "VSPD_LIM", "VNUM_LAN", "DEATHS", "DR_DRINK", "SPEEDREL", "ROLLOVER"]
    for column in numeric_cols:
        if column in vehicle.columns:
            vehicle[column] = pd.to_numeric(vehicle[column], errors="coerce")
    grouped = vehicle.groupby("ST_CASE")
    summary = grouped.size().rename("vehicle_count").to_frame()
    agg_map = {
        "TRAV_SP": "mean",
        "VSPD_LIM": "mean",
        "VNUM_LAN": "mean",
        "DEATHS": "sum",
        "DR_DRINK": "sum",
        "SPEEDREL": "max",
        "ROLLOVER": "max",
    }
    for column, how in agg_map.items():
        if column in vehicle.columns:
            summary[f"veh_{column.lower()}"] = getattr(grouped[column], how)()
    mode_cols = ["BODY_TYP", "VPAVETYP", "VSURCOND", "VTRAFWAY", "VALIGN", "VPROFILE", "VTRAFCON", "ACC_TYPE", "IMPACT1"]
    for column in mode_cols:
        if column in vehicle.columns:
            summary[f"mode_{column.lower()}"] = grouped[column].apply(lambda s: s.mode(dropna=True).iloc[0] if not s.dropna().empty else np.nan)
    return summary.reset_index()


def build_person_level_frame(year_tables: dict[str, pd.DataFrame], source_tag: str) -> pd.DataFrame:
    person = year_tables["person"].copy()
    accident = year_tables["accident"].copy()
    vehicle_summary = collapse_vehicle_to_crash(year_tables["vehicle"].copy())

    if person.empty or accident.empty or "ST_CASE" not in person.columns or "ST_CASE" not in accident.columns:
        return pd.DataFrame()

    if "INJ_SEV" not in person.columns:
        return pd.DataFrame()
    person["INJ_SEV"] = pd.to_numeric(person["INJ_SEV"], errors="coerce")
    person["kabco_ordinal"] = map_inj_sev_to_ordinal(person["INJ_SEV"])
    person["severity_label"] = ordinal_to_label(person["kabco_ordinal"])
    person["severe_or_fatal"] = build_severe_or_fatal(person["kabco_ordinal"])
    person["fatal_vs_rest"] = build_fatal_vs_rest(person["kabco_ordinal"])

    if {"DEATH_TM", "DEATH_DA"}.issubset(person.columns):
        person["survival_class"] = survival_time_class(person["DEATH_TM"], person["DEATH_DA"])
    else:
        person["survival_class"] = pd.NA

    person = person.dropna(subset=["kabco_ordinal"]).copy()

    merged = person.merge(accident, on="ST_CASE", how="left", suffixes=("", "_acc"))
    if not vehicle_summary.empty:
        merged = merged.merge(vehicle_summary, on="ST_CASE", how="left")
    merged["source"] = source_tag
    return merged


def _assign_design_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill WEIGHT / PSU / STRATUM for FARS records and validate CRSS records.

    FARS is a fatal-crash census: every person record contributes weight 1
    and the crash itself (ST_CASE) is treated as its own PSU. CRSS records
    already carry the public-use file's WEIGHT / PSU / STRATUM columns;
    we coerce them to numeric and complain loudly if anything is missing.
    """
    out = frame.copy()
    is_fars = out["record_origin"].astype(str) == "fars"
    is_crss = out["record_origin"].astype(str) == "crss"

    # WEIGHT: numeric, > 0 for valid CRSS rows, exactly 1.0 for FARS rows
    if "WEIGHT" not in out.columns:
        out["WEIGHT"] = pd.NA
    out["WEIGHT"] = pd.to_numeric(out["WEIGHT"], errors="coerce")
    out.loc[is_fars, "WEIGHT"] = 1.0
    crss_bad_weight = is_crss & (out["WEIGHT"].isna() | (out["WEIGHT"] <= 0))
    if crss_bad_weight.any():
        raise RuntimeError(
            f"Found {int(crss_bad_weight.sum())} CRSS records with "
            "missing or non-positive WEIGHT. Likely the upstream "
            "PERSON_KEEP_COLS whitelist no longer includes WEIGHT, or "
            "the raw CRSS file changed schema."
        )

    # PSU: integer identifier for primary sampling unit. FARS records
    # become their own PSU; this is the conventional "PSU = unique crash"
    # treatment used by the survey package's `ids = ~ST_CASE` recipe.
    if "PSU" not in out.columns:
        out["PSU"] = pd.NA
    out.loc[is_fars, "PSU"] = out.loc[is_fars, "ST_CASE"]
    crss_no_psu = is_crss & out["PSU"].isna()
    if crss_no_psu.any():
        raise RuntimeError(
            f"{int(crss_no_psu.sum())} CRSS records missing PSU. "
            "Re-check fusion-stage column whitelist."
        )

    # STRATUM: separate FARS rows into their own per-year stratum so that
    # CRSS-PSU codes can never collide with FARS-PSU codes downstream.
    if "STRATUM" not in out.columns:
        out["STRATUM"] = pd.NA
    out["STRATUM"] = out["STRATUM"].astype("string")
    fars_strata = "fars_" + out.loc[is_fars, "source"].astype(str)
    out.loc[is_fars, "STRATUM"] = fars_strata.values
    crss_no_stratum = is_crss & out["STRATUM"].isna()
    if crss_no_stratum.any():
        raise RuntimeError(
            f"{int(crss_no_stratum.sum())} CRSS records missing STRATUM."
        )

    return out


def fuse_fars_and_crss(fars_frame: pd.DataFrame, crss_frame: pd.DataFrame) -> pd.DataFrame:
    """Yasmin 2015 fusion: replace CRSS fatal observations with FARS detailed records.

    Post-fusion, WEIGHT/PSU/STRATUM are filled in so that downstream
    estimation can run weighted SML and design-based variance.
    """
    if crss_frame.empty:
        out = fars_frame.assign(record_origin="fars_only")
        return _assign_design_columns(out.assign(record_origin="fars"))
    crss_non_fatal = crss_frame[crss_frame["kabco_ordinal"] != 4].copy()
    crss_non_fatal["record_origin"] = "crss"
    fars_block = fars_frame.copy()
    fars_block["record_origin"] = "fars"
    combined = pd.concat([fars_block, crss_non_fatal], ignore_index=True, sort=False)
    return _assign_design_columns(combined)


def write_diagnostics(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dist = frame.groupby("severity_label", dropna=False).size().reset_index(name="count")
    dist["share"] = dist["count"] / dist["count"].sum()
    dist.to_csv(output_dir / "fusion_severity_distribution.csv", index=False)

    if "record_origin" in frame.columns:
        origin = frame.groupby(["record_origin", "severity_label"], dropna=False).size().reset_index(name="count")
        origin.to_csv(output_dir / "fusion_record_origin_by_severity.csv", index=False)


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    fars_blocks: list[pd.DataFrame] = []
    crss_blocks: list[pd.DataFrame] = []
    missing_crss_years: list[int] = []

    for year in config.years:
        fars_zip = find_zip(config.fars_root, "FARS", year)
        if fars_zip is None:
            raise FileNotFoundError(f"FARS zip for {year} not found under {config.fars_root}")
        fars_blocks.append(build_person_level_frame(load_year_tables(fars_zip, year), source_tag=f"FARS-{year}"))

        if not config.fars_only:
            crss_zip = find_zip(config.crss_root, "CRSS", year)
            if crss_zip is None:
                missing_crss_years.append(year)
                continue
            crss_blocks.append(build_person_level_frame(load_year_tables(crss_zip, year), source_tag=f"CRSS-{year}"))

    fars_frame = pd.concat(fars_blocks, ignore_index=True, sort=False) if fars_blocks else pd.DataFrame()
    crss_frame = pd.concat(crss_blocks, ignore_index=True, sort=False) if crss_blocks else pd.DataFrame()

    if missing_crss_years and not config.fars_only:
        message = f"WARNING: CRSS zips not found for years {missing_crss_years}. Output will be FARS-only."
        print(message)
        if config.require_crss:
            raise FileNotFoundError(message)

    fused = fuse_fars_and_crss(fars_frame, crss_frame)
    fused_path = config.output_dir / "kabco_person_level.csv"
    fused.to_csv(fused_path, index=False)
    write_diagnostics(fused, config.output_dir)

    print(f"Person-level KABCO dataset written: {fused_path}")
    print(f"Total person records: {len(fused):,}")
    if "severity_label" in fused.columns:
        print(fused["severity_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
