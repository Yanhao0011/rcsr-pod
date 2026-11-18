"""Data-validity cleaning helpers used by build_kabco_modeling_data.py.

The three functions are copied verbatim from the helper module used for the reported runs; nothing else of that
module is needed by the RCSR-POD pipeline.
"""
from __future__ import annotations

import pandas as pd


def coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    coerced = frame.copy()
    for column in columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def fill_missing(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    filled = frame.copy()
    for column in columns:
        series = filled[column]
        if pd.api.types.is_numeric_dtype(series):
            filled[column] = series.fillna(series.median())
        else:
            mode = series.mode(dropna=True)
            filled[column] = series.fillna(mode.iloc[0] if not mode.empty else 0)
    return filled


def clean_special_codes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in columns:
        if column not in cleaned:
            continue
        series = pd.to_numeric(cleaned[column], errors="coerce")
        if column == "LONGITUD":
            series = series.mask((series < -180) | (series > 180))
        elif column == "LATITUDE":
            series = series.mask((series < -90) | (series > 90))
        elif column == "avg_person_age":
            series = series.mask((series < 0) | (series > 120))
        elif column in {"veh_trav_sp", "veh_vspd_lim"}:
            series = series.mask((series < 0) | (series >= 97))
        elif column.startswith("mode_") or column in {
            "RUR_URB",
            "REL_ROAD",
            "ROUTE",
            "FUNC_SYS",
            "TYP_INT",
            "WRK_ZONE",
            "HARM_EV",
            "MAN_COLL",
        }:
            series = series.mask(series >= 97)
        cleaned[column] = series
    return cleaned
