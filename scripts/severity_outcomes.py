"""Severity outcome construction utilities.

Provides the 5-level KABCO injury severity ladder used by the upgraded
modeling pipeline. Replaces the legacy fatal-only count outcome.

KABCO mapping aligned to FARS INJ_SEV codes (also valid for CRSS PER_INJSEV):
    0 -> No Apparent Injury (O)
    1 -> Possible Injury (C)
    2 -> Suspected Minor Injury (B)
    3 -> Suspected Serious Injury (A)
    4 -> Fatal Injury (K)
    5 -> Injured, Severity Unknown
    6 -> Died Prior to Crash
    9 -> Unknown / Not Reported

We map to a 5-level ordinal outcome:
    0 no_injury, 1 possible_injury, 2 minor_injury, 3 serious_injury, 4 fatal_injury
"""

from __future__ import annotations

import pandas as pd


KABCO_LABELS = {
    0: "no_injury",
    1: "possible_injury",
    2: "minor_injury",
    3: "serious_injury",
    4: "fatal_injury",
}

LABEL_TO_ORDINAL = {label: code for code, label in KABCO_LABELS.items()}

INJ_SEV_TO_KABCO = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: pd.NA,
    6: pd.NA,
    9: pd.NA,
}

BINARY_SEVERE_OR_FATAL = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
BINARY_FATAL_VS_REST = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}


def map_inj_sev_to_ordinal(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(INJ_SEV_TO_KABCO)
    return pd.to_numeric(mapped, errors="coerce")


def ordinal_to_label(series: pd.Series) -> pd.Series:
    return series.map(KABCO_LABELS).astype("string")


def build_severe_or_fatal(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(BINARY_SEVERE_OR_FATAL), errors="coerce")


def build_fatal_vs_rest(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(BINARY_FATAL_VS_REST), errors="coerce")


def survival_time_class(death_tm: pd.Series, death_da: pd.Series) -> pd.Series:
    """Survival-time stratification of fatal injuries (literature 1, Yasmin 2015).

    Returns:
        0 immediate (DEATH_TM available, < 60 minutes)
        1 same-day non-immediate
        2 1-7 days
        3 8-30 days
        pd.NA otherwise
    """
    minutes = pd.to_numeric(death_tm, errors="coerce")
    days = pd.to_numeric(death_da, errors="coerce")
    bucket = pd.Series(pd.NA, index=minutes.index, dtype="object")
    bucket[minutes < 60] = 0
    bucket[(minutes >= 60) & (days.fillna(0) == 0)] = 1
    bucket[(days >= 1) & (days <= 7)] = 2
    bucket[(days >= 8) & (days <= 30)] = 3
    return bucket
