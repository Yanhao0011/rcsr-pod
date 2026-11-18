#!/usr/bin/env bash
# Step 1. FARS-CRSS fusion and data-validity cleaning.
# Input : data/FARS{2021,2022,2023}NationalCSV.zip and data/CRSS{2021,2022,2023}NationalCSV.zip (public NHTSA files)
# Output: outputs/kabco_person_level.csv (fused person-level frame, 649,808 records)
#         outputs/kabco_person_level_clean.csv (cleaned frame read by run_v3_frame.py)
# Usage : bash pipeline/run_data.sh            (PYTHON=<interpreter> to override "python")
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA="${RCPO_DATA:-$ROOT/data}"
OUT="${RCPO_ROOT:-$ROOT}/outputs"
mkdir -p "$OUT"
cd "$ROOT/scripts"
$PY -u build_fars_crss_fusion.py --fars-root "$DATA" --crss-root "$DATA" --output-dir "$OUT" --require-crss || exit 1
$PY -u build_kabco_modeling_data.py --fused-path "$OUT/kabco_person_level.csv" --output-dir "$OUT" || exit 1
