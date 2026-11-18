#!/usr/bin/env bash
# Step 2. Application (manuscript Section 4 and Appendix C), in the order of the reported run.
# Input : outputs/kabco_person_level_clean.csv (pipeline/run_data.sh) and data/CRSS{2021,2022,2023}NationalCSV.zip
# Output: outputs/v3/ (result files listed in README.md) and one log per step in outputs/v3/logs/
# Usage : bash pipeline/run_application.sh     (PYTHON=<interpreter>, RCPO_ROOT=<root> and RCPO_DATA=<dir> optional)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RCPO_ROOT="${RCPO_ROOT:-$ROOT}"
PY="${PYTHON:-python}"
V3="$RCPO_ROOT/outputs/v3"; L="$V3/logs"
mkdir -p "$L"
cd "$ROOT/scripts"
# de-duplicated frame, recoding, selection controls and the full-frame Firth cut fits
$PY -u run_v3_frame.py pavement,infrastructure > "$L/frame.log" 2>&1 || exit 1
# RCSR-POD departures and simultaneous bounds (B = 1000 draws per covariance variant)
$PY -u run_v3_application.py --pathways pavement,infrastructure --B 1000 --block 32 > "$L/application.log" 2>&1 || exit 1
# coding-invariant (covariate-level) family, Holm/BH, delta sweep, held-out specification sets
$PY -u run_v3_primary.py > "$L/primary.log" 2>&1 || exit 1
# Brant/Wald comparators (feature and covariate level), BIC rule, ML separation check
$PY -u run_v3_brant.py > "$L/brant.log" 2>&1 || exit 1
# five-fold crash-grouped cross-validation of the partial-PO specifications
$PY -u run_v3_heldout.py --summary "$V3/heldout_specs.json" --folds 5 --maxiter 1500 > "$L/heldout.log" 2>&1
# sensitivity analyses: source strata, CRSS design weights, refit check and PO-fit reference, minimum level size 500
for m in strata weighted refit nmin; do
  $PY -u run_v3_sensitivity.py --mode "$m" --summary "$V3/v3_primary_summary.json" --B 1000 > "$L/sens_$m.log" 2>&1
done
# year-pooling likelihood-ratio check (Appendix B)
$PY -u run_v3_yearpool.py > "$L/yearpool.log" 2>&1
# KABCO distribution by source (Table 2)
$PY -u describe_frame.py > "$L/describe_frame.log" 2>&1
touch "$V3/APPLICATION_DONE"
