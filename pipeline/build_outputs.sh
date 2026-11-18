#!/usr/bin/env bash
# Step 4. Appendix tables (LaTeX) from a results directory.
# Usage : bash pipeline/build_outputs.sh [results_dir]
#         results_dir defaults to outputs/v3 (a fresh run); use results/v3 for the results shipped with the repository.
# Output: outputs/tables/v3_tab_{sim,app,sens}.tex
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
RES="${1:-$ROOT/outputs/v3}"
case "$RES" in /*|[A-Za-z]:*) ;; *) RES="$PWD/$RES" ;; esac
export RCSR_RESULTS="$RES"
export RCSR_TABLES="${RCSR_TABLES:-$ROOT/outputs/tables}"
cd "$ROOT/scripts"
$PY -u build_v3_supp_tables.py || exit 1
echo "tables -> $RCSR_TABLES"
