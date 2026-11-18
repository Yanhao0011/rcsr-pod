#!/usr/bin/env bash
# Step 3. Monte Carlo study (manuscript Section 3 and Appendix A).
# Settings (B, replications R, N grid) are those recorded in results/v3/sim29/*_meta.json; the reported run used
# 16 worker processes and the default master seed of sim_v3.py (--seed 20261015). Each experiment draws its
# replication seeds from that master seed, so the results do not depend on the number of workers.
# Output: outputs/v3/sim29/sim_<exp>_summary.csv, sim_<exp>_long.csv, <exp>_meta.json, log_<exp>.txt
# Usage : bash pipeline/run_simulations.sh [experiment ...]   (default: all; JOBS=<workers>, PYTHON=<interpreter>)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RCPO_ROOT="${RCPO_ROOT:-$ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY="${PYTHON:-python}"
J="${JOBS:-16}"
O="$RCPO_ROOT/outputs/v3/sim29"
mkdir -p "$O"
cd "$ROOT/scripts"
S="$PY -u sim_v3.py --outdir $O --jobs $J"
run() {   # run <experiment> <args...>
  local e="$1"; shift
  $S --experiment "$e" "$@" > "$O/log_$e.txt" 2>&1 || echo "experiment $e failed, see $O/log_$e.txt"
}
EXPS="${*:-sep inv frm e1 e4 e2 e5 scale pow}"
for e in $EXPS; do
  case "$e" in
    sep)   run sep   --reps 300 --B 500  --n-grid 5000,25000,100000 ;;
    inv)   run inv   --reps 200 --B 500  --n-grid 25000,100000 ;;
    frm)   run frm   --reps 200 --B 500  --n-grid 25000,100000 --truth-pop 4000000 ;;
    e1)    run e1    --reps 200 --B 500  --n-grid 1000,5000,25000,100000 ;;
    e4)    run e4    --reps 200 --B 500  --n-grid 1000,5000,25000,100000 --post ;;
    e2)    run e2    --reps 200 --B 500  --n-grid 25000,100000 ;;
    e5)    run e5    --reps 200 --B 500  --n-grid 5000,25000,100000 --truth-pop 3000000 ;;
    scale) run scale --reps 50  --B 500  --n-grid 100000,500000 ;;
    pow)   run pow   --reps 300 --B 1000 --n-grid 5000,25000,100000 ;;
    *) echo "unknown experiment: $e"; exit 1 ;;
  esac
done
touch "$O/SIMULATIONS_DONE"
