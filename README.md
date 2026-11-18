# RCSR-POD: companion code

Code and result files for the manuscript on the Relevance-Calibrated, Separation-Robust Proportional-Odds Diagnostic
(RCSR-POD), applied to fused FARS–CRSS person records for 2021–2023.

RCSR-POD measures each covariate's departure from proportional odds (PO) as the mean absolute change in the fitted
probability of exceeding each cut when its cut-specific coefficients are replaced by the GLS-pooled coefficients, with the
cut intercept re-profiled so that the marginal rate is unchanged. The cut logits are Firth-penalized. A covariate is
flagged when a simultaneous one-sided lower confidence bound on its largest departure, from a parametric bootstrap of
the joint crash-cluster covariance of all cuts, exceeds a relevance threshold δ (0.01 in the manuscript).

The repository contains:

* the pipeline that builds the analysis frame from the public NHTSA files and runs the application;
* the Monte Carlo study;
* the generator of the appendix tables;
* the small result files behind every table and figure (`results/v3/`), so that the appendix tables can be
  regenerated without rerunning the computations.

The plotting code of the manuscript figures is not included.

## Repository layout

```
pipeline/                    shell drivers, run from anywhere (paths are relative to the repository root)
  run_data.sh                step 1: FARS-CRSS fusion and data cleaning
  run_application.sh         step 2: application (Section 4, Appendix C)
  run_simulations.sh         step 3: Monte Carlo study (Section 3, Appendix A)
  build_outputs.sh           step 4: appendix tables
scripts/
  download_fars.py, download_crss.py      optional download helpers (data/)
  build_fars_crss_fusion.py, severity_outcomes.py
                             person-level FARS-CRSS fusion and the KABCO outcome
  build_kabco_modeling_data.py, frame_cleaning.py
                             data-validity cleaning and merging of rare levels
  run_v3_frame.py            de-duplicated frame, recoding, selection controls, full-frame Firth cut fits
  run_v3_application.py      RCSR-POD departures, bootstrap bounds, multiplicity control
  run_v3_primary.py          covariate-level (coding-invariant) family, delta sweep, held-out specification sets
  run_v3_brant.py            Brant/Wald comparators, BIC rule, separation check
  run_v3_heldout.py          crash-grouped cross-validation of the partial-PO specifications
  run_v3_sensitivity.py      source strata, design weights, refit check, PO-fit reference, minimum level size
  run_v3_yearpool.py         year-pooling likelihood-ratio check
  describe_frame.py          KABCO distribution of the frame by source (Table 2)
  sim_v3.py                  Monte Carlo study (all experiments)
  rcsr_v3.py                 departure, GLS pooling, max-t bounds (numba kernels)
  run_task_a.py, run_departure_diag.py, po_diagnostic_core.py, departure_core.py, run_partial_po_fit.py,
  run_simboot.py, run_simboot3.py, sim_po_diagnostic.py, sim_departure.py, sim_departure_e5c.py
                             shared building blocks (design encoding, Firth logit, covariances,
                             penalized partial-PO likelihood, data-generating processes)
  build_v3_supp_tables.py    appendix tables (LaTeX)
hpc/                         PBS job templates (placeholders for project, queue, walltime, interpreter)
results/v3/                  result files of the reported run (application)
results/v3/sim29/            simulation summaries and metadata
results/v3/b29_nocontrol/    application without the selection controls (Table C.4 only, see Notes)
data/                        raw NHTSA zips (not tracked)
outputs/                     everything the pipeline writes (not tracked)
```

## Environment

The analyses were run in Python 3.13 (manuscript Appendix B.4). `requirements.txt` pins the packages that the code
imports to the versions of the workstation environment of the reported runs (Python 3.13.15): numpy 2.5.2,
scipy 1.18.0, pandas 3.0.5, numba 0.67.0, joblib 1.5.3 and statsmodels 0.14.6.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The drivers are bash scripts (on Windows, use Git Bash or WSL).

Environment variables (all optional):

| Variable | Default | Used by |
|---|---|---|
| `PYTHON` | `python` | all drivers |
| `RCPO_ROOT` | repository root | analysis scripts: inputs and outputs under `<root>/outputs` |
| `RCPO_DATA` | `<root>/data` | location of the NHTSA zips |
| `JOBS` | `16` | worker processes of the simulations |
| `RCSR_RESULTS` | `outputs/v3` | results read by the table generator |
| `RCSR_NOCTRL` | `results/v3/b29_nocontrol` | comparison run of Table C.4 |
| `RCSR_TABLES` | `outputs/tables` | generator output |

## Data

The data are public. Download the national CSV files of FARS 2021, 2022 and 2023 and of CRSS 2021, 2022 and 2023 from
the NHTSA file downloads page, https://www.nhtsa.gov/file-downloads, and place them in `data/` under these names:

```
data/FARS2021NationalCSV.zip  data/FARS2022NationalCSV.zip  data/FARS2023NationalCSV.zip
data/CRSS2021NationalCSV.zip  data/CRSS2022NationalCSV.zip  data/CRSS2023NationalCSV.zip
```

`python scripts/download_fars.py` and `python scripts/download_crss.py` fetch them from the NHTSA static download
server. The CRSS helper also fetches the CRSS auxiliary files, which the analysis does not use. If NHTSA changes the
download URLs, download the files manually. The zips used were 34–35 MB (FARS) and 50–59 MB (CRSS) each.

No raw data and no person-level frame are included. The pipeline writes three person-level CSV files of
180–232 MB each to `outputs/`.

## Running the pipeline

Run from the repository root. Each step reads only what the previous steps wrote.

| Step | Command | Main outputs |
|---|---|---|
| 1 Data | `bash pipeline/run_data.sh` | `outputs/kabco_person_level.csv` (649,808 records), `outputs/kabco_person_level_clean.csv` |
| 2 Application | `bash pipeline/run_application.sh` | `outputs/v3/` (frame of 645,050 records, Firth fits, all application results), logs in `outputs/v3/logs/` |
| 3 Simulations | `bash pipeline/run_simulations.sh [experiment ...]` | `outputs/v3/sim29/` |
| 4 Tables | `bash pipeline/build_outputs.sh [results_dir]` | `outputs/tables/v3_tab_{sim,app,sens}.tex` |

Steps 2 and 3 are independent. Step 4 reads `outputs/v3` by default. To regenerate the appendix tables
from the shipped results without running steps 1–3:

```
bash pipeline/build_outputs.sh results/v3
```

In the environment above, this reproduces the three appendix table files of the manuscript byte for byte.

Step 2 runs, in this order: `run_v3_frame.py`, `run_v3_application.py --B 1000 --block 32`, `run_v3_primary.py`,
`run_v3_brant.py`, `run_v3_heldout.py --folds 5 --maxiter 1500`, `run_v3_sensitivity.py --mode {strata, weighted,
refit, nmin} --B 1000`, `run_v3_yearpool.py` and `describe_frame.py`.

Step 3 runs the nine experiments of `sim_v3.py` with the settings recorded in `results/v3/sim29/*_meta.json`:

| Experiment | `--experiment` | N grid | R | B |
|---|---|---|---|---|
| Boundary and power | `pow` | 5,000; 25,000; 100,000 | 300 | 1,000 |
| Sub-relevance departures | `e1` | 1,000; 5,000; 25,000; 100,000 | 200 | 500 |
| Planted gaps | `e2` | 25,000; 100,000 | 200 | 500 |
| Many covariates (with `--post`) | `e4` | 1,000; 5,000; 25,000; 100,000 | 200 | 500 |
| Coding invariance | `inv` | 25,000; 100,000 | 200 | 500 |
| Separation | `sep` | 5,000; 25,000; 100,000 | 300 | 500 |
| Frame versus mechanism (`--truth-pop 4000000`) | `frm` | 25,000; 100,000 | 200 | 500 |
| Extended design (`--truth-pop 3000000`) | `e5` | 5,000; 25,000; 100,000 | 200 | 500 |
| Application scale | `scale` | 100,000; 500,000 | 50 | 500 |

On a PBS cluster, `hpc/rcsr_application.pbs` and `hpc/rcsr_simulations.pbs` wrap steps 2 and 3. Fill in the
placeholders (`<your_project>`, `<your_queue>`, `<hh:mm:ss>`, `<path/to/python>`), create `logs/` and submit from the
repository root.

## Manuscript tables

All files are in `outputs/v3/` after a run and in `results/v3/` as shipped (simulation files in the `sim29/`
subdirectory). The appendix tables are written by `scripts/build_v3_supp_tables.py`. Tables 1–4 of the main text
are typeset in the manuscript from the files listed. Figures 2–4 plot the simulation summaries in `sim29/`,
`v3_{infrastructure,pavement}.csv`, `brant_v3_{infrastructure,pavement}.csv`, `v3_primary_summary.json` and
`firth_fit_{infrastructure,pavement}.npz`.

| Item | Label | Produced by | Output file / source |
|---|---|---|---|
| Table 1 | `tab:simdesign` | `sim_v3.py` via `run_simulations.sh` | design of each experiment in `sim_v3.py`; N and R from `sim29/*_meta.json` |
| Table 2 | `tab:sample` | `describe_frame.py` | `frame_kabco_by_source.csv` (after a run; totals also in `frame_v3_report.json`) |
| Table 3 | `tab:gol` | `run_v3_brant.py`, `run_v3_primary.py`, `run_v3_sensitivity.py`, `run_v3_heldout.py` | Brant and grouped-Wald counts and BIC rule: `brant_v3_summary.json`; RCSR-POD counts: `v3_primary_summary.json`; within FARS / CRSS: `sens_strata_summary.json`; design-weighted: `sens_weighted_summary.json`; frame terms: `v3_primary_{infrastructure,pavement}.csv`; cut-specific parameters: `k_params` of each specification minus that of PO in `heldout_v3_summary.csv` |
| Table 4 | `tab:heldout` | `run_v3_heldout.py` | `heldout_v3_summary.csv` (`k_params`, `logloss_trunc`, `brier`, `rps`, `neg_share`); Share = (log loss of "PO + frame terms" − log loss of the specification) / (log loss of "PO + frame terms" − log loss of "Marginal Brant"), with `logloss_trunc` |
| Table A.1 | `stab:dgp` | `sim_v3.py` | data-generating processes (`build_*` functions of `sim_v3.py`; e1/e2 from `sim_po_diagnostic.simulate_ordinal`, e4 constants from `sim_departure.py`, e5 from `sim_departure_e5c.build_e5c`) |
| Table A.2 | `stab:pow` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_pow_summary.csv` |
| Table A.3 | `stab:e1` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_e1_summary.csv` |
| Table A.4 | `stab:e2` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_e2_summary.csv` |
| Table A.5 | `stab:e4` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_e4_summary.csv` |
| Table A.6 | `stab:post` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_e4_summary.csv` (columns written with `--post`) |
| Table A.7 | `stab:inv` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_inv_summary.csv`, `sim29/inv_meta.json` |
| Table A.8 | `stab:sep` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_sep_summary.csv` |
| Table A.9 | `stab:frm` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_frm_summary.csv` |
| Table A.10 | `stab:e5` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_e5_summary.csv` |
| Table A.11 | `stab:scale` | `build_v3_supp_tables.py` | `v3_tab_sim.tex` from `sim29/sim_scale_summary.csv` |
| Table C.1 | `stab:prim_infra` | `build_v3_supp_tables.py` | `v3_tab_app.tex` from `v3_primary_infrastructure.csv` |
| Table C.2 | `stab:prim_pavem` | `build_v3_supp_tables.py` | `v3_tab_app.tex` from `v3_primary_pavement.csv` |
| Table C.3 | `stab:covwald` | `build_v3_supp_tables.py` | `v3_tab_app.tex` from `brant_v3_covariates_{infrastructure,pavement}.csv` |
| Table C.4 | `stab:noctrl` | `build_v3_supp_tables.py` | `v3_tab_app.tex` from `b29_nocontrol/v3_primary_*.csv` (without controls) and `v3_primary_*.csv` (with controls) |
| Table C.5 | `stab:dummy_infra` | `build_v3_supp_tables.py` | `v3_tab_app.tex` from `v3_infrastructure.csv` |
| Table C.6 | `stab:dummy_pavem` | `build_v3_supp_tables.py` | `v3_tab_app.tex` from `v3_pavement.csv` |
| Table C.7 | `stab:strata` | `build_v3_supp_tables.py` | `v3_tab_sens.tex` from `sens_strata.csv`, `v3_primary_summary.json` |
| Table C.8 | `stab:weighted` | `build_v3_supp_tables.py` | `v3_tab_sens.tex` from `sens_weighted.csv`, `v3_primary_summary.json` |
| Table C.9 | `stab:refit` | `build_v3_supp_tables.py` | `v3_tab_sens.tex` from `sens_refit.csv` |
| Table C.10 | `stab:poref` | `build_v3_supp_tables.py` | `v3_tab_sens.tex` from `sens_poref_{infrastructure,pavement}.csv`, `sens_refit_summary.json` |
| Table C.11 | `stab:heldout` | `build_v3_supp_tables.py` | `v3_tab_sens.tex` from `heldout_v3_{infrastructure,pavement}.csv` |

Numbers quoted in the text come from the same files, for example the year-pooling test (`yearpool_v3.json`), the
frame construction and the FLIC intercept check (`frame_v3_report.json`) and the minimum-level-size check
(`sens_nmin_summary.json`).

## Compute

Simulation run times, from `results/v3/sim29/*_meta.json` (field `seconds`), with 16 worker processes:

| Experiment | pow | sep | e4 | e5 | e1 | scale | frm | e2 | inv | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Hours | 25.6 | 5.5 | 1.6 | 1.4 | 1.4 | 1.0 | 0.9 | 0.7 | 0.6 | 38.8 |

Step 1 and the first script of step 2 (`run_v3_frame.py`) take about 5 min on a workstation (fusion 132 s, cleaning
18 s, frame and Firth cut fits 103 s). In the environment above they reproduce the person-level files and the Firth
fits of the reported run exactly.

Application, from the logs of the reported run: the frame and the Firth cut fits took 75 s; the bootstrap of
`run_v3_application.py` (three covariance variants, B = 1,000 each) took about 1.2 h for the pavement set and 1.6 h for
the infrastructure set; the cross-validation fits of `run_v3_heldout.py` took about 2.1 h in total. In the
sensitivity logs, the four cut fits of one analysis take at most 161 s and one partial-PO refit at most 116 s; total
times of the sensitivity runs are not recorded. The PBS templates request 16 cores and 96 GB of memory. Adjust these
requests to your cluster.

## Seeds

All random numbers come from `numpy.random.default_rng` with fixed seeds:

* `run_v3_application.py --seed 20261015`: the bootstrap of the i-th pathway in `--pathways` uses seed + i
  (pavement 20261015, infrastructure 20261016).
* `run_v3_heldout.py --seed 20261015`: assignment of crashes to folds.
* `run_v3_sensitivity.py --seed 20261016`: one generator per call (one call per mode).
* `sim_v3.py --seed 20261015`: master generator of each experiment. Every replication receives its own seed from it,
  and its bootstrap draws use fixed offsets of that seed, so the results do not depend on `--jobs`. The calibrations
  of the true departures use fixed seeds set in the code.
* `build_kabco_modeling_data.py --random-state 42`: only the pathway subsamples `kabco_*_sample.csv`, which the
  RCSR-POD analysis does not use.

## Notes

* `results/v3` holds the result files of the reported run. `results/v3/sim29` holds the summaries and metadata of
  the reported simulations. The per-replication files `sim_*_long.csv` (about 24 MB in total) are not included.
  `run_simulations.sh` writes them.
* Table C.4 compares the reported analysis with an earlier run of the same pipeline before the two selection controls
  (crash size and person type) were added. The current code always includes the controls, so the "without controls"
  columns are read from the shipped `results/v3/b29_nocontrol/v3_primary_*.csv` (set `RCSR_NOCTRL` to use other
  files).
* `run_simulations.sh` records the settings of the reported simulation run as they are stored in its metadata
  (B, R and the N grid). The 16 workers come from its logs, `--post` from the columns of the e4 summary, and
  `--truth-pop` from the frame fractions in `frm_meta.json` and `e5_meta.json`. The seed is the default of `sim_v3.py`.

## License

MIT; see `LICENSE`.
