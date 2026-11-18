"""Monte-Carlo harness for the RC-PO proportional-odds diagnostic (Route A).

This is the *simulation spine* of the methods paper. It compares the standard
Brant test against the proposed relevance-calibrated, separation-robust PO
diagnostic (RC-PO) under data-generating processes (DGPs) where the *truth* is
known.

Flagship experiment (E1) — size / false-positive control
---------------------------------------------------------
Under a DGP in which every covariate is *practically* proportional-odds
(true max between-threshold coefficient gap  Delta_true <= delta), any covariate
flagged as a PO violator is a FALSE POSITIVE. We sweep the sample size N and
record the false-positive PO-flag rate for each method.

Two regimes, controlled by ``--departure``:
  * ``0.0``        : EXACT proportional odds. Sanity/calibration check — the
                     standard Brant test is *correctly* sized here (~alpha), and
                     RC-PO should be at or below alpha. This proves RC-PO does
                     not break calibration.
  * small (e.g. 0.04): TRIVIAL, practically-negligible PO departures. This is the
                     real pathology: as N grows, the standard Brant test detects
                     these substantively-null departures and its flag rate climbs
                     toward 100%, while RC-PO — which requires the lower bound of
                     a one-sided CI on the effect size to exceed delta — stays low.

The diagnostic logic (threshold-specific logits + Wald-style comparison) is kept
deliberately consistent with the production implementation in
``scripts/build_gol_ordered_models.py`` so the simulation validates the same
procedure that is applied to the FARS-CRSS data.

Usage
-----
    # fast smoke run (default): small N grid, few reps, writes CSV
    python scripts/sim_po_diagnostic.py

    # the two flagship regimes
    python scripts/sim_po_diagnostic.py --departure 0.0
    python scripts/sim_po_diagnostic.py --departure 0.04

    # heavier run closer to publication scale
    python scripts/sim_po_diagnostic.py --reps 300 --n-grid 1000,5000,25000,100000,650000

Outputs
-------
    outputs/sim_po_e1_dep{D}.csv          per-(N, rep, method) flag rates
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from scipy.special import expit
from po_diagnostic_core import firth_logit, ame_shift_table  # canonical RC-PO primitives

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

# KABCO-like cumulative intercepts: marginal P(Y>t) at x=0 decreasing along the
# severity ladder, roughly matching the unweighted modeling-sample distribution
# (P(Y>0)~0.49, P(Y>1)~0.38, P(Y>2)~0.27, P(Y>3)~0.20).
DEFAULT_ALPHAS = np.array([-0.036, -0.506, -0.987, -1.418])

# Base (proportional-odds) coefficients. First block = continuous covariates,
# second block = sparse one-hot dummies (these drive the separation channel).
DEFAULT_BETA_CONT = np.array([0.35, -0.28, 0.22, -0.18, 0.30])
DEFAULT_DUMMY_PREVALENCE = np.array([0.02, 0.01, 0.006, 0.003])  # rare FARS-like codes
DEFAULT_BETA_DUMMY = np.array([0.40, -0.45, 0.50, -0.55])


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

def simulate_ordinal(rng: np.random.Generator, n: int, *,
                     alphas: np.ndarray = DEFAULT_ALPHAS,
                     beta_cont: np.ndarray = DEFAULT_BETA_CONT,
                     dummy_prev: np.ndarray = DEFAULT_DUMMY_PREVALENCE,
                     beta_dummy: np.ndarray = DEFAULT_BETA_DUMMY,
                     departure: float = 0.0,
                     departure_pattern: np.ndarray | None = None,
                     gap_targets: np.ndarray | None = None,
                     return_betat: bool = False
                     ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate ordinal data from a (near-)proportional-odds cumulative model.

    P(Y > t | x) = logistic(alpha_t + x' beta_t),  beta_t = beta + departure * d_t

    departure == 0 -> exact proportional odds (beta_t identical across t).
    departure  > 0 -> trivial per-threshold perturbations of magnitude ~departure
                      (all true gaps stay below the relevance threshold delta when
                      ``departure`` is chosen small).

    ``gap_targets`` (E2): if given, a length-p array specifying the EXACT true max
    between-threshold gap for each covariate. Threshold coefficients are spread as
    beta_{j,t} = base_beta[j] + linspace(-g_j/2, +g_j/2, k_thresh), so
    true_max_gap[j] == g_j exactly. This overrides ``departure``/``departure_pattern``
    and lets the caller plant known substantive (g>delta) and trivial (g<=delta)
    violators to measure power and false-positive rate separately.

    Returns (X, y, true_max_gap) where true_max_gap[j] is the true max pairwise
    |beta_{j,s} - beta_{j,t}| for covariate j (so the caller knows the ground truth).
    """
    k_thresh = len(alphas)                      # K-1 thresholds
    p_cont = len(beta_cont)
    p_dum = len(dummy_prev)
    p = p_cont + p_dum

    # Design matrix: continuous ~ N(0,1) (already "standardized"), sparse dummies.
    Xc = rng.standard_normal((n, p_cont))
    Xd = np.column_stack([(rng.random(n) < q).astype(float) for q in dummy_prev])
    X = np.column_stack([Xc, Xd])

    base_beta = np.concatenate([beta_cont, beta_dummy])      # (p,)

    if gap_targets is not None:
        # Exact per-covariate gap control (E2): spread each gap linearly across
        # the K-1 thresholds so max(beta_t) - min(beta_t) == g_j.
        gap_targets = np.asarray(gap_targets, dtype=float)
        spread = np.linspace(-0.5, 0.5, k_thresh)              # (k_thresh,)
        beta_t = base_beta[:, None] + gap_targets[:, None] * spread[None, :]
    else:
        # Per-threshold departures d_t (p, k_thresh). Fixed pattern (seedable) so
        # the ground truth is identical across reps; only sampling noise varies.
        if departure_pattern is None:
            # deterministic, mean-zero-ish pattern in [-1, 1]
            idx = np.arange(p)[:, None] + np.arange(k_thresh)[None, :]
            departure_pattern = np.sin(idx * 1.3) * 0.5 + np.cos(idx * 0.7) * 0.5
        d = departure * departure_pattern                      # (p, k_thresh)
        beta_t = base_beta[:, None] + d                        # (p, k_thresh)

    # True max pairwise gap per covariate (ground truth Delta_true).
    true_max_gap = np.zeros(p)
    for j in range(p):
        vals = beta_t[j]
        true_max_gap[j] = float(np.max(vals) - np.min(vals))

    # Cumulative probabilities cp_t = P(Y > t).
    cp = np.empty((n, k_thresh))
    for t in range(k_thresh):
        eta = alphas[t] + X @ beta_t[:, t]
        cp[:, t] = 1.0 / (1.0 + np.exp(-eta))
    # Enforce monotone non-increasing cp across thresholds (small departures can
    # otherwise produce tiny violations for extreme x).
    cp = np.minimum.accumulate(cp, axis=1)

    # Class probabilities then sample y in {0,...,K-1}.
    probs = np.empty((n, k_thresh + 1))
    probs[:, 0] = 1.0 - cp[:, 0]
    for t in range(1, k_thresh):
        probs[:, t] = cp[:, t - 1] - cp[:, t]
    probs[:, k_thresh] = cp[:, k_thresh - 1]
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(axis=1, keepdims=True)

    u = rng.random(n)
    cum = np.cumsum(probs, axis=1)
    y = (u[:, None] > cum).sum(axis=1).astype(int)

    cols = [f"x_cont_{i}" for i in range(p_cont)] + [f"x_rare_{i}" for i in range(p_dum)]
    Xdf = pd.DataFrame(X, columns=cols)
    if return_betat:
        return Xdf, y, true_max_gap, beta_t, alphas
    return Xdf, y, true_max_gap


# ---------------------------------------------------------------------------
# Threshold-specific logit fits + diagnostics
# ---------------------------------------------------------------------------

def fit_threshold_logits(X: pd.DataFrame, y: np.ndarray, method: str
                         ) -> dict[int, tuple[pd.Series, pd.Series]]:
    """Fit K-1 binary logits at thresholds (Y>t). Returns {t: (params, bse)}
    over the feature columns (intercept dropped)."""
    levels = sorted(np.unique(y))
    if len(levels) < 3:
        return {}
    cols = list(X.columns)
    Xmat = np.column_stack([np.ones(len(X)), X.values])  # intercept + features
    out: dict[int, tuple[pd.Series, pd.Series]] = {}
    for t in levels[:-1]:
        by = (y > t).astype(int)
        if by.sum() == 0 or by.sum() == len(by):
            continue
        if method == "firth":
            beta, se, _ = firth_logit(Xmat, by)
            params = pd.Series(beta[1:], index=cols)
            bse = pd.Series(se[1:], index=cols)
        else:  # "mle" via statsmodels GLM (matches build_gol_ordered_models.py)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = sm.GLM(by, Xmat, family=sm.families.Binomial()).fit()
                params = pd.Series(res.params[1:], index=cols)
                bse = pd.Series(res.bse[1:], index=cols)
            except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
                params = pd.Series(np.nan, index=cols)
                bse = pd.Series(np.nan, index=cols)
        out[t] = (params, bse)
    return out


def diagnose(fits: dict[int, tuple[pd.Series, pd.Series]], features: list[str], *,
             delta: float, alpha_sig: float, stability_cap: float = 10.0) -> pd.DataFrame:
    """Per-feature flags from threshold-logit fits, for three diagnostics:

    brant_flag     : Wald chi2 across thresholds vs first threshold, p < alpha_sig
                     (the standard test).
    heuristic_flag : the ORIGINAL ad-hoc rule — stat_sig AND (max|beta| < cap)
                     AND (point max_gap > delta). The ``max|beta| < cap`` term is
                     the |beta|<10 stability band-aid; under separation it silently
                     DROPS variables whose coefficient is inflated.
    rcpo_flag      : the proposed test — exists a threshold pair (s, t) with
                     |b_s - b_t| - z_{1-alpha} * sqrt(se_s^2 + se_t^2) > delta
                     (one-sided minimum-effect / relevance test; auto-robust to
                     separation because exploded SEs drive the lower bound below
                     delta, with no magic cap).

    Also returns ``unstable`` (max|beta| >= cap or non-finite), used by E3 to
    quantify how often raw MLE separation triggers the band-aid.
    """
    z = stats.norm.ppf(1 - alpha_sig)
    ts = sorted(fits)
    rows = []
    for f in features:
        coefs = np.array([fits[t][0].get(f, np.nan) for t in ts], dtype=float)
        ses = np.array([fits[t][1].get(f, np.nan) for t in ts], dtype=float)
        finite = np.isfinite(coefs) & np.isfinite(ses) & (ses > 0)
        max_abs_coef = float(np.nanmax(np.abs(coefs))) if np.any(np.isfinite(coefs)) else np.inf
        unstable = (not np.isfinite(max_abs_coef)) or (max_abs_coef >= stability_cap)
        if finite.sum() < 2:
            rows.append({"feature": f, "brant_flag": False, "heuristic_flag": False,
                         "rcpo_flag": False, "max_gap": np.nan, "max_gap_lcb": np.nan,
                         "brant_pval": np.nan, "rcpo_pval": 1.0,
                         "max_abs_coef": max_abs_coef, "unstable": True})
            continue
        c = coefs[finite]
        s = ses[finite]
        # Standard Brant: Wald chi2 vs reference (first finite threshold).
        ref, ref_var = c[0], s[0] ** 2
        chi2 = 0.0
        df = 0
        for k in range(1, len(c)):
            var = s[k] ** 2 + ref_var
            if var > 0:
                chi2 += (c[k] - ref) ** 2 / var
                df += 1
        p_value = 1 - stats.chi2.cdf(chi2, df) if df > 0 else np.nan
        stat_sig = bool(np.isfinite(p_value) and (p_value < alpha_sig))
        # Point-estimate max pairwise gap (used by the heuristic) and the
        # relevance lower confidence bound (used by RC-PO).
        max_gap = 0.0
        max_lcb = -np.inf
        max_t = -np.inf                        # max standardized minimum-effect statistic
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                gap = abs(c[i] - c[j])
                se_gap = np.sqrt(s[i] ** 2 + s[j] ** 2)
                max_gap = max(max_gap, gap)
                max_lcb = max(max_lcb, gap - z * se_gap)
                if se_gap > 0:
                    max_t = max(max_t, (gap - delta) / se_gap)
        heuristic_flag = stat_sig and (not unstable) and (max_gap > delta)
        rcpo_flag = bool(max_lcb > delta)
        # One-sided minimum-effect p-value for H0: max gap <= delta, consistent with
        # the LCB test (rcpo_flag <=> rcpo_pval < alpha_sig). Feeds BH/Holm in E4.
        rcpo_pval = float(1.0 - stats.norm.cdf(max_t)) if np.isfinite(max_t) else 1.0
        rows.append({"feature": f, "brant_flag": stat_sig,
                     "heuristic_flag": bool(heuristic_flag), "rcpo_flag": rcpo_flag,
                     "max_gap": max_gap, "max_gap_lcb": max_lcb,
                     "brant_pval": float(p_value) if np.isfinite(p_value) else np.nan,
                     "rcpo_pval": rcpo_pval,
                     "max_abs_coef": max_abs_coef, "unstable": bool(unstable)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment E1
# ---------------------------------------------------------------------------

def _e1_task(n: int, rep: int, seed: int, *, departure: float, delta: float,
             alpha_sig: float, method: str):
    """One E1 replication (independent) -> a record dict, or None if unusable.
    NOTE: the ``brant_flag`` arm is the Wald PO test, i.e. the same selection
    criterion that gologit2 ``autofit`` uses on (unpenalized) ML fits; under the
    ``mle`` method this curve doubles as the autofit PO-relaxation rate."""
    rng = np.random.default_rng(seed)
    X, y, true_gap = simulate_ordinal(rng, n, departure=departure)
    features = list(X.columns)
    n_true_viol = int((true_gap > delta).sum())
    fits = fit_threshold_logits(X, y, method)
    if len(fits) < 2:
        return None
    diag = diagnose(fits, features, delta=delta, alpha_sig=alpha_sig)
    not_viol = true_gap <= delta
    mask = pd.Series(not_viol, index=features)
    denom = int(mask.sum())
    if denom == 0:
        return None
    d = diag.set_index("feature")
    idx = mask[mask].index
    return {"N": n, "rep": rep, "departure": departure,
            "n_true_violators": n_true_viol,
            "brant_fp_rate": float(d.loc[idx, "brant_flag"].sum()) / denom,
            "rcpo_fp_rate": float(d.loc[idx, "rcpo_flag"].sum()) / denom}


def run_e1(n_grid: list[int], reps: int, *, departure: float, delta: float,
           alpha_sig: float, method: str, seed: int, jobs: int = 1) -> pd.DataFrame:
    """For each N and rep: simulate (near-)PO data, run both diagnostics, record
    the per-rep false-positive PO-flag rate. ``jobs != 1`` parallelizes reps."""
    rng_master = np.random.default_rng(seed)
    tasks = [(n, rep, int(rng_master.integers(0, 2**63 - 1)))
             for n in n_grid for rep in range(reps)]
    kw = dict(departure=departure, delta=delta, alpha_sig=alpha_sig, method=method)
    if jobs == 1:
        rows = []
        for (n, rep, s) in tasks:
            rows.append(_e1_task(n, rep, s, **kw))
            if rep % 10 == 0:
                gc.collect()
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_e1_task)(n, rep, s, **kw) for (n, rep, s) in tasks)
    rows = [r for r in rows if r is not None]
    print(f"  E1 done: {len(rows)} usable reps across N-grid {n_grid}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("N").agg(
        brant_fp_mean=("brant_fp_rate", "mean"),
        brant_fp_sd=("brant_fp_rate", "std"),
        rcpo_fp_mean=("rcpo_fp_rate", "mean"),
        rcpo_fp_sd=("rcpo_fp_rate", "std"),
        reps=("rep", "count"),
    ).reset_index()
    return g


# ---------------------------------------------------------------------------
# Experiment E2 — power / discrimination under known planted partial-PO
# ---------------------------------------------------------------------------

# Per-covariate true max gaps. A spread from exact-PO (0) through trivial
# (< delta) to clearly substantive (>> delta), so we can read detection rate as
# a function of the true effect size. Length must match the DGP's covariate count
# (5 continuous + 4 sparse dummies = 9).
DEFAULT_E2_GAPS = np.array([0.00, 0.05, 0.08, 0.15, 0.30, 0.50, 0.00, 0.12, 0.40])

# Larger coefficient gaps for the PSC (probability-scale) power experiment, tuned
# so the resulting TRUE PSC gaps straddle delta_psc=0.01: the dense continuous
# carriers span ~0 to ~0.045, while the sparse one-hot carriers (cols 5-8) keep a
# tiny PSC despite large coefficients (prevalence-weighting), demonstrating that
# RC-PO(PSC) correctly ignores rare separating levels on the probability scale.
DEFAULT_E2PSC_GAPS = np.array([0.00, 0.30, 0.60, 1.00, 1.50, 1.50, 0.00, 0.80, 2.00])


def _e2_task(n: int, rep: int, seed: int, *, gaps, delta, alpha_sig, method):
    rng = np.random.default_rng(seed)
    X, y, true_gap = simulate_ordinal(rng, n, gap_targets=gaps)
    features = list(X.columns)
    fits = fit_threshold_logits(X, y, method)
    if len(fits) < 2:
        return []
    diag = diagnose(fits, features, delta=delta, alpha_sig=alpha_sig).set_index("feature")
    return [{"N": n, "rep": rep, "feature": f, "true_gap": float(true_gap[j]),
             "substantive": bool(true_gap[j] > delta),
             "brant_flag": bool(diag.loc[f, "brant_flag"]),
             "rcpo_flag": bool(diag.loc[f, "rcpo_flag"])}
            for j, f in enumerate(features)]


def run_e2(n_grid: list[int], reps: int, *, gaps: np.ndarray, delta: float,
           alpha_sig: float, method: str, seed: int, jobs: int = 1) -> pd.DataFrame:
    """For each N and rep: plant known per-covariate gaps, run both diagnostics,
    and record per-covariate detection (flag) outcomes alongside the true gap.
    ``jobs != 1`` parallelizes reps. One row per (N, rep, covariate)."""
    rng_master = np.random.default_rng(seed)
    tasks = [(n, rep, int(rng_master.integers(0, 2**63 - 1)))
             for n in n_grid for rep in range(reps)]
    kw = dict(gaps=gaps, delta=delta, alpha_sig=alpha_sig, method=method)
    if jobs == 1:
        nested = [_e2_task(n, rep, s, **kw) for (n, rep, s) in tasks]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_e2_task)(n, rep, s, **kw) for (n, rep, s) in tasks)
    records = [r for sub in nested for r in sub]
    print(f"  E2 done: {len(records)} covariate-rows across N-grid {n_grid}")
    return pd.DataFrame(records)


def summarize_e2(df: pd.DataFrame, delta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (by_N power/FP table, by_covariate detection table at max N)."""
    # Power = detection among substantive (true_gap>delta); FP = detection among trivial.
    def rates(sub: pd.DataFrame) -> pd.Series:
        subst = sub[sub["substantive"]]
        triv = sub[~sub["substantive"]]
        return pd.Series({
            "brant_power": subst["brant_flag"].mean() if len(subst) else np.nan,
            "rcpo_power": subst["rcpo_flag"].mean() if len(subst) else np.nan,
            "brant_fp": triv["brant_flag"].mean() if len(triv) else np.nan,
            "rcpo_fp": triv["rcpo_flag"].mean() if len(triv) else np.nan,
        })
    by_n = df.groupby("N")[["substantive", "brant_flag", "rcpo_flag"]].apply(rates).reset_index()
    n_max = df["N"].max()
    by_cov = (df[df["N"] == n_max]
              .groupby(["feature", "true_gap"])
              .agg(brant_detect=("brant_flag", "mean"),
                   rcpo_detect=("rcpo_flag", "mean"))
              .reset_index()
              .sort_values("true_gap"))
    return by_n, by_cov


# ---------------------------------------------------------------------------
# Experiment E3 — separation robustness: Firth vs the |beta|<10 band-aid
# ---------------------------------------------------------------------------

def _e3_task(prev: float, rep: int, seed: int, *, n: int, gap_rare: float,
             delta: float, alpha_sig: float):
    rng = np.random.default_rng(seed)
    p_cont = len(DEFAULT_BETA_CONT)
    p_dum = len(DEFAULT_BETA_DUMMY)
    gaps = np.concatenate([np.zeros(p_cont), np.full(p_dum, gap_rare)])
    strong_dummy_beta = np.full(p_dum, 3.0)
    dummy_prev = np.full(p_dum, prev)
    X, y, _ = simulate_ordinal(rng, n, dummy_prev=dummy_prev,
                               beta_dummy=strong_dummy_beta, gap_targets=gaps)
    features = list(X.columns)
    rare = [f for f in features if f.startswith("x_rare")]
    fits_mle = fit_threshold_logits(X, y, "mle")
    fits_firth = fit_threshold_logits(X, y, "firth")
    if len(fits_mle) < 2 or len(fits_firth) < 2:
        return []
    d_mle = diagnose(fits_mle, features, delta=delta, alpha_sig=alpha_sig).set_index("feature")
    d_firth = diagnose(fits_firth, features, delta=delta, alpha_sig=alpha_sig).set_index("feature")
    return [{"prevalence": prev, "rep": rep, "feature": f,
             "mle_unstable": bool(d_mle.loc[f, "unstable"]),
             "heuristic_detect": bool(d_mle.loc[f, "heuristic_flag"]),
             "rcpo_mle_detect": bool(d_mle.loc[f, "rcpo_flag"]),
             "rcpo_firth_detect": bool(d_firth.loc[f, "rcpo_flag"])}
            for f in rare]


def run_e3(prev_grid: list[float], reps: int, *, n: int, gap_rare: float,
           delta: float, alpha_sig: float, seed: int, jobs: int = 1) -> pd.DataFrame:
    """Plant a TRUE substantive PO violation (gap_rare > delta) on the rare
    one-hot dummies, then push their prevalence down to induce separation.

    For each rep we fit the threshold logits twice — by MLE and by Firth — and
    record, per rare dummy (a genuine violator that SHOULD be detected):
      * mle_unstable      : raw MLE blew up (|beta| >= cap) -> the band-aid fires
      * heuristic_detect  : original rule detects it (drops it when unstable)
      * rcpo_mle_detect   : RC-PO on MLE fits detects it
      * rcpo_firth_detect : RC-PO on Firth fits detects it
    As prevalence falls, the heuristic increasingly DROPS the real violator
    (false negative), while Firth keeps a finite estimate and recovers more."""
    rng_master = np.random.default_rng(seed)
    tasks = [(prev, rep, int(rng_master.integers(0, 2**63 - 1)))
             for prev in prev_grid for rep in range(reps)]
    kw = dict(n=n, gap_rare=gap_rare, delta=delta, alpha_sig=alpha_sig)
    if jobs == 1:
        nested = [_e3_task(prev, rep, s, **kw) for (prev, rep, s) in tasks]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_e3_task)(prev, rep, s, **kw) for (prev, rep, s) in tasks)
    records = [r for sub in nested for r in sub]
    print(f"  E3 done: {len(records)} rare-dummy rows across prevalence grid {prev_grid}")
    return pd.DataFrame(records)


def summarize_e3(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("prevalence")
              .agg(mle_unstable=("mle_unstable", "mean"),
                   heuristic_detect=("heuristic_detect", "mean"),
                   rcpo_mle_detect=("rcpo_mle_detect", "mean"),
                   rcpo_firth_detect=("rcpo_firth_detect", "mean"),
                   reps=("rep", "count"))
              .reset_index()
              .sort_values("prevalence", ascending=False))


# ---------------------------------------------------------------------------
# PSC arm — validate the ACTUAL applied estimand (probability-scale PSC + delta-
# method LCB > delta), with ground truth on the PSC scale. Closes the gap that
# E1/E2/E3 above validate only the coefficient-scale variant.
# ---------------------------------------------------------------------------

def _fit_firth_const(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Threshold-specific Firth logits retaining the intercept as 'const', the
    form ``ame_shift_table`` expects."""
    levels = sorted(np.unique(y))
    if len(levels) < 3:
        return {}
    cols = list(X.columns)
    Xmat = np.column_stack([np.ones(len(X)), X.values])
    idx = ["const"] + cols
    out = {}
    for t in levels[:-1]:
        by = (y > t).astype(int)
        if by.sum() == 0 or by.sum() == len(by):
            continue
        beta, se, _ = firth_logit(Xmat, by)
        out[t] = (pd.Series(beta, index=idx), pd.Series(se, index=idx))
    return out


def _true_psc_gaps(X: pd.DataFrame, alphas: np.ndarray, beta_t: np.ndarray) -> dict:
    """Ground-truth PSC gap per covariate, computed from the TRUE threshold
    coefficients over the realized design (same shift formula as ame_shift_table)."""
    Xv = X.values
    feats = list(X.columns)
    k = len(alphas)
    shifts = np.zeros((len(feats), k))
    for t in range(k):
        eta = alphas[t] + Xv @ beta_t[:, t]
        p = expit(eta)
        for jc in range(len(feats)):
            p0 = expit(eta - beta_t[jc, t] * Xv[:, jc])
            shifts[jc, t] = float(np.mean(p - p0))
    return {f: (max(abs(shifts[jc, i] - shifts[jc, j])
                    for i in range(k) for j in range(i + 1, k)) if k >= 2 else 0.0)
            for jc, f in enumerate(feats)}


def _wald_po_flag(fits: dict, feats: list, alpha_sig: float) -> dict:
    """Significance PO (Brant/Wald) flag computed on the SAME Firth fits, as the
    over-flagging contrast to RC-PO(PSC)."""
    ts = sorted(fits)
    out = {}
    for f in feats:
        c = np.array([float(fits[t][0].get(f, np.nan)) for t in ts])
        s = np.array([float(fits[t][1].get(f, np.nan)) for t in ts])
        ok = np.isfinite(c) & np.isfinite(s) & (s > 0)
        c, s = c[ok], s[ok]
        if len(c) < 2:
            out[f] = False; continue
        ref, rv = c[0], s[0] ** 2
        chi2 = float(np.sum([(c[k] - ref) ** 2 / (s[k] ** 2 + rv) for k in range(1, len(c))]))
        df = len(c) - 1
        pval = 1 - stats.chi2.cdf(chi2, df) if df > 0 else np.nan
        out[f] = bool(np.isfinite(pval) and pval < alpha_sig)
    return out


def _psc_eval(n, rep, seed, *, departure, gaps, delta_psc, alpha_sig):
    """One PSC replication -> per-covariate rows with the TRUE PSC gap, the
    significance flag, and the RC-PO(PSC) flag (delta-method LCB > delta_psc)."""
    rng = np.random.default_rng(seed)
    if gaps is not None:
        X, y, _, beta_t, alphas = simulate_ordinal(rng, n, gap_targets=gaps, return_betat=True)
    else:
        X, y, _, beta_t, alphas = simulate_ordinal(rng, n, departure=departure, return_betat=True)
    feats = list(X.columns)
    true_psc = _true_psc_gaps(X, alphas, beta_t)
    fits = _fit_firth_const(X, y)
    if len(fits) < 2:
        return []
    psc = ame_shift_table(fits, X, alpha_sig=alpha_sig, ci_method="delta")
    wald = _wald_po_flag(fits, feats, alpha_sig)
    return [{"N": n, "rep": rep, "feature": f, "true_psc_gap": float(true_psc[f]),
             "substantive": bool(true_psc[f] > delta_psc),
             "sig_flag": bool(wald.get(f, False)),
             "rcpo_psc_flag": bool(psc[f][1] > delta_psc)} for f in feats]


def run_e1psc(n_grid, reps, *, departure, delta_psc, alpha_sig, seed, jobs=1) -> pd.DataFrame:
    """Size / false-positive on the PROBABILITY scale: under (near-)PO data, the
    FP rate is among covariates whose TRUE PSC gap <= delta_psc."""
    rng_master = np.random.default_rng(seed)
    tasks = [(n, rep, int(rng_master.integers(0, 2**63 - 1))) for n in n_grid for rep in range(reps)]
    kw = dict(departure=departure, gaps=None, delta_psc=delta_psc, alpha_sig=alpha_sig)
    if jobs == 1:
        nested = [_psc_eval(n, rep, s, **kw) for (n, rep, s) in tasks]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=jobs, verbose=5)(delayed(_psc_eval)(n, rep, s, **kw) for (n, rep, s) in tasks)
    rows = []
    for sub in nested:
        if not sub:
            continue
        triv = [r for r in sub if not r["substantive"]]
        if not triv:
            continue
        rows.append({"N": sub[0]["N"], "rep": sub[0]["rep"], "departure": departure,
                     "n_psc_violators": sum(r["substantive"] for r in sub),
                     "sig_fp_rate": np.mean([r["sig_flag"] for r in triv]),
                     "rcpo_psc_fp_rate": np.mean([r["rcpo_psc_flag"] for r in triv])})
    return pd.DataFrame(rows)


def run_e2psc(n_grid, reps, *, gaps, delta_psc, alpha_sig, seed, jobs=1) -> pd.DataFrame:
    """Power / discrimination on the PROBABILITY scale: per-covariate detection
    vs the TRUE PSC gap (should step at delta_psc for RC-PO(PSC))."""
    rng_master = np.random.default_rng(seed)
    tasks = [(n, rep, int(rng_master.integers(0, 2**63 - 1))) for n in n_grid for rep in range(reps)]
    kw = dict(departure=0.0, gaps=gaps, delta_psc=delta_psc, alpha_sig=alpha_sig)
    if jobs == 1:
        nested = [_psc_eval(n, rep, s, **kw) for (n, rep, s) in tasks]
    else:
        from joblib import Parallel, delayed
        nested = Parallel(n_jobs=jobs, verbose=5)(delayed(_psc_eval)(n, rep, s, **kw) for (n, rep, s) in tasks)
    return pd.DataFrame([r for sub in nested for r in sub])


# ---------------------------------------------------------------------------
# Experiment E4 — MANY covariates tested at once: does relevance calibration +
# multiplicity control (BH-FDR / Holm) hold its family-wise error while the
# applied case tests eight simultaneous violators? Validates the FDR/Holm columns
# of the production table, which E1-E3 never exercise.
# ---------------------------------------------------------------------------

# 20 covariates (12 continuous + 8 sparse dummies). 8 are TRUE substantive
# violators (gap > delta = 0.10); the other 12 are relevance-null (gap <= delta),
# deliberately a MIX of exact PO (gap = 0) and trivial departures (0 < gap < delta)
# so the standard Brant test over-flags the trivial ones at large N while RC-PO does not.
E4_BETA_CONT = np.array([0.30, -0.25, 0.20, -0.15, 0.28, -0.22, 0.18, -0.12,
                         0.24, -0.20, 0.16, -0.26])                       # 12
E4_DUMMY_PREV = np.array([0.03, 0.02, 0.015, 0.01, 0.025, 0.012, 0.008, 0.02])  # 8
E4_BETA_DUMMY = np.array([0.40, -0.42, 0.45, -0.38, 0.36, -0.44, 0.34, -0.40])  # 8
#                       continuous (12): 5 violators (>0.10), 7 relevance-null (<=0.10)
E4_GAPS = np.array([0.00, 0.05, 0.15, 0.03, 0.25, 0.07, 0.35, 0.06, 0.20, 0.00, 0.08, 0.30,
#                       dummies (8): 3 violators, 5 relevance-null
                    0.00, 0.18, 0.04, 0.22, 0.00, 0.40, 0.05, 0.00])


def _e4_task(n, rep, seed, *, delta, alpha_sig, alpha_mult, method):
    """One E4 replication -> per-method family-wise outcome record. We test all
    20 covariates simultaneously and apply BH-FDR and Holm across them."""
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(seed)
    X, y, true_gap = simulate_ordinal(rng, n, beta_cont=E4_BETA_CONT,
                                      dummy_prev=E4_DUMMY_PREV, beta_dummy=E4_BETA_DUMMY,
                                      gap_targets=E4_GAPS)
    feats = list(X.columns)
    fits = fit_threshold_logits(X, y, method)
    if len(fits) < 2:
        return None
    d = diagnose(fits, feats, delta=delta, alpha_sig=alpha_sig).set_index("feature")
    is_viol = np.array([true_gap[j] > delta for j in range(len(feats))])   # ground truth
    n_viol = int(is_viol.sum())
    n_null = int((~is_viol).sum())

    rp = d["rcpo_pval"].to_numpy(float)
    bp = d["brant_pval"].to_numpy(float)
    rp = np.where(np.isfinite(rp), rp, 1.0)
    bp = np.where(np.isfinite(bp), bp, 1.0)

    def outcomes(reject: np.ndarray, tag: str):
        V = int((reject & ~is_viol).sum())            # false rejections (relevance-null)
        S = int((reject & is_viol).sum())             # true detections
        R = int(reject.sum())
        return {f"{tag}_fwer": int(V > 0),             # any false rejection (family-wise)
                f"{tag}_fdp": (V / R if R else 0.0),   # realized false-discovery proportion
                f"{tag}_power": (S / n_viol if n_viol else np.nan)}

    def bh(p):   return multipletests(p, alpha=alpha_mult, method="fdr_bh")[0]
    def holm(p): return multipletests(p, alpha=alpha_mult, method="holm")[0]

    rec = {"N": n, "rep": rep, "n_viol": n_viol, "n_null": n_null}
    rec.update(outcomes(bp < alpha_sig, "brant_raw"))
    rec.update(outcomes(holm(bp), "brant_holm"))
    rec.update(outcomes(rp < alpha_sig, "rcpo_raw"))
    rec.update(outcomes(bh(rp), "rcpo_bh"))
    rec.update(outcomes(holm(rp), "rcpo_holm"))
    return rec


def run_e4(n_grid, reps, *, delta, alpha_sig, alpha_mult, method, seed, jobs=1) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    tasks = [(n, rep, int(rng_master.integers(0, 2**63 - 1)))
             for n in n_grid for rep in range(reps)]
    kw = dict(delta=delta, alpha_sig=alpha_sig, alpha_mult=alpha_mult, method=method)
    if jobs == 1:
        rows = [_e4_task(n, rep, s, **kw) for (n, rep, s) in tasks]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_e4_task)(n, rep, s, **kw) for (n, rep, s) in tasks)
    rows = [r for r in rows if r is not None]
    print(f"  E4 done: {len(rows)} usable reps across N-grid {n_grid}")
    return pd.DataFrame(rows)


def summarize_e4(df: pd.DataFrame) -> pd.DataFrame:
    methods = ["brant_raw", "brant_holm", "rcpo_raw", "rcpo_bh", "rcpo_holm"]
    agg = {}
    for m in methods:
        agg[f"{m}_fwer"] = (f"{m}_fwer", "mean")     # P(>=1 false flag) — Holm target <= alpha
        agg[f"{m}_fdr"] = (f"{m}_fdp", "mean")       # E[false/rejected] — BH target <= alpha
        agg[f"{m}_power"] = (f"{m}_power", "mean")
    return df.groupby("N").agg(**agg, reps=("rep", "count")).reset_index()


# ---------------------------------------------------------------------------
# Experiment E4-PSC — the SAME many-covariate family-wise test as E4, but on the
# applied PROBABILITY-scale PSC estimand (delta-method LCB > delta_psc) with
# BH/Holm control, so the multiplicity validation matches the procedure used on
# the data exactly rather than by a scale-transfer argument.
# ---------------------------------------------------------------------------

# 20 covariates. True PSC gaps straddle delta_psc: the dense continuous carriers
# with larger coefficient gaps clear it; small-gap continuous and the sparse
# one-hot dummies (large coef but tiny population footprint) stay PSC-null.
E4PSC_BETA_CONT = np.array([0.30, -0.25, 0.20, -0.15, 0.28, -0.22, 0.18, -0.12,
                            0.24, -0.20, 0.16, -0.26])                      # 12
E4PSC_DUMMY_PREV = np.array([0.03, 0.02, 0.015, 0.01, 0.025, 0.012, 0.008, 0.02])  # 8
E4PSC_BETA_DUMMY = np.array([0.40, -0.42, 0.45, -0.38, 0.36, -0.44, 0.34, -0.40])  # 8
# NOTE 2026-08-30: this coefficient-scale-tuned e4psc design realizes only ~1-3 true
# PSC violators (coef->PSC map saturates) AND its ground truth via _true_psc_gaps is
# inconsistent with the monotonicity-clipped DGP. Use scripts/run_task_b.py (corrected,
# monotone-respecting, ground truth = estimable PSC) for the PSC-scale family-wise result.
#                       continuous (12): 8 PSC violators (coef gap >=1.0 -> PSC>delta), 4 null
E4PSC_GAPS = np.array([1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 1.10, 1.30, 0.00, 0.20, 0.15, 0.10,
#                       dummies (8): large coef gaps but rare -> tiny PSC -> all null
                       0.80, 0.60, 1.00, 0.50, 0.90, 0.70, 0.40, 0.80])


def _psc_stats(firth_full, X, delta, alpha_sig):
    """Per-covariate PSC point gap, delta-method LCB, and one-sided minimum-effect
    p-value (H0: PSC gap <= delta), consistent with ame_shift_table's delta bound."""
    ts = sorted(firth_full); feats = list(X.columns); Xv = X.values
    z = stats.norm.ppf(1 - alpha_sig)
    per_t = {}
    for t in ts:
        beta, se = firth_full[t]
        const = float(beta.get("const", 0.0))
        bvec = beta.reindex(feats).to_numpy(dtype=float)
        per_t[t] = (bvec, se, const + Xv @ bvec)
    out = {}
    for jc, f in enumerate(feats):
        xj = Xv[:, jc]; shifts = []; dses = []
        for t in ts:
            bvec, se, eta = per_t[t]; bj = bvec[jc]; se_bj = float(se.get(f, np.nan))
            eta0 = eta - bj * xj
            p0 = np.clip(1/(1+np.exp(-eta0)), 1e-12, 1-1e-12)
            p = np.clip(1/(1+np.exp(-eta)), 1e-12, 1-1e-12)
            shifts.append(float(np.mean(p - p0)))
            # dPSC/dbeta_j at the FULL predictor (see po_diagnostic_core); the
            # stripped p0 form mis-evaluated the derivative.
            g = float(np.mean(xj * p * (1 - p)))
            dses.append(abs(g) * se_bj if np.isfinite(se_bj) else np.nan)
        s = np.array(shifts); d = np.array(dses); fin = np.isfinite(s) & np.isfinite(d)
        s = s[fin]; d = d[fin]
        if len(s) < 2:
            out[f] = (0.0, -np.inf, 1.0); continue
        max_gap = 0.0; max_lcb = -np.inf; max_t = -np.inf
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                gap = abs(s[i] - s[j]); se_gap = np.sqrt(d[i] ** 2 + d[j] ** 2)
                max_gap = max(max_gap, gap); max_lcb = max(max_lcb, gap - z * se_gap)
                if se_gap > 0:
                    max_t = max(max_t, (gap - delta) / se_gap)
        pval = float(1 - stats.norm.cdf(max_t)) if np.isfinite(max_t) else 1.0
        out[f] = (max_gap, max_lcb, pval)
    return out


def _brant_pvals(firth_full, feats, alpha_sig):
    """Coefficient-scale Brant/Wald PO p-value per covariate on the same fits."""
    ts = sorted(firth_full); out = {}
    for f in feats:
        c = np.array([float(firth_full[t][0].get(f, np.nan)) for t in ts])
        se = np.array([float(firth_full[t][1].get(f, np.nan)) for t in ts])
        ok = np.isfinite(c) & np.isfinite(se) & (se > 0)
        c, se = c[ok], se[ok]
        if len(c) < 2:
            out[f] = 1.0; continue
        ref, rv = c[0], se[0] ** 2
        chi2 = float(np.sum([(c[k] - ref) ** 2 / (se[k] ** 2 + rv) for k in range(1, len(c))]))
        dfree = len(c) - 1
        out[f] = float(1 - stats.chi2.cdf(chi2, dfree)) if dfree > 0 else 1.0
    return out


def _e4psc_task(n, rep, seed, *, delta_psc, alpha_sig, alpha_mult):
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(seed)
    X, y, _, beta_t, alphas = simulate_ordinal(rng, n, beta_cont=E4PSC_BETA_CONT,
                                               dummy_prev=E4PSC_DUMMY_PREV,
                                               beta_dummy=E4PSC_BETA_DUMMY,
                                               gap_targets=E4PSC_GAPS, return_betat=True)
    feats = list(X.columns)
    fits = _fit_firth_const(X, y)
    if len(fits) < 2:
        return None
    psc = _psc_stats(fits, X, delta_psc, alpha_sig)
    bp = _brant_pvals(fits, feats, alpha_sig)
    true_psc = _true_psc_gaps(X, alphas, beta_t)
    is_viol = np.array([true_psc[f] > delta_psc for f in feats])
    n_viol = int(is_viol.sum())
    rp = np.array([psc[f][2] for f in feats], float)
    bpv = np.array([bp[f] for f in feats], float)
    rp = np.where(np.isfinite(rp), rp, 1.0); bpv = np.where(np.isfinite(bpv), bpv, 1.0)

    def outcomes(reject, tag):
        V = int((reject & ~is_viol).sum()); S = int((reject & is_viol).sum()); R = int(reject.sum())
        return {f"{tag}_fwer": int(V > 0), f"{tag}_fdp": (V / R if R else 0.0),
                f"{tag}_power": (S / n_viol if n_viol else np.nan)}

    def bh(p):   return multipletests(p, alpha=alpha_mult, method="fdr_bh")[0]
    def holm(p): return multipletests(p, alpha=alpha_mult, method="holm")[0]
    rec = {"N": n, "rep": rep, "n_viol": n_viol}
    rec.update(outcomes(bpv < alpha_sig, "brant_raw"))
    rec.update(outcomes(holm(bpv), "brant_holm"))
    rec.update(outcomes(rp < alpha_sig, "rcpo_raw"))
    rec.update(outcomes(bh(rp), "rcpo_bh"))
    rec.update(outcomes(holm(rp), "rcpo_holm"))
    return rec


def run_e4psc(n_grid, reps, *, delta_psc, alpha_sig, alpha_mult, seed, jobs=1) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    tasks = [(n, rep, int(rng_master.integers(0, 2**63 - 1))) for n in n_grid for rep in range(reps)]
    kw = dict(delta_psc=delta_psc, alpha_sig=alpha_sig, alpha_mult=alpha_mult)
    if jobs == 1:
        rows = [_e4psc_task(n, rep, s, **kw) for (n, rep, s) in tasks]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_e4psc_task)(n, rep, s, **kw) for (n, rep, s) in tasks)
    rows = [r for r in rows if r is not None]
    print(f"  E4-PSC done: {len(rows)} usable reps across N-grid {n_grid}")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte-Carlo for the RC-PO PO diagnostic.")
    p.add_argument("--experiment", choices=["e1", "e2", "e3", "e4", "e4psc", "e1psc", "e2psc"],
                   default="e1",
                   help="e1/e2/e3 = coefficient-scale; e4 = many-covariate family-wise / FDR "
                        "control; e4psc = the same on the applied probability-scale PSC "
                        "estimand; e1psc/e2psc = probability-scale size/power")
    p.add_argument("--alpha-mult", type=float, default=0.05,
                   help="E4 multiplicity level for BH-FDR / Holm across covariates")
    p.add_argument("--delta-psc", type=float, default=0.01,
                   help="relevance threshold on the probability (PSC) scale, matching production")
    p.add_argument("--prev-grid", default="0.02,0.008,0.004,0.002,0.001",
                   help="E3 rare-dummy prevalence grid")
    p.add_argument("--e3-n", type=int, default=20000, help="E3 fixed sample size")
    p.add_argument("--rare-gap", type=float, default=0.50,
                   help="E3 true PO-violation gap planted on the rare dummies")
    p.add_argument("--n-grid", default="1000,5000,25000,100000",
                   help="comma-separated sample sizes")
    p.add_argument("--reps", type=int, default=80, help="repetitions per N")
    p.add_argument("--departure", type=float, default=0.04,
                   help="0.0 = exact PO; small (~0.04) = trivial departures (flagship)")
    p.add_argument("--delta", type=float, default=0.10,
                   help="relevance threshold (standardized-coefficient units)")
    p.add_argument("--alpha-sig", type=float, default=0.05, help="test level")
    p.add_argument("--method", choices=["mle", "firth"], default="mle",
                   help="threshold-logit estimator")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel workers for E1 reps (1 = serial, backward-compatible)")
    return p.parse_args()


def main() -> None:
    # Allow Unicode (Δ, δ, α) in console output on Windows cp1252 terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = parse_args()
    n_grid = [int(s) for s in args.n_grid.split(",") if s.strip()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment == "e1psc":
        print(f"[E1-PSC] departure={args.departure}  delta_psc={args.delta_psc}  "
              f"reps={args.reps}  N-grid={n_grid}")
        df = run_e1psc(n_grid, args.reps, departure=args.departure, delta_psc=args.delta_psc,
                       alpha_sig=args.alpha_sig, seed=args.seed, jobs=args.jobs)
        if df.empty:
            print("No usable replications."); return
        tag = f"dep{args.departure}".replace(".", "p")
        path = OUTPUT_DIR / f"sim_po_e1psc_{tag}.csv"; df.to_csv(path, index=False)
        g = df.groupby("N").agg(sig_fp=("sig_fp_rate", "mean"),
                                rcpo_psc_fp=("rcpo_psc_fp_rate", "mean"),
                                reps=("rep", "count")).reset_index()
        print("\n=== E1-PSC (false-positive rate among PSC-trivial covariates; "
              "probability-scale estimand) ===")
        print(g.to_string(index=False,
              formatters={c: (lambda v: f"{v:.3f}") for c in ["sig_fp", "rcpo_psc_fp"]}))
        print(f"\nWrote {path}")
        return

    if args.experiment == "e2psc":
        print(f"[E2-PSC] delta_psc={args.delta_psc}  reps={args.reps}  N-grid={n_grid}")
        df = run_e2psc(n_grid, args.reps, gaps=DEFAULT_E2PSC_GAPS, delta_psc=args.delta_psc,
                       alpha_sig=args.alpha_sig, seed=args.seed, jobs=args.jobs)
        if df.empty:
            print("No usable replications."); return
        path = OUTPUT_DIR / "sim_po_e2psc.csv"; df.to_csv(path, index=False)
        nmax = int(df["N"].max())
        bycov = (df[df["N"] == nmax].groupby("feature").agg(
            true_psc=("true_psc_gap", "mean"), sig=("sig_flag", "mean"),
            rcpo_psc=("rcpo_psc_flag", "mean")).reset_index().sort_values("true_psc"))
        print(f"\n=== E2-PSC per-covariate detection at N={nmax} (delta_psc={args.delta_psc}) ===")
        print(bycov.to_string(index=False,
              formatters={c: (lambda v: f"{v:.3f}") for c in ["true_psc", "sig", "rcpo_psc"]}))
        print(f"\nWrote {path}")
        return

    if args.experiment == "e1":
        print(f"[E1] departure={args.departure}  delta={args.delta}  method={args.method}  "
              f"reps={args.reps}  N-grid={n_grid}")
        df = run_e1(n_grid, args.reps, departure=args.departure, delta=args.delta,
                    alpha_sig=args.alpha_sig, method=args.method, seed=args.seed,
                    jobs=args.jobs)
        if df.empty:
            print("No usable replications — check the DGP / N grid.")
            return
        tag = f"dep{args.departure}".replace(".", "p")
        raw_path = OUTPUT_DIR / f"sim_po_e1_{tag}.csv"
        df.to_csv(raw_path, index=False)
        summary = summarize(df)
        print("\n=== E1 summary (mean false-positive PO-flag rate) ===")
        print(summary.to_string(index=False,
              formatters={c: (lambda v: f"{v:.3f}") for c in
                          ["brant_fp_mean", "brant_fp_sd", "rcpo_fp_mean", "rcpo_fp_sd"]}))
        print(f"\nWrote {raw_path}")
        return

    if args.experiment == "e4psc":
        print(f"[E4-PSC] many-covariate family-wise control on the PROBABILITY-scale PSC  "
              f"delta_psc={args.delta_psc}  reps={args.reps}  N-grid={n_grid}  "
              f"alpha_mult={args.alpha_mult}")
        df = run_e4psc(n_grid, args.reps, delta_psc=args.delta_psc, alpha_sig=args.alpha_sig,
                       alpha_mult=args.alpha_mult, seed=args.seed, jobs=args.jobs)
        if df.empty:
            print("No usable replications."); return
        raw_path = OUTPUT_DIR / "sim_po_e4psc.csv"; df.to_csv(raw_path, index=False)
        summary = summarize_e4(df)
        path2 = OUTPUT_DIR / "sim_po_e4psc_summary.csv"; summary.to_csv(path2, index=False)
        cols = [c for c in summary.columns if c.endswith(("_fwer", "_fdr", "_power"))]
        print(f"\n=== E4-PSC family-wise error / FDR / power vs N (probability-scale PSC, "
              f"delta_psc={args.delta_psc}); mean true violators = {df['n_viol'].mean():.1f}/20 ===")
        print(summary.to_string(index=False,
              formatters={c: (lambda v: f"{v:.3f}") for c in cols}))
        print(f"\nWrote {raw_path} and {path2}")
        return

    if args.experiment == "e4":
        print(f"[E4] many-covariate family-wise control  delta={args.delta}  "
              f"method={args.method}  reps={args.reps}  N-grid={n_grid}  "
              f"alpha_mult={args.alpha_mult}  (8 true violators of 20 covariates)")
        df = run_e4(n_grid, args.reps, delta=args.delta, alpha_sig=args.alpha_sig,
                    alpha_mult=args.alpha_mult, method=args.method, seed=args.seed,
                    jobs=args.jobs)
        if df.empty:
            print("No usable replications."); return
        raw_path = OUTPUT_DIR / "sim_po_e4.csv"; df.to_csv(raw_path, index=False)
        summary = summarize_e4(df)
        path2 = OUTPUT_DIR / "sim_po_e4_summary.csv"; summary.to_csv(path2, index=False)
        cols = [c for c in summary.columns if c.endswith(("_fwer", "_fdr", "_power"))]
        print("\n=== E4 family-wise error (FWER), false-discovery rate (FDR), and power "
              "vs N; 8 true violators among 20 covariates ===")
        print(summary.to_string(index=False,
              formatters={c: (lambda v: f"{v:.3f}") for c in cols}))
        print(f"\nWrote {raw_path} and {path2}")
        return

    if args.experiment == "e3":
        prev_grid = [float(s) for s in args.prev_grid.split(",") if s.strip()]
        print(f"[E3] N={args.e3_n}  rare-gap={args.rare_gap}  delta={args.delta}  "
              f"reps={args.reps}  prevalence-grid={prev_grid}")
        df = run_e3(prev_grid, args.reps, n=args.e3_n, gap_rare=args.rare_gap,
                    delta=args.delta, alpha_sig=args.alpha_sig, seed=args.seed,
                    jobs=args.jobs)
        if df.empty:
            print("No usable replications — check the DGP / prevalence grid.")
            return
        raw_path = OUTPUT_DIR / "sim_po_e3.csv"
        df.to_csv(raw_path, index=False)
        summary = summarize_e3(df)
        print("\n=== E3 (detection of a TRUE rare-dummy violator + MLE separation rate) ===")
        print(summary.to_string(index=False,
              formatters={c: (lambda v: f"{v:.3f}") for c in
                          ["mle_unstable", "heuristic_detect", "rcpo_mle_detect",
                           "rcpo_firth_detect"]}))
        print(f"\nWrote {raw_path}")
        return

    # ---- E2: power vs true effect size ----
    print(f"[E2] delta={args.delta}  method={args.method}  reps={args.reps}  "
          f"N-grid={n_grid}\n     planted gaps={list(DEFAULT_E2_GAPS)}")
    df = run_e2(n_grid, args.reps, gaps=DEFAULT_E2_GAPS, delta=args.delta,
                alpha_sig=args.alpha_sig, method=args.method, seed=args.seed,
                jobs=args.jobs)
    if df.empty:
        print("No usable replications — check the DGP / N grid.")
        return
    raw_path = OUTPUT_DIR / "sim_po_e2.csv"
    df.to_csv(raw_path, index=False)
    by_n, by_cov = summarize_e2(df, args.delta)
    print("\n=== E2 power (detection among substantive Δ>δ) and FP (among trivial) ===")
    print(by_n.to_string(index=False,
          formatters={c: (lambda v: f"{v:.3f}") for c in
                      ["brant_power", "rcpo_power", "brant_fp", "rcpo_fp"]}))
    print("\n=== E2 per-covariate detection at max N ===")
    print(by_cov.to_string(index=False,
          formatters={c: (lambda v: f"{v:.3f}") for c in
                      ["true_gap", "brant_detect", "rcpo_detect"]}))
    print(f"\nWrote {raw_path}")


if __name__ == "__main__":
    main()
