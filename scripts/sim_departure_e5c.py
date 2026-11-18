#!/usr/bin/env python3
"""Extended Monte Carlo design for the departure diagnostic (pre-submission review item: 'simulations
too narrow'): crash-level clustering, outcome-dependent selection with design weights, a correlated
categorical block with a nearly collinear nested dummy, and a rare fatal cut.

Population model (per person i in crash c):
    P(y_i > t | x_i, u_c) = Lambda(alpha_t + x_i' beta_t + u_c),  t = 0..3,  u_c ~ N(0, sigma_u^2),
    alpha chosen so that P(y > t | x = 0, u = 0) ~ (0.49, 0.30, 0.12, 0.03)   (rare fatal cut),
    crashes of 1-3 persons (0.45 / 0.40 / 0.15);
    covariates: three N(0,1) regressors with common coefficients (0.30, -0.25, 0.20); one four-level
    categorical variable (reference 0.70, levels 0.15 / 0.10 / 0.05) coded as dummies A, B, C with A
    common (0.40), B drifting linearly from ell_B at t = 0 to 0 at t = 3 (substantive, calibrated to
    3*delta on the selected frame), C drifting from ell_C to 0 (boundary, calibrated to delta); and a
    nested dummy 1{level in {A, B}} with 1% of its entries flipped (nearly collinear block), common 0.20.
Selection (outcome-dependent, as in a fused fatal-census / crash-sample frame): records are kept with
    probability (0.25, 0.25, 0.60, 1.00, 1.00) for y = 0..4; design weight = 1 / selection probability.
The diagnostic runs UNWEIGHTED on the selected records (as in the application). Ground truth is the
departure implied by the pseudo-true cut-specific coefficients of the selected frame, obtained from a
Firth fit on a 3-million-person population draw (the frame-induced departure of population-PO
covariates is therefore part of the truth and is reported). Bounds: marginal delta-method, simultaneous
max-t from the model covariance, and simultaneous max-t from the crash-cluster sandwich.
Outputs: sim_e5c_long.csv, sim_e5c_summary.csv, e5c_meta.json
Usage:   python sim_departure_e5c.py --outdir outputs/v2/sim_ext --reps 300 --n-grid 5000,25000,100000 --B 200 --jobs 32
"""
from __future__ import annotations
import argparse, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_departure as SD
import run_simboot3 as S3
from run_simboot import chol_jitter
from departure_core import departure_profile, true_departure, logistic
warnings.filterwarnings("ignore")

DELTA, ALPHA, Z = SD.DELTA, SD.ALPHA, SD.Z
E5_ALPHAS = np.array([-0.040, -0.847, -1.992, -3.476])     # P(y>t | x=0, u=0) ~ (0.49, 0.30, 0.12, 0.03)
E5_SEL = np.array([0.25, 0.25, 0.60, 1.00, 1.00])           # selection probability by outcome level
E5_SIG_U = 0.5                                              # crash random-effect SD
E5_LEVELS = np.array([0.70, 0.15, 0.10, 0.05])              # categorical variable: reference, A, B, C
FEATS = ["x_cont_1", "x_cont_2", "x_cont_3", "cat_A", "cat_B", "cat_C", "nested_AB"]
FTYPE = ["null_cont", "null_cont", "null_cont", "null_dummy", "substantive", "boundary", "null_nested"]


def build_e5c(rng, n_pop, lo_b, lo_c):
    sizes = rng.choice([1, 2, 3], size=n_pop, p=[0.45, 0.40, 0.15])
    cum = np.cumsum(sizes); ncr = int(np.searchsorted(cum, n_pop)) + 1; sizes = sizes[:ncr]; n = int(sizes.sum())
    crash = np.repeat(np.arange(ncr), sizes); u = rng.normal(0.0, E5_SIG_U, ncr)[crash]
    Xc = rng.standard_normal((n, 3))
    lev = rng.choice(4, size=n, p=E5_LEVELS)
    A = (lev == 1).astype(float); Bd = (lev == 2).astype(float); C = (lev == 3).astype(float)
    nested = ((lev == 1) | (lev == 2)).astype(float); flip = rng.random(n) < 0.01; nested[flip] = 1.0 - nested[flip]
    X = np.column_stack([Xc, A, Bd, C, nested]); p = X.shape[1]
    bt = np.zeros((p, 4)); bt[0] = 0.30; bt[1] = -0.25; bt[2] = 0.20; bt[3] = 0.40
    bt[4] = np.linspace(lo_b, 0.0, 4); bt[5] = np.linspace(lo_c, 0.0, 4); bt[6] = 0.20
    cp = np.column_stack([logistic(E5_ALPHAS[t] + X @ bt[:, t] + u) for t in range(4)])
    cp = np.minimum.accumulate(cp, axis=1)
    pr = np.column_stack([1 - cp[:, 0], cp[:, 0] - cp[:, 1], cp[:, 1] - cp[:, 2], cp[:, 2] - cp[:, 3], cp[:, 3]])
    pr = np.clip(pr, 1e-12, None); pr /= pr.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > np.cumsum(pr, 1)).sum(1)
    keep = rng.random(n) < E5_SEL[y]
    return X[keep], y[keep], crash[keep], 1.0 / E5_SEL[y[keep]], bt


TRUTH_POP, CAL_POP = 3_000_000, 600_000


def pseudo_truth(lo_b, lo_c, n_pop=None, seed=777):
    """Pseudo-true cut-specific coefficients of the selected frame (Firth fit on a large draw) and the
    implied departure per feature."""
    n_pop = n_pop or TRUTH_POP
    rng = np.random.default_rng(seed); X, y, crash, w, bt = build_e5c(rng, n_pop, lo_b, lo_c)
    thr, B, Sg, eta, covs, Xm, A = SD.firth_fits(X, y)
    Dtrue, _ = true_departure(A, B.T, X)
    return Dtrue, A, B, float(len(y) / n_pop)


def calibrate_e5c():
    """bisection on ell_C (boundary, target delta) and ell_B (substantive, target 3*delta) on the selected frame"""
    lo_b, lo_c = 1.0, 0.3
    for _round in range(2):
        a, b = 0.0, 3.0
        for _ in range(14):
            m = 0.5 * (a + b); D, _, _, _ = pseudo_truth(lo_b, m, n_pop=CAL_POP)
            if D[5] < DELTA: a = m
            else: b = m
        lo_c = 0.5 * (a + b)
        a, b = 0.0, 4.0
        for _ in range(14):
            m = 0.5 * (a + b); D, _, _, _ = pseudo_truth(m, lo_c, n_pop=CAL_POP)
            if D[4] < 3 * DELTA: a = m
            else: b = m
        lo_b = 0.5 * (a + b)
    return lo_b, lo_c


def boot_lcb(Xm, X, A, B, covs, profs, Dtrue, Bdraws, rng, thr):
    """max-t simultaneous bounds for every feature from B joint draws per cut (given covariances)."""
    T = len(thr); m = X.shape[1]; Ls = [chol_jitter(c) for c in covs]
    beta_b = []; eta_b = []
    for i in range(T):
        full = np.concatenate([[A[i]], B[i]]); bb = full[:, None] + Ls[i] @ rng.standard_normal((m + 1, Bdraws))
        beta_b.append(bb); eta_b.append(Xm @ bb)
    lam_b = [logistic(e) for e in eta_b]
    out = []
    for j in range(m):
        xj = X[:, j]; w = profs[j]["w"]; Dhat = profs[j]["D"]
        bj = np.stack([beta_b[i][j + 1] for i in range(T)]); bbar_b = (w[:, None] * bj).sum(0)
        Db = np.zeros((T, Bdraws))
        for i in range(T):
            e_mk = eta_b[i] - xj[:, None] * bj[i][None, :]
            Db[i] = (lam_b[i] - logistic(e_mk + xj[:, None] * bbar_b[None, :])).mean(0)
        r = S3.maxt(Dhat, Db, list(range(T)))
        out.append((r["lcb"], r["c_alpha"], r["p"]))
    return out


def e5c_task(n_target, rep, seed, lo_b, lo_c, Dtrue, Bdraws):
    rng = np.random.default_rng(seed)
    n_pop = int(n_target / 0.40 * 1.06)
    X, y, crash, w, bt = build_e5c(rng, n_pop, lo_b, lo_c)
    if len(y) > n_target: X, y, crash, w = X[:n_target], y[:n_target], crash[:n_target], w[:n_target]
    m = X.shape[1]
    thr, B, Sg, eta, covs, Xm, A = SD.firth_fits(X, y, want_cov=True)
    if len(thr) < 4: return []
    T = len(thr)
    codes, uniq = pd.factorize(crash); n_cl = len(uniq)
    covs_cl = [S3.cluster_cov(Xm, (y > t).astype(int), np.concatenate([[A[i]], B[i]]), covs[i], codes, n_cl) for i, t in enumerate(thr)]
    Scl = np.array([np.sqrt(np.clip(np.diag(c), 0, None))[1:] for c in covs_cl])
    profs = [departure_profile(eta, B[:, j], Sg[:, j], X[:, j]) for j in range(m)]
    profs_cl = [departure_profile(eta, B[:, j], Scl[:, j], X[:, j]) for j in range(m)]
    rb = np.random.default_rng(seed + 1)
    sim_m = boot_lcb(Xm, X, A, B, covs, profs, Dtrue, Bdraws, rb, thr)
    sim_c = boot_lcb(Xm, X, A, B, covs_cl, profs_cl, Dtrue, Bdraws, rb, thr)
    rows = []
    for j in range(m):
        Dhat = profs[j]["D"]; se = profs[j]["se_D"]; se_cl = profs_cl[j]["se_D"]
        lcb_d = float(np.max(np.abs(Dhat) - Z * se)); lcb_dcl = float(np.max(np.abs(Dhat) - Z * se_cl))
        rows.append(dict(N=n_target, rep=rep, feature=FEATS[j], ftype=FTYPE[j], Dmax=float(np.abs(Dhat).max()), Dtrue=float(Dtrue[j]),
                         lcb_delta=lcb_d, lcb_delta_cl=lcb_dcl, lcb_sim=sim_m[j][0], lcb_simcl=sim_c[j][0],
                         c_alpha=sim_m[j][1], c_alpha_cl=sim_c[j][1], p_sim=sim_m[j][2], p_simcl=sim_c[j][2],
                         flag_delta=int(lcb_d > DELTA), flag_delta_cl=int(lcb_dcl > DELTA), flag_sim=int(sim_m[j][0] > DELTA), flag_simcl=int(sim_c[j][0] > DELTA),
                         cover_delta=int(lcb_d <= Dtrue[j]), cover_delta_cl=int(lcb_dcl <= Dtrue[j]), cover_sim=int(sim_m[j][0] <= Dtrue[j]), cover_simcl=int(sim_c[j][0] <= Dtrue[j]),
                         se_ratio_cl=float(np.median(Scl[:, j] / np.maximum(Sg[:, j], 1e-12))), n_clusters=int(n_cl)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/v2/sim_ext")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20251231)
    ap.add_argument("--n-grid", default="5000,25000,100000")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--calib-pop", type=int, default=600_000)
    ap.add_argument("--truth-pop", type=int, default=3_000_000)
    a = ap.parse_args()
    global TRUTH_POP, CAL_POP
    TRUTH_POP, CAL_POP = a.truth_pop, a.calib_pop
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    grid = [int(x) for x in a.n_grid.split(",") if x.strip()]
    t0 = time.time()
    lo_b, lo_c = calibrate_e5c()
    Dtrue, A_star, B_star, sel_frac = pseudo_truth(lo_b, lo_c)
    print(f"[e5c calibration] ell_B={lo_b:.4f} ell_C={lo_c:.4f}; selected fraction {sel_frac:.3f}", flush=True)
    print("[e5c truth on the selected frame] " + ", ".join(f"{f}({ft})={d:.4f}" for f, ft, d in zip(FEATS, FTYPE, Dtrue)), flush=True)
    meta = dict(ell_B=lo_b, ell_C=lo_c, selected_fraction=sel_frac, Dtrue={f: float(d) for f, d in zip(FEATS, Dtrue)}, ftype=dict(zip(FEATS, FTYPE)),
                alphas_population=E5_ALPHAS.tolist(), alphas_pseudo=A_star.tolist(), beta_pseudo=B_star.tolist(), selection=E5_SEL.tolist(),
                sigma_u=E5_SIG_U, B=a.B, reps=a.reps)
    (outdir / "e5c_meta.json").write_text(json.dumps(meta, indent=1))
    rngm = np.random.default_rng(a.seed)
    tasks = [(n, r, int(rngm.integers(0, 2 ** 62)), lo_b, lo_c, Dtrue, a.B) for n in grid for r in range(a.reps)]
    df = pd.DataFrame([r for rows in SD.run(tasks, e5c_task, a.jobs) for r in rows])
    summ = df.groupby(["N", "ftype"]).agg(Dtrue=("Dtrue", "mean"), Dmax_mean=("Dmax", "mean"),
                                          flag_delta=("flag_delta", "mean"), flag_delta_cl=("flag_delta_cl", "mean"),
                                          flag_sim=("flag_sim", "mean"), flag_simcl=("flag_simcl", "mean"),
                                          cover_delta=("cover_delta", "mean"), cover_delta_cl=("cover_delta_cl", "mean"),
                                          cover_sim=("cover_sim", "mean"), cover_simcl=("cover_simcl", "mean"),
                                          c_alpha_med=("c_alpha", "median"), c_alpha_cl_med=("c_alpha_cl", "median"),
                                          se_ratio_cl=("se_ratio_cl", "median"), reps=("rep", "nunique")).reset_index()
    df.to_csv(outdir / "sim_e5c_long.csv", index=False); summ.to_csv(outdir / "sim_e5c_summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"DONE e5c [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
