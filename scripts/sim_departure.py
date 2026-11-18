#!/usr/bin/env python3
"""Monte Carlo validation of the DEPARTURE statistic (RCSR-POD v2).

Experiments (--experiment):
  e0   strong-effect exact-PO demonstration: default DGP plus one binary covariate with
       prevalence 0.68 and a COMMON coefficient (beta = -2 or -0.5 at every cut). Reports,
       by N, the flag rate of the previous plain-PSC-gap rule and of the departure rule
       for that covariate (the former flags a PO-consistent large effect; the latter does not).
  e1d  exact PO (departure 0) and trivial departures (0.04) on the default DGP: false-flag
       rate among covariates whose TRUE Dmax <= delta, Wald/Brant vs departure rule, by N.
  e2d  detection versus true Dmax (E2PSC gap design), N in {25,000, 100,000}.
  e4d  many-covariate family-wise test (Task-B DGP: 6 exact-PO + 6 trivial continuous
       nulls, 8 dense-dummy violators): Brant vs departure rule, raw / BH / Holm, by N.
  bnd  boundary and simultaneous-rule check: carriers calibrated to true Dmax = delta
       (size), exact PO, and 3*delta (power); marginal delta-method vs max-t bootstrap rule.
All decisions use delta = 0.01 on the probability scale and one-sided alpha = 0.05.
"""
from __future__ import annotations
import argparse, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_po_diagnostic as S
from po_diagnostic_core import firth_logit
from run_simboot import chol_jitter
from departure_core import departure_profile, summarize, true_departure, logistic
warnings.filterwarnings("ignore")

DELTA, ALPHA = 0.01, 0.05
Z = stats.norm.ppf(1 - ALPHA)
ALPHAS = S.DEFAULT_ALPHAS
N_GRID = [1000, 2500, 5000, 10000, 25000, 50000, 100000]


# ------------------------------------------------------------------ fitting
def firth_fits(X, y, want_cov=False, ridge=1e-6):
    n, p = X.shape; Xm = np.column_stack([np.ones(n), X])
    thr = [t for t in range(4) if 0 < (y > t).sum() < n]
    B = np.zeros((len(thr), p)); Sg = np.zeros((len(thr), p)); A = np.zeros(len(thr)); eta = []; covs = []
    for i, t in enumerate(thr):
        b, se, ok = firth_logit(Xm, (y > t).astype(int), ridge=ridge)
        B[i], Sg[i], A[i] = b[1:], se[1:], b[0]; eta.append(Xm @ b)
        if want_cov:
            pr = logistic(Xm @ b); w = pr * (1 - pr)
            info = (Xm * w[:, None]).T @ Xm + ridge * np.eye(p + 1)
            try: covs.append(np.linalg.inv(info))
            except np.linalg.LinAlgError: covs.append(np.linalg.pinv(info))
    return thr, B, Sg, eta, covs, Xm, A


def brant_p(B, Sg):
    """Coefficient-scale Wald/Brant p-value per feature (marginal form, cut 0 as reference)."""
    T, p = B.shape; out = np.ones(p)
    for j in range(p):
        chi2 = float(np.sum((B[1:, j] - B[0, j]) ** 2 / (Sg[1:, j] ** 2 + Sg[0, j] ** 2)))
        out[j] = 1 - stats.chi2.cdf(chi2, T - 1)
    return out


def feature_stats(eta, B, Sg, X):
    """Per feature: old PSC gap/LCB, Dmax/LCB, marginal-D p-value for H0: Dmax <= delta."""
    p = X.shape[1]; rows = []
    for j in range(p):
        prof = departure_profile(eta, B[:, j], Sg[:, j], X[:, j]); s = summarize(prof, ALPHA)
        Tobs = float(np.max((np.abs(prof["D"]) - DELTA) / np.maximum(prof["se_D"], 1e-12)))
        rows.append(dict(psc_gap=s["psc_gap"], psc_lcb=s["psc_lcb"], Dmax=s["Dmax"], Dmax_lcb=s["Dmax_lcb"],
                         p_D=float(1 - stats.norm.cdf(Tobs))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ e0
def e0_task(n, rep, seed, beta_strong):
    rng = np.random.default_rng(seed)
    xb = (rng.random(n) < 0.68).astype(float); xc = rng.standard_normal((n, 2))
    xr = np.column_stack([(rng.random(n) < q).astype(float) for q in (0.02, 0.01)])
    X = np.column_stack([xb, xc, xr]); beta = np.array([beta_strong, 0.35, -0.28, 0.40, -0.45])
    cp = np.column_stack([logistic(ALPHAS[t] + X @ beta) for t in range(4)])
    pr = np.column_stack([1 - cp[:, 0], cp[:, 0] - cp[:, 1], cp[:, 1] - cp[:, 2], cp[:, 2] - cp[:, 3], cp[:, 3]])
    y = (rng.random(n)[:, None] > np.cumsum(pr, 1)).sum(1)
    thr, B, Sg, eta, _, _, _ = firth_fits(X, y)
    if len(thr) < 2: return None
    st = feature_stats(eta, B, Sg, X).iloc[0]
    return dict(N=n, rep=rep, beta=beta_strong, psc_gap=st.psc_gap, psc_flag=int(st.psc_lcb > DELTA),
                Dmax=st.Dmax, D_flag=int(st.Dmax_lcb > DELTA), brant_flag=int(brant_p(B, Sg)[0] < ALPHA))


# ------------------------------------------------------------------ e1d / e2d
def e12_task(n, rep, seed, departure, gaps):
    rng = np.random.default_rng(seed)
    if gaps is not None:
        Xdf, y, _, beta_t, alphas = S.simulate_ordinal(rng, n, gap_targets=gaps, return_betat=True)
    else:
        Xdf, y, _, beta_t, alphas = S.simulate_ordinal(rng, n, departure=departure, return_betat=True)
    X = Xdf.values; feats = list(Xdf.columns)
    Dtrue, _ = true_departure(alphas, beta_t, X)
    thr, B, Sg, eta, _, _, _ = firth_fits(X, y)
    if len(thr) < 2: return []
    st = feature_stats(eta, B, Sg, X); bp = brant_p(B, Sg)
    return [dict(N=n, rep=rep, departure=departure, feature=f, Dmax_true=float(Dtrue[j]),
                 substantive=bool(Dtrue[j] > DELTA), brant_flag=int(bp[j] < ALPHA),
                 D_flag=int(st.Dmax_lcb[j] > DELTA), psc_flag=int(st.psc_lcb[j] > DELTA))
            for j, f in enumerate(feats)]


# ------------------------------------------------------------------ e4d (Task-B DGP)
P_CONT, P_DUM, DUMMY_PREV, N_EXACT = 12, 8, 0.40, 6


def build_taskb(rng, n):
    Xc = rng.standard_normal((n, P_CONT)); Xd = (rng.random((n, P_DUM)) < DUMMY_PREV).astype(float)
    X = np.column_stack([Xc, Xd]); bt = np.zeros((P_CONT + P_DUM, 4))
    for j in range(N_EXACT): bt[j] = 0.12
    for j in range(N_EXACT, P_CONT): bt[j] = np.linspace(0.35, 0.0, 4)
    for j in range(P_CONT, P_CONT + P_DUM): bt[j] = np.linspace(1.0, 0.0, 4)
    cp = np.column_stack([logistic(ALPHAS[t] + X @ bt[:, t]) for t in range(4)])
    cp = np.minimum.accumulate(cp, axis=1)
    pr = np.column_stack([1 - cp[:, 0], cp[:, 0] - cp[:, 1], cp[:, 1] - cp[:, 2], cp[:, 2] - cp[:, 3], cp[:, 3]])
    pr = np.clip(pr, 1e-12, None); pr /= pr.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > np.cumsum(pr, 1)).sum(1)
    return X, y, bt


def multitest(pv, alpha):
    from statsmodels.stats.multitest import multipletests
    return multipletests(pv, alpha=alpha, method="holm")[0], multipletests(pv, alpha=alpha, method="fdr_bh")[0]


def e4d_task(n, rep, seed):
    rng = np.random.default_rng(seed); X, y, bt = build_taskb(rng, n)
    Dtrue, _ = true_departure(ALPHAS, bt, X); is_viol = Dtrue > DELTA
    thr, B, Sg, eta, _, _, _ = firth_fits(X, y)
    if len(thr) < 2: return None
    st = feature_stats(eta, B, Sg, X); pD = st.p_D.to_numpy(); pB = brant_p(B, Sg)
    nv = int(is_viol.sum())

    def outc(rej, tag):
        V = int((rej & ~is_viol).sum()); Sd = int((rej & is_viol).sum()); R = int(rej.sum())
        return {f"{tag}_fwer": int(V > 0), f"{tag}_fdp": (V / R if R else 0.0), f"{tag}_power": (Sd / nv if nv else np.nan)}
    hB, bhB = multitest(pB, ALPHA); hD, bhD = multitest(pD, ALPHA)
    rec = dict(N=n, rep=rep, n_viol=nv, n_viol_nominal=P_DUM)
    rec.update(outc(pB < ALPHA, "brant_raw")); rec.update(outc(hB, "brant_holm"))
    rec.update(outc(pD < ALPHA, "rcpo_raw")); rec.update(outc(bhD, "rcpo_bh")); rec.update(outc(hD, "rcpo_holm"))
    return rec


# ------------------------------------------------------------------ bnd (boundary + simultaneous)
KT, KN = 6, 14


def build_bnd(rng, n, lo):
    Xd = (rng.random((n, KT)) < 0.40).astype(float); Xc = rng.standard_normal((n, KN))
    X = np.column_stack([Xd, Xc]); bt = np.zeros((KT + KN, 4))
    for j in range(KT): bt[j] = np.linspace(lo, 0.0, 4)
    for j in range(KT, KT + KN): bt[j] = 0.10
    cp = np.column_stack([logistic(ALPHAS[t] + X @ bt[:, t]) for t in range(4)])
    cp = np.minimum.accumulate(cp, axis=1)
    pr = np.column_stack([1 - cp[:, 0], cp[:, 0] - cp[:, 1], cp[:, 1] - cp[:, 2], cp[:, 2] - cp[:, 3], cp[:, 3]])
    pr = np.clip(pr, 1e-12, None); pr /= pr.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > np.cumsum(pr, 1)).sum(1)
    return X, y, bt


def carrier_true_dmax(lo, n=120000, seed=777):
    rng = np.random.default_rng(seed); X, _, bt = build_bnd(rng, n, lo)
    Dtrue, _ = true_departure(ALPHAS, bt, X)
    return float(np.mean(Dtrue[:KT]))


def calibrate(target, iters=18):
    lo_lo, lo_hi = 0.0, 3.0
    for _ in range(iters):
        lo = 0.5 * (lo_lo + lo_hi)
        if carrier_true_dmax(lo) < target: lo_lo = lo
        else: lo_hi = lo
    lo = 0.5 * (lo_lo + lo_hi)
    return lo, carrier_true_dmax(lo, seed=778)


def bnd_task(scenario, n, rep, seed, lo, Bdraws):
    rng = np.random.default_rng(seed); X, y, bt = build_bnd(rng, n, lo); m = X.shape[1]
    Dtrue, _ = true_departure(ALPHAS, bt, X)   # per-replication truth for coverage of the lower bounds
    thr, B, Sg, eta, covs, Xm, A = firth_fits(X, y, want_cov=True)
    if len(thr) < 2: return []
    T = len(thr)
    profs = [departure_profile(eta, B[:, j], Sg[:, j], X[:, j]) for j in range(m)]
    # shared draws per threshold (full vector incl. intercept); departure per draw for every feature
    rb = np.random.default_rng(seed + 1)
    Ls = [chol_jitter(c) for c in covs]
    beta_b = []; eta_b = []
    for i in range(T):
        full = np.concatenate([[A[i]], B[i]])
        bb = full[:, None] + Ls[i] @ rb.standard_normal((m + 1, Bdraws))   # (m+1, B)
        beta_b.append(bb); eta_b.append(Xm @ bb)                            # (n, B)
    lam_b = [logistic(e) for e in eta_b]
    rows = []
    for j in range(m):
        xj = X[:, j]; w = profs[j]["w"]; Dhat = profs[j]["D"]
        bj = np.stack([beta_b[i][j + 1] for i in range(T)])           # (T, B)
        bbar_b = (w[:, None] * bj).sum(0)                              # (B,)
        Db = np.zeros((T, Bdraws))
        for i in range(T):
            e_mk = eta_b[i] - xj[:, None] * bj[i][None, :]
            Db[i] = (lam_b[i] - logistic(e_mk + xj[:, None] * bbar_b[None, :])).mean(0)
        SEb = Db.std(axis=1, ddof=1); ok = SEb > 0
        tmax = np.max(np.abs(Db[ok] - Dhat[ok, None]) / SEb[ok, None], axis=0)
        c_alpha = float(np.quantile(tmax, 1 - ALPHA))
        lcb_sim = float(np.max(np.abs(Dhat[ok]) - c_alpha * SEb[ok]))
        lcb_delta = float(np.max(np.abs(Dhat) - Z * profs[j]["se_D"]))
        rows.append(dict(scenario=scenario, N=n, rep=rep, feature=j, feature_type="carrier" if j < KT else "exactPO",
                         flag_delta=int(lcb_delta > DELTA), flag_simboot=int(lcb_sim > DELTA), c_alpha=c_alpha,
                         Dmax=float(np.max(np.abs(Dhat))), lcb_delta=lcb_delta, lcb_sim=lcb_sim, Dtrue=float(Dtrue[j]),
                         cover_delta=int(lcb_delta <= Dtrue[j]), cover_sim=int(lcb_sim <= Dtrue[j])))
    return rows


# ------------------------------------------------------------------ main
def run(tasks, fn, jobs):
    if jobs == 1:
        return [fn(*t) for t in tasks]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=jobs, verbose=5)(delayed(fn)(*t) for t in tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["e0", "e1d", "e2d", "e4d", "bnd"])
    ap.add_argument("--outdir", default="outputs/departure_sim")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20251231)
    ap.add_argument("--n-grid", default=",".join(map(str, N_GRID)))
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    grid = [int(x) for x in a.n_grid.split(",") if x.strip()]
    rngm = np.random.default_rng(a.seed); t0 = time.time()
    seeds = lambda k: [int(s) for s in rngm.integers(0, 2 ** 62, size=k)]

    if a.experiment == "e0":
        tasks = [(n, r, s, b) for b in (-2.0, -0.5) for n in grid for r in range(a.reps) for s in seeds(1)]
        df = pd.DataFrame([r for r in run(tasks, e0_task, a.jobs) if r])
        summ = df.groupby(["beta", "N"]).agg(psc_gap=("psc_gap", "mean"), psc_flag=("psc_flag", "mean"),
                                             Dmax=("Dmax", "mean"), D_flag=("D_flag", "mean"),
                                             brant_flag=("brant_flag", "mean"), reps=("rep", "nunique")).reset_index()
    elif a.experiment in ("e1d", "e2d"):
        if a.experiment == "e1d":
            tasks = [(n, r, s, dep, None) for dep in (0.0, 0.04) for n in grid for r in range(a.reps) for s in seeds(1)]
        else:
            tasks = [(n, r, s, 0.0, S.DEFAULT_E2PSC_GAPS) for n in grid for r in range(a.reps) for s in seeds(1)]
        df = pd.DataFrame([r for rows in run(tasks, e12_task, a.jobs) for r in rows])
        if a.experiment == "e1d":
            triv = df[~df.substantive]
            summ = triv.groupby(["departure", "N"]).agg(brant_fp=("brant_flag", "mean"), D_fp=("D_flag", "mean"),
                                                        psc_fp=("psc_flag", "mean"), n_cov=("feature", "nunique"),
                                                        max_Dtrue=("Dmax_true", "max"), reps=("rep", "nunique")).reset_index()
        else:
            summ = df.groupby(["N", "feature"]).agg(Dmax_true=("Dmax_true", "mean"), substantive=("substantive", "mean"),
                                                    brant_det=("brant_flag", "mean"), D_det=("D_flag", "mean"),
                                                    psc_det=("psc_flag", "mean")).reset_index()
    elif a.experiment == "e4d":
        rng0 = np.random.default_rng(a.seed); X, y, bt = build_taskb(rng0, 120000)
        Dtrue, _ = true_departure(ALPHAS, bt, X)
        print(f"[e4d truth @120k] true Dmax: violators median {np.median(Dtrue[P_CONT:]):.4f} "
              f"(min {Dtrue[P_CONT:].min():.4f}); nulls max {Dtrue[:P_CONT].max():.4f}; n_viol={int((Dtrue > DELTA).sum())}", flush=True)
        tasks = [(n, r, s) for n in grid for r in range(a.reps) for s in seeds(1)]
        df = pd.DataFrame([r for r in run(tasks, e4d_task, a.jobs) if r])
        summ = df.groupby("N").mean(numeric_only=True).reset_index()
    else:
        lo_b, d_b = calibrate(DELTA); lo_s, d_s = calibrate(3 * DELTA)
        print(f"[bnd calibration] boundary lo={lo_b:.4f} (true Dmax {d_b:.4f}); substantive lo={lo_s:.4f} (true Dmax {d_s:.4f})", flush=True)
        scen = {"exactPO": 0.0, "boundary": lo_b, "substantive": lo_s}
        tasks = [(sc, n, r, s, lo, a.B) for sc, lo in scen.items() for n in grid for r in range(a.reps) for s in seeds(1)]
        df = pd.DataFrame([r for rows in run(tasks, bnd_task, a.jobs) for r in rows])
        summ = df.groupby(["scenario", "N", "feature_type"]).agg(delta_rate=("flag_delta", "mean"), simboot_rate=("flag_simboot", "mean"),
                                                                  c_alpha_med=("c_alpha", "median"), Dmax_mean=("Dmax", "mean"),
                                                                  cover_delta=("cover_delta", "mean"), cover_sim=("cover_sim", "mean"),
                                                                  lcb_delta_mean=("lcb_delta", "mean"), lcb_sim_mean=("lcb_sim", "mean"),
                                                                  Dtrue_mean=("Dtrue", "mean"), reps=("rep", "nunique")).reset_index()
        (outdir / "bnd_meta.json").write_text(json.dumps(dict(lo_boundary=lo_b, Dtrue_boundary=d_b, lo_substantive=lo_s, Dtrue_substantive=d_s, B=a.B)))
    df.to_csv(outdir / f"sim_{a.experiment}_long.csv", index=False)
    summ.to_csv(outdir / f"sim_{a.experiment}_summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"DONE {a.experiment} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
