#!/usr/bin/env python3
"""Penalized partial proportional-odds cumulative logit (manuscript Eq. 2) on the
full frame (pre-submission item C1: the partial-PO deliverable estimated with
FINITE coefficients, replacing the separation-divergent MLE profiles of Fig. 3b).

For each pathway three nested cumulative-logit specifications are fitted by
L-BFGS on the exact multinomial likelihood of the five ordered categories,
P(y > t | x) = Lambda(alpha_t + x1'beta + x2'beta_t), t = 0..3, with the same
L2 penalty as the manuscript's baseline (lambda = 1 on every slope, none on the
cuts; features are scaled without centring exactly as in the diagnostic):
    PO   : every feature common across the four cuts,
    PPO  : the reported credible set threshold-specific, the rest common,
    FULL : every feature threshold-specific (the generalized ordered logit).
Outputs partial_po_coefs_{pathway}.csv (threshold-specific coefficients of the
credible set under PPO and FULL, common PO coefficient for reference) and
partial_po_fit_summary.json (LL, k, AIC, BIC, share of records with a negative
fitted category probability, optimizer status).

Usage:
    python run_partial_po_fit.py --frame outputs/kabco_person_level_clean.csv \
        --outdir outputs/hardening2 --maxiter 600
"""
from __future__ import annotations
import argparse, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_a as TA
warnings.filterwarnings("ignore")

# threshold-specific set per pathway: filled from --sets (simboot3_summary.json: the
# substantive simultaneous credible set plus the frame terms), else these defaults
SIM_CREDIBLE = {"infrastructure": ["vehicle_count", "SEX", "mode_impact1_12.0", "RUR_URB_MISSING"],
                "pavement": ["vehicle_count", "mode_vsurcond_1.0", "veh_trav_sp", "SEX", "mode_vpavetyp_2.0",
                             "veh_vspd_lim", "mode_vpavetyp_MISSING"]}


def load_sets(path):
    import json
    js = json.loads(Path(path).read_text())
    out = {}
    for s in js:
        out[s["pathway"]] = list(dict.fromkeys(s["set_sim"] + s.get("frame_terms", [])))
    return out

T = 4  # cumulative cuts t = 0..3 for the five KABCO levels


def _lg(x):
    return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-12, 1 - 1e-12)


class PartialPO:
    def __init__(self, X1, X2, y, lam=1.0):
        self.X1 = X1; self.X2 = X2; self.y = y.astype(int); self.lam = lam
        self.n = len(y); self.p1 = 0 if X1 is None else X1.shape[1]
        self.p2 = 0 if X2 is None else X2.shape[1]
        self.masks_up = [(self.y == t + 1) for t in range(T)]   # eta_t is the lower edge of category t+1
        self.masks_dn = [(self.y == t) for t in range(T)]       # ... and the upper edge of category t
        self.last_neg = 0

    def unpack(self, th):
        a = th[:T]; b = th[T:T + self.p1]
        bt = th[T + self.p1:].reshape(T, self.p2) if self.p2 else np.zeros((T, 0))
        return a, b, bt

    def etas(self, th):
        a, b, bt = self.unpack(th)
        base = self.X1 @ b if self.p1 else 0.0
        return [a[t] + base + (self.X2 @ bt[t] if self.p2 else 0.0) for t in range(T)]

    def probs(self, th):
        F = [_lg(e) for e in self.etas(th)]
        P = np.empty(self.n)
        P[self.masks_dn[0]] = 1.0 - F[0][self.masks_dn[0]]
        for j in range(1, T):
            m = self.masks_up[j - 1]
            P[m] = F[j - 1][m] - F[j][m]
        m = self.masks_up[T - 1]
        P[m] = F[T - 1][m]
        return P, F

    EPS = 1e-4   # smooth floor: below EPS the log-probability is extended linearly
                 # (log Pc = log EPS + (P - EPS)/EPS), which keeps the gradient bounded
                 # when a few records sit where the cumulative curves cross.

    def objective(self, th):
        a, b, bt = self.unpack(th)
        P, F = self.probs(th)
        self.last_neg = int((P <= 0).sum())
        low = P <= self.EPS
        logPc = np.where(low, np.log(self.EPS) + (P - self.EPS) / self.EPS, np.log(np.where(low, 1.0, P)))
        dlog = np.where(low, 1.0 / self.EPS, 1.0 / np.where(low, 1.0, P))   # d logPc / dP
        ll = float(logPc.sum())
        # dlogP_i / d eta_t  =  dlog_i * dP_i/d eta_t
        A = []
        for t in range(T):
            f = F[t] * (1 - F[t])
            at = np.zeros(self.n)
            mu, md = self.masks_up[t], self.masks_dn[t]
            at[mu] = f[mu] * dlog[mu]
            at[md] -= f[md] * dlog[md]
            A.append(at)
        g_a = np.array([at.sum() for at in A])
        g_b = self.X1.T @ np.sum(A, axis=0) if self.p1 else np.zeros(0)
        g_bt = np.concatenate([self.X2.T @ A[t] for t in range(T)]) if self.p2 else np.zeros(0)
        pen = 0.5 * self.lam * (b @ b + float(np.sum(bt * bt)))
        grad = -np.concatenate([g_a, g_b, g_bt]) + self.lam * np.concatenate([np.zeros(T), b, bt.ravel()])
        return -ll + pen, grad

    def fit(self, th0, maxiter, maxcor=30):
        t0 = time.time()
        res = minimize(self.objective, th0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, maxfun=maxiter * 3, ftol=1e-14, gtol=1e-4, maxcor=maxcor))
        P, _ = self.probs(res.x)
        ll = float(np.log(np.maximum(P, 1e-10)).sum())
        k = T + self.p1 + T * self.p2
        return dict(theta=res.x, ll=ll, k=k, aic=2 * k - 2 * ll, bic=k * np.log(self.n) - 2 * ll,
                    neg_share=float((P <= 0).mean()), n_neg=int((P <= 0).sum()),
                    n_below_eps=int((P <= self.EPS).sum()),
                    nit=int(res.nit), status=str(res.message), gradnorm=float(np.abs(res.jac).max()),
                    seconds=time.time() - t0)


def run_pathway(pathway, frame, lam, maxiter, outdir, tag=""):
    print(f"\n=== {pathway} ===", flush=True)
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    enc, names = TA.encode_features(frame, feats)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    y = pd.to_numeric(frame["kabco_ordinal"], errors="coerce").astype("Int64").dropna().astype(int).to_numpy()
    enc = enc.loc[enc.index.isin(keep)].reset_index(drop=True)
    X = TA.scale_no_mean(enc.values).astype(np.float64)
    n, p = X.shape
    cred = [f for f in SIM_CREDIBLE[pathway] if f in names]
    idx2 = [names.index(f) for f in cred]
    idx1 = [j for j in range(p) if j not in idx2]
    print(f"    n={n} p={p}; threshold-specific set: {len(cred)} features", flush=True)
    alpha0 = np.array([np.log(np.mean(y > t) / (1 - np.mean(y > t))) for t in range(T)])

    # PO baseline
    po = PartialPO(X, None, y, lam)
    r_po = po.fit(np.concatenate([alpha0, np.zeros(p)]), maxiter)
    a_po, b_po, _ = po.unpack(r_po["theta"])
    print(f"    PO   : LL={r_po['ll']:.2f} k={r_po['k']} neg={r_po['n_neg']} nit={r_po['nit']} "
          f"gmax={r_po['gradnorm']:.2e} ({r_po['seconds']:.0f}s) {r_po['status']}", flush=True)

    # PPO: warm start from the PO solution
    ppo = PartialPO(X[:, idx1], X[:, idx2], y, lam)
    th0 = np.concatenate([a_po, b_po[idx1], np.tile(b_po[idx2], T)])
    r_ppo = ppo.fit(th0, maxiter)
    a_ppo, b_ppo, bt_ppo = ppo.unpack(r_ppo["theta"])
    print(f"    PPO  : LL={r_ppo['ll']:.2f} k={r_ppo['k']} neg={r_ppo['n_neg']} nit={r_ppo['nit']} "
          f"gmax={r_ppo['gradnorm']:.2e} ({r_ppo['seconds']:.0f}s) {r_ppo['status']}", flush=True)

    # FULL generalized ordered logit: warm start from PO
    full = PartialPO(None, X, y, lam)
    r_full = full.fit(np.concatenate([a_po, np.tile(b_po, T)]), maxiter)
    a_full, _, bt_full = full.unpack(r_full["theta"])
    print(f"    FULL : LL={r_full['ll']:.2f} k={r_full['k']} neg={r_full['n_neg']} nit={r_full['nit']} "
          f"gmax={r_full['gradnorm']:.2e} ({r_full['seconds']:.0f}s) {r_full['status']}", flush=True)

    rows = []
    for jj, f in enumerate(cred):
        j = names.index(f)
        for t in range(T):
            rows.append(dict(pathway=pathway, feature=f, label=TA.LABELS.get(f, f), threshold=t,
                             coef_ppo=bt_ppo[t, jj], coef_full=bt_full[t, j], coef_po=b_po[j]))
    pd.DataFrame(rows).to_csv(outdir / f"partial_po_coefs_{pathway}{tag}.csv", index=False)
    # also the common PPO coefficients of the PO-preserving features (for completeness)
    pd.DataFrame(dict(feature=[names[j] for j in idx1], coef_ppo_common=b_ppo,
                      coef_po=b_po[idx1])).to_csv(outdir / f"partial_po_common_{pathway}{tag}.csv", index=False)

    def strip(r):
        return {k: v for k, v in r.items() if k != "theta"}
    summ = dict(pathway=pathway, n=int(n), p=int(p), lam=lam, eps=PartialPO.EPS, credible=cred,
                cuts_ppo=a_ppo.tolist(), PO=strip(r_po), PPO=strip(r_ppo), FULL=strip(r_full),
                dAIC_PPO_vs_PO=r_ppo["aic"] - r_po["aic"], dAIC_FULL_vs_PPO=r_full["aic"] - r_ppo["aic"],
                dBIC_PPO_vs_PO=r_ppo["bic"] - r_po["bic"], dBIC_FULL_vs_PPO=r_full["bic"] - r_ppo["bic"],
                max_abs_coef_ppo=float(np.abs(bt_ppo).max()), max_abs_coef_full=float(np.abs(bt_full).max()))
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="outputs/kabco_person_level_clean.csv")
    ap.add_argument("--outdir", default="outputs/hardening2")
    ap.add_argument("--pathways", default="infrastructure,pavement")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--nrows", type=int, default=0, help="smoke test: random subsample size")
    ap.add_argument("--sets", default="", help="simboot3_summary.json to take the threshold-specific sets from")
    ap.add_argument("--eps", type=float, default=1e-4, help="smooth probability floor of the likelihood")
    ap.add_argument("--tag", default="", help="suffix for the output file names, e.g. _eps1e-6")
    a = ap.parse_args()
    PartialPO.EPS = a.eps; print(f"probability floor EPS = {PartialPO.EPS:g}", flush=True)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if a.sets:
        SIM_CREDIBLE.update(load_sets(a.sets)); print("threshold-specific sets:", SIM_CREDIBLE, flush=True)
    frame = pd.read_csv(a.frame, low_memory=False)
    if a.nrows:
        frame = frame.sample(n=a.nrows, random_state=1).reset_index(drop=True)
    print(f"frame: {len(frame)} rows", flush=True)
    summ = [run_pathway(pw.strip(), frame, a.lam, a.maxiter, outdir, a.tag) for pw in a.pathways.split(",")]
    (outdir / f"partial_po_fit_summary{a.tag}.json").write_text(json.dumps(summ, indent=1, default=float))
    print("\nDONE run_partial_po_fit", flush=True)


if __name__ == "__main__":
    main()
