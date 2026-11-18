#!/usr/bin/env python3
"""Sharp-null comparators on the v3 frame: Brant/Wald PO tests per feature from the full-frame Firth threshold fits,
(i) marginal-variance form (Eq. 3 of the manuscript; contrasts with cut 0, marginal variances) and (ii) the full
joint form with the stacked four-cut record-level sandwich covariance (all cross-cut covariances); plus the
unpenalized ML fatal-cut fit of the source indicator (separation check).
Outputs: outputs/v3/brant_v3_{pathway}.csv, outputs/v3/brant_v3_summary.json"""
from __future__ import annotations
import sys, json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v3_application as APP
from po_diagnostic_core import _xtwx
from run_v3_frame import SOURCE_TERM, load_frame, is_frame

OUT = Path(os.environ.get("RCPO_ROOT", HERE.parent)) / "outputs" / "v3"


def lg(v): return 0.5 * (1.0 + np.tanh(0.5 * v))


def ml_cut(Xm, by, iters=60, ridge=1e-8):
    b = np.zeros(Xm.shape[1]); hist = []
    for _ in range(iters):
        p = lg(Xm @ b); H = _xtwx(Xm, p * (1 - p)) + ridge * np.eye(Xm.shape[1])
        step = np.linalg.solve(H, Xm.T @ (by - p)); b = b + step; hist.append(float(np.abs(step).max()))
        if hist[-1] < 1e-8: break
    p = lg(Xm @ b); cov = np.linalg.pinv(_xtwx(Xm, p * (1 - p)))
    return b, np.sqrt(np.clip(np.diag(cov), 0, None)), hist


def main():
    frame = load_frame()
    crash = (frame["source"].astype(str) + "_" + frame["YEAR"].astype(str) + "_" + frame["ST_CASE"].astype(str)).to_numpy()
    summ = {}
    for pw in ["pavement", "infrastructure"]:
        z = np.load(OUT / f"firth_fit_{pw}.npz", allow_pickle=True)
        A, B, S, thr = z["alpha"], z["beta"], z["se"], list(z["thr"])
        Xv, names, y, codes, n_cl = APP.build(frame, pw, crash)
        n, p = Xv.shape; T = len(thr); q = p + 1; Xm = np.column_stack([np.ones(n), Xv])
        _, Vs = APP.covariances(Xm, y, A, B, thr, codes, n_cl); Vj = Vs["joint"]
        rows = []
        for k in range(p):
            chi_m = float(np.sum((B[1:, k] - B[0, k]) ** 2 / (S[1:, k] ** 2 + S[0, k] ** 2)))
            idx = [t * q + k + 1 for t in range(T)]; Vk = Vj[np.ix_(idx, idx)]
            C = np.zeros((T - 1, T)); C[:, 0] = -1; C[np.arange(T - 1), np.arange(1, T)] = 1
            d = C @ B[:, k]; chi_j = float(d @ np.linalg.pinv(C @ Vk @ C.T) @ d)
            rows.append(dict(feature=names[k], chi2_marginal=chi_m, p_marginal=float(stats.chi2.sf(chi_m, T - 1)),
                             chi2_joint=chi_j, p_joint=float(stats.chi2.sf(chi_j, T - 1))))
        df = pd.DataFrame(rows); df.to_csv(OUT / f"brant_v3_{pw}.csv", index=False)
        # covariate-level (grouped) joint Wald test of PO, the like-for-like comparator of the RCSR-POD units,
        # with three decision rules: p < 0.05, Holm over covariates, and a BIC-type rule chi2 > df log N
        import run_task_a as TA
        feats = TA.PATHWAY_FEATURES[pw] + TA.CONTROL_FEATURES
        U = [u for u in APP.units(names, feats, None) if u["kind"] == "group"]
        grouped = {u["unit"].replace(" [group]", "") for u in U}
        units = U + [dict(unit=nm, kind="feature", cols=[k]) for k, nm in enumerate(names)
                     if APP.parent_of(nm, feats) not in grouped and not is_frame(nm, pw)]
        C = np.zeros((T - 1, T)); C[:, 0] = -1; C[np.arange(T - 1), np.arange(1, T)] = 1
        crow = []
        for u in units:
            cols = u["cols"]; g = len(cols)
            idx = [t * q + k + 1 for t in range(T) for k in cols]                       # stacked (t, k) order
            Vk = Vj[np.ix_(idx, idx)]; K = np.kron(C, np.eye(g))
            d = K @ np.concatenate([B[t, cols] for t in range(T)])
            chi = float(d @ np.linalg.pinv(K @ Vk @ K.T) @ d); dfree = (T - 1) * g
            crow.append(dict(unit=u["unit"], n_cols=g, chi2=chi, df=dfree, p=float(stats.chi2.sf(chi, dfree)),
                             bic=bool(chi > dfree * np.log(n))))
        cv = pd.DataFrame(crow)
        from rcsr_v3 import multitest
        cv["holm"] = multitest(cv.p.values)[0]
        cv.to_csv(OUT / f"brant_v3_covariates_{pw}.csv", index=False)
        hf = multitest(df.p_joint.values)[0]
        bic_feat = df.chi2_joint > (T - 1) * np.log(n)
        js = SOURCE_TERM[pw]; j = names.index(js)
        b_ml, se_ml, hist = ml_cut(Xm, (y > 3).astype(float))
        summ[pw] = dict(p=int(p), brant_marginal=int((df.p_marginal < 0.05).sum()), brant_joint=int((df.p_joint < 0.05).sum()),
                        brant_joint_holm=int(hf.sum()), bic_features=int(bic_feat.sum()), bic_feature_set=df.loc[bic_feat, "feature"].tolist(),
                        covariates=int(len(cv)), cov_wald=int((cv.p < 0.05).sum()), cov_holm=int(cv.holm.sum()), cov_bic=int(cv.bic.sum()),
                        cov_bic_set=cv.loc[cv.bic, "unit"].tolist(),
                        source_term=js, source_fatal_ml_coef=float(b_ml[j + 1]), source_fatal_ml_se=float(se_ml[j + 1]),
                        source_fatal_ml_last_step=hist[-1], ml_iterations=len(hist),
                        source_fatal_firth_coef=float(B[3, j]), source_fatal_firth_se=float(S[3, j]),
                        ml_vs_firth_max_abs_coef_other=float(np.max(np.abs(np.delete(b_ml[1:], j) - np.delete(B[3], j)))))
        print(pw, json.dumps(summ[pw], indent=1), flush=True)
    (OUT / "brant_v3_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
