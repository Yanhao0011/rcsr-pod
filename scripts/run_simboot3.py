#!/usr/bin/env python3
"""Simultaneous max-t bootstrap for the DEPARTURE statistic (RCSR-POD v2), full frame.

Per pathway (design v2 from run_task_a: HARM_EV_12.0 removed; frame terms flagged):
  1. Firth threshold logits at the four cuts; model-based covariance (X'WX + ridge)^-1 and a
     crash-cluster-robust sandwich (cluster = source x year x ST_CASE, sparse aggregation).
  2. Point departure profiles for every feature: Dmax with the marginal delta-method bound
     (model SE and cluster SE), the non-fatal-ladder version (cuts 0-2), the previous PSC gap.
  3. For candidates (Dmax > delta/2 under either SE, plus forced sets), B coefficient draws
     per cut from the joint covariance (model, then cluster); the whole departure profile is
     recomputed per draw with the pooled coefficient re-pooled per draw (fixed precision
     weights); max-t over the cuts gives the simultaneous critical value c_alpha, the
     simultaneous lower bound, a max-t p-value (-> Holm / BH over the family), and the
     non-fatal version.
Outputs: simboot3_{pathway}.csv, firth_threshold_coefs_{pathway}.csv (with intercepts),
         simboot3_{pathway}_summary.json.

Usage: python run_simboot3.py --frame outputs/kabco_person_level_clean.csv --outdir outputs/v2
           --pathways infrastructure,pavement --B 1000 --block 50
"""
from __future__ import annotations
import argparse, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_a as TA
from run_simboot import fit_firth_cov, chol_jitter
from departure_core import departure_profile, summarize, lcb_max, logistic
warnings.filterwarnings("ignore")

DELTA, ALPHA = TA.DELTA, TA.ALPHA_SIG
Z = stats.norm.ppf(1 - ALPHA)
DELTA_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]


def multitest(p, alpha):
    p = np.asarray(p, dtype=float); m = len(p); order = np.argsort(p); ps = p[order]
    holm = np.zeros(m, dtype=bool)
    for k in range(m):
        if ps[k] <= alpha / (m - k): holm[order[k]] = True
        else: break
    bh = np.zeros(m, dtype=bool); below = np.where(ps <= alpha * np.arange(1, m + 1) / m)[0]
    if len(below): bh[order[:below.max() + 1]] = True
    return holm, bh


def cluster_cov(Xmat, y, beta, cov_model, codes, n_cl):
    p = logistic(Xmat @ beta); Sc = Xmat * (y - p)[:, None]
    M = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_cl, len(codes)))
    G = M @ Sc
    return cov_model @ (G.T @ G) @ cov_model


def maxt(Dhat, Db, cuts):
    """Dhat (T,), Db (T, B) over the selected cut indices -> dict(lcb, c_alpha, T, p, se)."""
    Dh = Dhat[cuts]; D = Db[cuts]
    se = D.std(axis=1, ddof=1); ok = se > 0
    if not ok.any():
        return dict(lcb=np.nan, c_alpha=np.nan, T=np.nan, p=1.0)
    tmax = np.max(np.abs(D[ok] - Dh[ok, None]) / se[ok, None], axis=0)
    c = float(np.quantile(tmax, 1 - ALPHA))
    lcb = float(np.max(np.abs(Dh[ok]) - c * se[ok]))
    Tobs = float(np.max((np.abs(Dh[ok]) - DELTA) / se[ok]))
    return dict(lcb=lcb, c_alpha=c, T=Tobs, p=float(np.mean(tmax >= Tobs)))


def bootstrap(Xmat, Xv, A, B, covs, thr, cand, prof_w, prof_w_nf, Bdraws, block, rng, tag):
    """Returns {j: (Db (T,B), Db_nf (Tnf,B))} for candidate column indices j."""
    T = len(thr); Ls = [chol_jitter(c) for c in covs]
    nf = [i for i in range(T) if thr[i] <= 2]
    out = {j: (np.empty((T, Bdraws)), np.empty((len(nf), Bdraws))) for j in cand}
    masks = {j: (Xv[:, j] != 0) for j in cand}
    done = 0; t0 = time.time()
    while done < Bdraws:
        bs = min(block, Bdraws - done)
        beta_b = []; eta_b = []; lam_b = []
        for i in range(T):
            full = np.concatenate([[A[i]], B[i]])
            bb = full[:, None] + Ls[i] @ rng.standard_normal((Xmat.shape[1], bs))
            e = Xmat @ bb; beta_b.append(bb); eta_b.append(e); lam_b.append(logistic(e))
        for j in cand:
            m = masks[j]; xj = Xv[m, j][:, None]; nfrac = m.mean()
            bj = np.stack([beta_b[i][j + 1] for i in range(T)])                 # (T, bs)
            bbar = (prof_w[j][:, None] * bj).sum(0)                             # (bs,)
            bbar_nf = (prof_w_nf[j][:, None] * bj[nf]).sum(0)
            for i in range(T):
                e_mk = eta_b[i][m] - xj * bj[i][None, :]
                out[j][0][i, done:done + bs] = (lam_b[i][m] - logistic(e_mk + xj * bbar[None, :])).sum(0) / len(m)
                if i in nf:
                    out[j][1][nf.index(i), done:done + bs] = (lam_b[i][m] - logistic(e_mk + xj * bbar_nf[None, :])).sum(0) / len(m)
        done += bs
        print(f"    [{tag}] bootstrap {done}/{Bdraws} ({time.time() - t0:.0f}s)", flush=True)
    return out


def run_pathway(pathway, frame, cluster_all, Bdraws, block, seed, outdir):
    print(f"\n=== {pathway} ===", flush=True)
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    enc, names = TA.encode_features(frame, feats)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    y = pd.to_numeric(frame["kabco_ordinal"], errors="coerce").astype("Int64").dropna().astype(int).to_numpy()
    enc = enc.loc[enc.index.isin(keep)].reset_index(drop=True)
    codes, uniq = pd.factorize(cluster_all[frame.index.isin(keep)]); n_cl = len(uniq)
    Xv = TA.scale_no_mean(enc.values).astype(np.float64); n, p = Xv.shape
    Xmat = np.column_stack([np.ones(n), Xv])
    print(f"    n={n} p={p} clusters={n_cl}", flush=True)
    thr = [t for t in sorted(np.unique(y))[:-1] if 0 < (y > t).sum() < n]; T = len(thr)
    A = np.zeros(T); B = np.zeros((T, p)); S = np.zeros((T, p)); Scl = np.zeros((T, p)); covs = []; covs_cl = []; eta = []
    coef_rows = []
    for i, t in enumerate(thr):
        by = (y > t).astype(int); b, se, cov, ok = fit_firth_cov(Xmat, by)
        vcl = cluster_cov(Xmat, by, b, cov, codes, n_cl); secl = np.sqrt(np.clip(np.diag(vcl), 0, None))
        A[i], B[i], S[i], Scl[i] = b[0], b[1:], se[1:], secl[1:]; covs.append(cov); covs_cl.append(vcl); eta.append(Xmat @ b)
        print(f"    firth t>{t}: n1={int(by.sum())} converged={ok} median SE ratio cluster/model {np.median(secl[1:] / np.maximum(se[1:], 1e-12)):.3f}", flush=True)
        coef_rows.append(dict(pathway=pathway, feature="(intercept)", label="(intercept)", threshold=t, beta=b[0], se=se[0], se_cluster=secl[0]))
        for jc, f in enumerate(names):
            coef_rows.append(dict(pathway=pathway, feature=f, label=TA.LABELS.get(f, f), threshold=t, beta=b[jc + 1], se=se[jc + 1], se_cluster=secl[jc + 1]))
    pd.DataFrame(coef_rows).to_csv(outdir / f"firth_threshold_coefs_{pathway}.csv", index=False)

    nf = [i for i in range(T) if thr[i] <= 2]
    rows = []; prof_w = {}; prof_w_nf = {}; prof_w_cl = {}; prof_w_nf_cl = {}; Dhat = {}; Dhat_nf = {}; Dhat_cl = {}; Dhat_nf_cl = {}
    for j, f in enumerate(names):
        xj = Xv[:, j]; prev = float(np.mean(xj != 0))
        pm = departure_profile(eta, B[:, j], S[:, j], xj); sm = summarize(pm, ALPHA)
        pc = departure_profile(eta, B[:, j], Scl[:, j], xj); sc = summarize(pc, ALPHA)
        pmn = departure_profile(eta, B[:, j], S[:, j], xj, cuts=nf); pcn = departure_profile(eta, B[:, j], Scl[:, j], xj, cuts=nf)
        Dnf, _, Dnf_lcb = lcb_max(pmn["D"], pmn["se_D"], ALPHA); Dnfc, _, Dnfc_lcb = lcb_max(pcn["D"], pcn["se_D"], ALPHA)
        prof_w[j], prof_w_nf[j], prof_w_cl[j], prof_w_nf_cl[j] = pm["w"], pmn["w"], pc["w"], pcn["w"]
        Dhat[j], Dhat_nf[j], Dhat_cl[j], Dhat_nf_cl[j] = pm["D"], pmn["D"], pc["D"], pcn["D"]
        rows.append(dict(pathway=pathway, feature=f, label=TA.LABELS.get(f, f), frame_term=TA.is_frame_term(f, prev),
                         prevalence=prev, bbar=sm["bbar"], coef_gap=float(B[:, j].max() - B[:, j].min()),
                         psc_gap=sm["psc_gap"], psc_lcb=sm["psc_lcb"],
                         Dmax=sm["Dmax"], Dmax_cut=thr[sm["Dmax_cut"]], Dmax_lcb=sm["Dmax_lcb"], Dmax_lcb_cl=sc["Dmax_lcb"],
                         Dmax_nf=Dnf, Dmax_nf_lcb=Dnf_lcb, Dmax_nf_lcb_cl=Dnfc_lcb,
                         **{f"D{thr[i]}": pm["D"][i] for i in range(T)}))
    df = pd.DataFrame(rows)
    forced = set(TA.CREDIBLE.get(pathway, [])) | set(TA.FRAME_TERMS.get(pathway, []))
    cand = [j for j, f in enumerate(names) if max(df.Dmax_lcb[j] + Z * 0, df.Dmax[j]) > 0.5 * DELTA or f in forced]
    print(f"    candidates: {len(cand)} of {p}", flush=True)
    rng = np.random.default_rng(seed)
    boot_m = bootstrap(Xmat, Xv, A, B, covs, thr, cand, prof_w, prof_w_nf, Bdraws, block, rng, f"{pathway}/model")
    boot_c = bootstrap(Xmat, Xv, A, B, covs_cl, thr, cand, prof_w_cl, prof_w_nf_cl, Bdraws, block, rng, f"{pathway}/cluster")

    for col in ["sim_lcb", "sim_c_alpha", "sim_T", "sim_p", "sim_nf_lcb", "sim_nf_p",
                "simcl_lcb", "simcl_c_alpha", "simcl_p", "simcl_nf_lcb", "simcl_nf_p"]:
        df[col] = np.nan
    df["sim_p"] = 1.0; df["sim_nf_p"] = 1.0; df["simcl_p"] = 1.0; df["simcl_nf_p"] = 1.0
    df["candidate"] = False
    allc = list(range(T))
    for j in cand:
        m_full = maxt(Dhat[j], boot_m[j][0], allc); m_nf = maxt(Dhat_nf[j], boot_m[j][1], list(range(len(nf))))
        c_full = maxt(Dhat_cl[j], boot_c[j][0], allc); c_nf = maxt(Dhat_nf_cl[j], boot_c[j][1], list(range(len(nf))))
        df.loc[j, ["candidate", "sim_lcb", "sim_c_alpha", "sim_T", "sim_p", "sim_nf_lcb", "sim_nf_p",
                   "simcl_lcb", "simcl_c_alpha", "simcl_p", "simcl_nf_lcb", "simcl_nf_p"]] = [
            True, m_full["lcb"], m_full["c_alpha"], m_full["T"], m_full["p"], m_nf["lcb"], m_nf["p"],
            c_full["lcb"], c_full["c_alpha"], c_full["p"], c_nf["lcb"], c_nf["p"]]
    for col in ["psc_lcb", "Dmax_lcb", "Dmax_lcb_cl", "Dmax_nf_lcb", "sim_lcb", "sim_nf_lcb", "simcl_lcb", "simcl_nf_lcb"]:
        df["credible_" + col] = df[col] > DELTA
    for pcol, tag in [("sim_p", "sim"), ("simcl_p", "simcl"), ("sim_nf_p", "sim_nf")]:
        h, b = multitest(df[pcol].fillna(1.0).values, ALPHA); df["holm_" + tag] = h; df["bh_" + tag] = b
    df = df.sort_values("Dmax", ascending=False)
    df.to_csv(outdir / f"simboot3_{pathway}.csv", index=False)

    sub = df[~df.frame_term]
    def cnt(col, d=None): return int((d if d is not None else sub)[col].sum())
    summ = dict(pathway=pathway, n=int(n), p=int(p), clusters=int(n_cl), B=int(Bdraws), candidates=len(cand),
                frame_terms=df.loc[df.frame_term, "feature"].tolist(),
                marginal_Dmax=cnt("credible_Dmax_lcb"), marginal_Dmax_cluster=cnt("credible_Dmax_lcb_cl"),
                simultaneous_Dmax=cnt("credible_sim_lcb"), simultaneous_Dmax_cluster=cnt("credible_simcl_lcb"),
                simultaneous_nonfatal=cnt("credible_sim_nf_lcb"), holm_sim=cnt("holm_sim"), bh_sim=cnt("bh_sim"),
                holm_sim_cluster=cnt("holm_simcl"), bh_sim_cluster=cnt("bh_simcl"),
                frame_credible_sim=cnt("credible_sim_lcb", df[df.frame_term]),
                previous_psc_marginal=cnt("credible_psc_lcb", df),
                sweep_sim_substantive={str(d): int((sub.sim_lcb > d).sum()) for d in DELTA_GRID},
                sweep_marginal_substantive={str(d): int((sub.Dmax_lcb > d).sum()) for d in DELTA_GRID},
                set_sim=sub.loc[sub.credible_sim_lcb, "feature"].tolist(), set_marginal=sub.loc[sub.credible_Dmax_lcb, "feature"].tolist(),
                set_sim_cluster=sub.loc[sub.credible_simcl_lcb, "feature"].tolist(), set_holm=sub.loc[sub.holm_sim, "feature"].tolist(),
                set_bh=sub.loc[sub.bh_sim, "feature"].tolist(),
                nonfatal_survivors=sub.loc[sub.credible_sim_lcb & sub.credible_sim_nf_lcb, "feature"].tolist(),
                c_alpha_range=[float(np.nanmin(df.sim_c_alpha)), float(np.nanmax(df.sim_c_alpha))],
                median_se_ratio_cluster_over_model=float(np.median(Scl / np.maximum(S, 1e-12))))
    print(json.dumps(summ, indent=1), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="outputs/kabco_person_level_clean.csv")
    ap.add_argument("--outdir", default="outputs/v2")
    ap.add_argument("--pathways", default="infrastructure,pavement")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20251231)
    ap.add_argument("--nrows", type=int, default=0)
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(a.frame, low_memory=False)
    if a.nrows: frame = frame.sample(n=a.nrows, random_state=1).reset_index(drop=True)
    cluster_all = (frame["source"].astype(str) + "_" + frame["YEAR"].astype(str) + "_" + frame["ST_CASE"].astype(str)).to_numpy()
    summ = [run_pathway(pw.strip(), frame, cluster_all, a.B, a.block, a.seed + i, outdir) for i, pw in enumerate(a.pathways.split(","))]
    (outdir / "simboot3_summary.json").write_text(json.dumps(summ, indent=1))
    print("\nDONE run_simboot3", flush=True)


if __name__ == "__main__":
    main()
