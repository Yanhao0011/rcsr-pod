# -*- coding: utf-8 -*-
"""RCSR-POD v3 application: coding-invariant departure on the full FARS+CRSS frame.

Uses the saved full-frame Firth fits (outputs/v3/firth_fit_{pathway}.npz, from run_v3_frame.py); the coefficient covariance is
rebuilt at the saved estimates as inv(X'WX + ridge I) (checked against the saved SEs) and a cluster sandwich
(cluster = source x YEAR x ST_CASE). For every encoded feature and every parent variable (all dummies of one
categorical jointly) computes the invariant departure profile over cuts, and for candidates a parametric
bootstrap (model and cluster covariance) giving one-sided max-t simultaneous bounds over all cuts and over the
non-fatal ladder (cuts 0-2), marginal bounds, p-values over a delta grid and Holm/BH.

Usage: python run_v3_application.py --pathways pavement,infrastructure --B 1000 --block 32
"""
from __future__ import annotations
import argparse, sys, json, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task_a as TA
import run_departure_diag as RD
from run_v3_frame import V3_DROP, SOURCE_TERM, load_frame, frame_units, is_frame
from run_simboot import chol_jitter
import rcsr_v3 as V_

ROOT = Path(os.environ.get("RCPO_ROOT", Path(__file__).resolve().parents[1]))
FIT = ROOT / "outputs" / "v3"
DELTA = TA.DELTA
GRID = [0.005, 0.01, 0.02, 0.03, 0.05]
RGRID = [0.02, 0.05, 0.10]           # cut-relative thresholds (share of the cut's mean fitted rate)
RIDGE = 1e-6


def lg(v):
    return 0.5 * (1.0 + np.tanh(0.5 * v))


def parent_of(name, feats):
    if name.endswith("_UNK") and name[:-4] in feats:          # unknown indicator of a de-sentineled continuous field
        return name[:-4]
    for c in sorted(feats, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def build(frame, pathway, cluster_all):
    drop = V3_DROP[pathway]
    Xv, names, y = RD.build_design(frame, pathway, drop=drop)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    codes, uniq = pd.factorize(cluster_all[frame.index.isin(keep)])
    return Xv, list(names), y, codes, len(uniq)


def covariances(Xmat, y, A, B, thr, codes, n_cl, chunk=60000):
    """Per-cut model covariances and the stacked four-cut sandwiches (record-level HC0 and crash cluster).
    Memory-light: residuals are kept as vectors, the meats are accumulated over row chunks aligned to cluster
    boundaries (records sorted by cluster), so no n x q score matrix is ever stored."""
    from po_diagnostic_core import _xtwx
    q = Xmat.shape[1]; T = len(thr); covs = []; R = np.empty((T, len(y)))
    for i, t in enumerate(thr):
        b = np.concatenate([[A[i]], B[i]]); p = lg(Xmat @ b)
        covs.append(np.linalg.inv(_xtwx(Xmat, p * (1 - p)) + RIDGE * np.eye(q)))
        R[i] = (y > t).astype(float) - p
    order = np.argsort(codes, kind="stable"); cs = codes[order]
    Mh = np.zeros((T * q, T * q)); Mc = np.zeros((T * q, T * q))
    start = 0; n = len(y)
    while start < n:
        stop = min(start + chunk, n)
        while stop < n and cs[stop] == cs[stop - 1]:
            stop += 1                                            # never split a cluster across chunks
        idx = order[start:stop]; Xc = Xmat[idx]
        Sc = np.hstack([Xc * R[i, idx][:, None] for i in range(T)])        # (m, T q)
        Mh += Sc.T @ Sc
        u, inv = np.unique(cs[start:stop], return_inverse=True)
        G = np.zeros((len(u), T * q)); np.add.at(G, inv, Sc)
        Mc += G.T @ G
        start = stop
    out = {"model": np.zeros((T * q, T * q))}
    for i in range(T):
        out["model"][i * q:(i + 1) * q, i * q:(i + 1) * q] = covs[i]
    for tag, Mt in [("joint", Mh), ("cluster", Mc)]:
        V = np.zeros((T * q, T * q))
        for i in range(T):
            for j in range(T):
                V[i * q:(i + 1) * q, j * q:(j + 1) * q] = covs[i] @ Mt[i * q:(i + 1) * q, j * q:(j + 1) * q] @ covs[j]
        out[tag] = 0.5 * (V + V.T)
    return covs, out


def units(names, feats, prev):
    """Analysis units: each encoded feature, plus each multi-dummy parent variable (all its dummies, including a
    MISSING level; the source indicator is kept alone)."""
    out = [dict(unit=f, kind="feature", cols=[j]) for j, f in enumerate(names)]
    par = {}
    for j, f in enumerate(names):
        if f in SOURCE_TERM.values() or f == "persons_in_crash":
            continue
        par.setdefault(parent_of(f, feats), []).append(j)
    for p_, cols in par.items():
        if len(cols) > 1:
            out.append(dict(unit=p_ + " [group]", kind="group", cols=cols))
    return out


def unit_weights(covs, cols, cuts):
    """GLS pooling weights (T', g, g) of a unit from the model-based per-cut covariance blocks."""
    c1 = np.array(cols) + 1
    return V_.gls_weights(np.array([covs[i][np.ix_(c1, c1)] for i in cuts]))


def profile(eta, B, covs, X, cols, cuts):
    w = unit_weights(covs, cols, cuts)
    D, bbar = V_.dinv_point([eta[i] for i in cuts], B[np.ix_(cuts, cols)], w, X[:, cols])
    return D, w


def bootstrap(Xmat, X, A, B, V, cand, wts, cutsets, Bd, block, rng, tag):
    """Joint draws of the stacked cut coefficients from V (T*q, T*q). wts[(u, cs)] = GLS weights (len(cuts), g, g).
    Returns {(u, cs): (len(cuts), Bd)}."""
    T = len(A); q = Xmat.shape[1]; L = chol_jitter(V)
    theta = np.concatenate([np.concatenate([[A[i]], B[i]]) for i in range(T)])
    out = {(u, cs): np.empty((len(cutsets[cs]), Bd)) for u in cand for cs in cutsets}
    XG = {u: V_.as_group(X[:, cand[u]]) for u in cand}
    done = 0; t0 = time.time()
    while done < Bd:
        bs = min(block, Bd - done)
        dr = theta[:, None] + L @ rng.standard_normal((T * q, bs))
        draws = [dr[i * q:(i + 1) * q] for i in range(T)]
        etaT = [np.ascontiguousarray((Xmat @ d).T) for d in draws]
        for u, cols in cand.items():
            bj = np.stack([draws[i][np.array(cols) + 1] for i in range(T)])        # (T, g, bs)
            for cs, cuts in cutsets.items():
                bbar = V_.pooled(wts[(u, cs)], bj[cuts])                            # (g, bs)
                for k, i in enumerate(cuts):
                    out[(u, cs)][k, done:done + bs] = V_.dinv_draws(etaT[i], XG[u], np.ascontiguousarray((bj[i] - bbar).T))
        done += bs
        if done % (4 * block) == 0 or done == Bd:
            print(f"    [{tag}] {done}/{Bd} ({time.time() - t0:.0f}s)", flush=True)
    return out


def run_pathway(pathway, frame, cluster_all, Bd, block, seed, outdir):
    print(f"\n=== {pathway} ===", flush=True)
    z = np.load(FIT / f"firth_fit_{pathway}.npz", allow_pickle=True)
    A, B, S, thr = z["alpha"], z["beta"], z["se"], [int(t) for t in z["thr"]]
    Xv, names, y, codes, n_cl = build(frame, pathway, cluster_all)
    assert names == list(z["names"]), "design/feature order mismatch"
    n, p = Xv.shape; T = len(thr); Xmat = np.column_stack([np.ones(n), Xv])
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    covs, Vs = covariances(Xmat, y, A, B, thr, codes, n_cl)
    q = p + 1
    se_re = np.array([np.sqrt(np.diag(c))[1:] for c in covs]); rel = np.abs(se_re / S - 1)
    se_of = {k: np.array([np.sqrt(np.clip(np.diag(V)[i * q:(i + 1) * q], 0, None))[1:] for i in range(T)]) for k, V in Vs.items()}
    print(f"    n={n} p={p} clusters={n_cl}; rebuilt-vs-saved SE max rel diff {rel.max():.2e} (median {np.median(rel):.2e}); "
          f"median SE ratio joint/model {np.median(se_of['joint'] / S):.3f}, cluster/model {np.median(se_of['cluster'] / S):.3f}", flush=True)
    eta = [Xmat @ np.concatenate([[A[i]], B[i]]) for i in range(T)]
    PBAR = np.array([float(np.mean(lg(e))) for e in eta]); KAP = V_.kappa_n(n)
    prev = [float(np.mean(Xv[:, j] != 0)) for j in range(p)]
    U = units(names, feats, prev)
    cutsets = {"all": list(range(T)), "nf": [i for i in range(T) if thr[i] <= 2]}

    # estimand: precision weights from the model-based Firth SEs (fixed across variance estimators)
    rows = []; Dh = {}; wts = {}
    t0 = time.time()
    for u in U:
        cols = u["cols"]; key = u["unit"]
        for cs, cuts in cutsets.items():
            Dh[(key, cs)], wts[(key, cs)] = profile(eta, B, covs, Xv, cols, cuts)
        D = Dh[(key, "all")]
        f0 = names[cols[0]]
        r = dict(pathway=pathway, unit=key, kind=u["kind"], n_cols=len(cols),
                 label=TA.LABELS.get(f0, f0) if u["kind"] == "feature" else key,
                 frame_term=bool(u["kind"] == "feature" and is_frame(f0, pathway)),
                 prevalence=prev[cols[0]] if u["kind"] == "feature" else float(np.mean((Xv[:, cols] != 0).any(1))),
                 Dmax=float(D.max()), Dmax_cut=thr[int(np.argmax(D))], Dmax_nf=float(Dh[(key, "nf")].max()),
                 coef_gap=float(np.ptp(B[:, cols], axis=0).max()))
        r.update({f"D{thr[i]}": float(D[i]) for i in range(T)})
        rows.append(r)
    print(f"    point profiles for {len(U)} units in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)

    forced = set(frame_units(pathway))
    cand = {u["unit"]: u["cols"] for u, r in zip(U, rows) if r["Dmax"] > 0.5 * GRID[0] or u["unit"] in forced}
    print(f"    bootstrap candidates: {len(cand)} of {len(U)} units", flush=True)
    rng = np.random.default_rng(seed)
    VARIANTS = [("cl", "cluster"), ("m", "model"), ("j", "joint")]       # cl = primary (joint crash-cluster sandwich)
    boots = {tag: bootstrap(Xmat, Xv, A, B, Vs[k], cand, wts, cutsets, Bd, block, rng, f"{pathway}/{k}") for tag, k in VARIANTS}

    new = {}
    for key in df.unit:
        rec = dict(candidate=key in cand)
        for tag, _ in VARIANTS:
            for cs in cutsets:
                pre = f"{tag}_{cs}"
                if key in cand:
                    res = {d: V_.maxt_onesided(Dh[(key, cs)], boots[tag][(key, cs)], d, kappa=KAP) for d in GRID}
                    r0 = res[DELTA]
                    rec.update({f"{pre}_lcb": r0["lcb"], f"{pre}_lcb_marg": r0["lcb_marg"], f"{pre}_c": r0["c"],
                                f"{pre}_seD": float(r0["se"][int(np.argmax(Dh[(key, cs)]))])})
                    rec.update({f"{pre}_p{d}": res[d]["p"] for d in GRID})
                    rec[f"{pre}_ncut"] = res[DELTA]["ncut"]
                    if tag == "cl" and cs == "all":
                        # cut-relative departure D_t / Pbar_t (Pbar_t: mean fitted P(y > t), held fixed)
                        rr = {d: V_.maxt_onesided(Dh[(key, cs)] / PBAR, boots[tag][(key, cs)] / PBAR[:, None], d, kappa=KAP) for d in RGRID}
                        rec.update({"rel_Dmax": float(np.max(Dh[(key, cs)] / PBAR)), "rel_lcb": rr[RGRID[1]]["lcb"],
                                    **{f"rel_p{d}": rr[d]["p"] for d in RGRID}, **{f"rel_lcb{d}": rr[d]["lcb"] for d in RGRID}})
                else:
                    rec.update({f"{pre}_lcb": np.nan, f"{pre}_lcb_marg": np.nan, f"{pre}_c": np.nan, f"{pre}_seD": np.nan})
                    rec.update({f"{pre}_p{d}": 1.0 for d in GRID})
        new[key] = rec
    df = pd.concat([df, pd.DataFrame([new[k] for k in df.unit])], axis=1)
    for d in GRID:
        for pre in ["cl_all", "m_all", "j_all", "cl_nf"]:
            for kind in ["feature", "group"]:
                msk = (df.kind == kind).values
                h, b = V_.multitest(df.loc[msk, f"{pre}_p{d}"].values)
                df.loc[msk, f"holm_{pre}_{d}"] = h; df.loc[msk, f"bh_{pre}_{d}"] = b
    df = df.sort_values(["kind", "Dmax"], ascending=[True, False])
    df.to_csv(outdir / f"v3_{pathway}.csv", index=False)

    sub = df[(df.kind == "feature") & ~df.frame_term]; grp = df[df.kind == "group"]
    cr = lambda d_, col, dl=DELTA: d_.loc[d_[col] > dl, "unit"].tolist()
    hs = lambda d_, col: d_.loc[d_[col].astype(bool), "unit"].tolist()
    summ = dict(pathway=pathway, n=int(n), p=int(p), clusters=int(n_cl), B=int(Bd), cuts=thr,
                se_rebuild_max_rel=float(rel.max()), candidates=len(cand),
                frame_terms=df.loc[df.frame_term, ["unit", "Dmax", "cl_all_lcb"]].to_dict("records"),
                point_above_delta=cr(sub, "Dmax"),
                simultaneous=cr(sub, "cl_all_lcb"), simultaneous_model=cr(sub, "m_all_lcb"), simultaneous_joint=cr(sub, "j_all_lcb"),
                marginal=cr(sub, "cl_all_lcb_marg"), nonfatal_simultaneous=cr(sub, "cl_nf_lcb"),
                holm=hs(sub, f"holm_cl_all_{DELTA}"), bh=hs(sub, f"bh_cl_all_{DELTA}"),
                groups_simultaneous=cr(grp, "cl_all_lcb"), groups_point=cr(grp, "Dmax"), groups_holm=hs(grp, f"holm_cl_all_{DELTA}"),
                sweep_simultaneous={str(d): int((sub.cl_all_lcb > d).sum()) for d in GRID},
                sweep_model={str(d): int((sub.m_all_lcb > d).sum()) for d in GRID},
                sweep_marginal={str(d): int((sub.cl_all_lcb_marg > d).sum()) for d in GRID},
                sweep_point={str(d): int((sub.Dmax > d).sum()) for d in GRID},
                sweep_holm={str(d): int(sub[f"holm_cl_all_{d}"].astype(bool).sum()) for d in GRID},
                sweep_groups={str(d): int((grp.cl_all_lcb > d).sum()) for d in GRID},
                c_range=[float(np.nanmin(df.cl_all_c)), float(np.nanmax(df.cl_all_c))], kappa=KAP, pbar=PBAR.tolist(),
                sweep_relative={str(d): int((sub[f"rel_lcb{d}"] > d).sum()) for d in RGRID},
                relative_set={str(d): sub.loc[sub[f"rel_lcb{d}"] > d, "unit"].tolist() for d in RGRID},
                median_se_ratio_cluster_over_model=float(np.median(se_of["cluster"] / np.maximum(S, 1e-12))),
                median_se_ratio_joint_over_model=float(np.median(se_of["joint"] / np.maximum(S, 1e-12))))
    print(json.dumps(summ, indent=1), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathways", default="pavement,infrastructure")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "v3"))
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    cluster_all = (frame["source"].astype(str) + "_" + frame["YEAR"].astype(str) + "_" + frame["ST_CASE"].astype(str)).to_numpy()
    summ = [run_pathway(pw.strip(), frame, cluster_all, a.B, a.block, a.seed + i, outdir) for i, pw in enumerate(a.pathways.split(","))]
    tag = "_".join(pw.strip() for pw in a.pathways.split(","))
    (outdir / f"v3_summary_{tag}_B{a.B}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
