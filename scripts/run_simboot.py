#!/usr/bin/env python3
"""Hardening analysis #2 - simultaneous-inference bootstrap for the max PSC gap.

The manuscript's LCB is a max over the six threshold-pairs of the marginal
delta-method bound with an UNADJUSTED z. A referee (GPT #1/#2) objects that a max
selected over six correlated, nonlinear contrasts is not automatically a 1-alpha
bound and that the marginal delta-method variance omits the joint coefficient
covariance. This computes a PROPER simultaneous (max-t) lower bound:

  * draw the full per-threshold coefficient vector from its joint Firth covariance
    Sigma^(t) = (X' W_t X + ridge)^-1  (not just the marginal se of beta_j) and
    recompute the WHOLE PSC profile per draw (full predictor eta);
  * for each threshold pair (s,t) form the bootstrap contrast D_st = PSC^s - PSC^t,
    its bootstrap SE, and the SIMULTANEOUS critical value
        c_alpha = quantile_{1-alpha} ( max_{s<t} |D_st^b - D_hat_st| / SE_st )
    which absorbs the six-fold selection AND the joint covariance; then
        LCB_sim = max_{s<t} ( |D_hat_st| - c_alpha * SE_st ).

Because c_alpha >= z_{1-alpha}, LCB_sim <= the marginal delta-method LCB, so the
credible set can only shrink: if it survives, the referee's selection/covariance
concern does not overturn the result. Credible iff LCB_sim > delta. ADDITIVE.

Optional part 2 (--with-size 1): boundary-size Monte Carlo (true max gap = delta)
reporting the empirical rejection rate of BOTH the delta-method and the max-t
simultaneous bound, confirming the simultaneous bound controls size at the
least-favorable boundary (companion to H4).

Usage:
    python run_simboot.py --frame outputs/kabco_person_level_clean.csv \
        --outdir outputs --B 1000 --jobs 32
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_a as TA
from po_diagnostic_core import firth_logit
warnings.filterwarnings("ignore")

DELTA = TA.DELTA
ALPHA = TA.ALPHA_SIG
Z = stats.norm.ppf(1 - ALPHA)


def _lg(x):
    return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-12, 1 - 1e-12)


def fit_firth_cov(Xmat, by, ridge=1e-6):
    """Firth beta (via po_diagnostic_core) + full covariance inv(X'WX+ridge) at beta."""
    beta, se, ok = firth_logit(Xmat, by, ridge=ridge)
    eta = Xmat @ beta
    p = _lg(eta)
    w = p * (1 - p)
    info = (Xmat * w[:, None]).T @ Xmat + ridge * np.eye(Xmat.shape[1])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    return beta, se, cov, ok


def chol_jitter(cov):
    """Cholesky with escalating jitter (covariance can be near-singular)."""
    d = np.diag(cov)
    base = 1e-12 * (d.mean() if d.mean() > 0 else 1.0)
    for k in range(8):
        try:
            return np.linalg.cholesky(cov + (base * 10 ** k) * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            continue
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 1e-10, None)))


def maxt_lcb(shift, boot, thr):
    """max-t simultaneous lower bound on max_{s<t}|PSC^s - PSC^t|.

    shift : {t: point PSC^(t)}          boot : {t: (B,) bootstrap PSC^(t)}
    Returns (Delta_hat, LCB_sim, c_alpha).
    """
    pairs = [(s, u) for i, s in enumerate(thr) for u in thr[i + 1:]]
    B = len(next(iter(boot.values())))
    Dhat, SE, dev = {}, {}, {}
    for (s, u) in pairs:
        Dhat[(s, u)] = shift[s] - shift[u]
        d = boot[s] - boot[u]
        SE[(s, u)] = float(np.std(d, ddof=1))
        dev[(s, u)] = d - Dhat[(s, u)]           # centered bootstrap contrast
    tmax = np.zeros(B)
    for (s, u) in pairs:
        if SE[(s, u)] > 0:
            tmax = np.maximum(tmax, np.abs(dev[(s, u)]) / SE[(s, u)])
    c_alpha = float(np.quantile(tmax, 1 - ALPHA))
    Delta = max(abs(Dhat[p]) for p in pairs)
    lcb = max(abs(Dhat[(s, u)]) - c_alpha * SE[(s, u)] for (s, u) in pairs)
    return Delta, lcb, c_alpha


def simboot_pathway(pathway, frame, B, rng, block=200):
    print(f"\n=== {pathway} (simultaneous max-t bootstrap, B={B}) ===", flush=True)
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    enc, names = TA.encode_features(frame, feats)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    y = pd.to_numeric(frame["kabco_ordinal"], errors="coerce").astype("Int64").dropna().astype(int).to_numpy()
    enc = enc.loc[enc.index.isin(keep)].reset_index(drop=True)
    Xdf = pd.DataFrame(TA.scale_no_mean(enc.values), columns=names)
    Xv = Xdf.values.astype(np.float64)
    n, p = Xv.shape
    Xmat = np.column_stack([np.ones(n), Xv])
    print(f"    design n={n} p={p}", flush=True)

    levels = sorted(np.unique(y))
    thr = [t for t in levels[:-1] if 0 < (y > t).sum() < n]
    beta_t, se_t, cov_t, eta_t = {}, {}, {}, {}
    for t in thr:
        by = (y > t).astype(int)
        b, se, cov, ok = fit_firth_cov(Xmat, by)
        beta_t[t] = b; se_t[t] = se; cov_t[t] = cov; eta_t[t] = Xmat @ b
        print(f"    firth t>{t}: n1={int(by.sum())} converged={ok}", flush=True)

    # point corrected PSC per feature: per-threshold shift + marginal delta LCB
    pt_shift, pt = {}, {}
    for jc, f in enumerate(names):
        xj = Xv[:, jc]
        shift, dse = {}, {}
        for t in thr:
            bj = beta_t[t][jc + 1]; se_bj = se_t[t][jc + 1]; eta = eta_t[t]; pf = _lg(eta)
            shift[t] = float(np.mean(pf - _lg(eta - bj * xj)))
            g = float(np.mean(xj * pf * (1 - pf)))          # corrected (full predictor)
            dse[t] = abs(g) * se_bj
        mg, mlcb = 0.0, -np.inf
        for i, s in enumerate(thr):
            for u in thr[i + 1:]:
                gp = abs(shift[s] - shift[u]); seg = np.sqrt(dse[s] ** 2 + dse[u] ** 2)
                mg = max(mg, gp); mlcb = max(mlcb, gp - Z * seg)
        pt_shift[f] = shift
        pt[f] = (mg, mlcb, float(np.mean(xj != 0)))

    cand = [f for f in names if pt[f][0] > 0.5 * DELTA or f in set(TA.CREDIBLE.get(pathway, []))]
    cand_jc = [names.index(f) for f in cand]
    print(f"    candidates carried into bootstrap: {len(cand)}", flush=True)

    Ls = {t: chol_jitter(cov_t[t]) for t in thr}
    boot = {f: {t: np.empty(B) for t in thr} for f in cand}
    done = 0
    while done < B:
        bs = min(block, B - done)
        for t in thr:
            z = rng.standard_normal((Xmat.shape[1], bs))
            draws = beta_t[t][:, None] + Ls[t] @ z              # (p+1, bs)
            eta_block = Xmat @ draws                             # (n, bs)
            lam = _lg(eta_block)
            for f, jc in zip(cand, cand_jc):
                bj = draws[jc + 1, :]
                xj = Xv[:, jc][:, None]
                boot[f][t][done:done + bs] = (lam - _lg(eta_block - xj * bj[None, :])).mean(axis=0)
        done += bs
        print(f"    bootstrap {done}/{B}", flush=True)

    rows = []
    for f in cand:
        mg, mlcb, prev = pt[f]
        Delta, lcb_sim, c_alpha = maxt_lcb(pt_shift[f], boot[f], thr)
        rows.append(dict(pathway=pathway, feature=f, label=TA.LABELS.get(f, f),
                         prevalence=prev, psc_gap=mg,
                         delta_lcb=mlcb, credible_delta="Y" if mlcb > DELTA else "N",
                         c_alpha=c_alpha, simboot_lcb=lcb_sim,
                         credible_simboot="Y" if lcb_sim > DELTA else "N", B=B))
    return pd.DataFrame(rows).sort_values("psc_gap", ascending=False)


# ----------------------------- optional part 2 ----------------------------- #
def boundary_size(B, reps, grid, jobs, seed):
    ALPHAS = np.array([-0.036, -0.506, -0.987, -1.418]); KT, KN = 6, 14

    def build(rng, n, lo):
        Xd = (rng.random((n, KT)) < 0.40).astype(float)
        Xc = rng.standard_normal((n, KN)); X = np.column_stack([Xd, Xc]); m = KT + KN
        bt = np.zeros((m, 4))
        for j in range(KT): bt[j] = np.linspace(lo, 0.0, 4)
        for j in range(KT, m): bt[j] = 0.10
        cp = np.empty((n, 4))
        for t in range(4): cp[:, t] = _lg(ALPHAS[t] + X @ bt[:, t])
        cp = np.minimum.accumulate(cp, axis=1)
        pr = np.empty((n, 5)); pr[:, 0] = 1 - cp[:, 0]
        for t in range(1, 4): pr[:, t] = cp[:, t - 1] - cp[:, t]
        pr[:, 4] = cp[:, 3]; pr = np.clip(pr, 1e-12, None); pr /= pr.sum(1, keepdims=True)
        u = rng.random(n); yy = (u[:, None] > np.cumsum(pr, axis=1)).sum(1).astype(int)
        return X, yy

    def gap_of(seed_, n, lo):
        rng = np.random.default_rng(seed_); X, yy = build(rng, n, lo)
        Xm = np.column_stack([np.ones(n), X]); thr = [t for t in range(4) if 0 < (yy > t).sum() < n]
        if len(thr) < 2: return np.nan
        sh = {}
        for t in thr:
            b, _, _, _ = fit_firth_cov(Xm, (yy > t).astype(int)); eta = Xm @ b
            sh[t] = np.array([np.mean(_lg(eta) - _lg(eta - b[j + 1] * X[:, j])) for j in range(KT)])
        S = np.vstack([sh[t] for t in thr]); g = 0.0
        for i in range(len(thr)):
            for u in range(i + 1, len(thr)): g = max(g, float(np.max(np.abs(S[i] - S[u]))))
        return g

    def calibrate(target, seed_=777, n=120000, iters=18):
        lo_lo, lo_hi = 0.0, 3.0
        for _ in range(iters):
            lo = .5 * (lo_lo + lo_hi)
            if gap_of(seed_, n, lo) < target: lo_lo = lo
            else: lo_hi = lo
        return .5 * (lo_lo + lo_hi)

    lo = calibrate(DELTA)
    print(f"[size] calibrated lo={lo:.4f} for boundary gap=delta={DELTA}", flush=True)

    def one(n, rep, s):
        rng = np.random.default_rng(s); X, yy = build(rng, n, lo)
        m = X.shape[1]; Xm = np.column_stack([np.ones(n), X]); thr = [t for t in range(4) if 0 < (yy > t).sum() < n]
        if len(thr) < 2: return None
        be, cv, et = {}, {}, {}
        for t in thr:
            b, se, cov, _ = fit_firth_cov(Xm, (yy > t).astype(int)); be[t] = b; cv[t] = cov; et[t] = Xm @ b
        rb = np.random.default_rng(s + 1)
        Ls = {t: chol_jitter(cv[t]) for t in thr}
        dflag, sflag = [], []
        for j in range(KT):
            xj = X[:, j]; shift, dse = {}, {}
            for t in thr:
                bj = be[t][j + 1]; eta = et[t]; pf = _lg(eta)
                shift[t] = float(np.mean(pf - _lg(eta - bj * xj)))
                dse[t] = abs(np.mean(xj * pf * (1 - pf))) * np.sqrt(cv[t][j + 1, j + 1])
            mlcb = -np.inf
            for i, s0 in enumerate(thr):
                for u in thr[i + 1:]:
                    gp = abs(shift[s0] - shift[u]); mlcb = max(mlcb, gp - Z * np.sqrt(dse[s0] ** 2 + dse[u] ** 2))
            dflag.append(mlcb > DELTA)
            bootj = {}
            for t in thr:
                z = rb.standard_normal((m + 1, B)); dr = be[t][:, None] + Ls[t] @ z
                eb = Xm @ dr; bootj[t] = (_lg(eb) - _lg(eb - xj[:, None] * dr[j + 1, :][None, :])).mean(0)
            _, lcb_sim, _ = maxt_lcb(shift, bootj, thr)
            sflag.append(lcb_sim > DELTA)
        return {"N": n, "rep": rep, "delta_flag_rate": float(np.mean(dflag)),
                "simboot_flag_rate": float(np.mean(sflag))}

    rngm = np.random.default_rng(seed)
    tasks = [(n, rep, int(rngm.integers(0, 2**63 - 1))) for n in grid for rep in range(reps)]
    if jobs == 1:
        rows = [one(n, r, s) for (n, r, s) in tasks]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs, verbose=5)(delayed(one)(n, r, s) for (n, r, s) in tasks)
    df = pd.DataFrame([r for r in rows if r])
    return df.groupby("N").agg(delta_flag_rate=("delta_flag_rate", "mean"),
                               simboot_flag_rate=("simboot_flag_rate", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="outputs/kabco_person_level_clean.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--pathways", default="infrastructure,pavement")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20251231)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--with-size", type=int, default=0)
    ap.add_argument("--size-reps", type=int, default=150)
    ap.add_argument("--size-B", type=int, default=400)
    ap.add_argument("--size-grid", default="2500,10000,50000")
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(a.frame, low_memory=False)
    print(f"frame: {len(frame)} rows", flush=True)
    rng = np.random.default_rng(a.seed)

    summary = []
    for pathway in [p.strip() for p in a.pathways.split(",")]:
        df = simboot_pathway(pathway, frame, a.B, rng)
        path = outdir / f"simboot_{pathway}.csv"
        df.to_csv(path, index=False)
        nd = int((df.credible_delta == "Y").sum()); nsb = int((df.credible_simboot == "Y").sum())
        flips = df.loc[(df.credible_delta == "Y") & (df.credible_simboot == "N")]
        print(f"    wrote {path.name}: credible delta={nd}, simboot={nsb}, dropped={len(flips)}")
        for _, r in flips.iterrows():
            print(f"      DROP {r['label']}: delta_lcb={r['delta_lcb']:.4f} -> simboot_lcb={r['simboot_lcb']:.4f} "
                  f"(c_alpha={r['c_alpha']:.2f})")
        summary.append((pathway, nd, nsb))

    print("\n=== simboot summary (credible: delta-method -> simultaneous max-t) ===")
    for pw, nd, nsb in summary:
        print(f"  {pw}: {nd} -> {nsb}")

    if a.with_size:
        grid = [int(x) for x in a.size_grid.split(",") if x.strip()]
        agg = boundary_size(a.size_B, a.size_reps, grid, a.jobs, a.seed)
        spath = outdir / "simboot_boundary_size.csv"
        agg.to_csv(spath, index=False)
        print("\n=== boundary size (true max gap = delta): flag rate should be <= alpha ===")
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"wrote {spath}")
    print("\nDONE run_simboot")


if __name__ == "__main__":
    main()
