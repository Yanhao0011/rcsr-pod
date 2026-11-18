#!/usr/bin/env python3
"""Monte Carlo study of the coding-invariant departure (RCSR-POD v3).

Every decision uses the invariant departure (rcsr_v3) with ground truth defined by the same functional at the
true (or pseudo-true) cut-specific coefficients, delta = 0.01, one-sided alpha = 0.05, and the parametric
bootstrap one-sided max-t simultaneous bound (B draws from the Firth covariance).

Experiments (--experiment):
  e1   exact PO and trivial departures (default DGP): false-flag rate of Brant vs RCSR-POD by N.
  e2   detection versus true departure (gap design), by N.
  e4   many-covariate family (6 exact-PO + 6 trivial continuous + 8 dense-dummy violators): raw / Holm / BH.
  pow  power curve and boundary size: carriers calibrated to true departure in {0, .5, 1, 1.5, 2, 3} x delta,
       with coverage of the simultaneous and marginal lower bounds.
  inv  coding invariance: an uncentred continuous covariate (mean 5) and a binary covariate with trivial
       departures (true value 0.5 delta), each analysed under three codings (as generated, centred, reference
       reversed); flag rates of the coefficient-replacement departure without re-profiling vs the invariant one.
  sep  structural separation: an indicator whose records never reach the fatal category (the frame-indicator
       situation of the application); Firth vs unpenalized ML bounds by prevalence and N.
  frm  frame versus mechanism: population-PO covariates observed through a fused frame (a census of severe
       records plus a probability sample that excludes fatal records) with a source indicator; flag rates for
       the indicator and the mechanism covariates against pseudo-truth on the fused frame.
  e5   extended design (crash clustering, outcome-dependent selection, correlated four-level categorical with
       a nearly collinear nested dummy, rare fatal cut), model and cluster-sandwich bootstrap, plus the
       categorical as a grouped unit.
"""
from __future__ import annotations
import argparse, sys, json, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_po_diagnostic as S
import sim_departure as SD
import sim_departure_e5c as E5
import run_simboot3 as S3
from run_simboot import chol_jitter
from departure_core import departure_profile, logistic
import rcsr_v3 as V
from scipy import stats
warnings.filterwarnings("ignore")

DELTA, ALPHA, Z = 0.01, 0.05, V.Z
ALPHAS = S.DEFAULT_ALPHAS
BLOCK = 50


# ------------------------------------------------------------------ truth and estimation
def true_dinv(alphas, bt, X, units):
    """bt (p, T) true cut-specific coefficients; units: list of column lists. Pooling uses the population analogue of
    the estimator's GLS weights (inverse expected-information covariance blocks). Returns (Dmax (U,), D (U, T))."""
    n, p = X.shape; T = len(alphas); Xm = np.column_stack([np.ones(n), X])
    eta = [np.ascontiguousarray(alphas[t] + X @ bt[:, t]) for t in range(T)]
    lam = [logistic(e) * (1 - logistic(e)) for e in eta]
    ridge = 1e-8 * np.eye(p + 1)
    inv = [np.linalg.inv((Xm * lam[t][:, None]).T @ Xm + ridge) for t in range(T)]
    D = np.zeros((len(units), T))
    for u, cols in enumerate(units):
        c1 = np.array(cols) + 1
        W = V.gls_weights(np.array([inv[t][np.ix_(c1, c1)] for t in range(T)]))
        bbar = V.pooled(W, bt[cols].T); XG = V.as_group(X[:, cols])
        for t in range(T):
            D[u, t] = V._dinv_core(eta[t], XG, np.ascontiguousarray(bt[cols, t] - bbar))
    return D.max(1), D


def joint_cov(Xm, ys, ps, covs, codes=None):
    """Stacked four-cut sandwich H^-1 M H^-1: record-level scores (codes None) or scores summed within clusters."""
    q = Xm.shape[1]; T = len(ys)
    Sc = [Xm * (ys[i] - ps[i])[:, None] for i in range(T)]
    if codes is not None:
        from scipy import sparse
        n_cl = int(codes.max()) + 1
        M = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_cl, len(codes)))
        Sc = [np.asarray(M @ s_) for s_ in Sc]
    Vj = np.zeros((T * q, T * q))
    for i in range(T):
        for j in range(i, T):
            blk = covs[i] @ (Sc[i].T @ Sc[j]) @ covs[j]
            Vj[i * q:(i + 1) * q, j * q:(j + 1) * q] = blk; Vj[j * q:(j + 1) * q, i * q:(i + 1) * q] = blk.T
    return 0.5 * (Vj + Vj.T)


def fits(X, y, crash=None):
    thr, B, Sg, eta, covs, Xm, A = SD.firth_fits(X, y, want_cov=True)
    out = dict(thr=thr, B=B, Sg=Sg, eta=eta, covs=covs, Xm=Xm, A=A, n=len(y))
    if len(thr):
        ys = [(y > t).astype(float) for t in thr]; ps = [logistic(e) for e in eta]
        out["V"] = joint_cov(Xm, ys, ps, covs)
        if crash is not None:
            out["Vcl"] = joint_cov(Xm, ys, ps, covs, pd.factorize(crash)[0])
    return out


def wald(F, units):
    """Joint-covariance Wald test of PO per unit (cut-specific vs cut-0 contrasts, stacked record-level sandwich):
    returns (p (U,), chi2 (U,), df (U,))."""
    B = F["B"]; T = B.shape[0]; q = B.shape[1] + 1; Vj = F["V"]
    C = np.zeros((T - 1, T)); C[:, 0] = -1; C[np.arange(T - 1), np.arange(1, T)] = 1
    pv, ch, dfs = [], [], []
    for cols in units:
        g = len(cols); idx = [t * q + k + 1 for t in range(T) for k in cols]; K = np.kron(C, np.eye(g))
        d = K @ np.concatenate([B[t, cols] for t in range(T)])
        c2 = float(d @ np.linalg.pinv(K @ Vj[np.ix_(idx, idx)] @ K.T) @ d); df_ = (T - 1) * g
        pv.append(float(stats.chi2.sf(c2, df_))); ch.append(c2); dfs.append(df_)
    return np.array(pv), np.array(ch), np.array(dfs)


def comparators(F, units, n):
    """flags of the sharp-null comparators: joint Wald p < alpha and the BIC-type rule chi2 > df log N"""
    pv, ch, dfs = wald(F, units)
    return (pv < ALPHA).astype(int), (ch > dfs * np.log(n)).astype(int), pv


def bounds(F, X, units, Bd, rng, cl=False, deltas=(DELTA,)):
    """Invariant departure + one-sided max-t (cut-selected) simultaneous and marginal bounds for every unit, with
    draws from the stacked joint sandwich (record-level, or crash-cluster when cl=True): the deployed procedure."""
    B, A, eta, Xm, covs = F["B"], F["A"], F["eta"], F["Xm"], F["covs"]; T = len(F["thr"]); q = Xm.shape[1]
    W = [V.gls_weights(np.array([covs[i][np.ix_(np.array(c) + 1, np.array(c) + 1)] for i in range(T)])) for c in units]
    Dh = [V.dinv_point(eta, B[:, cols], W[u], X[:, cols])[0] for u, cols in enumerate(units)]
    XG = [V.as_group(X[:, cols]) for cols in units]
    L = chol_jitter(F["Vcl"] if cl else F["V"]); theta = np.concatenate([np.concatenate([[A[i]], B[i]]) for i in range(T)])
    Db = [np.empty((T, Bd)) for _ in units]; kap = V.kappa_n(F["n"])
    done = 0
    while done < Bd:
        bs = min(BLOCK, Bd - done)
        dr = theta[:, None] + L @ rng.standard_normal((T * q, bs))
        draws = [dr[i * q:(i + 1) * q] for i in range(T)]; etaT = [np.ascontiguousarray((Xm @ d).T) for d in draws]
        for u, cols in enumerate(units):
            bj = np.stack([draws[i][np.array(cols) + 1] for i in range(T)])        # (T, g, bs)
            bbar = V.pooled(W[u], bj)
            for i in range(T):
                Db[u][i, done:done + bs] = V.dinv_draws(etaT[i], XG[u], np.ascontiguousarray((bj[i] - bbar).T))
        done += bs
    out = []
    for u in range(len(units)):
        r = {d: V.maxt_onesided(Dh[u], Db[u], d, kappa=kap) for d in deltas}; r0 = r[deltas[0]]
        r_all = V.maxt_onesided(Dh[u], Db[u], deltas[0])                          # without cut selection (for comparison)
        out.append(dict(Dmax=float(Dh[u].max()), lcb=r0["lcb"], lcb_marg=r0["lcb_marg"], c=r0["c"], ncut=r0["ncut"],
                        lcb_nosel=r_all["lcb"], c_nosel=r_all["c"], **{f"p{d}": r[d]["p"] for d in deltas}))
    return out


def gen_ordinal(rng, X, bt, alphas, u=None):
    T = len(alphas); off = 0 if u is None else u
    cp = np.column_stack([logistic(alphas[t] + X @ bt[:, t] + off) for t in range(T)])
    cp = np.minimum.accumulate(cp, axis=1)
    pr = np.column_stack([1 - cp[:, 0]] + [cp[:, t] - cp[:, t + 1] for t in range(T - 1)] + [cp[:, T - 1]])
    pr = np.clip(pr, 1e-12, None); pr /= pr.sum(1, keepdims=True)
    return (rng.random(len(X))[:, None] > np.cumsum(pr, 1)).sum(1)


def single(p): return [[j] for j in range(p)]


def _threads():
    try:
        import numba; numba.set_num_threads(1)
    except Exception:
        pass


# ------------------------------------------------------------------ e1 / e2
def e12_task(n, rep, seed, departure, gaps, Bd):
    _threads(); rng = np.random.default_rng(seed)
    if gaps is not None:
        Xdf, y, _, bt, al = S.simulate_ordinal(rng, n, gap_targets=gaps, return_betat=True)
    else:
        Xdf, y, _, bt, al = S.simulate_ordinal(rng, n, departure=departure, return_betat=True)
    X = Xdf.values; p = X.shape[1]; U = single(p)
    Dt, _ = true_dinv(al, bt, X, U)
    F = fits(X, y)
    if len(F["thr"]) < 4: return []
    res = bounds(F, X, U, Bd, np.random.default_rng(seed + 1)); wf, bf, _ = comparators(F, U, n)
    return [dict(N=n, rep=rep, departure=departure, feature=f, Dtrue=float(Dt[j]), substantive=bool(Dt[j] > DELTA),
                 brant_flag=int(wf[j]), bic_flag=int(bf[j]), flag=int(res[j]["lcb"] > DELTA), flag_marg=int(res[j]["lcb_marg"] > DELTA),
                 Dmax=res[j]["Dmax"], lcb=res[j]["lcb"], cover=int(res[j]["lcb"] <= Dt[j]))
            for j, f in enumerate(Xdf.columns)]


# ------------------------------------------------------------------ e4 (+ post-selection accuracy)
E4_TRIV = 0.12          # trivial continuous drift (0.12 -> 0 over the cuts): true departure about 0.6 delta
N_TEST = 20000


def build_e4(rng, n):
    Xc = rng.standard_normal((n, SD.P_CONT)); Xd = (rng.random((n, SD.P_DUM)) < SD.DUMMY_PREV).astype(float)
    X = np.column_stack([Xc, Xd]); bt = np.zeros((SD.P_CONT + SD.P_DUM, 4))
    bt[:SD.N_EXACT] = 0.12
    for j in range(SD.N_EXACT, SD.P_CONT): bt[j] = 0.12 + np.linspace(E4_TRIV, 0.0, 4)
    for j in range(SD.P_CONT, SD.P_CONT + SD.P_DUM): bt[j] = np.linspace(1.0, 0.0, 4)
    return X, gen_ordinal(np.random.default_rng(rng.integers(2 ** 62)), X, bt, ALPHAS), bt


def cat_probs(X, bt, alphas):
    T = len(alphas); cp = np.column_stack([logistic(alphas[t] + X @ bt[:, t]) for t in range(T)])
    cp = np.minimum.accumulate(cp, axis=1)
    return np.column_stack([1 - cp[:, 0]] + [cp[:, t] - cp[:, t + 1] for t in range(T - 1)] + [cp[:, T - 1]])


def post_selection(X, y, sets, Xte, Pte):
    """penalized partial-PO refit for each selected set; RMSE of the fitted category probabilities against the true
    ones on an independent test sample, and the share of test records with a non-positive fitted probability"""
    from run_partial_po_fit import PartialPO, T as TT
    p = X.shape[1]; out = {}
    alpha0 = np.array([np.log(np.mean(y > t) / (1 - np.mean(y > t))) for t in range(TT)])
    po = PartialPO(X, None, y, 1.0); th_po = po.fit(np.concatenate([alpha0, np.zeros(p)]), 400)["theta"]
    for name, idx2 in sets.items():
        idx1 = [j for j in range(p) if j not in idx2]
        if not idx2:
            th, m = th_po, po
        else:
            m = PartialPO(X[:, idx1] if idx1 else None, X[:, idx2], y, 1.0)
            th = m.fit(np.concatenate([th_po[:TT], th_po[TT:][idx1], np.tile(th_po[TT:][idx2], TT)]), 400)["theta"]
        mt = PartialPO(Xte[:, idx1] if idx1 else None, Xte[:, idx2] if idx2 else None, np.zeros(len(Xte), dtype=int), 1.0)
        Fc = [logistic(e) for e in mt.etas(th)]
        P = np.column_stack([1 - Fc[0]] + [Fc[t] - Fc[t + 1] for t in range(TT - 1)] + [Fc[TT - 1]])
        out[name] = (float(np.sqrt(np.mean((P - Pte) ** 2))), float(np.mean((P <= 0).any(1))), len(idx2))
    return out


def e4_task(n, rep, seed, Bd, post):
    _threads(); rng = np.random.default_rng(seed); X, y, bt = build_e4(rng, n)
    U = single(X.shape[1]); Dt, _ = true_dinv(ALPHAS, bt, X, U); viol = Dt > DELTA
    F = fits(X, y)
    if len(F["thr"]) < 4: return None
    res = bounds(F, X, U, Bd, np.random.default_rng(seed + 1))
    pD = np.array([r[f"p{DELTA}"] for r in res]); wf, bf, pB = comparators(F, U, n); nv = int(viol.sum())
    h, bh = V.multitest(pD); hB, _ = V.multitest(pB)
    rec = dict(N=n, rep=rep, n_viol=nv)
    lcbflag = np.array([r["lcb"] > DELTA for r in res])
    for rej, tag in [(pB < ALPHA, "brant_raw"), (hB, "brant_holm"), (bf.astype(bool), "bic"), (pD < ALPHA, "rcsr_raw"),
                     (h, "rcsr_holm"), (bh, "rcsr_bh"), (lcbflag, "rcsr_lcb")]:
        Vn = int((rej & ~viol).sum()); Sd = int((rej & viol).sum()); R = int(rej.sum())
        rec.update({f"{tag}_fwer": int(Vn > 0), f"{tag}_fdp": Vn / R if R else 0.0, f"{tag}_power": Sd / nv if nv else np.nan})
    if post:
        rt = np.random.default_rng(seed + 7); Xte, _, _ = build_e4(rt, N_TEST); Pte = cat_probs(Xte, bt, ALPHAS)
        sets = {"PO": [], "RCSR-POD": list(np.where(lcbflag)[0]), "Brant": list(np.where(pB < ALPHA)[0]),
                "BIC": list(np.where(bf)[0]), "GOL": list(range(X.shape[1])), "oracle": list(np.where(viol)[0])}
        for name, (rmse, neg, k) in post_selection(X, y, sets, Xte, Pte).items():
            rec.update({f"rmse_{name}": rmse, f"neg_{name}": neg, f"k_{name}": k})
    return rec


# ------------------------------------------------------------------ pow (boundary, power curve; two departure shapes)
KT, KN = 6, 4


def build_pow(rng, n, lo, shape="linear"):
    Xd = (rng.random((n, KT)) < 0.40).astype(float); Xc = rng.standard_normal((n, KN))
    X = np.column_stack([Xd, Xc]); bt = np.zeros((KT + KN, 4))
    for j in range(KT):
        # linear: coefficient falls from lo to 0 across the cuts; peak: common except at the fatal cut (one-cut maximum)
        bt[j] = np.linspace(lo, 0.0, 4) if shape == "linear" else np.array([0.3, 0.3, 0.3, 0.3 - lo])
    for j in range(KT, KT + KN): bt[j] = 0.10
    return X, gen_ordinal(rng, X, bt, ALPHAS), bt


def carrier_truth(lo, n=200000, seed=777, shape="linear"):
    rng = np.random.default_rng(seed); X, _, bt = build_pow(rng, n, lo, shape)
    return float(np.mean(true_dinv(ALPHAS, bt, X, [[j] for j in range(KT)])[0]))


def calibrate(target, fn=carrier_truth, hi=3.0, iters=20, **kw):
    if target <= 0: return 0.0, 0.0
    a, b = 0.0, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if fn(m, **kw) < target: a = m
        else: b = m
    lo = 0.5 * (a + b); return lo, fn(lo, seed=778, **kw)


def pow_task(mult, lo, n, rep, seed, Bd, shape="linear"):
    _threads(); rng = np.random.default_rng(seed); X, y, bt = build_pow(rng, n, lo, shape); U = single(X.shape[1])
    Dt, _ = true_dinv(ALPHAS, bt, X, U); F = fits(X, y)
    if len(F["thr"]) < 4: return []
    res = bounds(F, X, U, Bd, np.random.default_rng(seed + 1)); wf, bf, _ = comparators(F, U, n)
    return [dict(shape=shape, mult=mult, N=n, rep=rep, feature=j, ftype="carrier" if j < KT else "exactPO", Dtrue=float(Dt[j]),
                 Dmax=res[j]["Dmax"], lcb=res[j]["lcb"], lcb_marg=res[j]["lcb_marg"], c=res[j]["c"], c_nosel=res[j]["c_nosel"],
                 ncut=res[j]["ncut"], flag=int(res[j]["lcb"] > DELTA), flag_nosel=int(res[j]["lcb_nosel"] > DELTA),
                 flag_marg=int(res[j]["lcb_marg"] > DELTA), brant_flag=int(wf[j]), bic_flag=int(bf[j]),
                 cover=int(res[j]["lcb"] <= Dt[j]), cover_marg=int(res[j]["lcb_marg"] <= Dt[j])) for j in range(len(U))]


# ------------------------------------------------------------------ inv (coding invariance: single columns and a grouped categorical)
INV_MEAN = 5.0
INV_LEV = np.array([0.50, 0.25, 0.15, 0.10])          # four-level categorical, refit under each reference level


def build_inv(rng, n, lo_c, lo_b, lo_g=0.0):
    xc = INV_MEAN + rng.standard_normal(n); xb = (rng.random(n) < 0.40).astype(float); xo = rng.standard_normal((n, 2))
    lev = rng.choice(4, size=n, p=INV_LEV)
    G = np.column_stack([(lev == k).astype(float) for k in (1, 2, 3)])
    X = np.column_stack([xc, xb, xo, G]); bt = np.zeros((7, 4))
    bt[0] = 0.30 + np.linspace(lo_c, 0, 4); bt[1] = 0.50 + np.linspace(lo_b, 0, 4); bt[2] = 0.25; bt[3] = -0.20
    bt[4] = 0.40 + np.linspace(lo_g, 0, 4); bt[5] = -0.30 + np.linspace(0, lo_g, 4); bt[6] = 0.20
    al = ALPHAS - 0.30 * INV_MEAN - 0.20                       # keep the marginal cut rates near the default design
    return X, gen_ordinal(rng, X, bt, al), bt, al, lev


def inv_truth_c(lo, n=200000, seed=777):
    rng = np.random.default_rng(seed); X, _, bt, al, _ = build_inv(rng, n, lo, 0.0)
    return float(true_dinv(al, bt, X, [[0]])[0][0])


def inv_truth_b(lo, n=200000, seed=777):
    rng = np.random.default_rng(seed); X, _, bt, al, _ = build_inv(rng, n, 0.0, lo)
    return float(true_dinv(al, bt, X, [[1]])[0][0])


def inv_truth_g(lo, n=200000, seed=777):
    rng = np.random.default_rng(seed); X, _, bt, al, _ = build_inv(rng, n, 0.0, 0.0, lo)
    return float(true_dinv(al, bt, X, [[4, 5, 6]])[0][0])


def inv_task(n, rep, seed, lo_c, lo_b, lo_g, Bd):
    _threads(); rng = np.random.default_rng(seed); X, y, bt, al, lev = build_inv(rng, n, lo_c, lo_b, lo_g)
    Dt, _ = true_dinv(al, bt, X, [[0], [1], [4, 5, 6]]); F = fits(X, y)
    if len(F["thr"]) < 4: return []
    res = bounds(F, X, [[0], [1]], Bd, np.random.default_rng(seed + 1))
    rows = []
    for j, codings in [(0, [("as generated", 0.0), ("centered", float(X[:, 0].mean())), ("shifted +5", -5.0)]),
                       (1, [("as generated", 0.0), ("centered", float(X[:, 1].mean())), ("reference reversed", 1.0)])]:
        for name, c in codings:
            # Firth/logit are affine-equivariant: recoding x -> x - c leaves the slopes, SEs and eta unchanged
            prof = departure_profile(F["eta"], F["B"][:, j], F["Sg"][:, j], X[:, j] - c)
            lcb_old = float(np.max(np.abs(prof["D"]) - Z * prof["se_D"]))
            Dinv_rec = V.dinv_point(F["eta"], F["B"][:, j], V.precision_weights(F["Sg"][:, j]), X[:, j] - c)[0].max()
            rows.append(dict(N=n, rep=rep, feature=["uncentered continuous", "binary"][j], coding=name, Dtrue=float(Dt[j]),
                             D_old=float(np.abs(prof["D"]).max()), flag_old=int(lcb_old > DELTA), Dinv=float(Dinv_rec),
                             Dinv_shift=float(abs(Dinv_rec - res[j]["Dmax"])), lcb_inv=res[j]["lcb"], flag_inv=int(res[j]["lcb"] > DELTA)))
    # grouped categorical: refit the Firth logits and recompute the bound under each of the four reference levels
    base = X[:, :4]
    for ref in range(4):
        G = np.column_stack([(lev == k).astype(float) for k in range(4) if k != ref])
        Xr = np.column_stack([base, G]); Fr = fits(Xr, y)
        rr = bounds(Fr, Xr, [[4, 5, 6]], Bd, np.random.default_rng(seed + 2))[0]
        rows.append(dict(N=n, rep=rep, feature="categorical (group)", coding=f"reference level {ref}", Dtrue=float(Dt[2]),
                         D_old=np.nan, flag_old=np.nan, Dinv=rr["Dmax"], Dinv_shift=np.nan, lcb_inv=rr["lcb"], flag_inv=int(rr["lcb"] > DELTA)))
    return rows


# ------------------------------------------------------------------ frm (frame vs mechanism, with a crash-size covariate)
FRM_ALPHAS = np.array([-0.040, -0.847, -1.992, -3.476])
FRM_CRSS = 0.10                                             # sampling rate of non-fatal crashes
FRM_NAMES = ["x_cont_1", "x_cont_2", "x_bin_1", "x_bin_2", "crash_size"]


def build_frm(rng, n_pop):
    """Crash-level fused frame: a census of every person in a crash with at least one fatality (FARS-like)
    plus a probability sample of crashes without a fatality (CRSS-like; fatal crashes are excluded from it).
    crash_size = log(persons in the crash), PO with a small common coefficient in the population."""
    sizes = rng.choice([1, 2, 3, 4, 5], size=n_pop, p=[0.35, 0.35, 0.15, 0.10, 0.05])
    cum = np.cumsum(sizes); ncr = int(np.searchsorted(cum, n_pop)) + 1; sizes = sizes[:ncr]; n = int(sizes.sum())
    crash = np.repeat(np.arange(ncr), sizes); u = rng.normal(0.0, 0.5, ncr)[crash]
    X = np.column_stack([rng.standard_normal((n, 2)), (rng.random((n, 2)) < [0.3, 0.5]).astype(float), np.log(sizes)[crash]])
    bt = np.tile(np.array([0.30, -0.25, 0.40, -0.30, -0.10])[:, None], (1, 4))
    y = gen_ordinal(rng, X, bt, FRM_ALPHAS, u=u)
    fatal_cr = np.zeros(ncr, dtype=bool); fatal_cr[crash[y == 4]] = True
    census = fatal_cr[crash]; samp = (~census) & (rng.random(ncr) < FRM_CRSS)[crash]
    keep = census | samp
    return X[keep], census[keep].astype(float), y[keep]


def frm_truth(n_pop=4_000_000, seed=777):
    rng = np.random.default_rng(seed); X, s, y = build_frm(rng, n_pop)
    out = {}
    for tag, Xd in [("with_source", np.column_stack([X, s])), ("without_source", X)]:
        F = SD.firth_fits(Xd, y); A, B = F[6], F[1]
        out[tag] = true_dinv(A, B.T, Xd, single(Xd.shape[1]))[0]
    return out, len(y) / n_pop, float(s.mean())


def frm_task(n_target, rep, seed, truth, frac, Bd):
    _threads(); rng = np.random.default_rng(seed)
    X, s, y = build_frm(rng, int(n_target / frac * 1.05)); X, s, y = X[:n_target], s[:n_target], y[:n_target]
    rows = []
    for tag, Xd, names in [("with_source", np.column_stack([X, s]), FRM_NAMES + ["source"]), ("without_source", X, FRM_NAMES)]:
        F = fits(Xd, y)
        if len(F["thr"]) < 4: return []
        U = single(Xd.shape[1]); res = bounds(F, Xd, U, Bd, np.random.default_rng(seed + 1)); wf, bf, _ = comparators(F, U, n_target)
        rows += [dict(N=n_target, rep=rep, model=tag, feature=f, Dtrue=float(truth[tag][j]), Dmax=res[j]["Dmax"],
                      flag=int(res[j]["lcb"] > DELTA), brant_flag=int(wf[j]), bic_flag=int(bf[j])) for j, f in enumerate(names)]
    return rows


# ------------------------------------------------------------------ sep (structural and quasi-separation at the fatal cut)
SEP_PREV = [0.01, 0.03, 0.10]
SEP_STRENGTH = {"structural": -30.0, "quasi": -4.0}


def ml_fits(X, y, ridge=1e-6, iters=60):
    """Unpenalized ML threshold logits by Newton-Raphson; under separation the affected coefficient drifts towards
    -inf and its variance explodes, as in standard software."""
    n, p = X.shape; Xm = np.column_stack([np.ones(n), X]); thr = [t for t in range(4) if 0 < (y > t).sum() < n]
    B = np.zeros((len(thr), p)); Sg = np.zeros((len(thr), p)); A = np.zeros(len(thr)); eta = []; covs = []; sep = []
    for i, t in enumerate(thr):
        by = (y > t).astype(float); b = np.zeros(p + 1)
        for _ in range(iters):
            pr = logistic(Xm @ b); w = pr * (1 - pr)
            H = (Xm * w[:, None]).T @ Xm + ridge * np.eye(p + 1)
            step = np.linalg.solve(H, Xm.T @ (by - pr)); b = b + step
            if np.abs(step).max() < 1e-8: break
        pr = logistic(Xm @ b); w = pr * (1 - pr)
        cov = np.linalg.pinv((Xm * w[:, None]).T @ Xm + ridge * np.eye(p + 1))
        A[i], B[i], Sg[i] = b[0], b[1:], np.sqrt(np.clip(np.diag(cov)[1:], 0, None)); eta.append(Xm @ b); covs.append(cov)
        sep.append(bool(np.abs(b[1:]).max() > 10))
    out = dict(thr=thr, B=B, Sg=Sg, eta=eta, covs=covs, Xm=Xm, A=A, separated=any(sep), n=n)
    ys = [(y > t).astype(float) for t in thr]; ps = [logistic(e) for e in eta]
    out["V"] = joint_cov(Xm, ys, ps, covs)
    return out


def build_sep(rng, n, prev, strength="structural"):
    """x_s: binary indicator whose records never (structural) or rarely (quasi) reach the top category; common
    coefficient 0.30 at the lower cuts. Three further covariates satisfy PO."""
    xs = (rng.random(n) < prev).astype(float); Xo = np.column_stack([rng.standard_normal((n, 2)), (rng.random(n) < 0.4).astype(float)])
    X = np.column_stack([xs, Xo]); bt = np.zeros((4, 4))
    bt[0] = [0.30, 0.30, 0.30, SEP_STRENGTH[strength]]; bt[1] = 0.25; bt[2] = -0.20; bt[3] = 0.35
    return X, gen_ordinal(rng, X, bt, ALPHAS), bt


def sep_task(prev, strength, n, rep, seed, Bd):
    _threads(); rng = np.random.default_rng(seed); X, y, bt = build_sep(rng, n, prev, strength); U = single(X.shape[1])
    Dt, _ = true_dinv(ALPHAS, bt, X, U)
    Ff = fits(X, y); Fm = ml_fits(X, y)
    if len(Ff["thr"]) < 4: return []
    rf = bounds(Ff, X, U, Bd, np.random.default_rng(seed + 1)); rm = bounds(Fm, X, U, Bd, np.random.default_rng(seed + 1))
    wm, _, _ = comparators(Fm, U, n)
    events = int(((X[:, 0] == 1) & (y == 4)).sum())
    return [dict(strength=strength, prev=prev, N=n, rep=rep, feature=["separated indicator", "x_cont_1", "x_cont_2", "x_bin"][j],
                 Dtrue=float(Dt[j]), ml_separated=int(Fm["separated"]), top_events=events, flag_firth=int(rf[j]["lcb"] > DELTA),
                 flag_ml=int(rm[j]["lcb"] > DELTA), brant_ml=int(wm[j]), Dmax_firth=rf[j]["Dmax"], Dmax_ml=rm[j]["Dmax"],
                 se_ratio_ml_firth=float(np.nanmax(Fm["Sg"][:, j]) / max(np.nanmax(Ff["Sg"][:, j]), 1e-12))) for j in range(len(U))]


# ------------------------------------------------------------------ e5 (extended design)
E5_UNITS = [[0], [1], [2], [3], [4], [5], [6], [3, 4, 5]]
E5_NAMES = E5.FEATS + ["categorical [group]"]
E5_TYPES = E5.FTYPE + ["group"]


def e5_truth(lo_b, lo_c, n_pop, seed=777):
    rng = np.random.default_rng(seed); X, y, crash, w, bt = E5.build_e5c(rng, n_pop, lo_b, lo_c)
    F = SD.firth_fits(X, y); A, B = F[6], F[1]
    return true_dinv(A, B.T, X, E5_UNITS)[0], A, B, len(y) / n_pop


def e5_calibrate(n_pop):
    lo_b, lo_c = 1.0, 0.3
    for _ in range(2):
        a, b = 0.0, 3.0
        for _ in range(14):
            m = 0.5 * (a + b)
            if e5_truth(lo_b, m, n_pop)[0][5] < DELTA: a = m
            else: b = m
        lo_c = 0.5 * (a + b); a, b = 0.0, 4.0
        for _ in range(14):
            m = 0.5 * (a + b)
            if e5_truth(m, lo_c, n_pop)[0][4] < 3 * DELTA: a = m
            else: b = m
        lo_b = 0.5 * (a + b)
    return lo_b, lo_c


def e5_task(n_target, rep, seed, lo_b, lo_c, Dtrue, frac, Bd):
    _threads(); rng = np.random.default_rng(seed)
    X, y, crash, w, bt = E5.build_e5c(rng, int(n_target / frac * 1.06), lo_b, lo_c)
    X, y, crash = X[:n_target], y[:n_target], crash[:n_target]
    F = fits(X, y, crash=crash)
    if len(F["thr"]) < 4: return []
    rm = bounds(F, X, E5_UNITS, Bd, np.random.default_rng(seed + 1))
    rc = bounds(F, X, E5_UNITS, Bd, np.random.default_rng(seed + 2), cl=True)
    return [dict(N=n_target, rep=rep, unit=E5_NAMES[u], utype=E5_TYPES[u], Dtrue=float(Dtrue[u]), Dmax=rm[u]["Dmax"],
                 flag=int(rm[u]["lcb"] > DELTA), flag_cl=int(rc[u]["lcb"] > DELTA), flag_marg=int(rm[u]["lcb_marg"] > DELTA),
                 cover=int(rm[u]["lcb"] <= Dtrue[u]), cover_cl=int(rc[u]["lcb"] <= Dtrue[u]), c=rm[u]["c"], c_cl=rc[u]["c"])
            for u in range(len(E5_UNITS))]


# ------------------------------------------------------------------ scale (application-scale design)
SC_LEV1 = 30; SC_LEV2 = 20


def build_scale(rng, n, lo1, lo_peak):
    """N up to 500,000; six PO continuous regressors; one continuous carrier whose coefficient departs only at the fatal
    cut (one-cut maximum, lo_peak); a 30-level categorical (geometric level probabilities) whose non-reference
    coefficients drift jointly (lo1); a 20-level categorical with trivial drifts."""
    Xc = rng.standard_normal((n, 7))
    p1 = 0.85 ** np.arange(SC_LEV1); p1 /= p1.sum(); l1 = rng.choice(SC_LEV1, size=n, p=p1)
    p2 = 0.80 ** np.arange(SC_LEV2); p2 /= p2.sum(); l2 = rng.choice(SC_LEV2, size=n, p=p2)
    G1 = np.column_stack([(l1 == k).astype(float) for k in range(1, SC_LEV1)])
    G2 = np.column_stack([(l2 == k).astype(float) for k in range(1, SC_LEV2)])
    X = np.column_stack([Xc, G1, G2]); p = X.shape[1]; bt = np.zeros((p, 4))
    bt[:6] = np.array([0.30, -0.25, 0.20, -0.15, 0.10, 0.05])[:, None]
    bt[6] = [0.20, 0.20, 0.20, 0.20 - lo_peak]
    r1 = np.random.default_rng(11).normal(0, 0.4, SC_LEV1 - 1); s1 = np.random.default_rng(12).choice([-1, 1], SC_LEV1 - 1)
    for k in range(SC_LEV1 - 1): bt[7 + k] = r1[k] + s1[k] * np.linspace(lo1, 0, 4)
    r2 = np.random.default_rng(13).normal(0, 0.3, SC_LEV2 - 1)
    for k in range(SC_LEV2 - 1): bt[7 + SC_LEV1 - 1 + k] = r2[k] + np.linspace(0.03, 0, 4)
    return X, gen_ordinal(rng, X, bt, ALPHAS), bt


SC_UNITS = [[j] for j in range(7)] + [list(range(7, 7 + SC_LEV1 - 1)), list(range(7 + SC_LEV1 - 1, 7 + SC_LEV1 - 1 + SC_LEV2 - 1))]
SC_NAMES = [f"x_cont_{j}" for j in range(6)] + ["peak carrier", "categorical-30 [group]", "categorical-20 [group]"]


def scale_task(n, rep, seed, lo1, lo_peak, Bd, Dtrue):
    _threads(); rng = np.random.default_rng(seed); X, y, bt = build_scale(rng, n, lo1, lo_peak)
    F = fits(X, y)
    if len(F["thr"]) < 4: return []
    res = bounds(F, X, SC_UNITS, Bd, np.random.default_rng(seed + 1)); wf, bf, _ = comparators(F, SC_UNITS, n)
    return [dict(N=n, rep=rep, unit=SC_NAMES[u], Dtrue=float(Dtrue[u]), Dmax=res[u]["Dmax"], lcb=res[u]["lcb"], c=res[u]["c"],
                 c_nosel=res[u]["c_nosel"], flag=int(res[u]["lcb"] > DELTA), flag_nosel=int(res[u]["lcb_nosel"] > DELTA),
                 flag_marg=int(res[u]["lcb_marg"] > DELTA), brant_flag=int(wf[u]), bic_flag=int(bf[u]),
                 cover=int(res[u]["lcb"] <= Dtrue[u])) for u in range(len(SC_UNITS))]


# ------------------------------------------------------------------ main
def mcse(p, r):
    """Monte Carlo standard error of a rate from r replications"""
    return np.sqrt(np.clip(p * (1 - p), 0, None) / np.maximum(r, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["e1", "e2", "e4", "pow", "inv", "frm", "sep", "e5", "scale"])
    ap.add_argument("--outdir", default=str(HERE.parent / "outputs" / "v3" / "sim29"))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--n-grid", default="5000,25000,100000")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--truth-pop", type=int, default=3_000_000)
    ap.add_argument("--post", action="store_true", help="e4: post-selection accuracy of the refitted partial-PO models")
    ap.add_argument("--shape", default="linear,peak", help="pow: departure shapes")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    grid = [int(x) for x in a.n_grid.split(",") if x.strip()]
    rngm = np.random.default_rng(a.seed); sd = lambda: int(rngm.integers(0, 2 ** 62)); t0 = time.time(); meta = {}
    run = lambda tasks, fn: SD.run(tasks, fn, a.jobs)
    ex = a.experiment
    if ex in ("e1", "e2"):
        if ex == "e1":
            tasks = [(n, r, sd(), dep, None, a.B) for dep in (0.0, 0.04) for n in grid for r in range(a.reps)]
        else:
            tasks = [(n, r, sd(), 0.0, S.DEFAULT_E2PSC_GAPS, a.B) for n in grid for r in range(a.reps)]
        df = pd.DataFrame([r for rows in run(tasks, e12_task) for r in rows])
        if ex == "e1":
            summ = df[~df.substantive].groupby(["departure", "N"]).agg(brant_fp=("brant_flag", "mean"), bic_fp=("bic_flag", "mean"),
                   rcsr_fp=("flag", "mean"), rcsr_marg_fp=("flag_marg", "mean"), max_Dtrue=("Dtrue", "max"), n_cov=("feature", "nunique"),
                   reps=("rep", "nunique")).reset_index()
        else:
            summ = df.groupby(["N", "feature"]).agg(Dtrue=("Dtrue", "mean"), substantive=("substantive", "mean"), brant_det=("brant_flag", "mean"),
                   bic_det=("bic_flag", "mean"), rcsr_det=("flag", "mean"), rcsr_marg_det=("flag_marg", "mean"), cover=("cover", "mean")).reset_index()
    elif ex == "e4":
        X, _, bt = build_e4(np.random.default_rng(a.seed), 200000)
        Dt = true_dinv(ALPHAS, bt, X, single(X.shape[1]))[0]
        meta = dict(Dtrue=Dt.tolist(), n_viol=int((Dt > DELTA).sum()))
        print(f"[e4 truth] violators min {Dt[SD.P_CONT:].min():.4f}; nulls max {Dt[:SD.P_CONT].max():.4f}", flush=True)
        df = pd.DataFrame([r for r in run([(n, r, sd(), a.B, a.post) for n in grid for r in range(a.reps)], e4_task) if r])
        summ = df.groupby("N").mean(numeric_only=True).reset_index()
    elif ex == "pow":
        mults = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]; shapes = [s_.strip() for s_ in a.shape.split(",")]
        cal = {(sh, m): calibrate(m * DELTA, shape=sh) for sh in shapes for m in mults}
        meta = {f"{sh}_{m}": dict(lo=v[0], Dtrue=v[1]) for (sh, m), v in cal.items()}; print("[pow calibration]", meta, flush=True)
        tasks = [(m, cal[(sh, m)][0], n, r, sd(), a.B, sh) for sh in shapes for m in mults for n in grid for r in range(a.reps)]
        df = pd.DataFrame([r for rows in run(tasks, pow_task) for r in rows])
        summ = df.groupby(["shape", "mult", "N", "ftype"]).agg(Dtrue=("Dtrue", "mean"), rate=("flag", "mean"), rate_nosel=("flag_nosel", "mean"),
               rate_marg=("flag_marg", "mean"), brant=("brant_flag", "mean"), bic=("bic_flag", "mean"), cover=("cover", "mean"),
               cover_marg=("cover_marg", "mean"), c_med=("c", "median"), c_nosel_med=("c_nosel", "median"), ncut_mean=("ncut", "mean"),
               lcb_mean=("lcb", "mean"), n_tests=("rep", "size"), reps=("rep", "nunique")).reset_index()
        summ["mcse_rate"] = mcse(summ.rate, summ.n_tests)
    elif ex == "inv":
        lo_c, d_c = calibrate(0.5 * DELTA, inv_truth_c, hi=0.5); lo_b, d_b = calibrate(0.5 * DELTA, inv_truth_b, hi=2.0)
        lo_g, d_g = calibrate(1.5 * DELTA, inv_truth_g, hi=3.0)
        meta = dict(lo_c=lo_c, Dtrue_c=d_c, lo_b=lo_b, Dtrue_b=d_b, lo_g=lo_g, Dtrue_g=d_g, mean_c=INV_MEAN); print("[inv calibration]", meta, flush=True)
        df = pd.DataFrame([r for rows in run([(n, r, sd(), lo_c, lo_b, lo_g, a.B) for n in grid for r in range(a.reps)], inv_task) for r in rows])
        g = df[df.feature == "categorical (group)"]
        spread = g.groupby(["N", "rep"]).agg(Dsp=("Dinv", lambda v: v.max() - v.min()), Lsp=("lcb_inv", lambda v: v.max() - v.min()),
                                             Fsp=("flag_inv", lambda v: int(v.max() != v.min())))
        meta["group_refit_spread"] = dict(max_Dinv_range=float(spread.Dsp.max()), median_Dinv_range=float(spread.Dsp.median()),
                                          max_lcb_range=float(spread.Lsp.max()), share_decision_differs=float(spread.Fsp.mean()))
        summ = df.groupby(["feature", "coding", "N"]).agg(Dtrue=("Dtrue", "mean"), D_old=("D_old", "mean"), flag_old=("flag_old", "mean"),
               Dinv=("Dinv", "mean"), max_Dinv_shift=("Dinv_shift", "max"), lcb_inv=("lcb_inv", "mean"), flag_inv=("flag_inv", "mean"),
               reps=("rep", "nunique")).reset_index()
    elif ex == "frm":
        truth, frac, share = frm_truth(a.truth_pop)
        meta = dict(frac=frac, census_share=share, **{k: v.tolist() for k, v in truth.items()}); print("[frm truth]", meta, flush=True)
        df = pd.DataFrame([r for rows in run([(n, r, sd(), truth, frac, a.B) for n in grid for r in range(a.reps)], frm_task) for r in rows])
        summ = df.groupby(["model", "feature", "N"]).agg(Dtrue=("Dtrue", "mean"), Dmax=("Dmax", "mean"), rate=("flag", "mean"),
               brant=("brant_flag", "mean"), bic=("bic_flag", "mean"), reps=("rep", "nunique")).reset_index()
    elif ex == "sep":
        tasks = [(pv, st, n, r, sd(), a.B) for st in SEP_STRENGTH for pv in SEP_PREV for n in grid for r in range(a.reps)]
        df = pd.DataFrame([r for rows in run(tasks, sep_task) for r in rows])
        summ = df.groupby(["strength", "feature", "prev", "N"]).agg(Dtrue=("Dtrue", "mean"), ml_separated=("ml_separated", "mean"),
               top_events=("top_events", "mean"), rate_firth=("flag_firth", "mean"), rate_ml=("flag_ml", "mean"), brant_ml=("brant_ml", "mean"),
               se_ratio=("se_ratio_ml_firth", "median"), reps=("rep", "nunique")).reset_index()
    elif ex == "scale":
        def tr(lo1, lo_peak, n=300000, seed=777):
            X, _, bt = build_scale(np.random.default_rng(seed), n, lo1, lo_peak); return true_dinv(ALPHAS, bt, X, SC_UNITS)[0]
        lo1, _ = calibrate(2.0 * DELTA, lambda m, seed=777: float(tr(m, 0.0, seed=seed)[7]), hi=2.0, iters=14)
        lop, _ = calibrate(1.5 * DELTA, lambda m, seed=777: float(tr(0.0, m, seed=seed)[6]), hi=3.0, iters=14)
        Dt = tr(lo1, lop, n=500000, seed=778)
        meta = dict(lo_group=lo1, lo_peak=lop, Dtrue=dict(zip(SC_NAMES, Dt.tolist()))); print("[scale calibration]", meta, flush=True)
        df = pd.DataFrame([r for rows in run([(n, r, sd(), lo1, lop, a.B, Dt) for n in grid for r in range(a.reps)], scale_task) for r in rows])
        summ = df.groupby(["N", "unit"]).agg(Dtrue=("Dtrue", "mean"), Dmax=("Dmax", "mean"), rate=("flag", "mean"), rate_nosel=("flag_nosel", "mean"),
               rate_marg=("flag_marg", "mean"), brant=("brant_flag", "mean"), bic=("bic_flag", "mean"), cover=("cover", "mean"),
               c_med=("c", "median"), c_nosel_med=("c_nosel", "median"), reps=("rep", "nunique")).reset_index()
    else:
        lo_b, lo_c = e5_calibrate(600_000); Dt, A_, B_, frac = e5_truth(lo_b, lo_c, a.truth_pop)
        meta = dict(ell_B=lo_b, ell_C=lo_c, frac=frac, Dtrue=dict(zip(E5_NAMES, Dt.tolist())))
        print("[e5 calibration]", meta, flush=True)
        df = pd.DataFrame([r for rows in run([(n, r, sd(), lo_b, lo_c, Dt, frac, a.B) for n in grid for r in range(a.reps)], e5_task) for r in rows])
        summ = df.groupby(["N", "unit", "utype"]).agg(Dtrue=("Dtrue", "mean"), Dmax=("Dmax", "mean"), rate=("flag", "mean"),
               rate_cl=("flag_cl", "mean"), rate_marg=("flag_marg", "mean"), cover=("cover", "mean"), cover_cl=("cover_cl", "mean"),
               c_med=("c", "median"), c_cl_med=("c_cl", "median"), reps=("rep", "nunique")).reset_index()
    meta.update(B=a.B, reps=a.reps, grid=grid, seconds=time.time() - t0)
    (out / f"{ex}_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    df.to_csv(out / f"sim_{ex}_long.csv", index=False); summ.to_csv(out / f"sim_{ex}_summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"DONE {ex} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
