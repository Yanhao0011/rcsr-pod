#!/usr/bin/env python3
"""Departure-based probability-scale contribution (D-PSC) for the RCSR-POD diagnostic.

Motivation (2026-09-06 finding): the between-threshold gap of the plain PSC,
    PSC_k^(t) = mean_i[ L(eta_i^(t)) - L(eta_i^(t) - beta_k^(t) x_ik) ],
is NOT zero under exact proportional odds, because the logistic link maps a common
coefficient to different probability shifts at cuts with different baseline
probabilities. The departure statistic removes that nonlinearity floor:

    D_k^(t) = mean_i[ L(eta_i^(t)) - L(eta_i^(t) - (beta_k^(t) - bbar_k) x_ik) ],
    bbar_k  = sum_t w_t beta_k^(t) / sum_t w_t,   w_t = 1 / SE(beta_k^(t))^2,

i.e. the amount by which P(y > t) would be misstated for feature k if its cut-specific
coefficient were replaced by the precision-pooled (proportional-odds) coefficient.
Under exact PO, beta_k^(t) - bbar_k -> 0 for every cut and D_k^(t) -> 0.

Summary statistics per feature:
    Dmax_k = max_t |D_k^(t)|            (maximum probability-scale departure; primary)
    Dgap_k = max_{s<t} |D_k^(s) - D_k^(t)|  (between-cut gap of the departure; secondary)
Each gets a delta-method standard error (cuts treated as independent, weights held fixed)
and a one-sided lower (1 - alpha) bound; a feature is a credible violator iff LCB > delta.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def logistic(x):
    return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-12, 1 - 1e-12)


def pooled_weights(se):
    w = 1.0 / np.maximum(np.asarray(se, dtype=float), 1e-12) ** 2
    return w / w.sum()


def departure_profile(eta, beta_k, se_k, xj, x_eval=None, cuts=None):
    """Per-cut departures for one feature.

    eta    : list over cuts of the FULL linear predictor (n,) at each cut
    beta_k : (T,) cut-specific coefficients of the feature; se_k : (T,) their SEs
    xj     : (n,) the feature column as used in the fit (removed from eta to get eta_{-k})
    x_eval : (n,) covariate values at which the departure is evaluated (default xj);
             use a constant vector for the per-unit (discrete-change) variant
    cuts   : optional subset of cut indices over which pooling and evaluation are done
             (e.g. [0, 1, 2] for the non-fatal ladder)
    Returns dict(D, se_D, psc, se_psc, bbar, w) over the selected cuts.
    """
    idx = list(range(len(eta))) if cuts is None else list(cuts)
    T = len(idx)
    beta_k = np.asarray(beta_k, dtype=float)[idx]; se_k = np.asarray(se_k, dtype=float)[idx]
    xe = xj if x_eval is None else x_eval
    w = pooled_weights(se_k); bbar = float(np.sum(w * beta_k))
    D = np.zeros(T); psc = np.zeros(T); g_full = np.zeros(T); g_pool = np.zeros(T)
    for i, t in enumerate(idx):
        e_mk = eta[t] - beta_k[i] * xj
        pf = logistic(e_mk + beta_k[i] * xe); p0 = logistic(e_mk); pp = logistic(e_mk + bbar * xe)
        psc[i] = float(np.mean(pf - p0))
        D[i] = float(np.mean(pf - pp))
        g_full[i] = float(np.mean(xe * pf * (1 - pf)))
        g_pool[i] = float(np.mean(xe * pp * (1 - pp)))
    se_psc = np.abs(g_full) * se_k
    # G[t, s] = dD_t / dbeta_s  (weights held fixed)
    G = -np.outer(g_pool, w)
    G[np.arange(T), np.arange(T)] += g_full
    se_D = np.sqrt((G ** 2) @ (se_k ** 2))
    return dict(D=D, se_D=se_D, psc=psc, se_psc=se_psc, bbar=bbar, w=w, G=G, se_k=se_k, cuts=idx)


def discrete_change_profile(eta, beta_k, se_k, xj, cuts=None):
    """Per-unit (discrete-change) departure: binary feature -> departure evaluated at
    x = 1 for every record (the scaled one-value); continuous feature -> D(x + 1 SD) - D(x).
    Returns dict(D, se_D) over the selected cuts."""
    nz = xj[xj != 0]
    is_binary = len(nz) > 0 and np.allclose(nz, nz[0])
    if is_binary:
        one = np.full_like(xj, float(nz[0]))
        prof = departure_profile(eta, beta_k, se_k, xj, x_eval=one, cuts=cuts)
        return dict(D=prof["D"], se_D=prof["se_D"], kind="binary")
    a = departure_profile(eta, beta_k, se_k, xj, x_eval=xj + 1.0, cuts=cuts)
    b = departure_profile(eta, beta_k, se_k, xj, x_eval=xj, cuts=cuts)
    G = a["G"] - b["G"]
    return dict(D=a["D"] - b["D"], se_D=np.sqrt((G ** 2) @ (a["se_k"] ** 2)), kind="continuous")


def lcb_max(D, se_D, alpha=0.05):
    """max_t |D_t| and its one-sided lower bound."""
    z = stats.norm.ppf(1 - alpha)
    D = np.asarray(D); se_D = np.asarray(se_D)
    t = int(np.argmax(np.abs(D)))
    return float(abs(D[t])), t, float(np.max(np.abs(D) - z * se_D))


def summarize(prof, alpha=0.05):
    """Dmax / Dgap / PSC-gap point estimates and one-sided lower bounds."""
    z = stats.norm.ppf(1 - alpha)
    D, sD, psc, sP = prof["D"], prof["se_D"], prof["psc"], prof["se_psc"]
    T = len(D)
    tstar = int(np.argmax(np.abs(D)))
    Dmax = float(abs(D[tstar])); Dmax_lcb = float(max(abs(D[t]) - z * sD[t] for t in range(T)))
    pairs = [(s, t) for s in range(T) for t in range(s + 1, T)]
    Dgap = max(abs(D[s] - D[t]) for s, t in pairs)
    Dgap_lcb = max(abs(D[s] - D[t]) - z * np.sqrt(sD[s] ** 2 + sD[t] ** 2) for s, t in pairs)
    Pgap = max(abs(psc[s] - psc[t]) for s, t in pairs)
    Pgap_lcb = max(abs(psc[s] - psc[t]) - z * np.sqrt(sP[s] ** 2 + sP[t] ** 2) for s, t in pairs)
    return dict(Dmax=Dmax, Dmax_cut=tstar, Dmax_lcb=Dmax_lcb, Dgap=float(Dgap), Dgap_lcb=float(Dgap_lcb),
                psc_gap=float(Pgap), psc_lcb=float(Pgap_lcb), bbar=prof["bbar"])


def departure_table(eta, B, S, Xv, names, alpha=0.05, delta=0.01, labels=None):
    """All features. eta: list over cuts of (n,) predictors; B, S: (T, p) coefficient / SE
    arrays (feature order = names); Xv: (n, p) design without intercept."""
    rows = []
    T, p = B.shape
    for j, f in enumerate(names):
        prof = departure_profile(eta, B[:, j], S[:, j], Xv[:, j])
        s = summarize(prof, alpha)
        s.update(feature=f, label=(labels or {}).get(f, f), prevalence=float(np.mean(Xv[:, j] != 0)),
                 coef_gap=float(B[:, j].max() - B[:, j].min()),
                 **{f"D{t}": prof["D"][t] for t in range(T)}, **{f"seD{t}": prof["se_D"][t] for t in range(T)},
                 credible_Dmax=bool(s["Dmax_lcb"] > delta), credible_Dgap=bool(s["Dgap_lcb"] > delta),
                 credible_psc=bool(s["psc_lcb"] > delta))
        rows.append(s)
    return pd.DataFrame(rows)


def true_departure(alphas, beta_true, X):
    """Ground-truth departure per feature from TRUE cut-specific coefficients.

    alphas: (T,) true cut intercepts; beta_true: (p, T); X: (n, p).
    The pooled coefficient uses the population analogue of the estimator's precision weights:
    w_t = 1 / [I_t^{-1}]_{jj}, the inverse of the expected-information variance of beta_j^(t)
    (expected information I_t = X'W_tX with an intercept column, evaluated at the true parameters).
    (Before 2026-09-06 the weights were the information DIAGONAL sum_i x_ij^2 lambda_i^(t), which
    ignores the correlation of x_j with the intercept and the other columns and therefore
    defined a slightly different estimand; the coverage check of the boundary design exposed it.)
    Returns (Dmax (p,), Dgap (p,)).
    """
    n, p = X.shape; T = len(alphas)
    Xm = np.column_stack([np.ones(n), X])
    eta = [alphas[t] + X @ beta_true[:, t] for t in range(T)]
    lam = [logistic(e) * (1 - logistic(e)) for e in eta]
    ridge = 1e-8 * np.eye(Xm.shape[1])   # keeps a column that is all-zero in a small replication invertible (weight -> 0)
    invdiag = np.array([np.diag(np.linalg.inv((Xm * lam[t][:, None]).T @ Xm + ridge))[1:] for t in range(T)])   # (T, p)
    Dmax = np.zeros(p); Dgap = np.zeros(p)
    for j in range(p):
        xj = X[:, j]
        w = 1.0 / np.maximum(invdiag[:, j], 1e-300); w = w / w.sum()
        bbar = float(np.sum(w * beta_true[j]))
        D = np.array([float(np.mean(logistic(eta[t]) - logistic(eta[t] - (beta_true[j, t] - bbar) * xj))) for t in range(T)])
        Dmax[j] = np.abs(D).max()
        Dgap[j] = max(abs(D[s] - D[t]) for s in range(T) for t in range(s + 1, T))
    return Dmax, Dgap
