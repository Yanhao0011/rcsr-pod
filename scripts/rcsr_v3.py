# -*- coding: utf-8 -*-
"""RCSR-POD v3 core: coding-invariant probability-scale departure.

For feature (or variable group) k at cumulative cut t, with Firth linear predictor eta_i^(t), cut-specific
coefficient(s) b_t and the pooled (proportional-odds) reference bbar = sum_s w_s b_s (precision weights),

    D_k^(t) = (1/N) sum_i | L(eta_i) - L(eta_i - (b_t - bbar)' x_i + a_t) |,

where the intercept shift a_t is re-profiled so that the sample-mean fitted P(y > t) is unchanged:
    (1/N) sum_i L(eta_i - (b_t - bbar)' x_i + a_t) = (1/N) sum_i L(eta_i).
Properties: D = 0 iff b_t = bbar (for non-degenerate x); invariant to any affine recoding of x (location
shifts and reference-level swaps are absorbed by a_t, scale by b); measures how much a common coefficient
would redistribute the fitted threshold probabilities across records once the cut's marginal rate is kept.
Delta_k = max_t D_k^(t).

Pooling: generalized-least-squares weights over cuts (gls_weights), so grouped units are exactly invariant to
the choice of reference level. Inference: parametric bootstrap from the joint (stacked) Firth coefficient
covariance. One-sided max-t over the selected cuts A = {t : Dhat_t >= max_s Dhat_s - kappa_N se_t}:
    c = q_{1-alpha}( max_{t in A} (D*_t - Dhat_t)/se_t ),  LCB = max_{t in A} (Dhat_t - c se_t),
    p = P*( max_{t in A} (D*_t - Dhat_t)/se_t >= max_{t in A} (Dhat_t - delta)/se_t ).
The marginal (per-cut, unadjusted) bound uses z_{1-alpha} in place of c.
"""
from __future__ import annotations
import numpy as np
import numba as nb
from scipy import stats

ALPHA = 0.05
Z = float(stats.norm.ppf(1 - ALPHA))


@nb.njit(cache=True, fastmath=False)
def _lg(v):
    if v >= 0:
        e = np.exp(-v); return 1.0 / (1.0 + e)
    e = np.exp(v); return e / (1.0 + e)


@nb.njit(cache=True)
def _dinv_core(eta, XG, d):
    """eta (n,), XG (n,g), d (g,) coefficient differences b_t - bbar -> re-profiled mean |L(eta) - L(eta - XG d + a)|."""
    n = eta.shape[0]; g = XG.shape[1]
    sh = np.empty(n); base = np.empty(n)
    target = 0.0
    for i in range(n):
        s = 0.0
        for k in range(g):
            s += XG[i, k] * d[k]
        sh[i] = s; base[i] = _lg(eta[i]); target += base[i]
    target /= n
    a = 0.0
    for it in range(40):
        gsum = 0.0; h = 0.0
        for i in range(n):
            p = _lg(eta[i] - sh[i] + a); gsum += p; h += p * (1.0 - p)
        gsum = gsum / n - target; h /= n
        if h <= 1e-300: break
        step = gsum / h
        if step > 5.0: step = 5.0
        if step < -5.0: step = -5.0
        a -= step
        if abs(step) < 1e-11: break
    s = 0.0
    for i in range(n):
        s += abs(base[i] - _lg(eta[i] - sh[i] + a))
    return s / n


@nb.njit(cache=True, parallel=True)
def dinv_draws(etaT, XG, dB):
    """etaT (bs, n) linear predictors of bs draws at one cut; XG (n, g); dB (bs, g) -> (bs,) departures."""
    bs = etaT.shape[0]
    out = np.empty(bs)
    for b in nb.prange(bs):
        out[b] = _dinv_core(etaT[b], XG, dB[b])
    return out


def as_group(X_k):
    X_k = np.asarray(X_k, dtype=np.float64)
    return np.ascontiguousarray(X_k[:, None] if X_k.ndim == 1 else X_k)


def gls_weights(V):
    """V (T, g, g): per-cut covariance blocks of a unit's cut-specific coefficients. Returns matrix weights
    W (T, g, g) with W_t = (sum_s V_s^-1)^-1 V_t^-1, so that bbar = sum_t W_t b_t is the generalized-least-squares
    (minimum-distance) pooled coefficient. For one column this is the precision weighting; for a group it is
    equivariant under any linear reparameterisation of the columns (e.g. a change of reference level)."""
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None, None]
    T, g, _ = V.shape
    ridge = 1e-12 * max(float(np.mean([np.trace(v) for v in V])) / g, 1e-300)
    Vi = np.array([np.linalg.inv(V[t] + ridge * np.eye(g)) for t in range(T)])
    S = np.linalg.inv(Vi.sum(axis=0))
    return np.einsum("gh,thk->tgk", S, Vi)


def as_matrix_weights(w, T, g):
    """accept matrix weights (T,g,g) or column-wise weights (T,g)/(T,) -> (T,g,g)"""
    w = np.asarray(w, dtype=float)
    if w.ndim == 3:
        return w
    w = w.reshape(T, g)
    return np.einsum("tg,gh->tgh", w, np.eye(g))


def pooled(W, B_k):
    """W (T,g,g); B_k (T,g) -> (g,); or B_k (T,g,bs) -> (g,bs)."""
    return np.einsum("tgh,th...->g...", W, B_k)


def dinv_point(eta_list, B_k, w, X_k):
    """Point profile. eta_list: list over cuts of (n,); B_k (T,) or (T,g); w matrix weights (T,g,g) from
    gls_weights (column-wise weights are accepted for single columns); X_k (n,) or (n,g). Returns D (T,), bbar."""
    T = len(eta_list)
    B_k = np.asarray(B_k, dtype=float).reshape(T, -1); g = B_k.shape[1]
    W = as_matrix_weights(w, T, g)
    bbar = pooled(W, B_k); XG = as_group(X_k)
    D = np.array([_dinv_core(np.ascontiguousarray(e, dtype=np.float64), XG, np.ascontiguousarray(B_k[t] - bbar))
                  for t, e in enumerate(eta_list)])
    return D, bbar


def precision_weights(se):
    w = 1.0 / np.maximum(np.asarray(se, dtype=float), 1e-12) ** 2
    return w / w.sum(axis=0)


def kappa_n(n):
    """cut-selection tuning constant (Andrews and Soares 2010, BIC-type choice): sqrt(log N)"""
    return float(np.sqrt(np.log(max(n, 3))))


def maxt_onesided(Dhat, Db, delta, alpha=ALPHA, kappa=None):
    """Dhat (T,), Db (T, Bd). One-sided simultaneous max-t over cuts, with optional cut selection: only cuts whose
    departure is within kappa bootstrap standard deviations of the largest one enter the max (a cut far below the
    maximum cannot be the maximiser, but a near-zero cut's folded, right-skewed bootstrap distribution would inflate
    the critical value). The critical value is an order statistic (method='higher'), so LCB > delta iff p < alpha
    up to bootstrap discreteness. The marginal bound uses z_{1-alpha} over all cuts."""
    se = Db.std(axis=1, ddof=1); ok = se > 0
    if not ok.any():
        return dict(lcb=np.nan, lcb_marg=np.nan, c=np.nan, p=1.0, se=se, ncut=0)
    Dm = float(np.max(Dhat[ok]))
    sel = ok.copy()
    if kappa is not None:
        sel = ok & (Dhat >= Dm - kappa * np.where(ok, se, np.inf))
    tmax = np.max((Db[sel] - Dhat[sel, None]) / se[sel, None], axis=0)
    c = float(np.quantile(tmax, 1 - alpha, method="higher"))
    lcb = float(np.max(Dhat[sel] - c * se[sel]))
    lcb_m = float(np.max(Dhat[ok] - Z * se[ok]))
    Tobs = float(np.max((Dhat[sel] - delta) / se[sel]))
    return dict(lcb=lcb, lcb_marg=lcb_m, c=c, p=float(np.mean(tmax >= Tobs)), se=se, ncut=int(sel.sum()))


def multitest(p, alpha=ALPHA):
    p = np.asarray(p, dtype=float); m = len(p); order = np.argsort(p); ps = p[order]
    holm = np.zeros(m, dtype=bool)
    for k in range(m):
        if ps[k] <= alpha / (m - k): holm[order[k]] = True
        else: break
    bh = np.zeros(m, dtype=bool); below = np.where(ps <= alpha * np.arange(1, m + 1) / m)[0]
    if len(below): bh[order[:below.max() + 1]] = True
    return holm, bh


# ---------------------------------------------------------------- record-weighted variant (design-weighted sensitivity)
@nb.njit(cache=True)
def _dinv_core_w(eta, XG, d, wt):
    """As _dinv_core with record weights wt (sum normalised inside): weighted mean fitted rate preserved and
    weighted mean absolute change."""
    n = eta.shape[0]; g = XG.shape[1]
    sh = np.empty(n); base = np.empty(n)
    W = 0.0; target = 0.0
    for i in range(n):
        s = 0.0
        for k in range(g):
            s += XG[i, k] * d[k]
        sh[i] = s; base[i] = _lg(eta[i]); target += wt[i] * base[i]; W += wt[i]
    target /= W
    a = 0.0
    for it in range(40):
        gsum = 0.0; h = 0.0
        for i in range(n):
            p = _lg(eta[i] - sh[i] + a); gsum += wt[i] * p; h += wt[i] * p * (1.0 - p)
        gsum = gsum / W - target; h /= W
        if h <= 1e-300: break
        step = gsum / h
        if step > 5.0: step = 5.0
        if step < -5.0: step = -5.0
        a -= step
        if abs(step) < 1e-11: break
    s = 0.0
    for i in range(n):
        s += wt[i] * abs(base[i] - _lg(eta[i] - sh[i] + a))
    return s / W


@nb.njit(cache=True, parallel=True)
def dinv_draws_w(etaT, XG, dB, wt):
    bs = etaT.shape[0]
    out = np.empty(bs)
    for b in nb.prange(bs):
        out[b] = _dinv_core_w(etaT[b], XG, dB[b], wt)
    return out
