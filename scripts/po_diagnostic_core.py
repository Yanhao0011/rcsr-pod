"""Shared core for the RC-PO proportional-odds diagnostic.

Two reusable primitives consumed by both the simulation harness
(``sim_po_diagnostic.py``) and the production GOL pipeline
(``build_gol_ordered_models.py``), so the numerical routines cannot drift apart:

* ``firth_logit`` — Firth (1993) penalized logistic regression (Jeffreys-prior
  bias reduction), giving finite estimates under quasi/complete separation.
  Replaces the ad-hoc ``|beta| < 10`` stability band-aid.
* ``relevance_lcb`` — the one-sided lower confidence bound on the maximum
  between-threshold coefficient gap, the test statistic behind the
  relevance-calibrated PO decision (flag iff this bound exceeds delta).
"""

from __future__ import annotations

import numpy as np
from scipy import stats



def _xtwx(X, w, chunk=60000):
    """X' diag(w) X accumulated in row chunks (avoids an n x p temporary)."""
    out = np.zeros((X.shape[1], X.shape[1]))
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]; out += (Xc * w[i:i + chunk, None]).T @ Xc
    return out


def _leverage(X, w, cov, chunk=60000):
    """diagonal of W^1/2 X cov X' W^1/2, in row chunks"""
    h = np.empty(X.shape[0])
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]; h[i:i + chunk] = np.einsum("ij,ij->i", Xc @ cov, Xc) * w[i:i + chunk]
    return h

def _penalized_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                      ridge: float) -> tuple[float, np.ndarray]:
    """Firth penalized log-likelihood l*(b) = l(b) + 0.5*log|I(b)| and the
    ridge-stabilized information matrix I(b). Overflow-safe via logaddexp."""
    eta = X @ beta
    # log-likelihood = sum_i [y_i*eta_i - log(1 + exp(eta_i))], stable form.
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    pr = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    w = pr * (1 - pr)
    info = _xtwx(X, w) + ridge * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(info)
    pen = 0.5 * logdet if sign > 0 else -np.inf
    return ll + pen, info


def firth_logit(X: np.ndarray, y: np.ndarray, *, max_iter: int = 200,
                tol: float = 1e-7, ridge: float = 1e-6
                ) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit a Firth-penalized logistic regression by damped modified-score IRLS.

    Maximizes the penalized log-likelihood l*(b) = l(b) + 0.5*log|I(b)|
    (Firth 1993; Heinze & Schemper 2002). The penalty guarantees a FINITE
    maximizer even under quasi/complete separation, where ordinary MLE diverges
    to ±inf. Convergence to that finite root is enforced by **step-halving**
    against the penalized log-likelihood (as in logistf / brglm2); without it a
    raw Newton step overshoots and blows up on sparse one-hot designs. The
    information matrix is ridge-stabilized (``ridge``·I) so it stays positive
    definite when one-hot columns are near-collinear.

    Parameters
    ----------
    X : (n, p) design matrix INCLUDING an intercept column.
    y : (n,) binary {0,1} response.

    Returns
    -------
    beta : (p,) penalized coefficients (finite).
    se   : (p,) standard errors from the penalized Fisher information.
    ok   : bool, whether the iteration converged within ``max_iter``.
    """
    p = X.shape[1]
    beta = np.zeros(p)
    ll_old, info = _penalized_loglik(X, y, beta, ridge)
    ok = False
    for _ in range(max_iter):
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(info)
        eta = X @ beta
        pr = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        w = pr * (1 - pr)
        h = _leverage(X, w, cov)                            # hat leverages
        u = X.T @ (y - pr + h * (0.5 - pr))                # Firth modified score
        step = cov @ u
        # Step-halving: accept the largest fraction of the Newton step that does
        # not decrease the penalized log-likelihood. Guarantees monotone ascent.
        lam = 1.0
        accepted = False
        for _ in range(40):
            cand = beta + lam * step
            ll_new, info_new = _penalized_loglik(X, y, cand, ridge)
            if np.isfinite(ll_new) and ll_new >= ll_old - 1e-9:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break                                           # at (or near) the optimum
        beta = cand
        info = info_new
        converged = abs(ll_new - ll_old) < tol * (abs(ll_old) + tol)
        ll_old = ll_new
        if converged:
            ok = True
            break
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    return beta, se, ok


def ame_shift_table(firth_full: dict, X, *, alpha_sig: float = 0.05,
                    ci_method: str = "delta", n_boot: int = 400,
                    seed: int = 20260602) -> dict:
    """Population-attributable probability-shift PO effect size (the PSC, the
    probability-scale contribution).

    For each threshold t and covariate j, the average contribution of j to the
    fitted cumulative probability P(Y>t) across the realized sample is

        shift_{j,t} = (1/n) sum_i [ Lambda(eta_i) - Lambda(eta_i - beta_{j,t} x_{ij}) ]

    where eta_i is the full Firth linear predictor at threshold t. The PO effect
    size is the maximum between-threshold gap  Delta_j = max_{s<t} |shift_{j,s} -
    shift_{j,t}|, on the probability scale. Unlike the raw-coefficient gap, this
    quantity scales with a covariate's *population footprint*: a sparse one-hot
    level that separates has a large coefficient but contributes to only a tiny
    fraction of observations, so its shift (<= its prevalence) is small and it is
    NOT flagged. Dense covariates (e.g. vehicle count) that genuinely move the
    predicted probability across the whole sample retain a large shift.

    Parameters
    ----------
    firth_full : {t: (beta, se)} with ``beta``/``se`` pandas Series indexed by
        the feature columns plus a leading ``'const'`` intercept entry.
    X : (n, p) feature DataFrame (no intercept column), same columns as ``beta``
        minus ``'const'``.

    The one-sided lower confidence bound on ``Delta_j`` is, by default, the
    delta-method bound maxed over threshold pairs (``ci_method='delta'``): flag
    when *any* pair's lower bound ``|shift_s - shift_t| - z_{1-alpha} * se_gap``
    exceeds delta. This is conservative and well-justified for a max-of-gaps
    statistic. A percentile coefficient bootstrap is also available
    (``ci_method='bootstrap'``) but is NOT used as the default: bootstrapping the
    *maximum* gap is anti-conservative here because the max is already selected at
    the point estimate, so the resampled max distribution is biased upward and the
    lower quantile sits too high (an uncorrected selection-bias artefact). Both
    keep the cross-coefficient independence approximation also used by the Brant
    test.

    Returns
    -------
    dict feature -> (ame_max_gap, ame_lcb). RC-PO(PSC) flags a credible violation
    iff ``ame_lcb > delta``.
    """
    import numpy as _np
    ts = sorted(firth_full)
    feats = list(X.columns)
    Xv = X.values
    z = stats.norm.ppf(1 - alpha_sig)
    rng = _np.random.default_rng(seed)

    # Precompute per-threshold linear predictor and probabilities.
    per_t = {}
    for t in ts:
        beta, se = firth_full[t]
        const = float(beta.get("const", 0.0))
        bvec = beta.reindex(feats).to_numpy(dtype=float)
        eta = const + Xv @ bvec
        p = _np.clip(1.0 / (1.0 + _np.exp(-eta)), 1e-12, 1 - 1e-12)
        per_t[t] = (bvec, se, eta, p)

    def _logistic(x):
        return _np.clip(1.0 / (1.0 + _np.exp(-x)), 1e-12, 1 - 1e-12)

    out = {}
    for jc, f in enumerate(feats):
        xj = Xv[:, jc]
        point_shifts = []
        delta_ses = []
        boot_shifts = []          # list of (n_boot,) draws per finite threshold
        ok_thresholds = []
        for t in ts:
            bvec, se, eta, p = per_t[t]
            bj = bvec[jc]
            se_bj = float(se.get(f, _np.nan))
            eta0 = eta - bj * xj                       # remove covariate j's contribution
            p0 = _logistic(eta0)
            shift = float(_np.mean(p - p0))
            point_shifts.append(shift)
            # delta-method gradient dPSC/dbeta_j. PSC = mean_i[Lambda(eta_i) -
            # Lambda(eta_i - beta_j x_ij)]; only the FIRST term depends on beta_j
            # (eta contains beta_j x_ij, the stripped term does not), so
            # d/dbeta_j PSC = mean_i[x_ij * lambda(eta_i)] — evaluate at the FULL
            # predictor p = Lambda(eta), not the stripped p0.
            g = float(_np.mean(xj * p * (1 - p)))      # delta-method gradient (full predictor)
            delta_ses.append(abs(g) * se_bj if _np.isfinite(se_bj) else _np.nan)
            if ci_method == "bootstrap" and _np.isfinite(se_bj) and se_bj > 0:
                # Resample beta_j; recompute shift for each draw (vectorized over draws).
                bdraw = bj + se_bj * rng.standard_normal(n_boot)          # (B,)
                eta_b = eta0[:, None] + xj[:, None] * bdraw[None, :]      # (n, B)
                p_b = _logistic(eta_b)
                p0_b = _logistic(eta0)[:, None]
                boot_shifts.append((p_b - p0_b).mean(axis=0))            # (B,)
                ok_thresholds.append(True)
            else:
                ok_thresholds.append(_np.isfinite(shift))
        ps = _np.array(point_shifts, dtype=float)
        finite = _np.isfinite(ps)
        if finite.sum() < 2:
            out[f] = (0.0, -_np.inf)
            continue
        s = ps[finite]
        max_gap = max(abs(s[i] - s[j]) for i in range(len(s)) for j in range(i + 1, len(s)))

        if ci_method == "bootstrap" and len(boot_shifts) >= 2:
            B = _np.column_stack(boot_shifts)                            # (n_boot, n_finite_t)
            # per-draw max between-threshold gap
            gaps = _np.zeros(B.shape[0])
            for i in range(B.shape[1]):
                for j in range(i + 1, B.shape[1]):
                    gaps = _np.maximum(gaps, _np.abs(B[:, i] - B[:, j]))
            lcb = float(_np.quantile(gaps, alpha_sig))                   # one-sided lower bound
        else:
            dse = _np.array(delta_ses, dtype=float)[finite]
            max_lcb = -_np.inf
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    gap = abs(s[i] - s[j])
                    se_gap = _np.sqrt(dse[i] ** 2 + dse[j] ** 2)
                    max_lcb = max(max_lcb, gap - z * se_gap)
            lcb = max_lcb
        out[f] = (float(max_gap), float(lcb))
    return out


def relevance_lcb(coefs: np.ndarray, ses: np.ndarray, *,
                  alpha_sig: float = 0.05) -> tuple[float, float]:
    """One-sided lower confidence bound on the max between-threshold gap.

    Given a variable's threshold-specific coefficients and standard errors,
    return ``(max_gap, max_lcb)`` where

        max_gap = max_{s<t} |b_s - b_t|
        max_lcb = max_{s<t} ( |b_s - b_t| - z_{1-alpha} * sqrt(se_s^2 + se_t^2) )

    The RC-PO rule flags a *credible* PO violation iff ``max_lcb > delta``. The
    se-inflated, separation-affected pairs drive their own lower bound below
    delta automatically, so no magnitude cap is needed. NaN/non-finite entries
    are dropped; returns ``(0.0, -inf)`` if fewer than two finite thresholds.
    """
    coefs = np.asarray(coefs, dtype=float)
    ses = np.asarray(ses, dtype=float)
    finite = np.isfinite(coefs) & np.isfinite(ses) & (ses > 0)
    c = coefs[finite]
    s = ses[finite]
    if len(c) < 2:
        return 0.0, -np.inf
    z = stats.norm.ppf(1 - alpha_sig)
    max_gap = 0.0
    max_lcb = -np.inf
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            gap = abs(c[i] - c[j])
            se_gap = np.sqrt(s[i] ** 2 + s[j] ** 2)
            max_gap = max(max_gap, gap)
            max_lcb = max(max_lcb, gap - z * se_gap)
    return float(max_gap), float(max_lcb)
