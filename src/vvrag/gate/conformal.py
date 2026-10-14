"""Conformal abstention gate (Section 8.2, contribution C4).

Goal: tau such that among ANSWERED questions (score >= tau) the error rate is
<= alpha with probability >= 1 - delta (exchangeable calib/test).

Method: risk-controlling threshold selection with fixed-sequence testing
(Learn-then-Test style): walk candidates strictest-first; tau* is the
smallest (loosest) passing threshold; STOP at the first failure (no
multiplicity correction); if even the strictest fails, tau = +inf.

DEVIATION from Section 8.2 step 2 (recorded in docs/decisions.md): the guide
says candidates = sorted unique calibration scores, descending. That is
degenerate: the strictest candidate (tau = max score) accepts ~1 point, and
the Clopper-Pearson bound with n_acc=1 is 1-delta = 0.9 > alpha for any sane
alpha, so the walk ALWAYS stops immediately with tau=+inf (always abstain).
We instead walk a coarse fixed grid (default 0..1 step 0.05), where the
strictest thresholds accept enough points for the bound to have power.
Set grid=None to recover the literal (degenerate) spec behavior.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_GRID = np.arange(0.0, 1.0 + 1e-9, 0.05)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided (1-delta) upper confidence bound for a Binomial rate."""
    from scipy.stats import beta

    if n <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if k < 0:
        raise ValueError("k must be >= 0")
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def select_tau(
    scores: np.ndarray,
    errors: np.ndarray,
    alpha: float = 0.10,
    delta: float = 0.10,
    grid: np.ndarray | None = DEFAULT_GRID,
    min_n: int = 30,
) -> tuple[float, dict]:
    """Returns (tau_star, info). tau_star = +inf means always abstain.

    Grid points accepting fewer than min_n calibration questions are skipped
    (neither tested nor selected): they are coverage-useless, and testing them
    would burn the walk's power on noise. Skipping preserves the
    fixed-sequence argument (fewer tested hypotheses, same stop rule).
    """
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=int)
    if scores.shape != errors.shape or scores.size == 0:
        raise ValueError("scores/errors must be non-empty and same shape")
    cands = sorted(set(scores.tolist()), reverse=True) if grid is None else sorted(
        [float(g) for g in grid], reverse=True
    )
    info: dict = {"candidates": [], "n_calib": int(scores.size)}
    tau_star = math.inf
    for tau in cands:
        accepted = scores >= tau
        n_acc = int(accepted.sum())
        if n_acc < min_n:
            info["candidates"].append({"tau": tau, "n_acc": n_acc, "k_err": None,
                                       "ucb": None, "skipped": True})
            continue  # too few answers to certify; neither stop nor select
        k_err = int(errors[accepted].sum())
        ucb = clopper_pearson_upper(k_err, n_acc, delta)
        info["candidates"].append({"tau": tau, "n_acc": n_acc, "k_err": k_err, "ucb": ucb})
        if ucb <= alpha:
            tau_star = float(tau)  # keep loosening while it passes
        else:
            break  # fixed-sequence: stop at first failure
    info["tau_star"] = tau_star
    return tau_star, info


def split_conformal_tau(
    correct_scores: np.ndarray, delta_prime: float = 0.10
) -> float:
    """Secondary method: tau so a correct test answer is accepted w.p. >= 1-delta'.

    Nonconformity of correct answers: 1 - score. q_hat = ceil((n+1)(1-dp))/n
    quantile; accept iff score >= 1 - q_hat.
    """
    s = np.sort(np.asarray(correct_scores, dtype=float))
    n = s.size
    if n == 0:
        return math.inf
    k = min(n - 1, int(math.ceil((n + 1) * (1.0 - delta_prime))) - 1)
    k = max(0, k)
    return float(s[k])


def apply_gate(score: float, tau: float) -> bool:
    """Returns True = answer, False = abstain. tau=+inf always abstains."""
    return float(score) >= float(tau)


def accepted_error(scores: np.ndarray, errors: np.ndarray, tau: float) -> tuple[float, float]:
    """Returns (error_rate_among_accepted, coverage). No accepts -> (0.0, 0.0)."""
    accepted = np.asarray(scores) >= tau
    cov = float(accepted.mean())
    if not accepted.any():
        return 0.0, cov
    return float(np.asarray(errors)[accepted].mean()), cov
