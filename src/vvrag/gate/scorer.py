"""Confidence scorer (Section 8.1): logistic regression + hand-weighted baseline.

Input Features -> score in [0, 1] (higher = more likely correct).
FEATURE_ORDER (verify/features.py) is the column order. Label = 1 if primary
answer correct (token-F1 >= threshold, default 0.5), else 0.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from vvrag.verify.features import to_vector


def hand_weighted(agreement: float, entail_prob: float | None,
                   heat_concentration: float,
                   localize_score: float = 0.0,
                   evidence_valid: bool = False,
                   quote_match: float = 0.0,
                   examiner_ran: bool | float | None = None,
                   examiner_supported: float | None = None,
                   quote_seen_match: float | None = None,
                   details_ok: float | None = None) -> float:
    """Hand-weighted confidence (rec 6: localization is a gate signal).

    Legacy 6-arg calls keep the exact old formula. When any Run 9 examiner
    kwarg is provided (guideline s4.7: the examiner's verdict is the strongest
    feature), the 1.0-normalized formula below is used instead.
    """
    entail = 0.5 if entail_prob is None else entail_prob
    if (examiner_ran is None and examiner_supported is None
            and quote_seen_match is None and details_ok is None):
        base = float(0.25 * agreement + 0.25 * entail + 0.10 * heat_concentration
                     + 0.20 * float(localize_score or 0.0)
                     + 0.20 * float(quote_match or 0.0))
        if evidence_valid:
            base += 0.05
        else:
            base -= 0.05
        return float(max(0.0, min(1.0, base)))
    ran = 1.0 if examiner_ran else 0.0
    ex = 0.5 if examiner_supported is None else float(examiner_supported)
    qs = 0.5 if quote_seen_match is None else float(quote_seen_match)
    det = 1.0 if details_ok is None else float(details_ok)
    base = float(0.15 * agreement + 0.10 * entail + 0.05 * heat_concentration
                 + 0.10 * float(localize_score or 0.0)
                 + 0.15 * float(quote_match or 0.0)
                 + 0.30 * ex * ran
                 + 0.10 * qs * ran
                 + 0.05 * det)
    if evidence_valid:
        base += 0.05
    else:
        base -= 0.05
    return float(max(0.0, min(1.0, base)))



class LogisticScorer:
    def __init__(self) -> None:
        self._scaler = None
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticScorer":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._scaler = StandardScaler().fit(X)
        self._clf = LogisticRegression(max_iter=2000).fit(self._scaler.transform(X), y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None or self._scaler is None:
            raise RuntimeError("scorer not fitted")
        return self._clf.predict_proba(self._scaler.transform(np.asarray(X, dtype=float)))[:, 1]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "clf": self._clf}, f)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticScorer":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        sc = cls()
        sc._scaler, sc._clf = blob["scaler"], blob["clf"]
        return sc


def features_to_matrix(features) -> np.ndarray:
    return np.array([to_vector(f) for f in features], dtype=float)
