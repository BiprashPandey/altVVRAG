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
                   quote_match: float = 0.0) -> float:
    """Hand-weighted confidence (rec 6: localization is a gate signal).

    New kwargs default to 0/False so old 3-arg calls keep working; the
    learned scorer picks up the new columns via FEATURE_ORDER automatically.
    `quote_match` (suggestion #4) is a deterministic verifier signal, so it
    carries real weight.
    """
    entail = 0.5 if entail_prob is None else entail_prob
    base = float(0.25 * agreement + 0.25 * entail + 0.10 * heat_concentration
                 + 0.20 * float(localize_score or 0.0)
                 + 0.20 * float(quote_match or 0.0))
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
