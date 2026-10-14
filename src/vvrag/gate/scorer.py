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
                  heat_concentration: float) -> float:
    entail = 0.5 if entail_prob is None else entail_prob
    return float(0.4 * agreement + 0.4 * entail + 0.2 * heat_concentration)


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
