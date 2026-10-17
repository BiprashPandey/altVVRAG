"""Gate tests (Section 8, Phase 7 acceptance incl. empirical guarantee check)."""
import numpy as np

from vvrag.gate.conformal import (
    accepted_error,
    apply_gate,
    clopper_pearson_upper,
    select_tau,
    split_conformal_tau,
)
from vvrag.gate.fixed import apply_fixed
from vvrag.gate.scorer import LogisticScorer, features_to_matrix, hand_weighted
from vvrag.verify.features import build_features


def test_cp_bound_sanity():
    assert clopper_pearson_upper(0, 100, 0.10) < 0.05
    assert clopper_pearson_upper(100, 100, 0.10) == 1.0
    assert clopper_pearson_upper(0, 0, 0.10) == 0.0


def test_perfect_separation_accepts_all_correct():
    rng = np.random.default_rng(0)
    good = rng.uniform(0.8, 1.0, 100)
    bad = rng.uniform(0.0, 0.3, 100)
    scores = np.concatenate([good, bad])
    errors = np.array([0] * 100 + [1] * 100)
    tau, _ = select_tau(scores, errors, alpha=0.10, delta=0.10)
    assert np.isfinite(tau)
    err, cov = accepted_error(scores, errors, tau)
    assert err == 0.0
    assert cov == 0.5  # all + only correct accepted


def test_random_scores_abstain_heavily():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 200)
    errors = rng.integers(0, 2, 200)
    tau, _ = select_tau(scores, errors, alpha=0.05, delta=0.10)
    _, cov = accepted_error(scores, errors, tau)
    assert np.isinf(tau) or cov < 0.2


def test_empirical_guarantee_check():
    """~200 random calib/test splits; violation fraction ~= <= delta."""
    rng = np.random.default_rng(2)
    N = 1000
    y = rng.integers(0, 2, N)  # 1 = correct
    scores = np.where(y == 1, rng.beta(8, 2, N), rng.beta(2, 8, N))
    errors = 1 - y
    alpha, delta, viol, counted = 0.10, 0.10, 0, 0
    for _ in range(200):
        idx = rng.permutation(N)
        cal, tst = idx[:500], idx[500:]
        tau, _ = select_tau(scores[cal], errors[cal], alpha, delta)
        n_acc = int((scores[tst] >= tau).sum())
        if n_acc < 30:  # degenerate split: gate answers ~nobody; skip
            continue
        err, _ = accepted_error(scores[tst], errors[tst], tau)
        counted += 1
        viol += err > alpha
    assert counted > 150, counted
    assert viol / counted <= delta + 0.05, (viol, counted)


def test_split_conformal_variant():
    rng = np.random.default_rng(3)
    tau = split_conformal_tau(rng.uniform(0.6, 1.0, 100), delta_prime=0.10)
    assert 0.0 <= tau <= 1.0
    assert split_conformal_tau(np.array([])) == float("inf")


def test_fixed_and_gate():
    assert apply_fixed(0.6, 0.5) and not apply_fixed(0.4, 0.5)
    assert apply_gate(0.9, 0.5) and not apply_gate(0.1, float("inf"))


def test_scorers():
    # rec 6: localization is a gate signal; rec 4 adds quote_match
    # weights 0.25/0.25/0.10/0.20/0.20 (+/-0.05 evidence)
    import pytest as _pt
    assert hand_weighted(1.0, 1.0, 1.0) == _pt.approx(0.55)  # no localize/quote, evidence invalid
    assert hand_weighted(1.0, 1.0, 1.0, 1.0, True, 1.0) == _pt.approx(1.0)  # perfectly localized
    assert hand_weighted(0.0, None, 0.0) == _pt.approx(0.075)  # missing entail -> 0.5*0.25 - 0.05
    assert hand_weighted(0.0, None, 0.0, 1.0, True, 1.0) > 0.5  # localize+quote rescue low score
    rng = np.random.default_rng(4)
    feats = [build_features(agreement=rng.random(), entail_min=rng.random(),
                            retrieval_margin=rng.random(), heat_concentration=rng.random(),
                            n_claims=2, crop_has_text=True) for _ in range(60)]
    X = features_to_matrix(feats)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # separable by construction
    sc = LogisticScorer().fit(X, y)
    s = sc.predict_score(X)
    assert ((s > 0.5).astype(int) == y).mean() > 0.9
    import tempfile, os

    p = os.path.join(tempfile.mkdtemp(), "scorer.pkl")
    sc.save(p)
    assert (LogisticScorer.load(p).predict_score(X) == s).all()
