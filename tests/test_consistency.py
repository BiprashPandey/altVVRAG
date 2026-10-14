"""Verification tests: consistency (fixed lists), NLI (live CPU model), features."""
from vvrag.reader.claims import Claim
from vvrag.schemas import ReaderSample
from vvrag.verify.consistency import agreement
from vvrag.verify.features import FEATURE_ORDER, build_features, to_vector
from vvrag.verify.nli import FakeNLI, aggregate_entailment, label_order
from vvrag.verify.vlm_crop_check import check_claims, vlm_crop_supports


def _s(i, ans):
    return ReaderSample(sample_id=i, answer=ans,
                        claims=[Claim(text=ans)], raw=ans)


def test_agreement_fixed_lists():
    a, d = agreement([_s(0, "42"), _s(1, "42"), _s(2, "42")])
    assert (a, d) == (1.0, 1)
    a, d = agreement([_s(0, "42 units"), _s(1, "42 units!"), _s(2, "oranges")])
    assert a == 2 / 3 and d == 3  # raw strings differ, normalized answers match x2
    a, d = agreement([_s(0, "revenue was 42 million"), _s(1, "revenue 42 million")])
    assert a == 1.0  # F1 rule
    assert agreement([]) == (0.0, 0)


def test_fake_nli_overlap():
    nli = FakeNLI()
    assert nli.entail_probs("the chart shows 42 units", ["42 units shown"]) == [2 / 3]


def test_nli_label_order_and_live_inference():
    from vvrag.verify.nli import NLIEntailer

    nli = NLIEntailer()
    nli.load()
    assert label_order(nli._model) == {"contradiction": 0, "entailment": 1, "neutral": 2}
    probs = nli.entail_probs(
        "The bar chart shows total revenue of 42 million dollars in 2024.",
        ["Revenue was 42 million dollars.", "The company sells only apples."],
    )
    assert probs[0] > 0.9, probs
    assert probs[1] < 0.2, probs
    # windowing: entailing sentence buried in neutral filler still found
    long_premise = " ".join([f"Filler sentence number {i} about nothing." for i in range(60)])
    long_premise += " The bar chart shows total revenue of 42 million dollars."
    w = nli.entail_probs(long_premise, ["Revenue was 42 million dollars."])
    assert w[0] > 0.9, w
    assert aggregate_entailment([0.9, 0.4]) == (0.4, 0.65)
    assert aggregate_entailment([]) == (0.5, 0.5)


def test_features_missing_entail_encoding():
    f = build_features(agreement=1.0, entail_min=None, retrieval_margin=0.5,
                       heat_concentration=0.7, n_claims=2, crop_has_text=False)
    assert f.entail_prob == 0.5 and not f.crop_has_text
    assert to_vector(f) == [1.0, 0.5, 0.5, 0.7, 2.0, 0.0]
    assert FEATURE_ORDER == ["agreement", "entail_prob", "retrieval_margin",
                             "heat_concentration", "n_claims", "crop_has_text"]


def test_vlm_check_disabled_and_mock():
    from vvrag.reader.client import SequenceMockReader

    assert check_claims(None, None, ["c"], enabled=False) == [None]
    r = SequenceMockReader(['{"verdict": "yes"} yes'])
    assert vlm_crop_supports(r, "img.png", "claim") is True
    r2 = SequenceMockReader(["no"])
    assert vlm_crop_supports(r2, "img.png", "claim") is False
    r3 = SequenceMockReader(["maybe"])
    assert vlm_crop_supports(r3, "img.png", "claim") is None
