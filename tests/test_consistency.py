"""Verification tests: consistency (fixed lists), NLI (live CPU model), features."""
from vvrag.reader.claims import Claim
from vvrag.schemas import ReaderSample
from vvrag.verify.consistency import agreement
from vvrag.verify.features import FEATURE_ORDER, build_features, to_vector
from vvrag.verify.nli import FakeNLI, aggregate_entailment, label_order, minimal_entailing_span
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
    assert to_vector(f) == [1.0, 0.5, 0.5, 0.7, 2.0, 0.0, 0.0, 0.0, 0.0,
                            1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0]
    assert FEATURE_ORDER == ["agreement", "entail_prob", "retrieval_margin",
                             "heat_concentration", "n_claims", "crop_has_text",
                             "localize_score", "evidence_valid", "quote_match",
                             "answer_supported", "cited_is_top1", "visual_margin",
                             "citation_resolved", "support_type", "examiner_ran",
                             "examiner_supported", "quote_seen_match", "details_ok"]
    g = build_features(agreement=1.0, entail_min=0.9, heat_concentration=0.5,
                       localize_score=1.0, evidence_valid=True, quote_match=0.9)
    v = to_vector(g)
    assert v[6:9] == [1.0, 1.0, 0.9]          # localize, evidence_valid, quote_match
    assert v[-4:] == [0.0, 0.5, 0.5, 1.0]     # examiner not run -> neutral defaults
    h = build_features(agreement=0.5, citation_resolved=0.0, support_type=0.0,
                       examiner_ran=True, examiner_supported=0.0,
                       quote_seen_match=0.0, details_ok=0.0)
    vh = to_vector(h)
    assert vh[-4:] == [1.0, 0.0, 0.0, 0.0]
    assert to_vector(h.to_dict())[-4:] == [1.0, 0.0, 0.0, 0.0]  # dict input path


def test_minimal_entailing_span_shrinks():
    # rec 3: two-stage narrow — last passing window is minimal support
    def entail(window: str, claim: str) -> float:
        return 1.0 if "derived from reports" in window else 0.0
    sents = ["Header noise.", "Data derived from reports here.",
             "Footer noise."]
    shrunk = minimal_entailing_span("claim", sents, entail)
    assert shrunk == ["Data derived from reports here."]
    assert minimal_entailing_span("claim", ["nothing here"], entail) == []
    assert minimal_entailing_span("", sents, entail) == []


def test_vlm_check_disabled_and_mock():
    from vvrag.reader.client import SequenceMockReader

    assert check_claims(None, None, ["c"], enabled=False) == [None]
    r = SequenceMockReader(['{"verdict": "yes"} yes'])
    assert vlm_crop_supports(r, "img.png", "claim") is True
    r2 = SequenceMockReader(["no"])
    assert vlm_crop_supports(r2, "img.png", "claim") is False
    r3 = SequenceMockReader(["maybe"])
    assert vlm_crop_supports(r3, "img.png", "claim") is None
    # realistic JSON-mode replies (GeminiReader forces application/json):
    assert vlm_crop_supports(
        SequenceMockReader(['{\n  "answer": "yes"\n}']), "img.png", "c") is True
    assert vlm_crop_supports(
        SequenceMockReader(['"no"']), "img.png", "c") is False
    assert vlm_crop_supports(
        SequenceMockReader(['Yes, the crop shows it.']), "img.png", "c") is True
