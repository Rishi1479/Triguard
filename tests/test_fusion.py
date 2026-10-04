import io
import itertools
import random

import numpy as np
import pytest
from PIL import Image

from ml.common import config
from ml.common.schemas import FinalResult, ModalityResult, RedFlag
from ml.fusion.cross_modal import cross_modal_checks, extract_roles, extract_salaries, known_companies
from ml.fusion.explain import build_reasons
from ml.fusion.fuser import Fuser, FuserArtifact, feature_vector, modality_probs, tune_thresholds, weighted_average
from ml.image import synth_letters
from ml.image.analyzer import ImageAnalyzer
from ml.pipeline import NoUsableInputError, TriGuardPipeline
from ml.text.analyzer import TextAnalyzer
from ml.url.analyzer import URLAnalyzer


def flag(code="X", sev="medium", ev="e"):
    return RedFlag(code=code, message=f"msg {code}", severity=sev, evidence=ev)


def res(mod, p, flags=(), **extracted):
    return ModalityResult(modality=mod, probability=p, red_flags=list(flags), extracted=extracted)


# ------------------------------------------------------------------ cross-modal
def test_known_companies_and_parsers():
    assert set(known_companies(["Infosys", "Kestrel Labs", "TCS"])) == {"Infosys", "TCS"}
    assert extract_salaries("CTC Rs 450,000 per annum") == [450_000]
    assert extract_salaries("salary ₹35,000 per month") == [420_000]
    assert extract_roles("We are hiring a Data Analyst at Infosys.") == ["Data Analyst"]


def test_company_url_mismatch_and_email_flags():
    text = res("text", .5, companies=["TCS"], emails=["hr@gmail.com"])
    url = res("url", .9, registered_domain="tcs-hiring.online")
    codes = {f.code: f.severity for f in cross_modal_checks(text, ModalityResult.unavailable("image"), [url])}
    assert codes["COMPANY_URL_MISMATCH"] == "high" and codes["FREE_EMAIL_FOR_NAMED_COMPANY"] == "high"


def test_consistent_case_has_no_flags():
    text = res("text", .1, companies=["TCS"], emails=["careers@tcs.com"])
    url = res("url", .05, registered_domain="tcs.com")
    assert cross_modal_checks(text, ModalityResult.unavailable("image"), [url]) == []
    ats = res("url", .05, registered_domain="lever.co")
    assert cross_modal_checks(text, ModalityResult.unavailable("image"), [ats]) == []


def test_company_mismatch_between_text_and_image():
    t, i = res("text", .5, companies=["TCS"]), res("image", .5, companies=["Infosys"])
    assert "COMPANY_MISMATCH_TEXT_IMAGE" in {f.code for f in cross_modal_checks(t, i, [])}


def test_qr_url_mismatch():
    img = res("image", .5, qr_links=["http://evil-pay.xyz/p"])
    url = res("url", .2, registered_domain="tcs.com")
    assert "QR_URL_MISMATCH" in {f.code for f in cross_modal_checks(ModalityResult.unavailable("text"), img, [url])}
    ok = res("image", .5, qr_links=["https://www.tcs.com/apply"])
    assert cross_modal_checks(ModalityResult.unavailable("text"), ok, [url]) == []


def test_salary_and_role_mismatch():
    t = res("text", .1)
    i = res("image", .1, ocr_text="Offer for the position of Software Engineer. Annual compensation Rs 900,000")
    codes = {f.code for f in cross_modal_checks(t, i, [], raw_text="We are hiring a Content Writer. Rs 300,000 per annum")}
    assert {"SALARY_MISMATCH", "ROLE_MISMATCH"} <= codes
    same = {f.code for f in cross_modal_checks(t, i, [], raw_text="Hiring a Software Engineer. Rs 900,000 per annum")}
    assert not same & {"SALARY_MISMATCH", "ROLE_MISMATCH"}


# ------------------------------------------------------------------ fuser
def test_weighted_average_only_available():
    r = [res("text", 0.8), ModalityResult.unavailable("image"), res("url", 0.2)]
    w = config.FUSION_WEIGHTS
    assert weighted_average(r, []) == pytest.approx((w["text"] * .8 + w["url"] * .2) / (w["text"] + w["url"]))
    assert weighted_average([ModalityResult.unavailable("text")], []) == 0.0
    assert weighted_average(r, [flag(sev="high")]) > weighted_average(r, [])


def test_multiple_url_results_use_max_and_feature_vector():
    r = [res("text", .3), res("url", .1), res("url", .9)]
    assert modality_probs(r)["url"] == .9 and modality_probs(r)["image"] is None
    v = feature_vector(r, [flag(sev="high"), flag(sev="low")])
    assert v == [.3, 0.0, .9, 1, 0, 1, 1, 1]


def test_labels_and_forced_suspicious():
    f = Fuser(models_dir="/nonexistent")
    assert f.mode == "weighted_average"
    assert f.label(0.1) == "Safe" and f.label(0.5) == "Suspicious" and f.label(0.9) == "Likely Scam"
    low = res("text", 0.05, [flag("PAYMENT_REQUEST", "high")])
    prob, label = f.decide([low], [])
    assert label == "Suspicious" and prob >= f.art.t_low
    assert f.decide([res("text", 0.05, [flag("URGENCY", "medium")])], [])[1] == "Safe"


def test_tune_thresholds_recall_favouring():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(300), np.ones(100)].astype(int)
    p = np.r_[rng.beta(2, 8, 300), rng.beta(6, 3, 100)]
    lo, hi = tune_thresholds(y, p)
    assert 0 < lo < hi < 1 and hi - lo >= 0.1 - 1e-9
    assert (p[y == 1] >= lo).mean() >= config.RECALL_TARGET_LOW - 0.02
    assert tune_thresholds([0, 0, 0], [.1, .2, .3]) == (config.SAFE_THRESHOLD, config.SCAM_THRESHOLD)


def test_meta_artifact_used(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    X = np.array([[0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0]] * 10, dtype=float)
    y = np.array([0, 1] * 10)
    (tmp_path / "fusion").mkdir()
    joblib.dump(FuserArtifact(meta=LogisticRegression().fit(X, y), t_low=.3, t_high=.7), tmp_path / "fusion" / "fuser.joblib")
    f = Fuser(tmp_path)
    assert f.mode == "meta" and f.probability([res("text", 1.0)], []) > f.probability([res("text", 0.0)], [])


# ------------------------------------------------------------------ explain
def test_reasons_ranked_capped_and_deduped():
    flags = [flag(f"C{i}", s) for i, s in enumerate(["low", "high", "medium", "high", "low", "medium", "low", "high"])]
    r = build_reasons([res("text", .9, flags)], [flag("CROSS", "high", "z")], "Likely Scam")
    assert len(r) == config.MAX_REASONS and r[0].startswith("High risk – Mismatch")
    assert [x.split(" –")[0] for x in r][:5] == ["High risk"] * 4 + ["Warning"]


def test_reasons_include_lime_shap_and_empty_case():
    t = ModalityResult(modality="text", probability=.6, explanation={"lime_top_words": [{"word": "fee", "weight": .4}, {"word": "job", "weight": -.1}]})
    u = ModalityResult(modality="url", probability=.6, explanation={"shap_top_features": [{"feature": "suspicious_tld", "value": 1, "shap": .5}]})
    out = " ".join(build_reasons([t, u], [], "Suspicious"))
    assert "“fee”" in out and "domain extension" in out
    assert "No scam warning" in build_reasons([res("text", .01)], [], "Safe")[0]


# ------------------------------------------------------------------ pipeline: all 7 combinations
class FakeOCR:
    name = "fake"

    def read(self, img):
        return ("Congratulations! You are selected as Data Analyst at TCS. No interview required. Pay refundable "
                "registration fee Rs 1,500 on UPI. Email tcshr1@gmail.com")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("empty")
    text = TextAnalyzer(d)
    return TriGuardPipeline(text=text, image=ImageAnalyzer(text, ocr=FakeOCR(), visual=None),
                            url=URLAnalyzer(d, enable_live=False), fuser=Fuser(d))


@pytest.fixture(scope="module")
def png():
    img = synth_letters.make_letter(True, random.Random(1), np.random.default_rng(1))[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SCAM_TEXT = "TCS is hiring a Data Analyst. Pay a registration fee of Rs 2,000 on UPI. No interview. Contact tcshr@gmail.com"
SCAM_URL = "http://tcs-hiring.online/apply"


@pytest.mark.parametrize("use", [c for n in (1, 2, 3) for c in itertools.combinations(("text", "image", "url"), n)])
def test_all_input_combinations(pipeline, png, use):
    kw = {"text": SCAM_TEXT if "text" in use else None, "image": png if "image" in use else None,
          "url": SCAM_URL if "url" in use else None}
    r = pipeline.analyze(**kw)
    assert isinstance(r, FinalResult) and 0 <= r.risk_score <= 100
    avail = {m.modality for m in r.modality_results if m.available}
    assert set(use) <= avail | {"url"}  # extra URLs found in content may add a url result
    for m in use:
        assert any(x.modality == m and x.available for x in r.modality_results)
    assert r.label in {"Suspicious", "Likely Scam"} and 1 <= len(r.reasons) <= config.MAX_REASONS


def test_genuine_text_is_safe_and_no_input_errors(pipeline):
    ok = pipeline.analyze(text="We are hiring a Python developer. Two interview rounds. Apply on our careers page.",
                          url="https://www.tcs.com/careers")
    assert ok.label == "Safe" and ok.risk_score < 30
    for kw in ({}, {"text": "  "}, {"url": ""}):
        with pytest.raises(NoUsableInputError):
            pipeline.analyze(**kw)
    with pytest.raises(NoUsableInputError):
        pipeline.analyze(image=b"garbage")
    with pytest.raises(NoUsableInputError):
        pipeline.analyze(url="javascript:alert(1)")


def test_urls_in_content_are_auto_analyzed_and_cross_flags(pipeline):
    r = pipeline.analyze(text="TCS is hiring a Data Analyst, apply at http://tcs-hiring.online/apply", url="https://www.tcs.com/careers")
    srcs = [m.extracted.get("source") for m in r.modality_results if m.modality == "url"]
    assert "found_in_content" in srcs
    assert any(f.code == "COMPANY_URL_MISMATCH" for f in r.cross_modal_flags) or any(
        m.extracted.get("impersonates") for m in r.modality_results if m.modality == "url")


def test_failing_modality_does_not_crash(tmp_path):
    class Boom:
        def analyze(self, *a, **k):
            raise RuntimeError("boom")
    p = TriGuardPipeline(text=Boom(), image=Boom(), url=URLAnalyzer(tmp_path, enable_live=False), fuser=Fuser(tmp_path))
    r = p.analyze(text="hello world job", url="http://tcs-hiring.online")
    assert r.modality_results[0].available is False and r.modality_results[2].available


def test_qr_decoder_survives_opencv_errors(monkeypatch):
    import cv2
    from ml.image import qr
    monkeypatch.setattr(qr, "_pyzbar", None)

    class Det:
        def detectAndDecodeMulti(self, g): raise cv2.error("x")
        def detectAndDecode(self, g): raise cv2.error("y")
    monkeypatch.setattr(cv2, "QRCodeDetector", Det)
    assert qr.decode_qr(np.zeros((50, 50), np.uint8)) == []
