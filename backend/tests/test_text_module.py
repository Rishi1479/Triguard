import joblib
import numpy as np
import pytest

from ml.common import config
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import expected_calibration_error
from ml.common.schemas import ModalityResult
from ml.common.synthetic import make_emscad_like
from ml.common.textprep import merge_columns
from ml.text.analyzer import TextAnalyzer, boost_with_rules
from ml.text.models import BiLSTMClassifier, TfidfLR, Vocab

SCAM = "Pay a refundable registration fee of Rs 1,500 to confirm your seat. No interview, selection guaranteed. Contact hr99@gmail.com"
GENUINE = "We are hiring a Software Engineer in Bengaluru. Two interview rounds, apply on our careers page."


@pytest.fixture(scope="module")
def data():
    raw = make_emscad_like(300, 60)
    return merge_columns(raw).tolist(), raw["fraudulent"].to_numpy()


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory, data):
    texts, y = data
    d = tmp_path_factory.mktemp("models") / "text_tfidf_lr"
    d.mkdir()
    lr = TfidfLR().fit(texts[:300], y[:300])
    cal = PlattCalibrator().fit(lr.predict_proba(texts[300:]), y[300:])
    joblib.dump({"model": lr, "calibrator": cal, "threshold": 0.5}, d / "model.joblib")
    return d.parent


def test_rules_only_fallback(tmp_path):
    an = TextAnalyzer(models_dir=tmp_path)
    assert an.model_name == "rules_only"
    r = an.analyze(SCAM)
    assert r.available and r.modality == "text" and r.probability > config.RULES_ONLY_PRIOR
    assert {f.code for f in r.red_flags} >= {"PAYMENT_REQUEST", "NO_INTERVIEW", "FREE_EMAIL_HR"}
    assert "hr99@gmail.com" in r.extracted["emails"]
    assert an.analyze(GENUINE).probability == pytest.approx(config.RULES_ONLY_PRIOR)


def test_tfidf_model_scores_and_explains(models_dir):
    an = TextAnalyzer(models_dir=models_dir)
    assert an.model_name == "text_tfidf_lr"
    s, g = an.analyze(SCAM), an.analyze(GENUINE)
    assert 0 <= g.probability <= 1 and s.probability > g.probability
    words = s.explanation["lime_top_words"]
    assert 0 < len(words) <= config.LIME_TOP_WORDS and {"word", "weight"} <= set(words[0])
    assert "lime_top_words" not in an.analyze(SCAM, explain=False).explanation


@pytest.mark.parametrize("bad", [None, "", "   \n"])
def test_empty_text_unavailable(models_dir, bad):
    r = TextAnalyzer(models_dir=models_dir).analyze(bad)
    assert r == ModalityResult.unavailable("text")


def test_huge_and_nonenglish_text(models_dir):
    an = TextAnalyzer(models_dir=models_dir)
    assert 0 <= an.analyze("hiring engineer " * 20000, explain=False).probability <= 1
    r = an.analyze("नौकरी के लिए पंजीकरण शुल्क जमा करें 日本語のテキスト", explain=False)
    assert 0 <= r.probability <= 1


def test_rule_boost_math():
    from ml.common.schemas import RedFlag
    f = [RedFlag(code="X", message="m", severity="high")]
    assert boost_with_rules(0.0, f) == pytest.approx(config.RULE_BOOST["high"])
    assert boost_with_rules(0.5, []) == 0.5
    assert boost_with_rules(1.0, f) == 1.0


def test_boost_off(models_dir):
    on = TextAnalyzer(models_dir=models_dir).analyze(SCAM, explain=False).probability
    off = TextAnalyzer(models_dir=models_dir, use_rules_boost=False).analyze(SCAM, explain=False).probability
    assert on >= off


def test_corrupt_hybrid_falls_back(models_dir, tmp_path):
    import shutil
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    (d / "text_hybrid_lgbm").mkdir()
    (d / "text_hybrid_lgbm" / "model.joblib").write_bytes(b"not a pickle")
    assert TextAnalyzer(models_dir=d).model_name == "text_tfidf_lr"


def test_only_the_three_text_models_are_servable():
    from ml.text.analyzer import _LOADERS
    assert config.TEXT_MODEL_PRIORITY == ("text_hybrid_lgbm", "text_tfidf_lr", "text_bilstm")
    assert set(_LOADERS) == set(config.TEXT_MODEL_PRIORITY)


def test_bilstm_trains_and_saves(data, tmp_path, monkeypatch):
    texts, y = data
    monkeypatch.setattr(config, "BILSTM_EPOCHS", 2)
    monkeypatch.setattr(config, "BILSTM_MAX_LEN", 60)
    clf = BiLSTMClassifier().fit(texts[:200], y[:200], texts[200:260], y[200:260])
    p = clf.predict_proba(texts[260:300])
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()
    clf.save(tmp_path / "bl")
    assert (tmp_path / "bl" / "model.pt").exists() and (tmp_path / "bl" / "vocab.json").exists()


@pytest.fixture(scope="module")
def with_bilstm(models_dir, data, tmp_path_factory):
    """models_dir copy that also holds a tiny trained Bi-LSTM artifact (as baselines.py writes it)."""
    import shutil
    texts, y = data
    d = tmp_path_factory.mktemp("bl") / "m"
    shutil.copytree(models_dir, d)
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "BILSTM_EPOCHS", 1)
    mp.setattr(config, "BILSTM_MAX_LEN", 60)
    try:
        clf = BiLSTMClassifier().fit(texts[:200], y[:200], texts[200:260], y[200:260])
        clf.save(d / "text_bilstm")
    finally:
        mp.undo()
    cal = PlattCalibrator().fit(clf.predict_proba(texts[260:]), y[260:])
    joblib.dump({"calibrator": cal, "threshold": 0.45}, d / "text_bilstm" / "calib.joblib")
    return d


def test_bilstm_not_chosen_by_default(with_bilstm):
    assert TextAnalyzer(models_dir=with_bilstm).model_name == "text_tfidf_lr"


def test_bilstm_served_when_preferred(with_bilstm):
    an = TextAnalyzer(models_dir=with_bilstm, prefer="text_bilstm")
    assert an.model_name == "text_bilstm" and an.threshold == 0.45
    r = an.analyze(SCAM)
    assert 0 <= r.probability <= 1 and r.explanation["model"] == "text_bilstm"
    assert r.explanation["lime_top_words"]  # LIME works through model_proba for any scorer


def test_search_order_puts_preferred_first():
    from ml.text.analyzer import search_order
    assert search_order(None)[-1] == "text_bilstm"
    assert search_order("text_bilstm")[0] == "text_bilstm"
    assert search_order("no_such_model") == list(config.TEXT_MODEL_PRIORITY)


def test_preferred_missing_falls_back(models_dir, caplog):
    an = TextAnalyzer(models_dir=models_dir, prefer="text_bilstm")
    assert an.model_name == "text_tfidf_lr"
    assert "preferred text model 'text_bilstm' is unavailable" in caplog.text


def test_incomplete_bilstm_is_skipped(with_bilstm, tmp_path):
    import shutil
    d = tmp_path / "m"
    shutil.copytree(with_bilstm, d)
    (d / "text_bilstm" / "model.pt").unlink()  # calib.joblib present, weights missing
    assert TextAnalyzer(models_dir=d, prefer="text_bilstm").model_name == "text_tfidf_lr"


def test_pipeline_passes_text_model(with_bilstm):
    from ml.pipeline import TriGuardPipeline
    p = TriGuardPipeline(models_dir=with_bilstm, enable_live=False, text_model="text_bilstm",
                         image=object(), url=object())
    assert p.text.model_name == "text_bilstm"


def test_vocab_uses_given_texts_only():
    v = Vocab.build(["alpha beta beta", "gamma"], size=10)
    assert v.encode("alpha zzz", max_len=4) == [v.token_to_id["alpha"], 1, 0, 0]


def test_platt_and_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p ** 3).astype(int)  # over-confident raw scores
    cal = PlattCalibrator().fit(p, y)
    assert expected_calibration_error(y, cal.predict(p)) < expected_calibration_error(y, p)
    assert np.array_equal(PlattCalibrator().fit(p, np.zeros(2000)).predict(p), p)  # single class -> identity
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0
