import json

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


def test_corrupt_distilbert_falls_back(models_dir, tmp_path):
    import shutil
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    (d / "text_distilbert").mkdir()
    (d / "text_distilbert" / "meta.json").write_text("{}")
    assert TextAnalyzer(models_dir=d).model_name == "text_tfidf_lr"


def test_tiny_distilbert_is_preferred(models_dir, tmp_path):
    import shutil
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    out = d / "text_distilbert"
    out.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "pay", "fee", "job", "hiring", "engineer"]
    (out / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(out / "vocab.txt")).save_pretrained(out)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=16, n_layers=1, n_heads=2, hidden_dim=32, num_labels=2)
    DistilBertForSequenceClassification(cfg).save_pretrained(out)
    (out / "meta.json").write_text(json.dumps({"max_len": 32, "threshold": 0.4, "platt_coef": 1.0, "platt_intercept": 0.0}))
    an = TextAnalyzer(models_dir=d)
    assert an.model_name == "text_distilbert" and an.threshold == 0.4
    r = an.analyze(SCAM, explain=False)
    assert 0 <= r.probability <= 1 and r.explanation["model"] == "text_distilbert"


def test_bilstm_trains_and_saves(data, tmp_path, monkeypatch):
    texts, y = data
    monkeypatch.setattr(config, "BILSTM_EPOCHS", 2)
    monkeypatch.setattr(config, "BILSTM_MAX_LEN", 60)
    clf = BiLSTMClassifier().fit(texts[:200], y[:200], texts[200:260], y[200:260])
    p = clf.predict_proba(texts[260:300])
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()
    clf.save(tmp_path / "bl")
    assert (tmp_path / "bl" / "model.pt").exists() and (tmp_path / "bl" / "vocab.json").exists()


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
