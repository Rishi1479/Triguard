import pandas as pd
import pytest

from ml.common import config
from ml.common.privacy import anonymize
from ml.common.textprep import clean_text, dedupe, merge_columns, normalized_hash, stratified_split
from ml.common.synthetic import make_emscad_like, make_urls
from ml.image import synth_letters
from ml.image.prepare_images import build_manifest
from ml.text import prepare_emscad
from ml.url.prepare_urls import normalize_url, to_standard


def test_clean_text_strips_html_placeholders_ws():
    raw = "<p>Hello&nbsp;<b>World</b></p>  #URL_abc123def#\n\n  apply   now #EMAIL_x1#"
    assert clean_text(raw) == "Hello World apply now"
    assert clean_text(float("nan")) == ""


def test_merge_columns_skips_missing():
    df = pd.DataFrame({"title": ["A"], "description": ["<i>B</i>"], "benefits": [None]})
    assert merge_columns(df).iloc[0] == "A B"


def test_dedupe_exact_and_near_and_label_conflict():
    df = pd.DataFrame({"text": ["Pay a fee!", "pay a  FEE", "Other text", "Other text."], "label": [0, 1, 0, 0]})
    out = dedupe(df)
    assert len(out) == 2
    assert out[out.text.str.lower().str.contains("fee")].label.iloc[0] == 1  # scam wins
    assert normalized_hash("A b!") == normalized_hash("a  B")


def test_split_stratified_and_disjoint():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(1000)], "label": [1] * 100 + [0] * 900})
    tr, va, te = stratified_split(df)
    assert (len(tr), len(va), len(te)) == (700, 150, 150)
    for part in (tr, va, te):
        assert part.label.mean() == pytest.approx(0.10, abs=0.01)
    assert not (set(tr.text) & set(va.text) | set(tr.text) & set(te.text) | set(va.text) & set(te.text))
    assert stratified_split(df)[0].equals(tr)  # deterministic


@pytest.mark.parametrize("raw,expected", [
    ("mail me at john.doe@gmail.com now", "mail me at [EMAIL] now"),
    ("call +91 9876543210 today", "call [PHONE] today"),
    ("call 98765-43210", "call [PHONE]"),
    ("Aadhaar 1234 5678 9012 ok", "Aadhaar [AADHAAR] ok"),
    ("PAN ABCDE1234F ok", "PAN [PAN] ok"),
    ("Salary 450000 per year in 2024", "Salary 450000 per year in 2024"),
])
def test_anonymize(raw, expected):
    assert anonymize(raw) == expected


def test_prepare_emscad_pipeline(tmp_path, monkeypatch):
    for attr in ("RAW_DIR", "PROCESSED_DIR"):
        monkeypatch.setattr(config, attr, tmp_path / attr)
    config.RAW_DIR.mkdir()
    make_emscad_like().to_csv(config.RAW_DIR / config.EMSCAD_FILES[0], index=False)
    splits = prepare_emscad.run()
    assert (tmp_path / "PROCESSED_DIR" / "text_train.csv").exists()
    assert not list(config.RAW_DIR.glob("SYNTHETIC_*"))
    texts = [set(s.text) for s in splits.values()]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2])
    assert all(0 < s.label.mean() < 0.3 for s in splits.values())


def test_synthetic_emscad_has_injected_duplicates():
    raw = make_emscad_like(200, 30)
    df, removed = prepare_emscad.build_dataset(raw)
    assert removed >= 30 and len(df) < len(raw)


def test_url_standardisation_formats():
    phish = to_standard(pd.DataFrame({"url": ["http://a.com"], "verified": ["yes"]}), "pt")
    assert phish.label.tolist() == [1]
    mal = to_standard(pd.DataFrame({"url": ["a", "b", "c"], "type": ["benign", "phishing", "malware"]}), "mp")
    assert mal.label.tolist() == [0, 1, 1]
    assert normalize_url("HTTPS://www.Example.com/") == "example.com"
    with pytest.raises(ValueError):
        to_standard(pd.DataFrame({"x": [1]}), "bad")
    assert set(make_urls(50).columns) == {"url", "type"}


def test_synth_letters_and_manifest(tmp_path):
    n = synth_letters.generate(n_each=6, out_dir=tmp_path)
    assert n == 12 and len(list((tmp_path / "scam").glob("SYNTHETIC_*.png"))) == 6
    df = build_manifest(tmp_path)
    assert len(df) == 12 and set(df.source) == {"SYNTHETIC"} and set(df.split) == {"train", "val", "test"}
    assert df.groupby("split").label.nunique().min() == 2
