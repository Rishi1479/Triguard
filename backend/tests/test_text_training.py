"""Tests for the EMSCAD loader (both releases) and training text models one at a time."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.common import config
from ml.common.synthetic import make_emscad_like
from ml.common.textprep import stratified_split
from ml.text import baselines, prepare_emscad


def _raw_v1(n: int = 4) -> pd.DataFrame:
    """Rows shaped like the original EMSCAD release (t/f labels, extra column)."""
    return pd.DataFrame({
        "title": ["Engineer"] * n, "company_profile": ["Acme builds things"] * n,
        "description": [f"Long enough description number {i} for the posting" for i in range(n)],
        "requirements": [""] * n, "benefits": [""] * n,
        "fraudulent": ["f", "t", "f", "t"][:n], "in_balanced_dataset": ["f"] * n,
    })


@pytest.mark.parametrize("vals, expected", [
    (["t", "f", "T", "true"], [1, 0, 1, 1]),
    ([1, 0, "1", "0"], [1, 0, 1, 0]),
    ([None, np.nan, "f", "t"], [0, 0, 0, 1]),
])
def test_to_binary_reads_both_releases(vals, expected):
    assert prepare_emscad.to_binary(pd.Series(vals)).tolist() == expected


def test_build_dataset_reads_tf_labels():
    df, _ = prepare_emscad.build_dataset(_raw_v1())
    assert df["label"].tolist() == [0, 1, 0, 1]
    assert list(df.columns) == ["text", "label"]


@pytest.mark.parametrize("name", config.EMSCAD_FILES)
def test_load_raw_accepts_either_file_name(tmp_path, monkeypatch, name):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    _raw_v1().to_csv(tmp_path / name, index=False)
    raw, synthetic = prepare_emscad.load_raw()
    assert not synthetic and len(raw) == 4


def test_load_raw_prefers_the_first_listed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    _raw_v1(4).to_csv(tmp_path / config.EMSCAD_FILES[0], index=False)
    _raw_v1(2).to_csv(tmp_path / config.EMSCAD_FILES[1], index=False)
    assert len(prepare_emscad.load_raw()[0]) == 4


# ------------------------------------------------------------------------------- per-model training
def test_resolve_expands_all_and_strips_prefix():
    assert baselines.resolve(None) == list(baselines.MODELS)
    assert baselines.resolve(["text_bilstm", "bilstm", "tfidf_lr"]) == ["bilstm", "tfidf_lr"]
    with pytest.raises(ValueError, match="unknown model"):
        baselines.resolve(["distilbert"])


def test_trainers_match_the_analyzer_models():
    assert {f"text_{n}" for n in baselines.MODELS} == set(config.TEXT_MODEL_PRIORITY)


@pytest.fixture
def processed(tmp_path, monkeypatch):
    df, _ = prepare_emscad.build_dataset(make_emscad_like(300, 60))
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    config.PROCESSED_DIR.mkdir()
    for name, part in zip(("train", "val", "test"), stratified_split(df)):
        part.to_csv(config.PROCESSED_DIR / f"text_{name}.csv", index=False)
    (config.PROCESSED_DIR / "SYNTHETIC_TEXT.flag").write_text("x")
    (tmp_path / "reports").mkdir()
    return tmp_path


def test_training_one_model_keeps_the_others_results(processed):
    models, reports = processed / "models", processed / "reports"
    (reports / "text_baselines.json").write_text(json.dumps({"synthetic": True, "bilstm": {"kept": 1}}))
    out = baselines.run(["tfidf_lr"], models_dir=models, reports_dir=reports)
    assert (models / "text_tfidf_lr" / "model.joblib").exists()
    assert not (models / "text_hybrid_lgbm").exists() and not (models / "text_bilstm").exists()
    saved = json.loads((reports / "text_baselines.json").read_text())
    assert saved["bilstm"] == {"kept": 1}
    assert "test" in saved["tfidf_lr"] and "test" in out["tfidf_lr"]


def test_results_from_the_other_kind_of_data_are_dropped(processed):
    reports = processed / "reports"
    (reports / "text_baselines.json").write_text(json.dumps({"synthetic": False, "bilstm": {"real": 1}}))
    baselines.run(["tfidf_lr"], models_dir=processed / "models", reports_dir=reports)
    saved = json.loads((reports / "text_baselines.json").read_text())
    assert "bilstm" not in saved and "test" in saved["tfidf_lr"]
