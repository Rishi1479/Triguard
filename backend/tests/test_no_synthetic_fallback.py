"""Data pipelines must fail loudly when real data is missing, never generate a SYNTHETIC stand-in."""
import pytest

from ml.common import config
from ml.fusion import train_fuser
from ml.image import compare_ocr
from ml.text import prepare_custom, prepare_emscad
from ml.url import prepare_urls


@pytest.fixture
def empty_dirs(tmp_path, monkeypatch):
    """Point every raw-data location at empty temporary folders."""
    for attr in ("RAW_DIR", "URL_RAW_DIR", "ALIGNED_DIR", "PROCESSED_DIR"):
        (tmp_path / attr).mkdir()
        monkeypatch.setattr(config, attr, tmp_path / attr)
    monkeypatch.setattr(config, "OCR_GT_FILE", tmp_path / "ocr_gt.csv")
    return tmp_path


@pytest.mark.parametrize("load", [
    prepare_emscad.load_raw, prepare_urls.load_raw, prepare_custom.run,
    train_fuser.load_aligned, compare_ocr.load_ground_truth,
], ids=["emscad", "urls", "custom", "aligned", "ocr"])
def test_missing_real_data_raises(empty_dirs, load):
    with pytest.raises(FileNotFoundError):
        load()
    assert not list(empty_dirs.rglob("SYNTHETIC_*"))
