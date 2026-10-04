import pytest

from ml.common import config


def test_split_ratios_sum_to_one():
    assert config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO == pytest.approx(1.0)


def test_thresholds_defaults():
    assert config.SEED == 42
    assert config.SAFE_THRESHOLD == 0.30 and config.SCAM_THRESHOLD == 0.70
    assert 0 < config.SAFE_THRESHOLD < config.SCAM_THRESHOLD < 1


def test_paths_anchored_at_repo_root():
    assert (config.ROOT_DIR / "requirements.txt").exists()
    assert config.RAW_DIR == config.ROOT_DIR / "data" / "raw"
