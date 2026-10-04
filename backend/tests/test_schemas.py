import pytest
from pydantic import ValidationError

from ml.common.schemas import FinalResult, ModalityResult, RedFlag


def flag(**kw):
    return RedFlag(**{"code": "FEE", "message": "Asks for fee", "severity": "high", **kw})


def test_redflag_defaults_and_validation():
    assert flag().evidence == ""
    with pytest.raises(ValidationError):
        flag(severity="critical")
    with pytest.raises(ValidationError):
        flag(code="")


def test_modality_result_defaults():
    r = ModalityResult(modality="text", probability=0.8)
    assert r.available is True
    assert r.red_flags == [] and r.extracted == {} and r.explanation == {}


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_probability_bounds(p):
    with pytest.raises(ValidationError):
        ModalityResult(modality="url", probability=p)


def test_invalid_modality():
    with pytest.raises(ValidationError):
        ModalityResult(modality="audio")


def test_unavailable_factory_and_no_shared_defaults():
    a, b = ModalityResult.unavailable("image"), ModalityResult.unavailable("image")
    assert a.available is False and a.probability == 0.0
    a.red_flags.append(flag())
    assert b.red_flags == []


def test_final_result_roundtrip():
    fr = FinalResult(
        label="Likely Scam",
        risk_score=91.5,
        modality_results=[ModalityResult(modality="text", probability=0.9, red_flags=[flag()])],
        reasons=["Asks for registration fee"],
        cross_modal_flags=[flag(code="DOMAIN_MISMATCH")],
    )
    assert FinalResult.model_validate_json(fr.model_dump_json()) == fr


def test_final_result_validation():
    with pytest.raises(ValidationError):
        FinalResult(label="Maybe", risk_score=50)
    with pytest.raises(ValidationError):
        FinalResult(label="Safe", risk_score=101)
