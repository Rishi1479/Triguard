"""Shared Pydantic models: every modality analyzer returns a ModalityResult."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]
Modality = Literal["text", "image", "url"]
Label = Literal["Safe", "Suspicious", "Likely Scam"]
EvidenceStrength = Literal["low", "normal"]


class RedFlag(BaseModel):
    """A single human-readable warning sign found in the input."""

    code: str = Field(min_length=1)
    message: str
    severity: Severity
    evidence: str = ""


class ModalityResult(BaseModel):
    """Output of one modality analyzer (text, image or url)."""

    modality: Modality
    probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Calibrated P(scam)")
    red_flags: list[RedFlag] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    available: bool = True

    @classmethod
    def unavailable(cls, modality: Modality) -> "ModalityResult":
        """Build the placeholder result for a modality whose input was not provided."""
        return cls(modality=modality, available=False)


class FinalResult(BaseModel):
    """Fused verdict across all available modalities.

    ``abstained`` marks a verdict the system does not stand behind: too little evidence, or a
    probability too close to the fence to call. ``label`` is still populated (so existing clients
    keep working) but should be presented as provisional.
    """

    label: Label
    risk_score: float = Field(ge=0.0, le=100.0, description="Calibrated risk, percent")
    modality_results: list[ModalityResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cross_modal_flags: list[RedFlag] = Field(default_factory=list)
    abstained: bool = Field(default=False, description="True when the evidence does not justify a firm verdict")
    evidence_strength: EvidenceStrength = Field(default="normal", description="'low' when abstained")
