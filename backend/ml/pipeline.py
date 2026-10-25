"""TriGuardPipeline: text + image + URL -> FinalResult. Models are loaded once at construction."""
from __future__ import annotations

import logging
from pathlib import Path

from ml.common import config
from ml.common.schemas import FinalResult, ModalityResult, RedFlag
from ml.fusion.cross_modal import cross_modal_checks
from ml.fusion.explain import build_reasons
from ml.fusion.fuser import Fuser
from ml.image.analyzer import ImageAnalyzer
from ml.text.analyzer import TextAnalyzer
from ml.url.analyzer import URLAnalyzer

log = logging.getLogger(__name__)

ABSTAIN_REASON = ("Not enough evidence for a confident verdict — treat this result as provisional "
                  "and add more detail (full posting text, the application link, or a clearer image).")


class NoUsableInputError(ValueError):
    """No provided input could be analysed (all empty, unreadable or malformed)."""


class AnalysisFailedError(RuntimeError):
    """Every analyzer that received input crashed (a server-side bug, not bad user input)."""


class TriGuardPipeline:
    """End-to-end analyzer. Pass pre-built analyzers to override (tests) or let it build defaults."""

    def __init__(self, models_dir: Path | None = None, enable_live: bool = True, ocr_backend: str = "auto",
                 text_model: str | None = None, text: TextAnalyzer | None = None,
                 image: ImageAnalyzer | None = None, url: URLAnalyzer | None = None,
                 fuser: Fuser | None = None) -> None:
        self.text = text or TextAnalyzer(models_dir, prefer=text_model)
        self.url = url or URLAnalyzer(models_dir, enable_live=enable_live)
        if image is None:
            from ml.image.ocr import get_engine
            image = ImageAnalyzer(self.text, ocr=get_engine(ocr_backend))
        self.image = image
        self.fuser = fuser or Fuser(models_dir)

    @staticmethod
    def _safe(fn, modality: str, *args, **kw) -> ModalityResult:
        try:
            return fn(*args, **kw)
        except Exception:  # one failing modality must not take down the whole analysis
            log.exception("%s analyzer failed", modality)
            res = ModalityResult.unavailable(modality)
            res.extracted["error"] = "analyzer_crashed"
            return res

    def _url_results(self, url: str | None, text_res: ModalityResult, image_res: ModalityResult,
                     explain: bool) -> list[ModalityResult]:
        """Analyze the provided URL plus any URLs found in the text/image (QR links included)."""
        results = [self._safe(self.url.analyze, "url", url, explain=explain)]
        seen = {(url or "").strip().lower()}
        extras = [u for r in (text_res, image_res) if r.available for u in r.extracted.get("urls", [])]
        for u in extras:
            if len(results) > config.MAX_EXTRA_URLS or u.strip().lower() in seen:
                continue
            seen.add(u.strip().lower())
            res = self._safe(self.url.analyze, "url", u, explain=False)
            if res.available:
                res.extracted["source"] = "found_in_content"
                results.append(res)
        return results

    def analyze_modalities(self, text: str | None = None, image: bytes | None = None, url: str | None = None,
                           explain: bool = True) -> tuple[list[ModalityResult], list[RedFlag]]:
        """Run all analyzers and cross-modal checks; returns ([text, image, url, extra urls...], cross flags)."""
        if not ((text and text.strip()) or image or (url and url.strip())):
            raise NoUsableInputError("Provide at least one of text, image or url.")
        text_res = self._safe(self.text.analyze, "text", text, explain=explain)
        image_res = self._safe(self.image.analyze, "image", image, explain=explain)
        url_results = self._url_results(url, text_res, image_res, explain)
        cross = cross_modal_checks(text_res, image_res, url_results, raw_text=text or "")
        return [text_res, image_res, *url_results], cross

    def analyze(self, text: str | None = None, image: bytes | None = None, url: str | None = None,
                explain: bool = True) -> FinalResult:
        """Analyze any combination of inputs and return the fused verdict."""
        results, cross = self.analyze_modalities(text, image, url, explain)
        if not any(r.available for r in results):
            if any(r.extracted.get("error") == "analyzer_crashed" for r in results):
                raise AnalysisFailedError("An internal analyzer failed; see server logs (models may need retraining).")
            issues = [f.message for r in results for f in r.red_flags] or ["nothing could be analysed"]
            raise NoUsableInputError("; ".join(dict.fromkeys(issues)))
        prob, label, abstained = self.fuser.decide_full(results, cross)
        reasons = build_reasons(results, cross, label)
        if abstained:
            reasons = [ABSTAIN_REASON, *reasons][: config.MAX_REASONS]
        return FinalResult(label=label, risk_score=round(100 * prob, 1), modality_results=results,
                           reasons=reasons, cross_modal_flags=cross,
                           abstained=abstained, evidence_strength="low" if abstained else "normal")
