"""Plain-English reasons: severity-ranked red flags topped up with the strongest LIME/SHAP evidence."""
from __future__ import annotations

from ml.common import config
from ml.common.schemas import ModalityResult, RedFlag

_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
_SEV_LABEL = {"high": "High risk", "medium": "Warning", "low": "Note"}
_SHAP_PHRASES = {
    "keyword_hits": "contains scam-style words such as 'login', 'verify', 'offer' or 'apply'",
    "keyword_in_host": "has job/verification keywords inside the domain name",
    "suspicious_tld": "ends in a domain extension often used by scam sites",
    "n_hyphens": "uses many hyphens, a common look-alike domain trick",
    "url_len": "is unusually long", "host_len": "has an unusually long domain name",
    "n_digits": "contains many digits", "digit_ratio": "has a high share of digits",
    "is_ip_host": "points to a raw IP address", "is_shortener": "is a shortened link",
    "subdomain_depth": "has many subdomain levels", "platform_link": "uses a generic form/chat link",
    "is_https": "does not use HTTPS", "entropy": "looks randomly generated",
}
_MODALITY_NAME = {"text": "Text", "image": "Image", "url": "Link"}


def _reason(flag: RedFlag, source: str) -> str:
    ev = f" (“{flag.evidence}”)" if flag.evidence else ""
    return f"{_SEV_LABEL[flag.severity]} – {source}: {flag.message}{ev}"


def _model_evidence(results: list[ModalityResult]) -> list[str]:
    out: list[str] = []
    for r in results:
        if not r.available:
            continue
        words = [w["word"] for w in r.explanation.get("lime_top_words", []) if w["weight"] > 0][:3]
        if words:
            out.append(f"Note – {_MODALITY_NAME[r.modality]}: the AI model's score was driven mostly by the words "
                       + ", ".join(f"“{w}”" for w in words) + ". Single words are only hints, not proof.")
        shap_terms = [_SHAP_PHRASES[s["feature"]] for s in r.explanation.get("shap_top_features", [])
                      if s["shap"] > 0 and s["feature"] in _SHAP_PHRASES][:2]
        if shap_terms:
            out.append("Note – Link: the AI model noticed that the web address " + " and ".join(shap_terms) + ".")
    return out


def build_reasons(results: list[ModalityResult], cross_flags: list[RedFlag], label: str,
                  max_reasons: int = config.MAX_REASONS) -> list[str]:
    """Up to ``max_reasons`` reasons: cross-modal and per-modality flags by severity, then model evidence."""
    tagged: list[tuple[int, int, str]] = []
    for f in cross_flags:
        tagged.append((_SEV_ORDER[f.severity], 0, _reason(f, "Mismatch between inputs")))
    for r in results:
        if r.available:
            for f in r.red_flags:
                tagged.append((_SEV_ORDER[f.severity], 1, _reason(f, _MODALITY_NAME[r.modality])))
    seen, reasons = set(), []
    for _, _, text in sorted(tagged, key=lambda t: (t[0], t[1])):
        if text not in seen:
            seen.add(text)
            reasons.append(text)
    reasons = reasons[:max_reasons]
    if len(reasons) < max_reasons:
        reasons += _model_evidence(results)[: max_reasons - len(reasons)]
    if not reasons:
        reasons = ["No scam warning signs were found in what you provided. Still verify the employer on its official website."
                   if label == config.LABEL_SAFE else "The model found the content statistically similar to known scams."]
    return reasons[:max_reasons]
