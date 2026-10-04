"""URLAnalyzer: LightGBM lexical model + impersonation and live-metadata rules -> ModalityResult."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from ml.common import config
from ml.common.schemas import ModalityResult, RedFlag, Severity
from ml.url.features import FEATURE_NAMES, PLATFORM_LINKS, extract_features, parse_host, registered_domain
from ml.url.impersonation import Impersonation, check_impersonation
from ml.url.live_checks import LiveChecker, LiveSignals

log = logging.getLogger(__name__)
_STRONG_IMPERSONATION = {"lookalike", "brand_in_domain", "brand_in_subdomain"}


def _flag(code: str, message: str, severity: Severity, evidence: str) -> RedFlag:
    return RedFlag(code=code, message=message, severity=severity, evidence=evidence)


def lexical_flags(url: str, feats: dict[str, float], imp: Impersonation | None) -> list[RedFlag]:
    """Rule flags that need no network access."""
    flags: list[RedFlag] = []
    host, _ = parse_host(url)
    if imp:
        strong = imp.kind in _STRONG_IMPERSONATION
        flags.append(_flag(
            "BRAND_IMPERSONATION" if strong else "BRAND_IN_PATH",
            f"Domain imitates {imp.company} (official site: {imp.real_domain})." if strong
            else f"URL path mentions {imp.company} but the site is not {imp.real_domain}.",
            "high" if strong else "low", host))
    if feats["is_ip_host"]:
        flags.append(_flag("IP_ADDRESS_HOST", "Link points to a raw IP address, not a company domain.", "medium", host))
    if feats["is_shortener"]:
        flags.append(_flag("URL_SHORTENER", "Shortened link hides the real destination.", "medium", host))
    low = url.lower()
    hit = next((p for p in PLATFORM_LINKS if p in low and p != "bit.ly"), None)
    if hit:
        flags.append(_flag("FORM_OR_MESSENGER_LINK", f"The application is collected through a generic form/chat link ({hit}) instead of the company's own careers website.",
                           "medium", hit))
    if feats["suspicious_tld"]:
        flags.append(_flag("SUSPICIOUS_TLD", "Domain ends in a TLD commonly abused for scams.", "low", host))
    if feats["n_at"]:
        flags.append(_flag("AT_SYMBOL", "'@' in the URL can disguise the real destination.", "medium", url[:80]))
    if feats["has_punycode"]:
        flags.append(_flag("PUNYCODE_DOMAIN", "Internationalised (punycode) domain may be a look-alike.", "medium", host))
    if not feats["is_https"] and not low.startswith("https"):
        flags.append(_flag("NO_HTTPS", "Link does not use HTTPS.", "low", url[:60]))
    return flags


def live_flags(sig: LiveSignals, domain: str) -> list[RedFlag]:
    """Rule flags from WHOIS/TLS lookups (inference-time only)."""
    flags: list[RedFlag] = []
    age = sig.domain_age_days
    if age is not None and age < config.NEW_DOMAIN_DAYS_MEDIUM:
        sev: Severity = "high" if age < config.NEW_DOMAIN_DAYS_HIGH else "medium"
        flags.append(_flag("NEW_DOMAIN", f"Domain {domain} was registered only {age} days ago.", sev, f"{age} days old"))
    if sig.ssl_ok is False:
        flags.append(_flag("SSL_INVALID", "No valid TLS certificate could be verified for this site.", "medium", domain))
    return flags


def boost(p: float, flags: list[RedFlag]) -> float:
    """p' = 1 - (1-p) * prod(1 - w[severity])."""
    keep = 1.0 - p
    for f in flags:
        keep *= 1.0 - config.URL_RULE_BOOST[f.severity]
    return float(1.0 - keep)


class URLAnalyzer:
    """Analyze an application URL without ever opening it."""

    def __init__(self, models_dir: Path | None = None, enable_live: bool = True,
                 live_checker: LiveChecker | None = None) -> None:
        folder = (Path(models_dir) if models_dir else config.MODELS_DIR) / "url_lgbm"
        self.blob = None
        if (folder / "model.joblib").exists():
            try:
                blob = joblib.load(folder / "model.joblib")
                blob["calibrator"].predict(blob["model"].predict_proba(np.zeros((1, len(FEATURE_NAMES))))[:, 1])  # self-test
                self.blob = blob
            except Exception as exc:  # e.g. saved with a different scikit-learn/LightGBM version: retrain
                log.warning("URL model in %s is unusable (%s); retrain with `python -m ml.url.train`", folder, exc)
        if self.blob is None:
            log.warning("no URL model in %s; using rules only", folder)
        self.live = (live_checker or LiveChecker()) if enable_live else None
        self._explainer = None

    @property
    def threshold(self) -> float:
        """Validation-tuned decision threshold."""
        return float(self.blob["threshold"]) if self.blob else config.DEFAULT_CLASS_THRESHOLD

    def model_proba(self, feats: dict[str, float]) -> float:
        """Calibrated model-only P(scam) from lexical features."""
        if self.blob is None:
            return config.RULES_ONLY_PRIOR
        x = np.array([[feats[k] for k in FEATURE_NAMES]])
        return float(self.blob["calibrator"].predict(self.blob["model"].predict_proba(x)[:, 1])[0])

    def _shap_top(self, feats: dict[str, float]) -> list[dict]:
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.blob["model"])
        x = np.array([[feats[k] for k in FEATURE_NAMES]])
        vals = self._explainer.shap_values(x)
        vals = vals[1] if isinstance(vals, list) else vals
        vals = np.asarray(vals).reshape(-1, len(FEATURE_NAMES))[0]
        order = np.argsort(-np.abs(vals))[: config.SHAP_TOP_K]
        return [{"feature": FEATURE_NAMES[i], "value": float(feats[FEATURE_NAMES[i]]), "shap": round(float(vals[i]), 4)}
                for i in order]

    def analyze(self, url: str | None, explain: bool = True) -> ModalityResult:
        """Score ``url``. Empty -> unavailable; unparseable -> unavailable with a MALFORMED_URL flag."""
        if url is None or not url.strip():
            return ModalityResult.unavailable("url")
        url = url.strip()[: config.MAX_URL_CHARS]
        try:
            host, _ = parse_host(url)
        except ValueError as exc:
            return ModalityResult(modality="url", available=False, extracted={"error": str(exc)},
                                  red_flags=[_flag("MALFORMED_URL", "Could not parse a valid web address.", "low", url[:80])])
        sig = self.live.check(url) if self.live else LiveSignals()
        target = sig.expanded_url or url  # analyse where a shortener really leads
        feats = extract_features(target)
        t_host, _ = parse_host(target)
        domain = registered_domain(t_host)
        imp = check_impersonation(target)
        flags = lexical_flags(url, extract_features(url), None) if target != url else []
        flags += lexical_flags(target, feats, imp)
        flags += live_flags(sig, domain)
        order = {"high": 0, "medium": 1, "low": 2}
        flags = sorted({(f.code, f.evidence): f for f in flags}.values(), key=lambda f: order[f.severity])
        p_model = self.model_proba(feats)
        explanation: dict = {"model": "url_lgbm" if self.blob else "rules_only", "model_probability": round(p_model, 4),
                             "threshold": self.threshold}
        if explain and self.blob is not None:
            try:
                explanation["shap_top_features"] = self._shap_top(feats)
            except Exception as exc:
                log.warning("SHAP failed: %s", exc)
        extracted = {"url": url, "host": t_host, "registered_domain": domain,
                     "impersonates": imp.company if imp and imp.kind in _STRONG_IMPERSONATION else None,
                     "expanded_url": sig.expanded_url, "domain_age_days": sig.domain_age_days,
                     "live_errors": sig.errors}
        return ModalityResult(modality="url", probability=min(max(boost(p_model, flags), 0.0), 1.0), red_flags=flags,
                              extracted=extracted, explanation=explanation)
