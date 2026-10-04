"""Cross-modal consistency checks between text, image (OCR/QR) and URL results."""
from __future__ import annotations

import re
from itertools import chain

from ml.common import config
from ml.common.red_flags import FREE_EMAIL_DOMAINS
from ml.common.schemas import ModalityResult, RedFlag, Severity
from ml.url.features import parse_host, registered_domain
from ml.url.impersonation import check_impersonation, load_companies

_NORM = re.compile(r"[^a-z0-9]+")
_MULT = [(r"lpa|lakhs? per annum|lakhs?", 100_000.0), (r"crores?", 10_000_000.0), (r"k", 1_000.0)]
_SALARY_RE = re.compile(
    r"(?:₹|rs\.?|inr|usd|\$)\s?(\d[\d,]*(?:\.\d+)?)\s?(lpa|lakhs?(?: per annum)?|crores?|k)?"
    r"(?:\s*(?:/|per|a)?\s*(month|monthly|pm|year|annum|yearly|pa|week|weekly|day|daily))?", re.I)
_PERIOD_MULT = {"month": 12, "monthly": 12, "pm": 12, "week": 52, "weekly": 52, "day": 365, "daily": 365}
_ROLE_RE = re.compile(r"(?:position|role|post|designation|as|hiring(?: a| an)?|for)\s+(?:of\s+)?(?:a\s+|an\s+|the\s+)?"
                      r"([A-Z][A-Za-z/&-]+(?:\s+[A-Z][A-Za-z/&-]+){0,3})")
_ROLE_STOP = frozenset({"At", "In", "The", "You", "We", "Our", "Your", "Pay", "India"})


def _flag(code: str, msg: str, sev: Severity, evidence: str) -> RedFlag:
    return RedFlag(code=code, message=msg, severity=sev, evidence=evidence)


def _norm(s: str) -> str:
    return _NORM.sub("", s.lower())


def known_companies(names: list[str]) -> dict[str, tuple[str, ...]]:
    """Map mentioned company names that exist in company_domains.csv to their official domains."""
    out: dict[str, tuple[str, ...]] = {}
    for name in names:
        n = _norm(name)
        for company, domains, brands in load_companies():
            if n and (n == _norm(company) or n in brands or any(len(b) >= 3 and b == n for b in brands)
                      or _norm(company) in n.split() or n.startswith(_norm(company))):
                out[company] = domains
    return out


def extract_salaries(text: str) -> list[float]:
    """Annualised salary figures mentioned in ``text``."""
    vals = []
    for m in _SALARY_RE.finditer(text):
        num, unit, period = float(m.group(1).replace(",", "")), (m.group(2) or "").lower(), (m.group(3) or "").lower()
        for pat, mult in _MULT:
            if re.fullmatch(pat, unit):
                num *= mult
                break
        vals.append(num * _PERIOD_MULT.get(period, 1))
    return [v for v in vals if v >= 1000]


def extract_roles(text: str) -> list[str]:
    """Job-title phrases (heuristic)."""
    roles = []
    for m in _ROLE_RE.finditer(text):
        words = [w for w in m.group(1).split() if w not in _ROLE_STOP]
        if words:
            roles.append(" ".join(words))
    return roles


def _domain_of(url: str) -> str | None:
    try:
        return registered_domain(parse_host(url)[0])
    except ValueError:
        return None


def _all(results: list[ModalityResult], key: str) -> list[str]:
    return list(dict.fromkeys(chain.from_iterable(r.extracted.get(key, []) or [] for r in results if r.available)))


def _company_checks(text_res, image_res, url_domains: list[str]) -> list[RedFlag]:
    flags: list[RedFlag] = []
    t_comp = known_companies(text_res.extracted.get("companies", [])) if text_res.available else {}
    i_comp = known_companies(image_res.extracted.get("companies", [])) if image_res.available else {}
    if t_comp and i_comp and not (set(t_comp) & set(i_comp)):
        flags.append(_flag("COMPANY_MISMATCH_TEXT_IMAGE", "The text and the image name different companies.", "high",
                           f"text: {', '.join(t_comp)} / image: {', '.join(i_comp)}"))
    named = {**t_comp, **i_comp}
    for dom in url_domains:
        if dom in config.ATS_DOMAINS:
            continue
        for company, official in named.items():
            if dom not in official:
                imp = check_impersonation("http://" + dom)
                sev: Severity = "high" if imp and imp.company == company else "medium"
                flags.append(_flag("COMPANY_URL_MISMATCH",
                                   f"The posting names {company} but links to {dom}, not its official site ({official[0]}).",
                                   sev, dom))
    return flags


def _email_checks(results: list[ModalityResult], named: dict[str, tuple[str, ...]]) -> list[RedFlag]:
    flags: list[RedFlag] = []
    for email in _all(results, "emails"):
        dom = email.rsplit("@", 1)[-1].lower()
        reg = registered_domain(dom)
        if reg in FREE_EMAIL_DOMAINS and named:
            flags.append(_flag("FREE_EMAIL_FOR_NAMED_COMPANY", f"Claims to be {next(iter(named))} but contacts you from a free "
                               f"e-mail address ({reg}).", "high", email))
        elif named and not any(reg in d for d in named.values()):
            imp = check_impersonation("http://" + reg)
            flags.append(_flag("HR_EMAIL_DOMAIN_MISMATCH", f"Recruiter e-mail domain {reg} does not match "
                               f"{next(iter(named))}'s official domain.", "high" if imp else "medium", email))
    return flags


def _qr_checks(image_res, stated_domains: list[str]) -> list[RedFlag]:
    flags = []
    for link in (image_res.extracted.get("qr_links", []) if image_res.available else []):
        dom = _domain_of(link)
        if dom and stated_domains and dom not in stated_domains:
            flags.append(_flag("QR_URL_MISMATCH", f"The QR code leads to {dom}, which differs from the link given "
                               f"({', '.join(stated_domains)}).", "high", link[:100]))
    return flags


def _content_checks(text_res, image_res, raw_text: str) -> list[RedFlag]:
    flags: list[RedFlag] = []
    if not (text_res.available and image_res.available):
        return flags
    t_sal, i_sal = extract_salaries(raw_text), \
        extract_salaries(image_res.extracted.get("ocr_text", ""))
    if t_sal and i_sal and max(max(t_sal), max(i_sal)) / max(min(max(t_sal), max(i_sal)), 1) > config.SALARY_MISMATCH_RATIO:
        flags.append(_flag("SALARY_MISMATCH", "The salary in the text differs from the salary in the image.", "medium",
                           f"text ≈ {max(t_sal):,.0f}/yr vs image ≈ {max(i_sal):,.0f}/yr"))
    t_roles, i_roles = extract_roles(raw_text), extract_roles(image_res.extracted.get("ocr_text", ""))
    if t_roles and i_roles:
        a, b = {w.lower() for r in t_roles for w in r.split()}, {w.lower() for r in i_roles for w in r.split()}
        if len(a & b) / len(a | b) < config.ROLE_MISMATCH_JACCARD:
            flags.append(_flag("ROLE_MISMATCH", "The job role in the text differs from the role in the image.", "medium",
                               f"text: {t_roles[0]} / image: {i_roles[0]}"))
    return flags


def cross_modal_checks(text_res: ModalityResult, image_res: ModalityResult, url_results: list[ModalityResult],
                       raw_text: str = "") -> list[RedFlag]:
    """All cross-modal red flags for one analysis, high severity first. ``raw_text`` is the submitted text (not stored)."""
    avail = [r for r in (text_res, image_res) if r.available]
    named = {**(known_companies(text_res.extracted.get("companies", [])) if text_res.available else {}),
             **(known_companies(image_res.extracted.get("companies", [])) if image_res.available else {})}
    url_domains = [d for d in (r.extracted.get("registered_domain") for r in url_results if r.available) if d]
    primary = url_domains[:1]
    flags = _company_checks(text_res, image_res, primary)
    flags += _email_checks(avail, named)
    flags += _qr_checks(image_res, primary or [d for d in (_domain_of(u) for u in (text_res.extracted.get("urls", []) if text_res.available else [])) if d])
    flags += _content_checks(text_res, image_res, raw_text)
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted({(f.code, f.evidence): f for f in flags}.values(), key=lambda f: order[f.severity])
