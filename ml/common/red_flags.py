"""Rule-based red flags and entity extraction for job/internship text.

Rules are deliberately conservative and negation-aware ("we never ask for any fee" must not fire).
"""
from __future__ import annotations

import re
from typing import Iterable

from ml.common.privacy import EMAIL_RE, PHONE_RE
from ml.common.schemas import RedFlag, Severity

FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "outlook.com", "hotmail.com",
    "live.com", "rediffmail.com", "aol.com", "protonmail.com", "proton.me", "icloud.com", "mail.com",
    "ymail.com", "zoho.in", "gmx.com", "yandex.com",
})
NEGATION_WINDOW = 45  # chars before a match scanned for negation
EVIDENCE_PAD = 25
_NEGATIONS = re.compile(
    r"\b(?:no|not|never|without|zero|don'?t|do not|does not|doesn'?t|won'?t|will not|neither|nor|free of|beware|"
    r"avoid|fraud|scam|never ask)\b", re.I)

_AMOUNT = r"(?:₹|rs\.?|inr|usd|\$|€|£)\s?\d[\d,]*(?:\.\d+)?"
_FEE_WORDS = (r"(?:registration|training|processing|security|refundable|joining|application|admission|kit|"
              r"verification|onboarding|interview|certificate|laptop|courier|document)\s+"
              r"(?:fees?|charges?|deposit|amount|cost)")
_FEE_RES = [
    re.compile(_FEE_WORDS, re.I),
    re.compile(r"\b(?:security|refundable|caution)\s+deposit\b", re.I),
    re.compile(rf"\b(?:pay|deposit|transfer|send|remit|submit)\b[^.\n]{{0,40}}?{_AMOUNT}", re.I),
    re.compile(rf"{_AMOUNT}[^.\n]{{0,30}}?\b(?:fee|charges?|deposit)\b", re.I),
    re.compile(r"\b(?:pay|paid|payment)\b[^.\n]{0,30}\b(?:via|by|through|on)\s+(?:upi|paytm|phonepe|gpay|google pay|western union|"
               r"bitcoin|crypto|gift cards?)\b", re.I),
    re.compile(r"\b(?:western union|moneygram|gift cards?)\b", re.I),
]
_URGENCY_STRONG = re.compile(
    r"\b(?:urgent(?:ly)?|hurry|limited seats?|last chance|offer (?:expires?|valid)[^.\n]{0,25}|"
    r"valid for \d+ hours?|only \d+ (?:seats?|vacancies|positions?) (?:left|remaining)|apply immediately|"
    r"act now|expires? (?:today|tonight)|joining (?:within|in) \d+ hours?|positions? fill fast)\b", re.I)
_URGENCY_MILD = re.compile(r"\b(?:immediate joining|join immediately|immediately)\b", re.I)
_NO_INTERVIEW = re.compile(
    r"\b(?:no|without|skip(?:ping)?)\s+(?:any\s+)?interview(?:s)?(?:\s+(?:needed|required|necessary))?\b|"
    r"\b(?:selection|job|placement|offer|joining)\s+(?:is\s+)?(?:100%\s+)?guaranteed\b|"
    r"\bguaranteed\s+(?:job|selection|placement|offer|joining)\b|\b100%\s+(?:job|placement)\b|"
    r"\bdirect\s+(?:joining|selection)\b|\bselected\s+without\b", re.I)
_PAY_PERIOD = r"(?:per|a|/|each)?\s?(?:day|daily|week|weekly|hour|hr)"
_UNREALISTIC_PAY = [
    re.compile(rf"\b(?:earn|income|make|salary|payout|get)\b[^.\n]{{0,25}}?{_AMOUNT}\s?(?:k)?\s*{_PAY_PERIOD}\b", re.I),
    re.compile(rf"{_AMOUNT}\s*{_PAY_PERIOD}\s+(?:guaranteed|assured)", re.I),
    re.compile(r"\b(?:unlimited|guaranteed|assured)\s+(?:income|earnings?|salary|payout|money)\b", re.I),
    re.compile(r"\bearn\s+(?:lakhs?|thousands?|crores?)\b", re.I),
    re.compile(r"\bdaily\s+(?:payment|payout|salary)\b", re.I),
]
_MESSENGER = re.compile(r"\b(?:whats\s?app|telegram|signal|wa\.me|t\.me)\b", re.I)
_MESSENGER_CONTEXT = re.compile(
    r"\b(?:only|contact|message|msg|join|group|channel|dm|chat|reach|number|call|apply)\b", re.I)
_SENSITIVE_ITEM = re.compile(
    r"\b(?:aadhaar|aadhar|pan\s*(?:card|number|no)?|bank\s+(?:account|details|statement)|account\s+(?:number|details)|"
    r"ifsc|otp|cvv|debit card|credit card|atm\s+pin|passport|password|ssn|social security|upi\s+pin)\b", re.I)
_REQUEST_VERB = re.compile(
    r"\b(?:send|share|provide|submit|give|enter|upload|email|mail|whatsapp|forward|tell|confirm|verify|"
    r"need|require[sd]?|send us|attach)\b", re.I)
_HR_CONTEXT = re.compile(r"\b(?:hr|human resources|recruit\w*|careers?|jobs?|hiring|apply|resume|cv|vacanc\w+|offer)\b", re.I)


def _snippet(text: str, start: int, end: int) -> str:
    return text[max(0, start - EVIDENCE_PAD): end + EVIDENCE_PAD].strip()


def _negated(text: str, start: int) -> bool:
    return bool(_NEGATIONS.search(text[max(0, start - NEGATION_WINDOW): start]))


def _flag(code: str, message: str, severity: Severity, evidence: str) -> RedFlag:
    return RedFlag(code=code, message=message, severity=severity, evidence=evidence)


def _first_unnegated(text: str, patterns: Iterable[re.Pattern]) -> re.Match | None:
    for pat in patterns:
        for m in pat.finditer(text):
            if not _negated(text, m.start()):
                return m
    return None


def _check_fee(text: str) -> list[RedFlag]:
    m = _first_unnegated(text, _FEE_RES)
    return [_flag("PAYMENT_REQUEST", "Asks you to pay a fee or deposit — genuine employers never charge candidates.",
                  "high", _snippet(text, m.start(), m.end()))] if m else []


def _check_urgency(text: str) -> list[RedFlag]:
    m = _first_unnegated(text, [_URGENCY_STRONG])
    if m:
        return [_flag("URGENCY", "Creates artificial urgency to rush your decision.", "medium",
                      _snippet(text, m.start(), m.end()))]
    m = _first_unnegated(text, [_URGENCY_MILD])
    return [_flag("URGENCY_MILD", "Mentions immediate joining/urgency.", "low",
                  _snippet(text, m.start(), m.end()))] if m else []


def _check_no_interview(text: str) -> list[RedFlag]:
    m = _first_unnegated(text, [_NO_INTERVIEW])
    return [_flag("NO_INTERVIEW", "Promises selection with no interview or a guaranteed job.", "high",
                  _snippet(text, m.start(), m.end()))] if m else []


def _check_pay(text: str) -> list[RedFlag]:
    m = _first_unnegated(text, _UNREALISTIC_PAY)
    return [_flag("UNREALISTIC_PAY", "Promises unrealistic or guaranteed earnings.", "medium",
                  _snippet(text, m.start(), m.end()))] if m else []


def _check_messenger(text: str) -> list[RedFlag]:
    for m in _MESSENGER.finditer(text):
        window = text[max(0, m.start() - 60): m.end() + 60]
        if _MESSENGER_CONTEXT.search(window) and not _negated(text, m.start()):
            return [_flag("MESSENGER_ONLY_CONTACT", "Recruitment via WhatsApp/Telegram instead of official channels.",
                          "medium", _snippet(text, m.start(), m.end()))]
    return []


def _check_free_email(text: str) -> list[RedFlag]:
    for m in EMAIL_RE.finditer(text):
        domain = m.group(0).rsplit("@", 1)[1].lower()
        if domain in FREE_EMAIL_DOMAINS:
            hr_like = bool(_HR_CONTEXT.search(text[max(0, m.start() - 120): m.end() + 60])
                           or re.search(r"hr|job|career|recruit|hiring", m.group(0).split("@")[0], re.I))
            if hr_like:
                return [_flag("FREE_EMAIL_HR", f"Recruiter uses a free e-mail provider ({domain}) instead of a company domain.",
                              "medium", m.group(0))]
    return []


def _check_sensitive(text: str) -> list[RedFlag]:
    for m in _SENSITIVE_ITEM.finditer(text):
        before = text[max(0, m.start() - 70): m.start()]
        after = text[m.end(): m.end() + 25]
        verb = _REQUEST_VERB.search(before) or re.match(
            r"\s*(?:copy|copies|details|number|card|photo|scan)?\s*(?:is|are)?\s*(?:required|needed)", after, re.I)
        if verb and not _negated(text, m.start()):
            code, msg = (("OTP_REQUEST", "Asks for an OTP/PIN — never share these with anyone.")
                         if re.search(r"otp|cvv|pin|password", m.group(0), re.I) else
                         ("SENSITIVE_INFO_REQUEST", "Asks for Aadhaar/PAN/bank or other sensitive details up front."))
            return [_flag(code, msg, "high", _snippet(text, m.start(), m.end()))]
    return []


_CHECKS = (_check_fee, _check_no_interview, _check_sensitive, _check_urgency, _check_pay,
           _check_messenger, _check_free_email)


def detect_red_flags(text: str) -> list[RedFlag]:
    """Run all rules over ``text``. Returns flags ordered high -> low severity."""
    if not text or not text.strip():
        return []
    flags = [f for check in _CHECKS for f in check(text)]
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(flags, key=lambda f: order[f.severity])


# ---------------------------------------------------------------- entity extraction
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"')\]]+|"
    r"\b(?:forms\.gle|bit\.ly|tinyurl\.com|t\.me|wa\.me|goo\.gl|rb\.gy|cutt\.ly|docs\.google\.com/forms)/[^\s<>\"')\]]+",
    re.I)
_BARE_DOMAIN_RE = re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|org|net|io|co|xyz|online|top|site|info|biz|link|work|click|app)\b(?:/[^\s]*)?", re.I)
_COMPANY_SUFFIX = re.compile(
    r"\b((?:[A-Z][\w&.-]*\s+){1,3}(?:Pvt\.?\s*Ltd\.?|Private Limited|Limited|Ltd\.?|Inc\.?|LLC|LLP|Technologies|Solutions|"
    r"Systems|Labs|Corp(?:oration)?|Consultancy|Services|Infotech))\b")
_COMPANY_VERBS = re.compile(
    r"\b([A-Z][\w&-]+(?:\s+[A-Z][\w&-]+){0,2})\s+(?:is|are)\s+(?:hiring|looking|seeking|inviting|recruiting)\b")
_COMPANY_AT = re.compile(r"\b(?:at|join|with|from|for)\s+([A-Z][\w&-]+(?:\s+[A-Z][\w&-]+){0,2})\b")
_STOP_COMPANY = frozenset({
    "We", "Our", "The", "You", "Your", "This", "That", "Candidates", "Apply", "Contact", "Email", "Please", "Pay",
    "India", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Mumbai", "Delhi", "Hr", "HR", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Whatsapp", "WhatsApp", "Telegram", "Aadhaar", "Congratulations",
    "Immediate", "Immediately", "Work", "Part", "Full", "A", "An", "Software", "Data", "Senior", "Junior",
})


def extract_urls(text: str) -> list[str]:
    """URLs (explicit scheme/www, known link shorteners/platforms, and bare domains), trailing punctuation stripped."""
    found = [m.group(0).rstrip(".,;:!?") for m in _URL_RE.finditer(text)]
    emails = {m.group(0).rsplit("@", 1)[1].lower() for m in EMAIL_RE.finditer(text)}
    for m in _BARE_DOMAIN_RE.finditer(text):
        cand = m.group(0).rstrip(".,;:!?")
        host = cand.split("/")[0].lower()
        if host in emails or any(host in u.lower() for u in found) or text[max(0, m.start() - 1): m.start()] == "@":
            continue
        found.append(cand)
    return list(dict.fromkeys(found))


def extract_companies(text: str) -> list[str]:
    """Heuristic company-name mentions (suffix words, '<X> is hiring', 'at <X>')."""
    cands: list[str] = []
    for rx in (_COMPANY_SUFFIX, _COMPANY_VERBS, _COMPANY_AT):
        for m in rx.finditer(text):
            name = m.group(1).strip(" .,-")
            words = name.split()
            while words and words[0] in _STOP_COMPANY:
                words.pop(0)
            name = " ".join(words)
            if len(name) >= 2 and name not in _STOP_COMPANY and not name.lower().startswith("http"):
                cands.append(name)
    return list(dict.fromkeys(cands))


def extract_entities(text: str) -> dict[str, list[str]]:
    """URLs, e-mails, phone numbers and company names found in ``text`` (for cross-modal checks)."""
    emails = list(dict.fromkeys(m.group(0) for m in EMAIL_RE.finditer(text)))
    phones = list(dict.fromkeys(m.group(0).strip() for m in PHONE_RE.finditer(text)))
    return {"urls": extract_urls(text), "emails": emails, "phones": phones, "companies": extract_companies(text)}
