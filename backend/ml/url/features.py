"""Lexical URL features (no network access). Used for training and inference."""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import tldextract

from ml.common import config

# Offline extractor: bundled public-suffix snapshot, never fetches the list.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

SUSPICIOUS_TLDS = frozenset({"xyz", "top", "online", "site", "click", "work", "tk", "ml", "ga", "cf", "gq", "info",
                             "biz", "link", "icu", "buzz", "live", "rest", "fit", "monster", "cyou", "support"})
# RFC 2606 / RFC 6761 reserved names: they can never resolve to a real employer's website, so a
# posting that links to one is using a placeholder or a fabricated address. Rules-only (a trained
# model cannot learn from them: no real-world URL corpus contains them).
RESERVED_TLDS = frozenset({"example", "test", "invalid", "localhost", "local", "internal", "lan", "home", "corp"})
RESERVED_DOMAINS = frozenset({"example.com", "example.net", "example.org"})
SHORTENERS = frozenset({"bit.ly", "tinyurl.com", "goo.gl", "t.co", "rb.gy", "cutt.ly", "is.gd", "ow.ly", "shorturl.at",
                        "tiny.cc", "buff.ly", "rebrand.ly", "bl.ink", "s.id", "lnkd.in"})
PLATFORM_LINKS = ("forms.gle", "docs.google.com/forms", "t.me", "wa.me", "api.whatsapp.com", "chat.whatsapp.com",
                  "telegram.me", "linktr.ee", "typeform.com", "forms.office.com", "bit.ly")
KEYWORDS = ("login", "verify", "job", "hiring", "apply", "offer", "career", "registration", "secure", "account",
            "update", "confirm", "recruit", "internship", "payment", "wallet", "bonus", "free")
_SPECIAL = {"dots": ".", "hyphens": "-", "underscores": "_", "slashes": "/", "question": "?", "equals": "=",
            "at": "@", "ampersand": "&", "percent": "%", "hash": "#"}
# Scheme and a leading "www." (only when a real domain follows it, so "www.com" stays intact).
_SCHEME_WWW_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\d*\.(?=[^/?#]*\.))?", re.I)

# No scheme-based feature (e.g. is_https): labelled corpora write benign and malicious URLs differently
# (Kaggle malicious_phish: 8% of benign URLs carry a scheme vs 96-100% of malware/defacement), so such a
# feature learns the dataset's formatting, not the URL. HTTPS is still checked by the NO_HTTPS rule flag.
FEATURE_NAMES: list[str] = [
    "url_len", "host_len", "path_len", "query_len", *[f"n_{k}" for k in _SPECIAL], "n_digits", "digit_ratio",
    "n_letters", "subdomain_depth", "is_ip_host", "suspicious_tld", "tld_len", "entropy", "host_entropy",
    "is_shortener", "keyword_hits", "keyword_in_host", "platform_link", "has_port", "n_params",
    "double_slash_in_path", "longest_token", "host_hyphen_digit", "n_host_tokens", "has_punycode",
]


def parse_host(url: str) -> tuple[str, str]:
    """Return (host, path+query) for ``url``, adding a scheme if missing. Raises ValueError if no host."""
    url = url.strip()[: config.MAX_URL_CHARS]
    if re.match(r"^(?:javascript|data|mailto|file|ftp|vbscript|tel|sms):", url, re.I):
        raise ValueError(f"unsupported URL scheme: {url[:20]!r}")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host or " " in host or ("." not in host and not is_ip(host)):
        raise ValueError(f"no valid host in URL: {url[:60]!r}")
    return host, parts.path + (("?" + parts.query) if parts.query else "")


def entropy(s: str) -> float:
    """Shannon entropy (bits/char) of ``s``."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def is_ip(host: str) -> bool:
    """True if ``host`` is a bare IPv4/IPv6 address rather than a domain name."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def registered_domain(host: str) -> str:
    """eTLD+1 of ``host`` (e.g. careers.google.com -> google.com); IPs/odd hosts returned as-is."""
    if is_ip(host):
        return host
    ext = _TLD(host)
    return ".".join(p for p in (ext.domain, ext.suffix) if p) or host


def is_reserved_domain(host: str) -> bool:
    """True if ``host`` sits under a reserved/placeholder name (``.example``, ``.test``, ``example.com``...)."""
    if is_ip(host):
        return False
    return host.rsplit(".", 1)[-1] in RESERVED_TLDS or registered_domain(host) in RESERVED_DOMAINS


def canonical_url(url: str) -> str:
    """``url`` without scheme, leading ``www.`` and trailing ``/`` (e.g. ``HTTPS://www.a.com/`` -> ``a.com``).

    Features are computed on this form so the same address scores the same however it was written.
    """
    return _SCHEME_WWW_RE.sub("", url.strip()[: config.MAX_URL_CHARS]).rstrip("/")


def extract_features(url: str) -> dict[str, float]:
    """Compute the numeric feature dict (keys == FEATURE_NAMES) for a URL string."""
    parse_host(url)  # rejects unsupported schemes / missing host before the scheme is stripped
    raw = canonical_url(url)
    host, rest = parse_host(raw)
    low = raw.lower()
    path, _, query = rest.partition("?")
    ext = _TLD(host) if not is_ip(host) else None
    sub = ext.subdomain if ext else ""
    f: dict[str, float] = {
        "url_len": len(raw), "host_len": len(host), "path_len": len(path), "query_len": len(query),
    }
    for name, ch in _SPECIAL.items():
        f[f"n_{name}"] = raw.count(ch)
    digits = sum(c.isdigit() for c in raw)
    f.update(
        n_digits=digits, digit_ratio=digits / max(len(raw), 1), n_letters=sum(c.isalpha() for c in raw),
        subdomain_depth=len([s for s in sub.split(".") if s]), is_ip_host=float(is_ip(host)),
        suspicious_tld=float(bool(ext) and ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        tld_len=len(ext.suffix) if ext else 0, entropy=entropy(raw), host_entropy=entropy(host),
        is_shortener=float(registered_domain(host) in SHORTENERS),
        keyword_hits=sum(low.count(k) for k in KEYWORDS), keyword_in_host=sum(k in host for k in KEYWORDS),
        platform_link=float(any(p in low for p in PLATFORM_LINKS)),
        has_port=float(bool(re.match(r"^[^/?#]*:\d+", low))),
        n_params=len([p for p in query.split("&") if p]), double_slash_in_path=float("//" in path),
        longest_token=max((len(t) for t in re.split(r"[^a-z0-9]+", low) if t), default=0),
        host_hyphen_digit=float("-" in host and any(c.isdigit() for c in host)),
        n_host_tokens=len([t for t in re.split(r"[.-]", host) if t]), has_punycode=float("xn--" in host),
    )
    return f


def features_matrix(urls: list[str]) -> list[list[float]]:
    """Feature rows in FEATURE_NAMES order; unparseable URLs get an all-zero row."""
    rows = []
    for u in urls:
        try:
            d = extract_features(u)
            rows.append([float(d[k]) for k in FEATURE_NAMES])
        except ValueError:
            rows.append([0.0] * len(FEATURE_NAMES))
    return rows
