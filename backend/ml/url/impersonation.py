"""Brand-impersonation / lookalike-domain detection against data/company_domains.csv."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from ml.common import config
from ml.url.features import parse_host, registered_domain

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t", "$": "s", "@": "a"})
MIN_SUBSTRING_BRAND_LEN = 4  # shorter brands (tcs, hcl, ibm) must match whole hyphen-separated tokens


@dataclass(frozen=True)
class Impersonation:
    """A suspected impersonation: which company, its real domain, and why we think so."""

    company: str
    real_domain: str
    kind: str  # "lookalike" | "brand_in_domain" | "brand_in_subdomain" | "brand_in_path"
    score: float


@lru_cache(maxsize=4)
def load_companies(path: Path | None = None) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    """(company, official registered domains, brand tokens) from the CSV."""
    df = pd.read_csv(path or config.COMPANY_DOMAINS_FILE).fillna("")
    out = []
    for _, r in df.iterrows():
        domains = tuple(registered_domain(d.strip()) for d in r["domains"].split(";") if d.strip())
        brands = {d.split(".")[0] for d in domains}
        brands |= {re.sub(r"[^a-z0-9]", "", a) for a in str(r["aliases"]).lower().split(";") if a.strip()}
        brands.add(re.sub(r"[^a-z0-9]", "", r["company"].lower()))
        out.append((r["company"], domains, tuple(sorted(b for b in brands if len(b) >= 2))))
    return tuple(out)


@lru_cache(maxsize=4)
def official_domains(path: Path | None = None) -> frozenset[str]:
    """Every registered domain listed in company_domains.csv (the set of employers we can recognise)."""
    return frozenset(d for _company, domains, _brands in load_companies(path) for d in domains)


def _label(domain: str) -> str:
    return domain.split(".")[0]


def _brand_in(label: str, brand: str) -> bool:
    tokens = re.split(r"[-_.]", label)
    if brand in tokens:
        return True
    return len(brand) >= MIN_SUBSTRING_BRAND_LEN and brand in label


def check_impersonation(url: str, companies_path: Path | None = None) -> Impersonation | None:
    """Return the best impersonation match for ``url`` or None. Official domains/subdomains never match."""
    host, rest = parse_host(url)
    reg = registered_domain(host)
    label = _label(reg)
    sub_part = host[: -len(reg)] if host.endswith(reg) else ""
    deleet = label.translate(_LEET)
    best: Impersonation | None = None
    for company, domains, brands in load_companies(companies_path):
        if reg in domains:
            return None  # genuine
        cands: list[Impersonation] = []
        for dom in domains:
            ratio = fuzz.ratio(deleet, _label(dom))
            if label != _label(dom) and (ratio >= config.LOOKALIKE_MIN_SCORE or deleet == _label(dom)):
                cands.append(Impersonation(company, dom, "lookalike", ratio))
        for b in brands:
            if _brand_in(deleet, b):
                cands.append(Impersonation(company, domains[0], "brand_in_domain", 90.0 + len(b) / 10))
            elif _brand_in(sub_part.translate(_LEET), b):
                cands.append(Impersonation(company, domains[0], "brand_in_subdomain", 80.0))
            elif len(b) >= MIN_SUBSTRING_BRAND_LEN and b in rest.lower():
                cands.append(Impersonation(company, domains[0], "brand_in_path", 50.0))
        for c in cands:
            if best is None or c.score > best.score:
                best = c
    return best
