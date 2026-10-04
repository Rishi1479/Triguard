"""Inference-time metadata lookups (WHOIS age, TLS certificate, shortener expansion).

Strictly metadata: no page is ever fetched or rendered (HEAD only, redirects read but not followed
automatically). These are RULE signals at inference; they are never used as training features.
Every lookup has a timeout and results are cached.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin

from ml.common import config
from ml.url.features import SHORTENERS, parse_host, registered_domain

log = logging.getLogger(__name__)
MAX_REDIRECT_HOPS = 3
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-check")


@dataclass
class LiveSignals:
    """Results of the live metadata lookups; ``None`` means unknown/failed."""

    domain_age_days: int | None = None
    ssl_ok: bool | None = None
    ssl_issuer: str | None = None
    expanded_url: str | None = None
    errors: list[str] = field(default_factory=list)


def _run_with_timeout(fn, *args, timeout: float):
    fut = _POOL.submit(fn, *args)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        fut.cancel()
        raise TimeoutError(f"{fn.__name__} timed out after {timeout}s")


def _whois_age_days(domain: str) -> int | None:
    import whois  # python-whois

    created = whois.whois(domain).creation_date
    if isinstance(created, list):
        created = min((c for c in created if c), default=None)
    if not isinstance(created, datetime):
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - created).days, 0)


def _tls_info(host: str, timeout: float) -> tuple[bool, str | None]:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout=timeout) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as tls:
            cert = tls.getpeercert()
    issuer = dict(x[0] for x in cert.get("issuer", ())).get("organizationName")
    return True, issuer


def _is_public_host(host: str) -> bool:
    """Refuse private/loopback targets (SSRF guard) before any outbound request."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def _expand(url: str, timeout: float) -> str | None:
    """Follow shortener redirects with HEAD only (max hops); returns the last Location or None."""
    import httpx

    current = url if "://" in url else "http://" + url
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECT_HOPS):
            host, _p = parse_host(current)
            if registered_domain(host) not in SHORTENERS or not _is_public_host(host):
                break
            resp = client.head(current)
            loc = resp.headers.get("location")
            if resp.status_code not in (301, 302, 303, 307, 308) or not loc:
                break
            current = urljoin(current, loc)
    return current if current != (url if "://" in url else "http://" + url) else None


class LiveChecker:
    """Cached, time-limited live lookups. Methods never raise; failures are recorded in ``errors``."""

    def __init__(self, timeout: float = config.LIVE_TIMEOUT_S, ttl: int = config.LIVE_CACHE_TTL_S) -> None:
        self.timeout, self.ttl = timeout, ttl
        self._cache: dict[tuple[str, str], tuple[float, object]] = {}

    def _cached(self, kind: str, key: str, fn):
        now = time.time()
        hit = self._cache.get((kind, key))
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        value = fn()
        self._cache[(kind, key)] = (now, value)
        return value

    def check(self, url: str) -> LiveSignals:
        """Run WHOIS, TLS and shortener expansion for ``url``."""
        sig = LiveSignals()
        try:
            host, _ = parse_host(url)
        except ValueError as exc:
            sig.errors.append(str(exc))
            return sig
        domain = registered_domain(host)
        if domain in SHORTENERS:
            try:
                sig.expanded_url = self._cached("expand", url, lambda: _run_with_timeout(_expand, url, self.timeout, timeout=self.timeout * MAX_REDIRECT_HOPS))
            except Exception as exc:
                sig.errors.append(f"expand: {exc}")
            if sig.expanded_url:
                try:
                    host, _ = parse_host(sig.expanded_url)
                    domain = registered_domain(host)
                except ValueError:
                    pass
        try:
            sig.domain_age_days = self._cached("whois", domain, lambda: _run_with_timeout(_whois_age_days, domain, timeout=self.timeout))
        except Exception as exc:
            sig.errors.append(f"whois: {exc}")
        try:
            sig.ssl_ok, sig.ssl_issuer = self._cached("tls", host, lambda: _run_with_timeout(_tls_info, host, self.timeout, timeout=self.timeout + 1))
        except Exception as exc:
            sig.ssl_ok = False
            sig.errors.append(f"tls: {exc}")
        return sig
