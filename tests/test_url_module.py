import joblib
import pytest

from ml.common import config
from ml.common.calibration import PlattCalibrator
from ml.url import live_checks
from ml.url.analyzer import URLAnalyzer, boost, live_flags
from ml.url.features import FEATURE_NAMES, entropy, extract_features, features_matrix, parse_host, registered_domain
from ml.url.impersonation import check_impersonation
from ml.url.live_checks import LiveChecker, LiveSignals


class FakeLive:
    def __init__(self, **kw):
        self.sig = LiveSignals(**kw)
        self.calls = []

    def check(self, url):
        self.calls.append(url)
        return self.sig


def test_features_keys_and_values():
    f = extract_features("http://tcs-hiring.online/apply?id=1&x=2")
    assert set(f) == set(FEATURE_NAMES)
    assert f["n_hyphens"] == 1 and f["n_question"] == 1 and f["n_params"] == 2
    assert f["suspicious_tld"] == 1 and f["is_https"] == 0 and f["keyword_hits"] >= 2
    assert extract_features("https://www.tcs.com/careers")["suspicious_tld"] == 0


def test_features_special_cases():
    assert extract_features("http://192.168.1.5/job.html")["is_ip_host"] == 1
    assert extract_features("https://bit.ly/3abc")["is_shortener"] == 1
    assert extract_features("https://forms.gle/AbC")["platform_link"] == 1
    assert extract_features("https://a.b.c.example.com/x")["subdomain_depth"] == 3
    assert extract_features("http://user@evil.com/")["n_at"] == 1
    assert entropy("") == 0 and entropy("aaaa") == 0 and entropy("abcd") == pytest.approx(2.0)


def test_scheme_less_and_bad_urls():
    assert parse_host("www.tcs.com/careers")[0] == "www.tcs.com"
    for bad in ["", "not a url", "javascript:alert(1)", "mailto:a@b.com", "localhost", "http://"]:
        with pytest.raises(ValueError):
            parse_host(bad)
    assert features_matrix(["bad url", "http://a.com"])[0] == [0.0] * len(FEATURE_NAMES)
    assert registered_domain("careers.google.co.in") == "google.co.in"


@pytest.mark.parametrize("url,kind", [
    ("http://tcs-hiring.online/apply", "brand_in_domain"),
    ("http://infosyss.com", "lookalike"),
    ("http://wipr0.com", "lookalike"),
    ("http://g00gle-careers.top", "brand_in_domain"),
    ("http://tcs.com.evil.xyz/login", "brand_in_subdomain"),
    ("https://evil.example/amazon/jobs", "brand_in_path"),
])
def test_impersonation_positive(url, kind):
    assert check_impersonation(url).kind == kind


@pytest.mark.parametrize("url", [
    "https://www.tcs.com/careers", "https://careers.google.com/jobs", "https://github.com", "https://hclub.in",
    "https://www.thecsstudio.com", "https://www.linkedin.com/company/amazon/jobs", "https://www.amazon.jobs/en",
])
def test_impersonation_negative(url):
    assert check_impersonation(url) is None


def test_company_csv_size():
    from ml.url.impersonation import load_companies
    assert len(load_companies()) >= 50


@pytest.fixture(scope="module")
def url_models(tmp_path_factory):
    import lightgbm as lgb
    import numpy as np
    from ml.common.synthetic import make_urls
    df = make_urls(300)
    x = np.array(features_matrix(df.url.tolist()))
    y = (df.type != "benign").astype(int).to_numpy()
    clf = lgb.LGBMClassifier(n_estimators=40, verbose=-1, random_state=0).fit(x, y)
    cal = PlattCalibrator().fit(clf.predict_proba(x)[:, 1], y)
    d = tmp_path_factory.mktemp("m") / "url_lgbm"
    d.mkdir()
    joblib.dump({"model": clf, "calibrator": cal, "threshold": 0.5, "features": FEATURE_NAMES}, d / "model.joblib")
    return d.parent


def test_analyzer_scam_vs_genuine(url_models):
    an = URLAnalyzer(models_dir=url_models, enable_live=False)
    s = an.analyze("http://tcs-hiring.online/apply?id=77")
    g = an.analyze("https://www.tcs.com/careers")
    assert s.available and s.probability > 0.7 > 0.3 > g.probability
    assert "BRAND_IMPERSONATION" in {f.code for f in s.red_flags}
    assert s.extracted["impersonates"] == "TCS" and s.extracted["registered_domain"] == "tcs-hiring.online"
    assert s.explanation["shap_top_features"] and "feature" in s.explanation["shap_top_features"][0]
    assert s.red_flags[0].severity == "high"


def test_analyzer_rules_only_without_model(tmp_path):
    r = URLAnalyzer(models_dir=tmp_path, enable_live=False).analyze("http://tcs-hiring.online")
    assert r.explanation["model"] == "rules_only" and r.probability > config.RULES_ONLY_PRIOR


@pytest.mark.parametrize("bad", [None, "", "  "])
def test_analyzer_empty(url_models, bad):
    assert not URLAnalyzer(models_dir=url_models, enable_live=False).analyze(bad).available


def test_analyzer_malformed_and_huge(url_models):
    an = URLAnalyzer(models_dir=url_models, enable_live=False)
    r = an.analyze("javascript:alert(1)")
    assert not r.available and r.red_flags[0].code == "MALFORMED_URL"
    assert 0 <= an.analyze("http://a.com/" + "x" * 100_000, explain=False).probability <= 1


def test_live_signals_become_flags(url_models):
    fake = FakeLive(domain_age_days=5, ssl_ok=False)
    r = URLAnalyzer(models_dir=url_models, live_checker=fake).analyze("https://some-company-careers.com/apply")
    codes = {f.code: f.severity for f in r.red_flags}
    assert codes["NEW_DOMAIN"] == "high" and codes["SSL_INVALID"] == "medium"
    assert r.extracted["domain_age_days"] == 5 and fake.calls
    old = live_flags(LiveSignals(domain_age_days=4000, ssl_ok=True), "x.com")
    assert old == []
    assert live_flags(LiveSignals(domain_age_days=100), "x.com")[0].severity == "medium"


def test_shortener_expansion_is_analysed(url_models):
    fake = FakeLive(expanded_url="http://tcs-hiring.online/apply")
    r = URLAnalyzer(models_dir=url_models, live_checker=fake).analyze("https://bit.ly/3abc")
    codes = {f.code for f in r.red_flags}
    assert {"URL_SHORTENER", "BRAND_IMPERSONATION"} <= codes
    assert r.extracted["expanded_url"] == "http://tcs-hiring.online/apply"


def test_boost():
    from ml.common.schemas import RedFlag
    assert boost(0.1, []) == pytest.approx(0.1)
    assert boost(0.0, [RedFlag(code="x", message="m", severity="high")]) == pytest.approx(config.URL_RULE_BOOST["high"])


# ------------------------------------------------ live checks (all network mocked)
def test_live_checker_caches_and_survives_failures(monkeypatch):
    calls = {"whois": 0}

    def whois(domain):
        calls["whois"] += 1
        return 12

    monkeypatch.setattr(live_checks, "_whois_age_days", whois)
    monkeypatch.setattr(live_checks, "_tls_info", lambda host, t: (_ for _ in ()).throw(OSError("refused")))
    lc = LiveChecker(timeout=1)
    s1, s2 = lc.check("http://example-job.com"), lc.check("http://example-job.com/other")
    assert s1.domain_age_days == 12 and s1.ssl_ok is False and any("tls" in e for e in s1.errors)
    assert calls["whois"] == 1 and s2.domain_age_days == 12  # cached by domain


def test_live_checker_timeout(monkeypatch):
    import time
    monkeypatch.setattr(live_checks, "_whois_age_days", lambda d: time.sleep(2))
    monkeypatch.setattr(live_checks, "_tls_info", lambda h, t: (True, "X"))
    t0 = time.time()
    s = LiveChecker(timeout=0.2).check("http://slow-whois.com")
    assert time.time() - t0 < 1.5 and s.domain_age_days is None and any("timed out" in e for e in s.errors)


def test_live_checker_bad_url_no_raise():
    assert "no valid host" in LiveChecker().check("not a url").errors[0]


def test_shortener_expand_head_only_and_ssrf_guard(monkeypatch):
    import httpx
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(302, headers={"location": "http://tcs-hiring.online/apply"})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(live_checks, "_is_public_host", lambda h: True)
    assert live_checks._expand("https://bit.ly/abc", 1) == "http://tcs-hiring.online/apply"
    assert seen == ["HEAD"]
    seen.clear()
    monkeypatch.setattr(live_checks, "_is_public_host", lambda h: False)
    assert live_checks._expand("https://bit.ly/abc", 1) is None and seen == []
    assert live_checks._expand("https://www.tcs.com", 1) is None and seen == []  # not a shortener: never contacted
