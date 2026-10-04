import numpy as np
import pytest

from ml.common.schemas import ModalityResult, RedFlag
from ml.common.synthetic import make_emscad_like, make_urls
from ml.common.textprep import merge_columns
from reports import evaluate_all as ev


def test_md_table_and_mean_std():
    t = ev.md_table(["a", "b"], [["x", 0.12345], ["y", 2]])
    assert t.splitlines()[0] == "| a | b |" and "| x | 0.123 |" in t and "| y | 2 |" in t
    ms = ev.mean_std([{m: 0.5 for m in ev.METRICS}, {m: 0.7 for m in ev.METRICS}])
    assert ms["f1"] == "0.600 ± 0.100"


def test_cv_text_and_url_run_with_folds():
    raw = make_emscad_like(120, 40)
    out = ev.cv_text(merge_columns(raw).tolist(), raw["fraudulent"].to_numpy(), bilstm=False)
    assert set(out) == {"TF-IDF + LR"} and all("±" in v for v in out["TF-IDF + LR"].values())
    u = make_urls(80)
    out = ev.cv_url(u["url"].tolist(), (u["type"] != "benign").astype(int).to_numpy())
    assert float(out["LightGBM (lexical)"]["f1"].split()[0]) > 0.8


def make_cases(n=60, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        y = int(i % 3 == 0)
        mk = lambda m, signal: ModalityResult(modality=m, probability=float(np.clip(rng.normal(0.75 if y else 0.25, signal), 0, 1)),
                                              available=bool(rng.random() > 0.2))
        cross = [RedFlag(code="X", message="m", severity="high")] if y and rng.random() > 0.3 else []
        cases.append({"case_id": i, "label": y, "results": [mk("text", .15), mk("image", .3), mk("url", .15)], "cross": cross})
    return cases


def test_ablation_rows_and_ordering():
    rows, scores, y, ci_rows = ev.ablation(make_cases())
    names = [r[0] for r in rows]
    assert names[:2] == ["Text only", "URL only"] and names[-1].startswith("Meta-classifier") and len(rows) == len(ev.ABLATION) + 1
    by = {r[0]: r for r in rows}
    pct = lambda n: float(by[n][1].rstrip("%"))  # coverage
    assert pct("All three") > pct("Text only") and pct("All three") >= 90
    assert by["All three"][6] > by["Image only"][6]  # more evidence ranks better (ROC-AUC)
    assert all(0 <= s.min() and s.max() <= 1 for s in scores.values()) and len(y) == 60
    # Every configuration also gets a bootstrap CI row, label first.
    assert [r[0] for r in ci_rows] == names
    assert all(len(r) == 1 + len(ev.CI_METRICS) for r in ci_rows)


@pytest.mark.parametrize("fn,flags,near,frag", [
    (True, [], False, "ordinary posting"), (True, ["URGENCY"], False, "score stayed low"),
    (False, ["URGENCY"], False, "wording that can also appear"), (False, [], False, "generic job vocabulary"),
    (True, [], True, "low margin"),
])
def test_error_notes(fn, flags, near, frag):
    assert frag in ev._note(fn, flags, near)
