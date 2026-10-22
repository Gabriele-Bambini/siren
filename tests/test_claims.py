"""Tests for scripts/claims.py: the machine check that manuscripts only contain sourced numbers."""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("claims", os.path.join(HERE, "scripts", "claims.py"))
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)

VALUES = {"e1/x/auprc": 0.31234, "e3/decision/status": "INCONCLUSIVE", "e1/x/n": 10356, "e1/x/bad": float("nan")}


def test_flatten_uses_slash_paths_and_list_indices():
    flat = claims.flatten({"a": {"b.c": [1, 2]}}, "e1")
    assert flat == {"e1/a/b.c/0": 1, "e1/a/b.c/1": 2}


def test_render_formats_and_reports_missing_and_nonfinite():
    text, missing = claims.render_text("AUPRC {{e1/x/auprc|.2f}}, n={{e1/x/n|,}}, {{nope}}, {{e1/x/bad|.2f}}", VALUES)
    assert text.startswith("AUPRC 0.31, n=10,356")
    assert "nope" in missing and any("non-finite" in m for m in missing)


def test_check_flags_handtyped_numbers_in_abstract_results_discussion_only():
    tpl = ("# T\n## Abstract\nWe obtain 0.31 here and {{e1/x/auprc|.2f}} there.\n"
           "## Methods\nLearning rate 0.05 and 300 trees are fine here.\n"
           "## Results\nSee 0.77, -0.012, 87% and 10,356 pairs; also E1, M2, chr19 and Table 3.\n"
           "## Discussion\nThe gain was 12 points.\n")
    errs = " ".join(claims.check_template(tpl, VALUES))
    for tok in ("0.31", "0.77", "-0.012", "87%", "10,356", "12"):
        assert tok in errs, tok
    assert "0.05" not in errs and "300" not in errs
    assert "E1" not in errs and "chr19" not in errs and "Table 3" not in errs


def test_citations_urls_dois_and_placeholders_are_exempt():
    tpl = ("## Results\nSee [Gschwind 2026], https://example.org/v1.0 and doi 10.1038/s41586-026-10781-4 "
           "with {{e1/x/n|,}} pairs <!-- note 2026 -->.\n")
    assert claims.check_template(tpl, VALUES) == []


def test_check_detects_status_mismatch_forbidden_verbs_and_banned_terms():
    tpl = ("# T\n## Abstract\n<!-- claim: e3/decision/status status=BENEFIT -->\nok\n"
           "## Discussion\n<!-- claim: e3/decision/status status=INCONCLUSIVE -->\nThis clearly shows a benefit.\n"
           "The specificity wall <!-- allow: specificity wall --> is fine here; a reference class is not.\n")
    errs = claims.check_template(tpl, VALUES)
    assert any("status mismatch" in e for e in errs)
    assert any("not allowed for a INCONCLUSIVE" in e for e in errs)
    assert not any("specificity wall" in e for e in errs)
    assert any("reference class" in e for e in errs)


def test_check_detects_stale_render():
    tpl = "## Results\nAUPRC {{e1/x/auprc|.2f}}\n"
    errs = claims.check_template(tpl, VALUES, rendered="## Results\nAUPRC 0.99\n")
    assert any("differs from the rendering" in e for e in errs)
    assert claims.check_template(tpl, VALUES, rendered="## Results\nAUPRC 0.31\n") == []


def test_fmt_value_rejects_nan():
    try:
        claims.fmt_value(float("nan"), ".2f")
    except ValueError:
        return
    assert not math.isfinite(float("nan"))
    raise AssertionError("NaN must be rejected")
