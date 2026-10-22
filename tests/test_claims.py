"""Tests for scripts/claims.py: the machine check that manuscripts only contain sourced numbers."""
import importlib.util
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("claims", os.path.join(HERE, "scripts", "claims.py"))
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)

VALUES = {"e1/x/auprc": 0.31234, "e3/decision/status": "INCONCLUSIVE", "e1/x/n": 10356}


def test_flatten_uses_slash_paths_and_list_indices():
    flat = claims.flatten({"a": {"b.c": [1, 2]}}, "e1")
    assert flat == {"e1/a/b.c/0": 1, "e1/a/b.c/1": 2}


def test_render_formats_and_reports_missing():
    text, missing = claims.render_text("AUPRC {{e1/x/auprc|.2f}}, n={{e1/x/n|,}}, {{nope}}", VALUES)
    assert text.startswith("AUPRC 0.31, n=10,356")
    assert missing == ["nope"]


def test_check_flags_handtyped_decimals_only_in_abstract_and_results():
    tpl = ("# T\n## Abstract\nWe obtain 0.31 here and {{e1/x/auprc|.2f}} there.\n"
           "## Methods\nLearning rate 0.05 is fine here.\n## Results\nSee 0.77.\n")
    errs = claims.check_template(tpl, VALUES)
    msgs = " ".join(errs)
    assert "0.31" in msgs and "0.77" in msgs and "0.05" not in msgs


def test_check_detects_status_mismatch_and_banned_terms():
    tpl = ("# T\n## Abstract\n<!-- claim: e3/decision/status status=BENEFIT -->\nok\n"
           "## Discussion\nThe specificity wall <!-- allow: specificity wall --> is fine here; a reference class is not.\n")
    errs = claims.check_template(tpl, VALUES)
    assert any("status mismatch" in e for e in errs)
    assert not any("specificity wall" in e for e in errs)
    assert any("reference class" in e for e in errs)


def test_check_detects_stale_render():
    tpl = "## Results\nAUPRC {{e1/x/auprc|.2f}}\n"
    errs = claims.check_template(tpl, VALUES, rendered="## Results\nAUPRC 0.99\n")
    assert any("differs from the rendering" in e for e in errs)
    assert claims.check_template(tpl, VALUES, rendered="## Results\nAUPRC 0.31\n") == []
