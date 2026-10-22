"""Tests for the conformal helpers in scripts/v2_e4_conformal.py (finite-sample quantile, BH selection)."""
import importlib.util
import os
import tempfile

import numpy as np
import pytest

os.environ["SIREN_ROOT"] = tempfile.mkdtemp()   # the script creates results/v2 under SIREN_ROOT at import time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("e4", os.path.join(HERE, "scripts", "v2_e4_conformal.py"))
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)


def test_conf_q_uses_finite_sample_rank():
    res = np.arange(1.0, 10.0)                      # n = 9, level 0.9 -> ceil(10 * 0.9) = 9th smallest
    assert e4.conf_q(res, 0.9) == pytest.approx(9.0)
    assert e4.conf_q(np.arange(1.0, 101.0), 0.9) == pytest.approx(91.0)   # ceil(101 * 0.9) = 91


def test_bh_select_step_up():
    p = np.array([0.001, 0.01, 0.2, 0.5])
    assert e4.bh_select(p, 0.05).tolist() == [True, True, False, False]
    assert e4.bh_select(np.array([0.9, 0.8]), 0.05).sum() == 0
    # step-up: a large p-value early in the order can be rescued by a later small one
    q = np.array([0.014, 0.0001, 0.02, 0.9])
    assert e4.bh_select(q, 0.05).tolist() == [True, True, True, False]
