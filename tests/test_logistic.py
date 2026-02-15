"""Tests for LogisticModel: eval, derivative correctness, and numerical gradient check."""

from __future__ import annotations

import math

from mlenergy_data.modeling import LogisticModel


def test_eval_at_midpoint():
    """At x = x0, sigmoid = 0.5, so y = b0 + L/2."""
    lp = LogisticModel(L=100.0, x0=5.0, k=1.0, b0=50.0)
    assert abs(lp.eval_x(5.0) - 100.0) < 1e-10


def test_eval_at_extremes():
    """Far from x0, sigmoid approaches 0 or 1."""
    lp = LogisticModel(L=100.0, x0=5.0, k=2.0, b0=10.0)
    # Very large x -> sigmoid -> 1 -> y -> b0 + L = 110
    assert abs(lp.eval_x(100.0) - 110.0) < 1e-5
    # Very small x -> sigmoid -> 0 -> y -> b0 = 10
    assert abs(lp.eval_x(-100.0) - 10.0) < 1e-5


def test_eval_batch():
    """eval(batch) should equal eval_x(log2(batch))."""
    lp = LogisticModel(L=200.0, x0=7.0, k=0.5, b0=30.0)
    for batch in [8, 16, 32, 64, 128, 256, 512]:
        x = math.log2(batch)
        assert abs(lp.eval(batch) - lp.eval_x(x)) < 1e-12


def test_derivative_numerical_gradient():
    """Analytical derivative should match numerical finite-difference gradient."""
    lp = LogisticModel(L=150.0, x0=6.0, k=1.5, b0=20.0)

    eps = 1e-7
    for x in [3.0, 5.0, 6.0, 7.0, 9.0]:
        analytical = lp.deriv_wrt_x(x)
        numerical = (lp.eval_x(x + eps) - lp.eval_x(x - eps)) / (2 * eps)
        assert abs(analytical - numerical) < 1e-4, (
            f"Gradient mismatch at x={x}: analytical={analytical:.8f}, numerical={numerical:.8f}"
        )


def test_derivative_sign():
    """Positive k and L means derivative is positive (increasing sigmoid)."""
    lp = LogisticModel(L=100.0, x0=5.0, k=1.0, b0=0.0)
    for x in [3.0, 5.0, 7.0]:
        assert lp.deriv_wrt_x(x) > 0

    # Negative L means derivative is negative
    lp_neg = LogisticModel(L=-100.0, x0=5.0, k=1.0, b0=200.0)
    for x in [3.0, 5.0, 7.0]:
        assert lp_neg.deriv_wrt_x(x) < 0


def test_derivative_peak_at_midpoint():
    """Derivative is maximized at x = x0."""
    lp = LogisticModel(L=100.0, x0=5.0, k=2.0, b0=0.0)
    d_mid = lp.deriv_wrt_x(5.0)
    d_off = lp.deriv_wrt_x(3.0)
    assert d_mid > d_off


def test_numerical_stability_large_input():
    """Should not overflow for large inputs."""
    lp = LogisticModel(L=100.0, x0=5.0, k=1.0, b0=0.0)
    assert math.isfinite(lp.eval_x(1000.0))
    assert math.isfinite(lp.eval_x(-1000.0))
    assert math.isfinite(lp.deriv_wrt_x(1000.0))
    assert math.isfinite(lp.deriv_wrt_x(-1000.0))
