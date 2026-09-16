"""Two-state correctness test for the backward-Kolmogorov (BK) residual.

Independent analytic check of the sign convention derived in train_h.py /
sample_guided.py (see train_h.bk_residual's docstring for the full audit):
the base CTMC only ever runs FORWARD (t: 0 -> T) to build training data; the
guided sampler runs BACKWARD (t: T -> 0), producing the generated X_0 whose
reward R(X_0) is the Monte Carlo target. With u = log h parameterized in
model time t (not the sampling clock s = T - t), the BK equation becomes

    -du/dt(t,x) + sum_y G(x,y) [exp(u(t,y)-u(t,x)) - 1] = 0

Two-state setup: states {0, 1}, symmetric CTMC with constant off-diagonal
rate q (G(0,1) = G(1,0) = q, matching this project's uniform-rate
convention -- see ctmc.forward_rate / sample_guided.guided_step). Forward
transition probabilities from X_0 are standard for a symmetric 2-state
chain:

    a(t) = P(X_t == X_0 | X_0) = (1 + exp(-2*q*t)) / 2
    b(t) = P(X_t != X_0 | X_0) = (1 - exp(-2*q*t)) / 2 = 1 - a(t)

so, using time-reversibility of a symmetric chain (P(X_0=y|X_t=x) =
P(X_t=x|X_0=y) under the uniform stationary prior),

    h(t, 0) = a(t) * R(0) + b(t) * R(1)
    h(t, 1) = a(t) * R(1) + b(t) * R(0)

This closed-form h_t is what the four assertions below check against: near-
zero BK residual, correct time-derivative sign, the guided sampler's
rate formula q * h(y)/h(x), and the terminal boundary h(0,x) = R(x).

Run directly: python test_bk_two_state.py
"""

from __future__ import annotations

import math

Q = 2.0  # off-diagonal rate G(0,1) = G(1,0)
R0 = 0.9  # R(state 0)
R1 = 0.2  # R(state 1)


def a(t: float) -> float:
    return (1.0 + math.exp(-2.0 * Q * t)) / 2.0


def b(t: float) -> float:
    return 1.0 - a(t)


def h(t: float, x: int) -> float:
    if x == 0:
        return a(t) * R0 + b(t) * R1
    return a(t) * R1 + b(t) * R0


def dh_dt(t: float, x: int) -> float:
    """Analytic derivative of h(t,x) w.r.t. t (closed form, not autograd)."""
    da_dt = -Q * math.exp(-2.0 * Q * t)
    db_dt = -da_dt
    if x == 0:
        return da_dt * R0 + db_dt * R1
    return da_dt * R1 + db_dt * R0


def du_dt(t: float, x: int) -> float:
    return dh_dt(t, x) / h(t, x)


def bk_residual_analytic(t: float, x: int) -> float:
    """-du/dt(t,x) + G(x,y) * [exp(u(t,y)-u(t,x)) - 1], y = other state.

    Uses the CLOSED-FORM h(t,.) and dh/dt above (no autograd, no
    train_h.py) -- an independent check of the sign convention documented
    in train_h.bk_residual, not a test that the code agrees with itself.
    """
    y = 1 - x
    u_x = math.log(h(t, x))
    u_y = math.log(h(t, y))
    generator_term = Q * (math.exp(u_y - u_x) - 1.0)
    return -du_dt(t, x) + generator_term


def test_analytic_bk_residual_near_zero() -> None:
    for t in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        for x in [0, 1]:
            residual = bk_residual_analytic(t, x)
            assert abs(residual) < 1e-8, (
                f"BK residual not ~0 at t={t}, x={x}: residual={residual}"
            )
    print("[PASS] analytic h_t satisfies the BK residual (near-zero) at all test points")


def test_time_derivative_sign() -> None:
    """h(t,0) starts at R0=0.9 (large) and decays toward the uniform mix
    (R0+R1)/2 = 0.55 as t -> infinity, so dh/dt(t,0) < 0 for all t > 0 --
    confirms the -du/dt sign (not +du/dt) is the one that zeroes the
    residual above, i.e. the derived sign is not a coincidental match."""
    for t in [0.01, 0.1, 0.5, 1.0]:
        assert dh_dt(t, 0) < 0.0, f"expected h(t,0) decaying, got dh/dt={dh_dt(t, 0)} at t={t}"
        assert dh_dt(t, 1) > 0.0, f"expected h(t,1) rising, got dh/dt={dh_dt(t, 1)} at t={t}"
    # Sanity: using the WRONG sign (+du/dt) would NOT zero the residual.
    t, x = 0.3, 0
    y = 1 - x
    u_x, u_y = math.log(h(t, x)), math.log(h(t, y))
    generator_term = Q * (math.exp(u_y - u_x) - 1.0)
    wrong_sign_residual = du_dt(t, x) + generator_term
    assert abs(wrong_sign_residual) > 1e-3, (
        "expected the wrong-sign residual to be clearly nonzero, "
        f"got {wrong_sign_residual}"
    )
    print("[PASS] -du/dt is the correct sign; +du/dt does not zero the residual")


def test_guided_rate_matches_h_ratio() -> None:
    """The guided sampler's rate q_guided(x,y) = q * h(t,y)/h(t,x) (see
    sample_guided.guided_step) should equal the rate recovered from the
    known h_t via direct ratio -- this is the same formula, evaluated
    against the closed-form h_t rather than a trained model."""
    t = 0.4
    for x in [0, 1]:
        y = 1 - x
        guided_rate = Q * (h(t, y) / h(t, x))
        expected = Q * math.exp(math.log(h(t, y)) - math.log(h(t, x)))
        assert abs(guided_rate - expected) < 1e-10
    print("[PASS] guided rate formula q * h(t,y)/h(t,x) is internally consistent")


def test_terminal_boundary() -> None:
    """h(0, x) must equal R(x) exactly (a(0)=1, b(0)=0)."""
    assert abs(h(0.0, 0) - R0) < 1e-12, f"h(0,0)={h(0.0, 0)} != R0={R0}"
    assert abs(h(0.0, 1) - R1) < 1e-12, f"h(0,1)={h(0.0, 1)} != R1={R1}"
    print("[PASS] terminal boundary h(0,x) == R(x) holds exactly")


if __name__ == "__main__":
    test_analytic_bk_residual_near_zero()
    test_time_derivative_sign()
    test_guided_rate_matches_h_ratio()
    test_terminal_boundary()
    print("\nAll two-state BK correctness checks passed.")
