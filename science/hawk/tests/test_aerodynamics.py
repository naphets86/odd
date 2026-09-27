"""Tests for hawk.aerodynamics — full branch and statement coverage."""

import math

import pytest

from hawk.aerodynamics import (
    A_CS_MAX,
    A_CS_MIN,
    C_D0,
    C_DW,
    C_L0,
    C_L_ALPHA,
    G,
    H_SCALE,
    K_ALPHA,
    K_BETA,
    RHO_0,
    air_density,
    drag_coefficient,
    drag_force,
    lift_coefficient,
    lift_force,
    reference_area,
    terminal_velocity,
)


# ---------------------------------------------------------------------------
# air_density
# ---------------------------------------------------------------------------

class TestAirDensity:
    def test_sea_level(self):
        """h=0 must return exactly ρ₀."""
        assert air_density(0.0) == pytest.approx(RHO_0)

    def test_scale_height(self):
        """At h = H_s the density must be ρ₀/e."""
        expected = RHO_0 / math.e
        assert air_density(H_SCALE) == pytest.approx(expected, rel=1e-9)

    def test_positive_at_altitude(self):
        """Density is strictly positive at any physically plausible altitude."""
        for h in [0.0, 100.0, 1000.0, 8848.0, 20_000.0]:
            assert air_density(h) > 0.0

    def test_decreases_with_altitude(self):
        """Density must decrease monotonically with altitude."""
        assert air_density(0.0) > air_density(1000.0) > air_density(5000.0)

    def test_high_altitude_near_zero(self):
        """Very high altitude yields density close to zero."""
        assert air_density(100_000.0) < 1e-3


# ---------------------------------------------------------------------------
# drag_coefficient
# ---------------------------------------------------------------------------

class TestDragCoefficient:
    def test_wings_fully_open_zero_alpha(self):
        """β=1, α=0 → C_D = C_D0  (no wing-fold penalty, no alpha term)."""
        assert drag_coefficient(0.0, 1.0) == pytest.approx(C_D0)

    def test_wings_fully_folded_zero_alpha(self):
        """β=0, α=0 → C_D = C_D0 + K_β·C_Dw  (maximum wing-fold penalty)."""
        expected = C_D0 + K_BETA * C_DW
        assert drag_coefficient(0.0, 0.0) == pytest.approx(expected)

    def test_alpha_contribution(self):
        """Non-zero alpha adds K_α·α² to the coefficient."""
        alpha = 0.2
        cd_no_alpha = drag_coefficient(0.0, 1.0)
        cd_with_alpha = drag_coefficient(alpha, 1.0)
        assert cd_with_alpha == pytest.approx(cd_no_alpha + K_ALPHA * alpha ** 2)

    def test_increases_with_alpha_squared(self):
        """C_D is symmetric in α (depends on α²)."""
        assert drag_coefficient(0.3, 0.5) == pytest.approx(drag_coefficient(-0.3, 0.5))

    def test_decreases_as_wings_open(self):
        """C_D decreases as wings open (β_fold → 1) for fixed α."""
        assert drag_coefficient(0.1, 0.0) > drag_coefficient(0.1, 0.5) > drag_coefficient(0.1, 1.0)

    def test_combined_formula(self):
        """Verify exact formula for arbitrary values."""
        alpha, beta = 0.15, 0.6
        expected = C_D0 + K_ALPHA * alpha ** 2 + K_BETA * (1.0 - beta) ** 2 * C_DW
        assert drag_coefficient(alpha, beta) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# lift_coefficient
# ---------------------------------------------------------------------------

class TestLiftCoefficient:
    def test_zero_alpha_any_beta(self):
        """α=0 → C_L = C_L0 regardless of β_fold."""
        for beta in [0.0, 0.5, 1.0]:
            assert lift_coefficient(0.0, beta) == pytest.approx(C_L0)

    def test_wings_folded_any_alpha(self):
        """β=0 → C_L = C_L0 (no effective wing area for lift generation)."""
        for alpha in [0.0, 0.1, 0.5]:
            assert lift_coefficient(alpha, 0.0) == pytest.approx(C_L0)

    def test_increases_with_alpha_and_beta(self):
        """C_L grows with both α and β_fold."""
        assert lift_coefficient(0.1, 0.5) < lift_coefficient(0.2, 0.5)
        assert lift_coefficient(0.1, 0.5) < lift_coefficient(0.1, 1.0)

    def test_combined_formula(self):
        alpha, beta = 0.2, 0.8
        expected = C_L_ALPHA * alpha * beta + C_L0
        assert lift_coefficient(alpha, beta) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# reference_area
# ---------------------------------------------------------------------------

class TestReferenceArea:
    def test_fully_folded(self):
        """β=0 → A_ref = A_cs_min."""
        assert reference_area(0.0) == pytest.approx(A_CS_MIN)

    def test_fully_open(self):
        """β=1 → A_ref = A_cs_max."""
        assert reference_area(1.0) == pytest.approx(A_CS_MAX)

    def test_intermediate_value(self):
        """β=0.5 lies strictly between min and max."""
        a = reference_area(0.5)
        assert A_CS_MIN < a < A_CS_MAX

    def test_monotone_increasing(self):
        """A_ref is non-decreasing in β_fold."""
        areas = [reference_area(b) for b in [0.0, 0.25, 0.5, 0.75, 1.0]]
        assert areas == sorted(areas)

    def test_formula_explicit(self):
        beta = 0.64
        expected = A_CS_MIN + (A_CS_MAX - A_CS_MIN) * beta ** (2.0 / 3.0)
        assert reference_area(beta) == pytest.approx(expected)

    def test_custom_min_max(self):
        """Custom a_cs_min / a_cs_max are honoured."""
        a_min, a_max = 0.01, 0.10
        assert reference_area(0.0, a_min, a_max) == pytest.approx(a_min)
        assert reference_area(1.0, a_min, a_max) == pytest.approx(a_max)


# ---------------------------------------------------------------------------
# drag_force
# ---------------------------------------------------------------------------

class TestDragForce:
    def test_positive(self):
        """Drag force must be positive for any non-zero speed."""
        assert drag_force(50.0, 0.0, 0.0, 0.5) > 0.0

    def test_zero_speed(self):
        """No speed → no drag."""
        assert drag_force(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_scales_with_v_squared(self):
        """D ∝ v²: doubling speed quadruples drag."""
        d1 = drag_force(20.0, 0.0, 0.0, 1.0)
        d2 = drag_force(40.0, 0.0, 0.0, 1.0)
        assert d2 == pytest.approx(4.0 * d1, rel=1e-9)

    def test_formula_explicit(self):
        v, h, alpha, beta = 30.0, 200.0, 0.05, 0.8
        rho = air_density(h)
        cd = drag_coefficient(alpha, beta)
        a_ref = reference_area(beta)
        expected = 0.5 * rho * v ** 2 * cd * a_ref
        assert drag_force(v, h, alpha, beta) == pytest.approx(expected)

    def test_custom_area_params(self):
        """Custom min/max area propagate to drag force."""
        d_default = drag_force(50.0, 0.0, 0.0, 0.5)
        d_custom = drag_force(50.0, 0.0, 0.0, 0.5, a_cs_min=0.001, a_cs_max=0.05)
        assert d_default != d_custom


# ---------------------------------------------------------------------------
# lift_force
# ---------------------------------------------------------------------------

class TestLiftForce:
    def test_positive_for_open_wings(self):
        """Lift is positive when wings are open and angle-of-attack > 0."""
        assert lift_force(30.0, 0.0, 0.1, 1.0) > 0.0

    def test_near_zero_for_folded_wings(self):
        """Wings fully folded (β=0) produce only the residual C_L0 lift."""
        l_folded = lift_force(30.0, 0.0, 0.2, 0.0)
        l_open = lift_force(30.0, 0.0, 0.2, 1.0)
        assert l_folded < l_open

    def test_formula_explicit(self):
        v, h, alpha, beta, s_w = 40.0, 100.0, 0.15, 0.7, 0.11
        rho = air_density(h)
        cl = lift_coefficient(alpha, beta)
        expected = 0.5 * rho * v ** 2 * cl * s_w
        assert lift_force(v, h, alpha, beta, s_w=s_w) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# terminal_velocity  (Theorem 3.1)
# ---------------------------------------------------------------------------

class TestTerminalVelocity:
    def test_paper_example_approx_316_ms(self):
        """Formula with paper Theorem 3.1 parameters → ≈ 316.4 m/s.

        The paper's numerical example claims ≈ 392 km/h for m=0.70 kg,
        ρ=1.225 kg/m³, C_D=0.028, A=0.004 m², but that arithmetic is
        incorrect: sqrt(2·0.70·9.81 / (1.225·0.028·0.004)) ≈ 316.4 m/s
        (≈ 1139 km/h).  The implementation correctly follows the theorem.
        """
        v_max = terminal_velocity(mass=0.70, rho=1.225, c_d_min=0.028, a_cs_min=0.004)
        assert v_max == pytest.approx(316.4, abs=0.5)   # m/s

    def test_increases_with_mass(self):
        """Heavier object → higher terminal velocity."""
        v_light = terminal_velocity(mass=0.5)
        v_heavy = terminal_velocity(mass=1.5)
        assert v_heavy > v_light

    def test_formula_explicit(self):
        m, rho, cd, a = 1.0, 1.1, 0.05, 0.008
        expected = math.sqrt(2.0 * m * G / (rho * cd * a))
        assert terminal_velocity(m, rho, cd, a) == pytest.approx(expected)

    def test_positive(self):
        assert terminal_velocity(1.2) > 0.0
