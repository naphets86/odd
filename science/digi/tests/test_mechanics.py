"""tests/test_mechanics.py – Tests for digi.mechanics"""
import numpy as np
import pytest
from digi.mechanics import (
    natural_frequency,
    damping_ratio,
    damped_frequency,
    damping_regime,
    free_oscillation,
    step_response_membrane,
    mechanical_impedance,
)


class TestNaturalFrequency:
    def test_simple_case(self):
        """k=1, m=1 → ω₀=1, f₀=1/(2π)"""
        omega_0, f_0 = natural_frequency(k=1.0, m=1.0)
        assert pytest.approx(omega_0) == 1.0
        assert pytest.approx(f_0) == 1.0 / (2 * np.pi)

    def test_frequency_relationship(self):
        """f₀ = ω₀/(2π)"""
        omega_0, f_0 = natural_frequency(k=4.0, m=1.0)
        assert pytest.approx(f_0) == omega_0 / (2 * np.pi)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            natural_frequency(k=0.0, m=1.0)

    def test_m_zero_raises(self):
        with pytest.raises(ValueError):
            natural_frequency(k=1.0, m=0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            natural_frequency(k=-1.0, m=1.0)

    def test_stiffer_spring_higher_frequency(self):
        _, f1 = natural_frequency(k=1.0, m=1.0)
        _, f2 = natural_frequency(k=4.0, m=1.0)
        assert f2 > f1

    def test_heavier_mass_lower_frequency(self):
        _, f1 = natural_frequency(k=1.0, m=1.0)
        _, f2 = natural_frequency(k=1.0, m=4.0)
        assert f2 < f1


class TestDampingRatio:
    def test_formula(self):
        """D = r / (2√(mk))"""
        r, m, k = 2.0, 1.0, 1.0
        D = damping_ratio(r, m, k)
        assert pytest.approx(D) == r / (2 * np.sqrt(m * k))

    def test_critical_damping(self):
        """r = 2√(mk) → D = 1"""
        m, k = 1.0, 4.0
        r = 2 * np.sqrt(m * k)
        D = damping_ratio(r, m, k)
        assert pytest.approx(D) == 1.0

    def test_zero_damping(self):
        assert damping_ratio(0.0, 1.0, 1.0) == 0.0

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            damping_ratio(-1.0, 1.0, 1.0)

    def test_invalid_m(self):
        with pytest.raises(ValueError):
            damping_ratio(1.0, 0.0, 1.0)


class TestDampedFrequency:
    def test_underdamped(self):
        """ωd = ω₀√(1 - D²)"""
        omega_0, D = 10.0, 0.5
        omega_d = damped_frequency(omega_0, D)
        assert pytest.approx(omega_d) == omega_0 * np.sqrt(1 - D**2)

    def test_critical_damping_returns_zero(self):
        assert damped_frequency(10.0, 1.0) == 0.0

    def test_overdamped_returns_zero(self):
        assert damped_frequency(10.0, 1.5) == 0.0

    def test_zero_damping_gives_omega0(self):
        assert pytest.approx(damped_frequency(5.0, 0.0)) == 5.0

    def test_damped_less_than_natural(self):
        omega_0, D = 10.0, 0.3
        omega_d = damped_frequency(omega_0, D)
        assert omega_d < omega_0


class TestDampingRegime:
    def test_underdamped(self):
        assert damping_regime(0.5) == "underdamped"

    def test_critically_damped(self):
        assert damping_regime(1.0) == "critically_damped"

    def test_overdamped(self):
        assert damping_regime(2.0) == "overdamped"

    def test_zero_damping(self):
        assert damping_regime(0.0) == "underdamped"


class TestFreeOscillation:
    def test_initial_displacement(self):
        """At t=0 solution must equal x0"""
        t = np.array([0.0])
        x = free_oscillation(t, omega_0=10.0, D=0.3, x0=2.0)
        assert pytest.approx(x[0]) == 2.0

    def test_decays_to_zero(self):
        """Amplitude must eventually decay to ~0"""
        t = np.linspace(0, 100, 10_000)
        x = free_oscillation(t, omega_0=2 * np.pi, D=0.1)
        assert abs(x[-1]) < 1e-5

    def test_always_finite(self):
        t = np.linspace(0, 10, 1000)
        x = free_oscillation(t, omega_0=5.0, D=0.05)
        assert np.all(np.isfinite(x))

    def test_underdamped_oscillates(self):
        """Underdamped signal should change sign"""
        t = np.linspace(0, 5, 500)
        x = free_oscillation(t, omega_0=2 * np.pi, D=0.05)
        signs = np.sign(x[x != 0])
        assert np.any(signs == 1) and np.any(signs == -1)


class TestStepResponseMembrane:
    def test_starts_at_zero(self):
        t = np.array([0.0])
        u = step_response_membrane(t, omega_0=10.0, D=0.5)
        assert pytest.approx(u[0], abs=1e-8) == 0.0

    def test_settles_to_U0(self):
        """Final value of step response = U0"""
        U0 = 2.5
        t = np.linspace(0, 100, 50_000)
        u = step_response_membrane(t, omega_0=2 * np.pi, D=0.3, U0=U0)
        assert pytest.approx(u[-1], rel=1e-4) == U0

    def test_critically_damped_settles(self):
        U0 = 1.0
        t = np.linspace(0, 50, 20_000)
        u = step_response_membrane(t, omega_0=2 * np.pi, D=1.0, U0=U0)
        assert pytest.approx(u[-1], rel=1e-4) == U0


class TestMechanicalImpedance:
    def test_at_resonance_pure_real(self):
        """At resonance omegas_0 = sqrt(k/m), Im(Z_m) ≈ 0"""
        m, r, k = 1.0, 0.1, 4.0
        omega_0 = np.sqrt(k / m)
        Z = mechanical_impedance(np.array([omega_0]), m, r, k)
        assert pytest.approx(Z[0].imag, abs=1e-9) == 0.0
        assert pytest.approx(Z[0].real) == r

    def test_returns_complex(self):
        Z = mechanical_impedance(np.array([1.0, 2.0, 3.0]), m=1.0, r=0.1, k=4.0)
        assert Z.dtype == complex
