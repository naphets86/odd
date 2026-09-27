"""tests/test_rlc.py – Tests for digi.rlc"""
import numpy as np
import pytest
from digi.rlc import (
    resonance_frequency,
    quality_factor,
    damping_coefficient,
    damping_ratio_rlc,
    impedance,
    impedance_magnitude,
    impedance_phase,
    current_transfer,
    bandpass_magnitude,
    step_response,
    bandwidth_3db,
)


class TestResonanceFrequency:
    def test_formula(self):
        """ω₀ = 1/√(LC)"""
        L, C = 1e-3, 1e-6  # 1 mH, 1 µF → f₀ ≈ 5.033 kHz
        omega_0, f_0 = resonance_frequency(L, C)
        assert pytest.approx(omega_0) == 1.0 / np.sqrt(L * C)
        assert pytest.approx(f_0) == omega_0 / (2 * np.pi)

    def test_both_zero_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            resonance_frequency(0.0, 1e-6)

    def test_L_zero_raises(self):
        with pytest.raises(ValueError):
            resonance_frequency(0.0, 1e-6)

    def test_C_zero_raises(self):
        with pytest.raises(ValueError):
            resonance_frequency(1e-3, 0.0)

    def test_returns_floats(self):
        omega_0, f_0 = resonance_frequency(1e-3, 1e-6)
        assert isinstance(omega_0, float)
        assert isinstance(f_0, float)


class TestQualityFactor:
    def test_formula(self):
        """Q = (1/R)√(L/C)"""
        R, L, C = 10.0, 1e-3, 1e-6
        Q = quality_factor(R, L, C)
        assert pytest.approx(Q) == (1.0 / R) * np.sqrt(L / C)

    def test_high_Q_for_small_R(self):
        Q_high = quality_factor(1.0, 1e-3, 1e-6)
        Q_low = quality_factor(100.0, 1e-3, 1e-6)
        assert Q_high > Q_low

    def test_R_zero_raises(self):
        with pytest.raises(ValueError):
            quality_factor(0.0, 1e-3, 1e-6)

    def test_positive(self):
        assert quality_factor(10.0, 1e-3, 1e-6) > 0


class TestDampingCoefficient:
    def test_formula(self):
        R, L = 100.0, 1e-3
        alpha = damping_coefficient(R, L)
        assert pytest.approx(alpha) == R / (2 * L)

    def test_L_zero_raises(self):
        with pytest.raises(ValueError):
            damping_coefficient(10.0, 0.0)


class TestDampingRatioRLC:
    def test_relation_to_q(self):
        """D_rlc = 1/(2Q)"""
        R, L, C = 10.0, 1e-3, 1e-6
        Q = quality_factor(R, L, C)
        D = damping_ratio_rlc(R, L, C)
        assert pytest.approx(D) == 1.0 / (2 * Q)

    def test_critical_at_d_equals_1(self):
        """D = 1 ↔ Q = 0.5"""
        L, C = 1.0, 1.0
        omega_0, _ = resonance_frequency(L, C)
        R_crit = 2 * np.sqrt(L / C)  # D=1 condition
        D = damping_ratio_rlc(R_crit, L, C)
        assert pytest.approx(D, abs=1e-9) == 1.0


class TestImpedance:
    def test_at_resonance_is_R(self):
        """At ω₀: Z = R (imaginary parts cancel)"""
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        Z = impedance(np.array([omega_0]), R, L, C)
        assert pytest.approx(Z[0].real, rel=1e-5) == R
        assert pytest.approx(Z[0].imag, abs=1e-8) == 0.0

    def test_inductive_above_resonance(self):
        """Above resonance impedance is inductive (Im Z > 0)"""
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        Z = impedance(np.array([2 * omega_0]), R, L, C)
        assert Z[0].imag > 0.0

    def test_capacitive_below_resonance(self):
        """Below resonance impedance is capacitive (Im Z < 0)"""
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        Z = impedance(np.array([0.5 * omega_0]), R, L, C)
        assert Z[0].imag < 0.0

    def test_returns_complex_array(self):
        Z = impedance(np.array([1.0, 2.0]), R=10.0, L=1e-3, C=1e-6)
        assert Z.dtype == np.complex128


class TestImpedanceMagnitude:
    def test_at_resonance_minimum(self):
        """|Z(ω₀)| = R is the minimum"""
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        omegas = np.array([0.5 * omega_0, omega_0, 2 * omega_0])
        mags = impedance_magnitude(omegas, R, L, C)
        assert mags[1] < mags[0]
        assert mags[1] < mags[2]
        assert pytest.approx(mags[1], rel=1e-5) == R

    def test_always_positive(self):
        omega = np.linspace(100, 1e7, 500)
        mags = impedance_magnitude(omega, R=10, L=1e-3, C=1e-6)
        assert np.all(mags > 0)


class TestImpedancePhase:
    def test_zero_at_resonance(self):
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        phi = impedance_phase(np.array([omega_0]), R, L, C)
        assert pytest.approx(phi[0], abs=1e-6) == 0.0

    def test_positive_above_resonance(self):
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        phi = impedance_phase(np.array([10 * omega_0]), R, L, C)
        assert phi[0] > 0.0

    def test_negative_below_resonance(self):
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        phi = impedance_phase(np.array([0.1 * omega_0]), R, L, C)
        assert phi[0] < 0.0


class TestBandpassMagnitude:
    def test_peak_is_one(self):
        R, L, C = 10.0, 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        omegas = np.linspace(omega_0 * 0.1, omega_0 * 10, 1000)
        mag = bandpass_magnitude(omegas, R, L, C)
        assert pytest.approx(np.max(mag), abs=1e-6) == 1.0

    def test_non_negative(self):
        omega = np.linspace(1000, 1e7, 200)
        mag = bandpass_magnitude(omega, R=10, L=1e-3, C=1e-6)
        assert np.all(mag >= 0)


class TestStepResponse:
    def test_initial_condition_zero(self):
        """Capacitor voltage starts at 0"""
        t = np.array([0.0])
        u_C, _ = step_response(t, R=10.0, L=1e-3, C=1e-6, U0=1.0)
        assert pytest.approx(u_C[0], abs=1e-6) == 0.0

    def test_settles_to_U0_underdamped(self):
        """Underdamped response settles to U0"""
        R, L, C, U0 = 10.0, 1e-3, 1e-6, 5.0
        t = np.linspace(0, 0.02, 50_000)
        u_C, _ = step_response(t, R, L, C, U0)
        assert pytest.approx(u_C[-1], rel=1e-3) == U0

    def test_overdamped_settles_to_U0(self):
        """Overdamped response also settles to U0"""
        L, C = 1e-3, 1e-6
        omega_0, _ = resonance_frequency(L, C)
        R_over = 10 * 2 * np.sqrt(L / C)  # very overdamped
        t = np.linspace(0, 1, 10_000)
        u_C, _ = step_response(t, R_over, L, C, U0=1.0)
        assert pytest.approx(u_C[-1], rel=1e-3) == 1.0

    def test_current_zero_at_equilibrium(self):
        """Current must decay to zero at steady state"""
        R, L, C = 10.0, 1e-3, 1e-6
        t = np.linspace(0, 0.05, 100_000)
        _, i = step_response(t, R, L, C, U0=1.0)
        assert abs(i[-1]) < 1e-6


class TestBandwidth3dB:
    def test_formula(self):
        """BW = R/(2πL)"""
        R, L, C = 10.0, 1e-3, 1e-6
        bw = bandwidth_3db(R, L, C)
        assert pytest.approx(bw) == R / (2 * np.pi * L)

    def test_inverse_Q_relationship(self):
        """BW = f₀/Q"""
        R, L, C = 10.0, 1e-3, 1e-6
        _, f_0 = resonance_frequency(L, C)
        Q = quality_factor(R, L, C)
        bw = bandwidth_3db(R, L, C)
        assert pytest.approx(bw, rel=1e-5) == f_0 / Q
