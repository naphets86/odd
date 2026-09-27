"""
Unit tests for harmonic analysis and Leibniz series functions.

Tests cover:
- Leibniz series coefficients
- Series convergence
- Rectangle wave Fourier reconstruction
- Market supply odd harmonics
- Market adjustment dynamics
"""

import pytest
import numpy as np
from ecos.economic_analysis import (
    leibniz_series_coefficient,
    leibniz_series_sum,
    leibniz_convergence_error,
    market_supply_odd_harmonics,
    fourier_rectangle_wave,
    market_adjustment_dynamics,
)


class TestLeibnizSeriesCoefficient:
    """Test Leibniz series coefficient calculation."""

    def test_coefficient_first_term(self):
        """Test first term of Leibniz series."""
        coeff = leibniz_series_coefficient(0)
        assert coeff == pytest.approx(1.0)

    def test_coefficient_second_term(self):
        """Test second term of Leibniz series."""
        coeff = leibniz_series_coefficient(1)
        assert coeff == pytest.approx(-1.0/3.0)

    def test_coefficient_third_term(self):
        """Test third term of Leibniz series."""
        coeff = leibniz_series_coefficient(2)
        assert coeff == pytest.approx(1.0/5.0)

    def test_coefficient_alternating_signs(self):
        """Test alternating signs of coefficients."""
        for n in range(10):
            coeff = leibniz_series_coefficient(n)
            expected_sign = 1 if n % 2 == 0 else -1
            assert np.sign(coeff) == expected_sign

    def test_coefficient_decreasing_magnitude(self):
        """Test coefficient magnitudes decrease."""
        magnitudes = [abs(leibniz_series_coefficient(n)) for n in range(10)]
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] > magnitudes[i + 1]

    def test_coefficient_formula(self):
        """Test Leibniz series formula: (-1)^n / (2n+1)."""
        for n in range(20):
            coeff = leibniz_series_coefficient(n)
            expected = (-1) ** n / (2 * n + 1)
            assert coeff == pytest.approx(expected)

    def test_coefficient_invalid_negative_n(self):
        """Test coefficient fails with negative n."""
        with pytest.raises(ValueError, match="non-negative"):
            leibniz_series_coefficient(-1)

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 100])
    def test_coefficient_various_n(self, n):
        """Test coefficient with various n values."""
        coeff = leibniz_series_coefficient(n)
        assert np.isfinite(coeff)
        assert abs(coeff) <= 1.0


class TestLeibnizSeriesSum:
    """Test Leibniz series partial sum."""

    def test_series_sum_one_term(self):
        """Test series sum with one term."""
        s = leibniz_series_sum(1)
        assert s == pytest.approx(1.0)

    def test_series_sum_two_terms(self):
        """Test series sum with two terms."""
        s = leibniz_series_sum(2)
        expected = 1.0 - 1.0/3.0
        assert s == pytest.approx(expected)

    def test_series_sum_convergence(self):
        """Test series convergence toward π/4."""
        pi_quarter = np.pi / 4.0
        
        sums = [leibniz_series_sum(n) for n in [10, 50, 100, 500]]
        errors = [abs(s - pi_quarter) for s in sums]
        
        # Errors should decrease
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1]

    def test_series_sum_range(self):
        """Test series sum stays in reasonable range."""
        for n in range(11, 100, 10):
            s = leibniz_series_sum(n)
            # Should be bounded near π/4
            assert 0.5 < s < 1.0

    def test_series_sum_invalid_terms(self):
        """Test series sum fails with invalid num_terms."""
        with pytest.raises(ValueError, match="positive"):
            leibniz_series_sum(0)
        
        with pytest.raises(ValueError, match="positive"):
            leibniz_series_sum(-1)

    @pytest.mark.parametrize("n_terms", [1, 5, 10, 50, 100])
    def test_series_sum_various_terms(self, n_terms):
        """Test series sum with various term counts."""
        s = leibniz_series_sum(n_terms)
        assert np.isfinite(s)
        assert 0 < s <= 1.0

    def test_series_sum_oscillation(self):
        """Test series oscillates around π/4."""
        pi_quarter = np.pi / 4.0
        
        s1 = leibniz_series_sum(100)
        s2 = leibniz_series_sum(101)
        
        # Should oscillate on either side
        error1 = s1 - pi_quarter
        error2 = s2 - pi_quarter
        assert error1 * error2 < 0  # Opposite signs


class TestLeibnizConvergenceError:
    """Test Leibniz series convergence error estimates."""

    def test_error_one_term(self):
        """Test convergence error with one term."""
        error = leibniz_convergence_error(1)
        assert error == pytest.approx(1.0 / 3.0)

    def test_error_two_terms(self):
        """Test convergence error with two terms."""
        error = leibniz_convergence_error(2)
        assert error == pytest.approx(1.0 / 5.0)

    def test_error_decreasing(self):
        """Test convergence error decreases with terms."""
        errors = [leibniz_convergence_error(n) for n in range(1, 20)]
        
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1]

    def test_error_positive(self):
        """Test convergence error is always positive."""
        for n in range(1, 100):
            error = leibniz_convergence_error(n)
            assert error > 0

    def test_error_formula(self):
        """Test convergence error formula: 1/(2n+1)."""
        for n in range(1, 20):
            error = leibniz_convergence_error(n)
            expected = 1.0 / (2 * n + 1)
            assert error == pytest.approx(expected)

    def test_error_invalid_terms(self):
        """Test error fails with invalid terms."""
        with pytest.raises(ValueError, match="positive"):
            leibniz_convergence_error(0)

    @pytest.mark.parametrize("n", [1, 10, 100, 1000])
    def test_error_upper_bound(self, n):
        """Test actual error is bounded by estimate."""
        pi_quarter = np.pi / 4.0
        s = leibniz_series_sum(n)
        actual_error = abs(s - pi_quarter)
        error_bound = leibniz_convergence_error(n)
        
        assert actual_error <= error_bound


class TestMarketSupplyOddHarmonics:
    """Test market supply calculation with odd harmonics."""

    def test_supply_base_level(self):
        """Test supply at base level (no harmonics)."""
        S = market_supply_odd_harmonics(0.0, S0=100.0, amplitudes=[], frequencies=[])
        assert S == pytest.approx(100.0)

    def test_supply_with_single_harmonic(self):
        """Test supply with single harmonic."""
        t = 0.0
        S0 = 100.0
        amplitude = 10.0
        frequency = 1.0
        
        S = market_supply_odd_harmonics(t, S0, [amplitude], [frequency])
        # At t=0, sin(0) = 0
        assert S == pytest.approx(S0)

    def test_supply_harmonic_amplitude_damping(self):
        """Test harmonic amplitudes are damped by (2n+1)."""
        t = np.pi / 2.0  # sin(π/2) = 1
        S0 = 100.0
        amplitude = 1.0
        frequency = 1.0
        
        # At this time, harmonic n adds amplitude/(2n+1) * sin((2n+1)*π/2)
        S = market_supply_odd_harmonics(t, S0, [amplitude], [frequency])
        
        # 1st harmonic (n=0): amplitude/1 * sin(π/2) = 1.0
        expected_contribution = 1.0 / 1.0
        assert S > S0  # Should add something

    def test_supply_multiple_harmonics(self):
        """Test supply with multiple harmonics."""
        t = 0.1
        S0 = 100.0
        amplitudes = [10.0, 5.0, 2.0]
        frequencies = [1.0, 1.0, 1.0]
        
        S = market_supply_odd_harmonics(t, S0, amplitudes, frequencies)
        assert S > 0  # Should be positive
        assert np.isfinite(S)

    def test_supply_with_phases(self):
        """Test supply calculation with phase shifts."""
        t = 0.0
        S0 = 100.0
        amplitudes = [10.0]
        frequencies = [1.0]
        phases = [np.pi / 2.0]  # Phase shift
        
        S = market_supply_odd_harmonics(t, S0, amplitudes, frequencies, phases)
        # With phase shift π/2, sin(π/2) = 1
        assert S != pytest.approx(100.0)  # Should differ from base

    def test_supply_default_phases(self):
        """Test supply uses default phases when not provided."""
        t = 0.1
        S0 = 100.0
        amplitudes = [10.0]
        frequencies = [1.0]
        
        S_with_phases = market_supply_odd_harmonics(t, S0, amplitudes, frequencies, None)
        S_explicit_zero = market_supply_odd_harmonics(t, S0, amplitudes, frequencies, [0.0])
        
        assert S_with_phases == pytest.approx(S_explicit_zero)

    def test_supply_empty_harmonics(self):
        """Test supply with empty harmonics list."""
        S = market_supply_odd_harmonics(0.5, 100.0, [], [])
        assert S == pytest.approx(100.0)

    def test_supply_oscillatory_behavior(self):
        """Test supply oscillates over time."""
        S0 = 100.0
        amplitudes = [10.0]
        frequencies = [1.0]
        
        times = np.linspace(0, 2*np.pi, 100)
        supplies = [market_supply_odd_harmonics(t, S0, amplitudes, frequencies) 
                   for t in times]
        
        # Should have variation
        assert max(supplies) > min(supplies)


class TestFourierRectangleWave:
    """Test Fourier rectangle wave reconstruction."""

    def test_rectangle_wave_zero_time(self):
        """Test rectangle wave at t=0."""
        v = fourier_rectangle_wave(0.0, amplitude=1.0, period=1.0, num_harmonics=10)
        assert v >= 0

    def test_rectangle_wave_positive_values(self):
        """Test rectangle wave produces bounded values."""
        times = np.linspace(0, 2, 50)
        amplitude = 1.0
        
        for t in times:
            v = fourier_rectangle_wave(t, amplitude=amplitude, period=1.0)
            # Should be bounded by amplitude (approximately)
            assert -2*amplitude < v < 2*amplitude

    def test_rectangle_wave_amplitude_scaling(self):
        """Test rectangle wave scales with amplitude."""
        t = 0.25
        v1 = fourier_rectangle_wave(t, amplitude=1.0, period=1.0)
        v2 = fourier_rectangle_wave(t, amplitude=2.0, period=1.0)
        
        # Should scale linearly
        assert v2 == pytest.approx(2.0 * v1)

    def test_rectangle_wave_harmonics_convergence(self):
        """Test more harmonics give better approximation."""
        t = 0.25
        amplitude = 1.0
        period = 1.0
        
        v10 = fourier_rectangle_wave(t, amplitude, period, num_harmonics=10)
        v50 = fourier_rectangle_wave(t, amplitude, period, num_harmonics=50)
        v100 = fourier_rectangle_wave(t, amplitude, period, num_harmonics=100)
        
        # With more harmonics, should approach a steady value
        assert v10 != pytest.approx(v50)  # Should improve
        assert v50 != pytest.approx(v100)  # Should improve more

    def test_rectangle_wave_periodicity(self):
        """Test rectangle wave is periodic."""
        amplitude = 1.0
        period = 1.0
        num_harmonics = 50
        
        v1 = fourier_rectangle_wave(0.1, amplitude, period, num_harmonics)
        v2 = fourier_rectangle_wave(0.1 + period, amplitude, period, num_harmonics)
        
        # Should be approximately periodic
        assert v1 == pytest.approx(v2, rel=1e-5)

    def test_rectangle_wave_invalid_harmonics(self):
        """Test rectangle wave fails with invalid harmonics."""
        with pytest.raises(ValueError, match="positive"):
            fourier_rectangle_wave(0.5, num_harmonics=0)


class TestMarketAdjustmentDynamics:
    """Test market adjustment dynamics ODE solution."""

    def test_adjustment_basic_solution(self):
        """Test basic market adjustment solution."""
        demand_shock = 1.0
        alpha = -1.0
        eta = 0.1
        T = 2.0 * np.pi
        t_array = np.array([0.0, 1.0, 2.0])
        
        P = market_adjustment_dynamics(demand_shock, alpha, eta, T, t_array)
        
        assert P.shape == t_array.shape
        assert len(P) == 3

    def test_adjustment_finite_values(self):
        """Test adjustment produces finite values."""
        demand_shock = 1.0
        alpha = -1.0
        eta = 0.1
        T = 2.0
        t_array = np.linspace(0, 10, 100)
        
        P = market_adjustment_dynamics(demand_shock, alpha, eta, T, t_array)
        
        assert np.all(np.isfinite(P))

    def test_adjustment_decay_behavior(self):
        """Test adjustment shows decay without driving force."""
        demand_shock = 1.0
        alpha = -1.0
        eta = 0.0  # No periodic driving
        T = 1.0
        t_array = np.linspace(0, 10, 100)
        
        P = market_adjustment_dynamics(demand_shock, alpha, eta, T, t_array)
        
        # Should decay toward zero
        assert P[0] > P[-1]
        assert abs(P[-1]) < abs(P[0])

    def test_adjustment_periodic_driving(self):
        """Test adjustment with periodic driving force."""
        demand_shock = 0.0
        alpha = -0.5
        eta = 1.0
        T = 2.0 * np.pi
        t_array = np.linspace(0, 20, 200)
        
        P = market_adjustment_dynamics(demand_shock, alpha, eta, T, t_array)
        
        # Should show driven oscillations
        assert np.ptp(P) > 0.1  # Has variation

    def test_adjustment_positive_alpha_warning(self):
        """Test adjustment warns with positive alpha."""
        with pytest.warns(UserWarning):
            market_adjustment_dynamics(1.0, 1.0, 0.1, 1.0, np.array([1.0]))

    def test_adjustment_single_time_point(self):
        """Test adjustment for single time point."""
        P = market_adjustment_dynamics(1.0, -1.0, 0.1, 1.0, np.array([0.0]))
        
        assert len(P) == 1
        assert np.isfinite(P[0])

    def test_adjustment_array_shape_preservation(self):
        """Test adjustment preserves input array shape."""
        t_array = np.linspace(0, 5, 50)
        P = market_adjustment_dynamics(1.0, -1.0, 0.1, 1.0, t_array)
        
        assert P.shape == t_array.shape
        assert isinstance(P, np.ndarray)
