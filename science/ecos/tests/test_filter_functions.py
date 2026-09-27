"""
Unit tests for economic filter model functions.

Tests cover:
- Low-pass filter calculations
- Frequency response
- Shock damping
- Stability conditions
- Cutoff frequencies
"""

import pytest
import numpy as np
from ecos.economic_analysis import (
    calculate_filter_cutoff_frequency,
    calculate_filter_magnitude,
    filter_shock_response,
    filter_stability_condition,
)


class TestFilterCutoffFrequency:
    """Test cutoff frequency calculations."""

    def test_cutoff_positive_tau(self):
        """Test cutoff frequency with positive tau."""
        tau = 1.0
        cutoff = calculate_filter_cutoff_frequency(tau)
        assert cutoff > 0
        expected = 1.0 / (2.0 * np.pi * tau)
        assert cutoff == pytest.approx(expected)

    def test_cutoff_small_tau(self):
        """Test cutoff frequency with small tau."""
        tau = 0.01
        cutoff = calculate_filter_cutoff_frequency(tau)
        assert cutoff > 0
        assert cutoff > calculate_filter_cutoff_frequency(1.0)

    def test_cutoff_large_tau(self):
        """Test cutoff frequency with large tau."""
        tau = 100.0
        cutoff = calculate_filter_cutoff_frequency(tau)
        assert cutoff > 0
        assert cutoff < calculate_filter_cutoff_frequency(1.0)

    def test_cutoff_inverse_relationship(self):
        """Test cutoff frequency is inversely proportional to tau."""
        tau1 = 1.0
        tau2 = 2.0
        cutoff1 = calculate_filter_cutoff_frequency(tau1)
        cutoff2 = calculate_filter_cutoff_frequency(tau2)
        assert cutoff1 == pytest.approx(2.0 * cutoff2)

    def test_cutoff_invalid_tau_zero(self):
        """Test cutoff frequency fails with tau = 0."""
        with pytest.raises(ValueError, match="positive"):
            calculate_filter_cutoff_frequency(0.0)

    def test_cutoff_invalid_tau_negative(self):
        """Test cutoff frequency fails with negative tau."""
        with pytest.raises(ValueError, match="positive"):
            calculate_filter_cutoff_frequency(-1.0)

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    def test_cutoff_various_taus(self, tau):
        """Test cutoff frequency with various tau values."""
        cutoff = calculate_filter_cutoff_frequency(tau)
        assert cutoff > 0
        assert cutoff == pytest.approx(1.0 / (2.0 * np.pi * tau))


class TestFilterMagnitude:
    """Test filter magnitude response."""

    def test_magnitude_zero_frequency(self):
        """Test magnitude at zero frequency."""
        tau = 1.0
        mag = calculate_filter_magnitude(0.0, tau)
        assert mag == pytest.approx(1.0)

    def test_magnitude_at_cutoff(self):
        """Test magnitude at cutoff frequency."""
        tau = 1.0
        f_cutoff = calculate_filter_cutoff_frequency(tau)
        mag = calculate_filter_magnitude(f_cutoff, tau)
        # At cutoff: |H| = 1/sqrt(2) ≈ 0.707
        assert mag == pytest.approx(1.0 / np.sqrt(2), rel=1e-10)

    def test_magnitude_decreasing_with_frequency(self):
        """Test magnitude decreases with frequency."""
        tau = 1.0
        frequencies = np.array([0.0, 0.5, 1.0, 2.0])
        magnitudes = [calculate_filter_magnitude(f, tau) for f in frequencies]
        
        # Magnitude should decrease monotonically
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] > magnitudes[i + 1]

    def test_magnitude_range(self):
        """Test magnitude stays in valid range."""
        tau = 1.0
        frequencies = np.linspace(0.0, 10.0, 100)
        
        for f in frequencies:
            mag = calculate_filter_magnitude(f, tau)
            assert 0 < mag <= 1.0

    def test_magnitude_high_frequency_limit(self):
        """Test magnitude approaches zero at high frequencies."""
        tau = 1.0
        mag = calculate_filter_magnitude(100.0, tau)
        assert mag < 0.1

    def test_magnitude_invalid_tau_zero(self):
        """Test magnitude fails with tau = 0."""
        with pytest.raises(ValueError, match="positive"):
            calculate_filter_magnitude(1.0, 0.0)

    def test_magnitude_invalid_tau_negative(self):
        """Test magnitude fails with negative tau."""
        with pytest.raises(ValueError, match="positive"):
            calculate_filter_magnitude(1.0, -1.0)

    def test_magnitude_tau_scaling(self):
        """Test magnitude changes correctly with tau."""
        f = 1.0
        tau1 = 1.0
        tau2 = 2.0
        
        mag1 = calculate_filter_magnitude(f, tau1)
        mag2 = calculate_filter_magnitude(f, tau2)
        
        # Both should be in (0, 1)
        assert 0 < mag1 < 1
        assert 0 < mag2 < 1
        # Larger tau gives stronger damping at the same frequency
        assert mag1 > mag2

    @pytest.mark.parametrize("f,tau", [
        (0.0, 1.0),
        (1.0, 0.5),
        (2.0, 1.0),
        (0.5, 2.0),
    ])
    def test_magnitude_known_values(self, f, tau):
        """Test magnitude with known mathematical values."""
        mag = calculate_filter_magnitude(f, tau)
        omega = 2.0 * np.pi * f
        expected = 1.0 / np.sqrt(1.0 + (omega * tau) ** 2)
        assert mag == pytest.approx(expected)


class TestFilterShockResponse:
    """Test shock response and damping."""

    def test_shock_response_zero_shock(self):
        """Test response to zero shock."""
        damped, damping_factor = filter_shock_response(0.0, 1.0, 1.0)
        assert damped == pytest.approx(0.0)
        assert damping_factor > 0

    def test_shock_response_low_frequency(self):
        """Test shock response at low frequency."""
        shock = 1.0
        frequency = 0.1
        tau = 1.0
        damped, damping_factor = filter_shock_response(shock, frequency, tau)
        
        # Low frequency should pass through mostly
        assert damped > 0.8 * shock
        assert damping_factor > 0.8

    def test_shock_response_high_frequency(self):
        """Test shock response at high frequency."""
        shock = 1.0
        frequency = 10.0
        tau = 1.0
        damped, damping_factor = filter_shock_response(shock, frequency, tau)
        
        # High frequency should be strongly damped
        assert damped < 0.2 * shock
        assert damping_factor < 0.2

    def test_shock_response_scaling(self):
        """Test shock response scales linearly with shock magnitude."""
        frequency = 1.0
        tau = 1.0
        
        damped1, df1 = filter_shock_response(1.0, frequency, tau)
        damped2, df2 = filter_shock_response(2.0, frequency, tau)
        
        # Damped magnitude should scale
        assert damped2 == pytest.approx(2.0 * damped1)
        # Damping factor should be same
        assert df1 == pytest.approx(df2)

    def test_shock_response_invalid_tau(self):
        """Test shock response fails with invalid tau."""
        with pytest.raises(ValueError, match="positive"):
            filter_shock_response(1.0, 1.0, 0.0)

    def test_shock_response_tuple_structure(self):
        """Test shock response returns proper tuple."""
        damped, damping = filter_shock_response(1.0, 1.0, 1.0)
        assert isinstance(damped, (float, np.floating))
        assert isinstance(damping, (float, np.floating))
        assert damped > 0
        assert 0 < damping <= 1.0


class TestFilterStabilityCondition:
    """Test filter stability analysis."""

    def test_stable_condition_large_period(self):
        """Test stability with large shock period."""
        shock_period = 100.0
        tau = 1.0
        # T >> 2π·τ, should be stable
        assert filter_stability_condition(shock_period, tau) is True

    def test_unstable_condition_small_period(self):
        """Test instability with small shock period."""
        shock_period = 0.1
        tau = 1.0
        # T << 2π·τ, should be unstable
        assert filter_stability_condition(shock_period, tau) is False

    def test_critical_stability_condition(self):
        """Test at critical stability point."""
        tau = 1.0
        critical_period = 2.0 * np.pi * tau
        # Exactly at boundary
        assert filter_stability_condition(critical_period, tau) is False
        # Just above
        assert filter_stability_condition(critical_period + 0.01, tau) is True

    def test_stability_with_different_taus(self):
        """Test stability varies with tau."""
        shock_period = 10.0
        
        tau_small = 0.5
        tau_large = 2.0
        
        stable_small = filter_stability_condition(shock_period, tau_small)
        stable_large = filter_stability_condition(shock_period, tau_large)
        
        # Larger tau makes resonance more likely
        assert stable_small == True
        assert stable_large == False

    def test_stability_invalid_tau(self):
        """Test stability fails with invalid tau."""
        with pytest.raises(ValueError, match="positive"):
            filter_stability_condition(10.0, 0.0)

    def test_stability_invalid_period(self):
        """Test stability fails with invalid period."""
        with pytest.raises(ValueError, match="positive"):
            filter_stability_condition(0.0, 1.0)

    def test_stability_inverse_relationship(self):
        """Test stability condition has inverse relationship."""
        tau = 1.0
        
        # For same tau, larger period -> more stable
        period1 = 10.0
        period2 = 50.0
        
        stable1 = filter_stability_condition(period1, tau)
        stable2 = filter_stability_condition(period2, tau)
        
        # Both should be stable given large periods
        assert stable1 and stable2

    @pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 5.0])
    def test_stability_across_taus(self, tau):
        """Test stability condition across various tau values."""
        # Large period should always be stable
        assert filter_stability_condition(100.0, tau) is True
        
        # Small period should always be unstable
        assert filter_stability_condition(0.1, tau) is False

    def test_stability_resonance_avoidance(self):
        """Test that stability condition correctly identifies resonance."""
        tau = 1.0
        
        # Dangerous frequencies (near natural frequency)
        dangerous_periods = np.linspace(5.0, 7.0, 5)
        for period in dangerous_periods:
            if period < 2.0 * np.pi * tau:
                assert filter_stability_condition(period, tau) is False
            else:
                assert filter_stability_condition(period, tau) is True
