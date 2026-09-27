"""
Integration and edge case tests for economic_analysis module.

Tests cover:
- Integration between different functions
- Edge cases and boundary conditions
- Error handling and validation
- Numerical stability
"""

import pytest
import numpy as np
from ecos.economic_analysis import (
    validate_omega,
    resonance_analysis,
    in_resonance_window,
    calculate_productivity,
    calculate_gini_coefficient,
    calculate_freedom_degree,
    calculate_predictability,
    calculate_growth_rate,
    interpret_regime,
    market_entropy,
    lyapunov_exponent_economic_system,
    calculate_filter_magnitude,
    filter_stability_condition,
    leibniz_series_sum,
    RESONANCE_WINDOW_CENTER,
    RESONANCE_WINDOW_MIN,
    RESONANCE_WINDOW_MAX,
)


class TestIntegrationResonanceWindow:
    """Integration tests for resonance window detection."""

    def test_resonance_and_regime_interpretation(self):
        """Test resonance detection matches regime interpretation."""
        # Optimal omega should be identified as optimal regime
        omega = RESONANCE_WINDOW_CENTER
        assert in_resonance_window(omega)
        interp = interpret_regime(omega)
        assert "optimal" in interp.lower() or "balance" in interp.lower()

    def test_resonance_and_metrics_consistency(self):
        """Test resonance window matches metrics consistency."""
        # Test points inside and outside window show different behaviors
        omega_in = 0.707
        omega_out = 0.5
        
        analysis_in = resonance_analysis(omega_in)
        analysis_out = resonance_analysis(omega_out)
        
        assert analysis_in.in_window is True
        assert analysis_out.in_window is False
        
        # Inside window should have better growth
        assert analysis_in.growth_rate > analysis_out.growth_rate

    def test_predictability_and_growth_correlation(self):
        """Test predictability correlates with growth in window."""
        omegas = np.linspace(RESONANCE_WINDOW_MIN, RESONANCE_WINDOW_MAX, 20)
        
        for omega in omegas:
            pred = calculate_predictability(omega)
            growth = calculate_growth_rate(omega)
            
            # Higher predictability should correlate with growth
            assert pred > 0.8  # High predictability in window
            assert growth > 0   # Positive growth in window


class TestIntegrationEntropy:
    """Integration tests for entropy calculations."""

    def test_entropy_and_uncertainty(self):
        """Test entropy reflects uncertainty."""
        # Maximum entropy at 0.5 (maximum uncertainty)
        e_half = market_entropy(0.5)
        e_certain = market_entropy(0.1)
        
        assert e_half > e_certain

    def test_entropy_and_freedom(self):
        """Test entropy relates to freedom of economic system."""
        # Low omega has lower freedom and entropy
        freedom_low = calculate_freedom_degree(0.2)
        freedom_high = calculate_freedom_degree(0.8)
        
        assert freedom_low < freedom_high


class TestIntegrationDynamics:
    """Integration tests for dynamic analysis."""

    def test_filter_stability_and_resonance(self):
        """Test filter stability relates to resonance conditions."""
        # System in resonance window should be more stable
        tau = 1.0
        large_period = 100.0  # Should be stable
        
        is_stable = filter_stability_condition(large_period, tau)
        assert is_stable is True

    def test_leibniz_convergence_to_pi_over_4(self):
        """Test Leibniz series converges to π/4 correctly."""
        pi_quarter = np.pi / 4.0
        
        for n_terms in [10, 50, 100, 500, 1000]:
            s = leibniz_series_sum(n_terms)
            error = abs(s - pi_quarter)
            
            # Error should decrease with more terms
            if n_terms > 10:
                s_prev = leibniz_series_sum(n_terms - 10)
                error_prev = abs(s_prev - pi_quarter)
                assert error < error_prev


class TestEdgeCasesOmega:
    """Edge case tests for omega parameter."""

    def test_edge_case_omega_exactly_zero(self):
        """Test all functions with omega = 0.0."""
        omega = 0.0
        
        validate_omega(omega)  # Should pass
        assert not in_resonance_window(omega)
        
        prod = calculate_productivity(omega)
        assert prod > 0
        
        freedom = calculate_freedom_degree(omega)
        assert freedom == pytest.approx(0.0)
        
        pred = calculate_predictability(omega)
        assert pred > 0
        
        entropy = market_entropy(omega)
        assert entropy == pytest.approx(0.0)

    def test_edge_case_omega_exactly_one(self):
        """Test all functions with omega = 1.0."""
        omega = 1.0
        
        validate_omega(omega)  # Should pass
        assert not in_resonance_window(omega)
        
        prod = calculate_productivity(omega)
        assert prod >= 0
        
        freedom = calculate_freedom_degree(omega)
        assert np.isinf(freedom)
        
        entropy = market_entropy(omega)
        assert entropy == pytest.approx(0.0)

    def test_edge_case_omega_just_inside_window(self):
        """Test functions just inside resonance window."""
        omega = RESONANCE_WINDOW_MIN + 1e-10
        
        assert in_resonance_window(omega)
        
        analysis = resonance_analysis(omega)
        assert analysis.in_window is True
        assert analysis.growth_rate > 0

    def test_edge_case_omega_just_outside_window(self):
        """Test functions just outside resonance window."""
        omega = RESONANCE_WINDOW_MIN - 1e-10
        
        assert not in_resonance_window(omega)
        
        analysis = resonance_analysis(omega)
        assert analysis.in_window is False


class TestEdgeCasesNumerical:
    """Edge case tests for numerical stability."""

    def test_very_small_freedom_degree(self):
        """Test freedom degree with very small omega."""
        omega = 1e-10
        freedom = calculate_freedom_degree(omega)
        
        # Should be very small but positive
        assert freedom > 0
        assert freedom < 1e-5

    def test_very_large_leibniz_series(self):
        """Test Leibniz series with many terms."""
        s = leibniz_series_sum(10000)
        pi_quarter = np.pi / 4.0
        
        error = abs(s - pi_quarter)
        assert error < 0.001  # Should be very close

    def test_filter_magnitude_extreme_frequencies(self):
        """Test filter magnitude with extreme frequencies."""
        tau = 1.0
        
        # Zero frequency
        mag_zero = calculate_filter_magnitude(0.0, tau)
        assert mag_zero == pytest.approx(1.0)
        
        # Very high frequency
        mag_high = calculate_filter_magnitude(1000.0, tau)
        assert mag_high < 1e-3

    def test_resonance_analysis_all_boundaries(self):
        """Test resonance analysis at all critical points."""
        critical_points = [
            0.0,
            RESONANCE_WINDOW_MIN,
            RESONANCE_WINDOW_CENTER,
            RESONANCE_WINDOW_MAX,
            1.0
        ]
        
        for omega in critical_points:
            analysis = resonance_analysis(omega)
            
            # All metrics should be finite
            assert np.isfinite(analysis.stability_score)
            assert np.isfinite(analysis.productivity)
            assert np.isfinite(analysis.freedom_degree) or np.isinf(analysis.freedom_degree)
            assert np.isfinite(analysis.predictability)
            assert np.isfinite(analysis.gini_coefficient)
            assert np.isfinite(analysis.growth_rate)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_validate_omega_boundary_values(self):
        """Test validate_omega at boundaries."""
        # Valid boundaries
        validate_omega(0.0)
        validate_omega(1.0)
        
        # Invalid outside boundaries
        with pytest.raises(ValueError):
            validate_omega(-1e-10)
        
        with pytest.raises(ValueError):
            validate_omega(1.0 + 1e-10)

    def test_filter_functions_invalid_inputs(self):
        """Test filter functions handle invalid inputs."""
        # Test various invalid tau values
        with pytest.raises(ValueError):
            calculate_filter_magnitude(1.0, 0.0)
        
        with pytest.raises(ValueError):
            calculate_filter_magnitude(1.0, -1.0)
        
        with pytest.raises(ValueError):
            filter_stability_condition(1.0, 0.0)
        
        with pytest.raises(ValueError):
            filter_stability_condition(0.0, 1.0)

    def test_regime_unknown_input(self):
        """Test regime functions with invalid input."""
        with pytest.raises(ValueError):
            from ecos.economic_analysis import estimate_historical_omega
            estimate_historical_omega("nonexistent_regime")

    def test_leibniz_invalid_parameters(self):
        """Test Leibniz functions with invalid parameters."""
        with pytest.raises(ValueError):
            leibniz_series_sum(0)
        
        with pytest.raises(ValueError):
            from ecos.economic_analysis import leibniz_series_coefficient
            leibniz_series_coefficient(-1)


class TestConsistencyAndMonotonicity:
    """Test consistency and monotonicity properties."""

    def test_freedom_strictly_increasing(self):
        """Test freedom degree is strictly increasing."""
        omegas = np.linspace(0.0, 0.99, 100)
        freedoms = [calculate_freedom_degree(omega) for omega in omegas]
        
        for i in range(len(freedoms) - 1):
            assert freedoms[i] < freedoms[i + 1]

    def test_predictability_peak_at_center(self):
        """Test predictability has single peak at center."""
        omegas = np.linspace(0.0, 1.0, 200)
        preds = [calculate_predictability(omega) for omega in omegas]
        
        max_idx = np.argmax(preds)
        max_omega = omegas[max_idx]
        
        # Maximum should be near resonance center
        assert abs(max_omega - RESONANCE_WINDOW_CENTER) < 0.05

    def test_gini_monotonic_tendency(self):
        """Test Gini coefficient shows monotonic tendency."""
        omegas = np.linspace(0.0, 1.0, 50)
        ginis = [calculate_gini_coefficient(omega) for omega in omegas]
        
        # Should generally increase (with some smoothing)
        assert ginis[-1] > ginis[0]

    def test_growth_rate_shape(self):
        """Test growth rate has expected shape."""
        omegas = np.linspace(0.0, 1.0, 100)
        growths = [calculate_growth_rate(omega) for omega in omegas]
        
        # Should be peaked in resonance window
        max_growth = max(growths)
        max_idx = growths.index(max_growth)
        max_omega = omegas[max_idx]
        
        assert RESONANCE_WINDOW_MIN < max_omega < RESONANCE_WINDOW_MAX


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_entropy_numerical_stability(self):
        """Test entropy is numerically stable."""
        # Test with values that might cause numerical issues
        for omega in [1e-10, 0.5, 1.0 - 1e-10]:
            entropy = market_entropy(omega)
            assert np.isfinite(entropy)
            assert entropy >= 0

    def test_lyapunov_numerical_stability(self):
        """Test Lyapunov exponent is numerically stable."""
        for omega in np.linspace(0.0, 0.99, 20):
            lam = lyapunov_exponent_economic_system(omega)
            assert np.isfinite(lam) or np.isinf(lam)

    def test_analysis_without_nans(self):
        """Test resonance analysis produces no NaNs."""
        omegas = np.linspace(0.0, 0.99, 100)
        
        for omega in omegas:
            analysis = resonance_analysis(omega)
            
            # Check for NaNs
            assert not np.isnan(analysis.stability_score)
            assert not np.isnan(analysis.productivity)
            assert not np.isnan(analysis.predictability)
            assert not np.isnan(analysis.gini_coefficient)
            assert not np.isnan(analysis.growth_rate)


class TestCrossModuleFunctionality:
    """Test cross-module functionality and interactions."""

    def test_complete_economic_analysis_pipeline(self):
        """Test complete pipeline of economic analysis."""
        omega = 0.7
        
        # Validate
        validate_omega(omega)
        
        # Analyze
        analysis = resonance_analysis(omega)
        assert analysis.in_window is True
        
        # Interpret
        interpretation = interpret_regime(omega)
        assert "optimal" in interpretation.lower() or "balance" in interpretation.lower()
        
        # Check entropy
        entropy = market_entropy(omega)
        assert 0 <= entropy <= np.log(2)
        
        # Check stability
        lyap = lyapunov_exponent_economic_system(omega)
        assert lyap < 0

    def test_crisis_to_optimal_transition(self):
        """Test transition from crisis to optimal regime."""
        # Crisis omega
        omega_crisis = 0.9
        analysis_crisis = resonance_analysis(omega_crisis)
        
        # Optimal omega
        omega_optimal = 0.7
        analysis_optimal = resonance_analysis(omega_optimal)
        
        # Check metrics improve
        assert analysis_optimal.growth_rate > analysis_crisis.growth_rate
        assert analysis_optimal.predictability > analysis_crisis.predictability
        assert analysis_optimal.stability_score > analysis_crisis.stability_score
