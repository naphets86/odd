"""
Unit tests for core economic analysis functions.

Tests cover:
- Resonance window validation
- Productivity calculations
- Stability metrics
- Freedom degree calculations
- Predictability measures
"""

import pytest
import numpy as np
from ecos.economic_analysis import (
    validate_omega,
    in_resonance_window,
    calculate_productivity,
    calculate_freedom_degree,
    calculate_predictability,
    calculate_price_variance,
    calculate_gini_coefficient,
    calculate_growth_rate,
    resonance_analysis,
    RESONANCE_WINDOW_MIN,
    RESONANCE_WINDOW_MAX,
    RESONANCE_WINDOW_CENTER,
    GOLDEN_RATIO_INV,
)


class TestValidateOmega:
    """Test omega validation function."""

    def test_valid_omega_zero(self):
        """Test validation with omega = 0."""
        validate_omega(0.0)  # Should not raise

    def test_valid_omega_half(self):
        """Test validation with omega = 0.5."""
        validate_omega(0.5)  # Should not raise

    def test_valid_omega_one(self):
        """Test validation with omega = 1.0."""
        validate_omega(1.0)  # Should not raise

    def test_valid_omega_resonance_center(self):
        """Test validation with omega at resonance center."""
        validate_omega(RESONANCE_WINDOW_CENTER)  # Should not raise

    def test_invalid_omega_negative(self):
        """Test validation with negative omega."""
        with pytest.raises(ValueError, match="Omega must be in"):
            validate_omega(-0.1)

    def test_invalid_omega_above_one(self):
        """Test validation with omega > 1."""
        with pytest.raises(ValueError, match="Omega must be in"):
            validate_omega(1.5)

    def test_invalid_omega_far_negative(self):
        """Test validation with far negative omega."""
        with pytest.raises(ValueError):
            validate_omega(-10.0)

    def test_invalid_omega_far_positive(self):
        """Test validation with far positive omega."""
        with pytest.raises(ValueError):
            validate_omega(100.0)


class TestResonanceWindow:
    """Test resonance window detection."""

    def test_below_window(self):
        """Test point below resonance window."""
        assert not in_resonance_window(0.5)

    def test_window_minimum(self):
        """Test at minimum of resonance window."""
        assert in_resonance_window(RESONANCE_WINDOW_MIN)

    def test_window_center(self):
        """Test at center of resonance window."""
        assert in_resonance_window(RESONANCE_WINDOW_CENTER)

    def test_window_maximum(self):
        """Test at maximum of resonance window."""
        assert in_resonance_window(RESONANCE_WINDOW_MAX)

    def test_above_window(self):
        """Test point above resonance window."""
        assert not in_resonance_window(0.9)

    def test_just_below_window(self):
        """Test just below resonance window."""
        assert not in_resonance_window(RESONANCE_WINDOW_MIN - 0.01)

    def test_just_above_window(self):
        """Test just above resonance window."""
        assert not in_resonance_window(RESONANCE_WINDOW_MAX + 0.01)

    def test_window_width(self):
        """Test window width matches expected range."""
        width = RESONANCE_WINDOW_MAX - RESONANCE_WINDOW_MIN
        assert 0.15 < width < 0.20


class TestProductivityCalculation:
    """Test productivity function."""

    def test_productivity_zero(self):
        """Test productivity at omega = 0."""
        prod = calculate_productivity(0.0)
        assert prod > 0
        assert prod == pytest.approx(1.0 - 2.0 * RESONANCE_WINDOW_CENTER ** 2)

    def test_productivity_center(self):
        """Test productivity at optimal center."""
        prod_center = calculate_productivity(RESONANCE_WINDOW_CENTER)
        assert prod_center > 0

    def test_productivity_monotonic(self):
        """Test productivity behavior is reasonable."""
        omegas = np.linspace(0.0, 0.95, 20)
        productivities = [calculate_productivity(omega) for omega in omegas]
        assert all(p >= 0 for p in productivities)

    def test_productivity_near_one(self):
        """Test productivity approaches zero near omega=1."""
        prod = calculate_productivity(0.99)
        assert 0 <= prod

    def test_productivity_lambda_parameter(self):
        """Test productivity with different lambda parameters."""
        prod_lambda1 = calculate_productivity(0.5, lambda_param=1.0)
        prod_lambda2 = calculate_productivity(0.5, lambda_param=2.0)
        assert prod_lambda1 >= prod_lambda2

    def test_productivity_nonnegativity(self):
        """Test productivity is never negative."""
        omegas = np.linspace(0.0, 0.99, 50)
        for omega in omegas:
            assert calculate_productivity(omega) >= 0


class TestFreedomDegree:
    """Test freedom degree calculation."""

    def test_freedom_zero(self):
        """Test freedom at omega = 0."""
        freedom = calculate_freedom_degree(0.0)
        assert freedom == pytest.approx(0.0)

    def test_freedom_half(self):
        """Test freedom at omega = 0.5."""
        freedom = calculate_freedom_degree(0.5)
        expected = -np.log(0.5)
        assert freedom == pytest.approx(expected)

    def test_freedom_monotonic_increase(self):
        """Test freedom increases monotonically."""
        omegas = np.linspace(0.0, 0.9, 20)
        freedoms = [calculate_freedom_degree(omega) for omega in omegas]
        for i in range(len(freedoms) - 1):
            assert freedoms[i] < freedoms[i + 1]

    def test_freedom_window_values(self):
        """Test freedom values at window boundaries."""
        f_min = calculate_freedom_degree(RESONANCE_WINDOW_MIN)
        f_center = calculate_freedom_degree(RESONANCE_WINDOW_CENTER)
        f_max = calculate_freedom_degree(RESONANCE_WINDOW_MAX)
        assert f_min < f_center < f_max

    @pytest.mark.parametrize("omega,expected", [
        (0.0, 0.0),
        (0.5, np.log(2)),
        (0.9, -np.log(0.1)),
    ])
    def test_freedom_known_values(self, omega, expected):
        """Test freedom with known mathematical values."""
        assert calculate_freedom_degree(omega) == pytest.approx(expected, rel=1e-10)


class TestPredictability:
    """Test predictability calculation."""

    def test_predictability_center_optimal(self):
        """Test predictability is maximum at center."""
        pred_center = calculate_predictability(RESONANCE_WINDOW_CENTER)
        assert pred_center == pytest.approx(1.0)

    def test_predictability_symmetric(self):
        """Test predictability is symmetric around center."""
        delta = 0.05
        pred_left = calculate_predictability(RESONANCE_WINDOW_CENTER - delta)
        pred_right = calculate_predictability(RESONANCE_WINDOW_CENTER + delta)
        assert pred_left == pytest.approx(pred_right, rel=1e-10)

    def test_predictability_range(self):
        """Test predictability stays in valid range."""
        omegas = np.linspace(0.0, 1.0, 100)
        for omega in omegas:
            pred = calculate_predictability(omega)
            assert 0 <= pred <= 1

    def test_predictability_boundaries(self):
        """Test predictability at window boundaries."""
        pred_min = calculate_predictability(RESONANCE_WINDOW_MIN)
        pred_max = calculate_predictability(RESONANCE_WINDOW_MAX)
        assert 0.8 < pred_min < 1.0
        assert 0.8 < pred_max < 1.0

    def test_predictability_far_from_center(self):
        """Test predictability decreases far from center."""
        pred_close = calculate_predictability(RESONANCE_WINDOW_CENTER + 0.01)
        pred_far = calculate_predictability(RESONANCE_WINDOW_CENTER + 0.3)
        assert pred_close > pred_far


class TestPriceVariance:
    """Test market price variance calculation."""

    def test_price_variance_zero_omega(self):
        """Test variance at omega = 0."""
        var = calculate_price_variance(0.0)
        assert var == pytest.approx(0.0)

    def test_price_variance_one_omega(self):
        """Test variance at omega = 1."""
        var = calculate_price_variance(1.0)
        assert var == pytest.approx(0.0)

    def test_price_variance_maximum_half(self):
        """Test variance is maximum at omega = 0.5."""
        var_half = calculate_price_variance(0.5)
        var_other = calculate_price_variance(0.4)
        assert var_half > var_other

    def test_price_variance_symmetric(self):
        """Test variance is symmetric around 0.5."""
        var_low = calculate_price_variance(0.3)
        var_high = calculate_price_variance(0.7)
        assert var_low == pytest.approx(var_high)

    def test_price_variance_gamma_scaling(self):
        """Test variance scales with gamma."""
        var_1 = calculate_price_variance(0.5, gamma=1.0)
        var_2 = calculate_price_variance(0.5, gamma=2.0)
        assert var_2 == pytest.approx(2.0 * var_1)

    def test_price_variance_nonnegativity(self):
        """Test variance is never negative."""
        omegas = np.linspace(0.0, 1.0, 50)
        for omega in omegas:
            assert calculate_price_variance(omega) >= 0


class TestGiniCoefficient:
    """Test Gini coefficient calculation."""

    def test_gini_zero(self):
        """Test Gini at omega = 0."""
        gini = calculate_gini_coefficient(0.0)
        assert 0.15 < gini < 0.25

    def test_gini_one(self):
        """Test Gini at omega = 1."""
        gini = calculate_gini_coefficient(1.0)
        assert 0.95 < gini <= 1.0

    def test_gini_in_window(self):
        """Test Gini in resonance window."""
        gini = calculate_gini_coefficient(RESONANCE_WINDOW_CENTER)
        assert 0.35 < gini < 0.45

    def test_gini_range(self):
        """Test Gini stays in valid range."""
        omegas = np.linspace(0.0, 1.0, 100)
        for omega in omegas:
            gini = calculate_gini_coefficient(omega)
            assert 0 <= gini <= 1.0

    def test_gini_monotonic_increase(self):
        """Test Gini generally increases with omega."""
        omegas = np.linspace(0.0, 1.0, 20)
        ginis = [calculate_gini_coefficient(omega) for omega in omegas]
        # Should generally increase (with some smoothing in window)
        assert ginis[-1] > ginis[0]

    @pytest.mark.parametrize("omega", [
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    ])
    def test_gini_continuity(self, omega):
        """Test Gini is continuous."""
        gini = calculate_gini_coefficient(omega)
        gini_delta = calculate_gini_coefficient(min(1.0, omega + 1e-8))
        assert abs(gini - gini_delta) < 1e-5


class TestGrowthRate:
    """Test economic growth rate calculation."""

    def test_growth_low_controlled_regime(self):
        """Test growth in controlled regime."""
        growth = calculate_growth_rate(0.3)
        assert -0.01 < growth < 0.02

    def test_growth_in_window(self):
        """Test growth is positive in resonance window."""
        growth = calculate_growth_rate(RESONANCE_WINDOW_CENTER)
        assert 0.02 < growth < 0.05

    def test_growth_crisis_regime(self):
        """Test growth is negative in crisis regime."""
        growth = calculate_growth_rate(0.9)
        assert growth < 0

    def test_growth_range(self):
        """Test growth stays in reasonable range."""
        omegas = np.linspace(0.0, 1.0, 50)
        for omega in omegas:
            growth = calculate_growth_rate(omega)
            assert -0.05 < growth < 0.05

    def test_growth_peak_in_window(self):
        """Test growth is maximized in window."""
        growth_window = calculate_growth_rate(RESONANCE_WINDOW_CENTER)
        growth_low = calculate_growth_rate(0.3)
        growth_high = calculate_growth_rate(0.9)
        assert growth_window > growth_low
        assert growth_window > growth_high


class TestResonanceAnalysis:
    """Test comprehensive resonance analysis."""

    def test_resonance_analysis_structure(self):
        """Test resonance analysis returns correct structure."""
        analysis = resonance_analysis(0.7)
        assert hasattr(analysis, 'omega')
        assert hasattr(analysis, 'in_window')
        assert hasattr(analysis, 'stability_score')
        assert hasattr(analysis, 'productivity')
        assert hasattr(analysis, 'freedom_degree')
        assert hasattr(analysis, 'predictability')
        assert hasattr(analysis, 'gini_coefficient')
        assert hasattr(analysis, 'growth_rate')
        assert hasattr(analysis, 'window_distance')

    def test_resonance_analysis_in_window(self):
        """Test resonance analysis detects window correctly."""
        analysis_in = resonance_analysis(RESONANCE_WINDOW_CENTER)
        assert analysis_in.in_window is True

        analysis_out = resonance_analysis(0.5)
        assert analysis_out.in_window is False

    def test_resonance_analysis_consistency(self):
        """Test resonance analysis is internally consistent."""
        omega = 0.7
        analysis = resonance_analysis(omega)
        
        assert analysis.omega == omega
        assert analysis.stability_score > 0
        assert analysis.freedom_degree > 0
        assert 0 <= analysis.predictability <= 1
        assert 0 <= analysis.gini_coefficient <= 1

    def test_resonance_analysis_at_boundaries(self):
        """Test resonance analysis at all boundaries."""
        omegas = [0.0, 0.5, RESONANCE_WINDOW_MIN, RESONANCE_WINDOW_CENTER, 
                 RESONANCE_WINDOW_MAX, 0.9, 1.0]
        
        for omega in omegas:
            analysis = resonance_analysis(omega)
            assert analysis is not None
            assert all(not np.isnan(v) for v in [
                analysis.stability_score,
                analysis.productivity,
                analysis.freedom_degree,
                analysis.predictability,
                analysis.gini_coefficient,
                analysis.growth_rate,
                analysis.window_distance
            ])

    @pytest.mark.parametrize("omega", [0.0, 0.25, 0.5, 0.707, 0.75, 1.0])
    def test_resonance_analysis_valid_metrics(self, omega):
        """Test resonance analysis produces valid metrics."""
        analysis = resonance_analysis(omega)
        
        # All metrics should be finite
        assert np.isfinite(analysis.stability_score)
        assert np.isfinite(analysis.productivity)
        assert np.isfinite(analysis.freedom_degree)
        assert np.isfinite(analysis.predictability)
        assert np.isfinite(analysis.gini_coefficient)
        assert np.isfinite(analysis.growth_rate)
        assert np.isfinite(analysis.window_distance)
