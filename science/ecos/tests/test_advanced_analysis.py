"""
Unit tests for advanced economic analysis functions.

Tests cover:
- Historical regime analysis
- Market entropy calculations
- Mutual information
- Lyapunov stability analysis
- Supply chain optimization
"""

import pytest
import numpy as np
from ecos.economic_analysis import (
    estimate_historical_omega,
    interpret_regime,
    market_entropy,
    mutual_information_market,
    lyapunov_exponent_economic_system,
    lyapunov_stability_margin,
    optimize_supply_chain_routing,
    RESONANCE_WINDOW_CENTER,
)


class TestHistoricalRegimes:
    """Test historical regime omega estimation."""

    def test_estimate_known_regime(self):
        """Test estimation for known regime."""
        omega, interpretation = estimate_historical_omega("switzerland")
        assert 0 <= omega <= 1
        assert isinstance(interpretation, str)
        assert "prosperity" in interpretation.lower() or "optimal" in interpretation.lower()

    def test_estimate_all_regimes(self):
        """Test all predefined regimes."""
        regimes = [
            "soviet_union", "east_germany", "north_korea", "switzerland",
            "scandinavia", "germany_longterm", "weimar_1923", "argentina_2001",
            "zimbabwe_2008", "uk_cornlaw_before", "uk_cornlaw_during", "uk_cornlaw_after"
        ]
        
        for regime in regimes:
            omega, interpretation = estimate_historical_omega(regime)
            assert 0 <= omega <= 1
            assert len(interpretation) > 0

    def test_estimate_controlled_regimes(self):
        """Test controlled economy regimes have low omega."""
        regimes = ["north_korea", "soviet_union", "east_germany"]
        omegas = [estimate_historical_omega(r)[0] for r in regimes]
        
        assert all(omega < 0.5 for omega in omegas)

    def test_estimate_prosperous_regimes(self):
        """Test prosperous economy regimes have moderate omega."""
        regimes = ["switzerland", "scandinavia", "germany_longterm"]
        omegas = [estimate_historical_omega(r)[0] for r in regimes]
        
        assert all(0.65 < omega < 0.75 for omega in omegas)

    def test_estimate_crisis_regimes(self):
        """Test crisis regimes have high omega."""
        regimes = ["weimar_1923", "argentina_2001", "zimbabwe_2008"]
        omegas = [estimate_historical_omega(r)[0] for r in regimes]
        
        assert all(omega > 0.8 for omega in omegas)

    def test_estimate_invalid_regime(self):
        """Test estimation fails for unknown regime."""
        with pytest.raises(ValueError, match="Unknown regime"):
            estimate_historical_omega("atlantis_2500")

    def test_estimate_case_insensitive(self):
        """Test regime names are case insensitive."""
        omega1, _ = estimate_historical_omega("Switzerland")
        omega2, _ = estimate_historical_omega("SWITZERLAND")
        omega3, _ = estimate_historical_omega("switzerland")
        
        assert omega1 == omega2 == omega3

    def test_estimate_space_handling(self):
        """Test regime names handle spaces."""
        # This tests that space replacement works
        omega, interp = estimate_historical_omega("uk cornlaw before")
        assert 0 <= omega <= 1


class TestRegimeInterpretation:
    """Test regime interpretation."""

    def test_interpret_extreme_control(self):
        """Test interpretation of extreme control regime."""
        interp = interpret_regime(0.1)
        assert "control" in interp.lower()
        assert "poverty" in interp.lower()

    def test_interpret_over_controlled(self):
        """Test interpretation of over-controlled regime."""
        interp = interpret_regime(0.4)
        assert "control" in interp.lower()
        assert "innovation" in interp.lower()

    def test_interpret_optimal_balance(self):
        """Test interpretation at optimal balance."""
        interp = interpret_regime(RESONANCE_WINDOW_CENTER)
        assert "optimal" in interp.lower() or "balance" in interp.lower()

    def test_interpret_toward_optimum(self):
        """Test interpretation moving toward optimum."""
        interp = interpret_regime(0.65)
        assert "optimum" in interp.lower() or "structure" in interp.lower()

    def test_interpret_toward_chaos(self):
        """Test interpretation moving toward chaos."""
        interp = interpret_regime(0.75)
        assert "chaos" in interp.lower() or "freedom" in interp.lower() or "predictability" in interp.lower()

    def test_interpret_early_crisis(self):
        """Test interpretation of early crisis."""
        interp = interpret_regime(0.82)
        assert "crisis" in interp.lower() or "instability" in interp.lower()

    def test_interpret_crisis_regime(self):
        """Test interpretation of crisis regime."""
        interp = interpret_regime(0.9)
        assert "crisis" in interp.lower()

    def test_interpret_extreme_chaos(self):
        """Test interpretation of extreme chaos."""
        interp = interpret_regime(0.99)
        assert "chaos" in interp.lower() or "collapse" in interp.lower()

    def test_interpret_returns_string(self):
        """Test interpretation returns non-empty string."""
        for omega in [0.0, 0.25, 0.5, 0.75, 1.0]:
            interp = interpret_regime(omega)
            assert isinstance(interp, str)
            assert len(interp) > 0


class TestMarketEntropy:
    """Test market entropy calculation."""

    def test_entropy_zero_omega(self):
        """Test entropy at omega = 0."""
        s = market_entropy(0.0)
        assert s == pytest.approx(0.0)

    def test_entropy_one_omega(self):
        """Test entropy at omega = 1."""
        s = market_entropy(1.0)
        assert s == pytest.approx(0.0)

    def test_entropy_half_omega(self):
        """Test entropy at omega = 0.5."""
        s = market_entropy(0.5)
        expected = 2.0 * 0.5 * np.log(2)
        assert s == pytest.approx(expected)

    def test_entropy_symmetric(self):
        """Test entropy is symmetric around 0.5."""
        e1 = market_entropy(0.3)
        e2 = market_entropy(0.7)
        assert e1 == pytest.approx(e2)

    def test_entropy_maximum_at_half(self):
        """Test entropy is maximum at omega = 0.5."""
        e_half = market_entropy(0.5)
        e_other = market_entropy(0.3)
        
        assert e_half > e_other

    def test_entropy_range(self):
        """Test entropy stays in valid range."""
        omegas = np.linspace(0.0, 1.0, 100)
        for omega in omegas:
            s = market_entropy(omega)
            assert 0 <= s <= np.log(2)

    def test_entropy_monotonic_to_half(self):
        """Test entropy increases to 0.5."""
        omegas = np.linspace(0.0, 0.5, 50)
        entropies = [market_entropy(omega) for omega in omegas]
        
        for i in range(len(entropies) - 1):
            assert entropies[i] <= entropies[i + 1]


class TestMutualInformation:
    """Test mutual information between markets."""

    def test_mutual_info_identical_markets(self):
        """Test mutual information for identical markets."""
        mi = mutual_information_market(0.5, 0.5, correlation=1.0)
        assert mi >= 0

    def test_mutual_info_independent_markets(self):
        """Test mutual information for independent markets."""
        mi = mutual_information_market(0.5, 0.5, correlation=0.0)
        assert mi >= 0

    def test_mutual_info_zero_omega(self):
        """Test mutual information with zero omega."""
        mi = mutual_information_market(0.0, 0.5)
        assert mi >= 0

    def test_mutual_info_nonnegativity(self):
        """Test mutual information is always non-negative."""
        omegas1 = np.linspace(0.0, 1.0, 20)
        omegas2 = np.linspace(0.0, 1.0, 20)
        
        for o1 in omegas1:
            for o2 in omegas2:
                mi = mutual_information_market(o1, o2)
                assert mi >= 0

    def test_mutual_info_correlation_scaling(self):
        """Test mutual information scales with correlation."""
        mi_low = mutual_information_market(0.5, 0.5, correlation=0.5)
        mi_high = mutual_information_market(0.5, 0.5, correlation=1.0)
        
        assert mi_high >= mi_low

    def test_mutual_info_invalid_correlation(self):
        """Test mutual information fails with invalid correlation."""
        with pytest.raises(ValueError, match="Correlation"):
            mutual_information_market(0.5, 0.5, correlation=-0.1)
        
        with pytest.raises(ValueError, match="Correlation"):
            mutual_information_market(0.5, 0.5, correlation=1.5)


class TestLyapunovExponent:
    """Test Lyapunov exponent calculations."""

    def test_lyapunov_zero_omega(self):
        """Test Lyapunov exponent at omega = 0."""
        lam = lyapunov_exponent_economic_system(0.0)
        assert lam == pytest.approx(-1.0)

    def test_lyapunov_half_omega(self):
        """Test Lyapunov exponent at omega = 0.5."""
        lam = lyapunov_exponent_economic_system(0.5)
        expected = -1.0 / (1.0 - 0.5)
        assert lam == pytest.approx(expected)

    def test_lyapunov_negative(self):
        """Test Lyapunov exponent is negative."""
        omegas = np.linspace(0.0, 0.9, 50)
        for omega in omegas:
            lam = lyapunov_exponent_economic_system(omega)
            assert lam < 0

    def test_lyapunov_decreasing_with_omega(self):
        """Test Lyapunov exponent becomes more negative with omega."""
        lam1 = lyapunov_exponent_economic_system(0.5)
        lam2 = lyapunov_exponent_economic_system(0.8)
        
        assert lam2 < lam1  # More negative

    def test_lyapunov_near_one(self):
        """Test Lyapunov exponent near omega = 1."""
        lam = lyapunov_exponent_economic_system(0.99)
        assert lam < -10  # Very negative


class TestLyapunovStabilityMargin:
    """Test Lyapunov stability margin."""

    def test_margin_at_zero(self):
        """Test stability margin at omega = 0."""
        margin = lyapunov_stability_margin(0.0)
        expected = np.exp(-1.0)
        assert margin == pytest.approx(expected)

    def test_margin_range(self):
        """Test stability margin is in (0, 1)."""
        omegas = np.linspace(0.0, 0.9, 50)
        for omega in omegas:
            margin = lyapunov_stability_margin(omega)
            assert 0 < margin <= 1.0

    def test_margin_decreasing_with_omega(self):
        """Test stability margin decreases with omega."""
        margins = [lyapunov_stability_margin(omega) 
                  for omega in np.linspace(0.0, 0.9, 20)]
        
        for i in range(len(margins) - 1):
            assert margins[i] >= margins[i + 1]

    def test_margin_stable_systems(self):
        """Test margin for stable systems."""
        # Low omega should have margin > some threshold
        margin = lyapunov_stability_margin(0.2)
        assert margin > 0.3


class TestSupplyChainOptimization:
    """Test supply chain routing optimization."""

    def test_optimization_simple_case(self):
        """Test optimization for simple supply-demand case."""
        supply = np.array([[100.0, 150.0],
                          [80.0, 100.0]])  # 2 suppliers, 2 time periods
        demand = np.array([[60.0, 70.0]])   # 1 location, 2 time periods
        costs = np.array([[1.0, 1.0],
                         [2.0, 2.0]])
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        assert hasattr(result, 'total_cost')
        assert hasattr(result, 'allocation')
        assert hasattr(result, 'supply_levels')
        assert hasattr(result, 'demand_satisfaction')
        assert hasattr(result, 'convergence')

    def test_optimization_results_nonnegative(self):
        """Test optimization results are non-negative."""
        supply = np.array([[100.0, 150.0], [80.0, 100.0]])
        demand = np.array([[60.0, 70.0]])
        costs = np.array([[1.0, 1.0], [2.0, 2.0]])
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        assert result.total_cost >= 0
        assert np.all(result.allocation >= 0)
        assert np.all(result.supply_levels >= 0)

    def test_optimization_allocation_shape(self):
        """Test optimization allocation has correct shape."""
        S, T = 3, 4
        supply = np.random.rand(S, T) * 100 + 50
        demand = np.random.rand(2, T) * 30
        costs = np.random.rand(S, T) * 5
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        assert result.allocation.shape == (S, T)

    def test_optimization_supply_constraint(self):
        """Test optimization respects supply constraints."""
        supply = np.array([[50.0, 50.0], [30.0, 30.0]])
        demand = np.array([[40.0, 40.0]])
        costs = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        # Sum of allocation should not exceed supply
        for i in range(supply.shape[0]):
            assert np.sum(result.allocation[i]) <= np.sum(supply[i]) + 1e-3

    def test_optimization_larger_problem(self):
        """Test optimization on larger problem."""
        S, T = 5, 10
        supply = np.random.rand(S, T) * 100 + 50
        demand = np.random.rand(3, T) * 50
        costs = np.random.rand(S, T) * 10
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        assert result.total_cost >= 0
        assert result.allocation.shape == (S, T)
        assert len(result.demand_satisfaction) == T

    def test_optimization_minimal_cost(self):
        """Test optimization finds minimal cost solution."""
        # Simple case with obvious optimal solution
        supply = np.array([[100.0]])  # 1 supplier
        demand = np.array([[50.0]])   # 1 location
        costs = np.array([[1.0]])     # Simple cost
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        # Cost should be approximately demand * cost
        expected_cost = 50.0 * 1.0
        assert result.total_cost == pytest.approx(expected_cost, rel=0.01)

    @pytest.mark.parametrize("S,T", [(2, 2), (3, 3), (4, 5)])
    def test_optimization_various_sizes(self, S, T):
        """Test optimization with various problem sizes."""
        supply = np.random.rand(S, T) * 100 + 50
        demand = np.random.rand(2, T) * 50
        costs = np.random.rand(S, T) * 5
        
        result = optimize_supply_chain_routing(supply, demand, costs)
        
        assert result.allocation.shape == (S, T)
        assert np.all(result.allocation >= 0)
