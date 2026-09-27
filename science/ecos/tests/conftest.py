"""
Pytest configuration and fixtures for economic_analysis test suite.

This module provides:
- Shared fixtures for test data
- Custom markers
- Test configuration
- Logging setup
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: Edge case tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "parametrize: Parametrized tests"
    )


# ============================================================================
# FIXTURES FOR TEST DATA
# ============================================================================

@pytest.fixture
def valid_omega_values():
    """Fixture providing valid omega values."""
    return [0.0, 0.25, 0.5, 0.618, 0.707, 0.796, 0.75, 0.9, 1.0]


@pytest.fixture
def resonance_window_omegas():
    """Fixture providing omegas in and around resonance window."""
    from ecos.economic_analysis import RESONANCE_WINDOW_MIN, RESONANCE_WINDOW_MAX, RESONANCE_WINDOW_CENTER
    
    return {
        'below': 0.5,
        'min': RESONANCE_WINDOW_MIN,
        'center': RESONANCE_WINDOW_CENTER,
        'max': RESONANCE_WINDOW_MAX,
        'above': 0.9,
    }


@pytest.fixture
def filter_parameters():
    """Fixture providing filter test parameters."""
    return {
        'tau_values': [0.1, 0.5, 1.0, 2.0, 5.0],
        'frequencies': [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
        'shock_magnitudes': [0.1, 1.0, 10.0],
        'shock_periods': [0.1, 1.0, 10.0, 100.0],
    }


@pytest.fixture
def supply_chain_data():
    """Fixture providing sample supply chain data."""
    S, T = 3, 4  # 3 suppliers, 4 time periods
    L = 2        # 2 demand locations
    
    supply = np.random.rand(S, T) * 100 + 50
    demand = np.random.rand(L, T) * 50
    costs = np.random.rand(S, T) * 10
    
    return {
        'supply': supply,
        'demand': demand,
        'costs': costs,
        'S': S,
        'T': T,
        'L': L,
    }


@pytest.fixture
def harmonic_parameters():
    """Fixture providing harmonic analysis parameters."""
    return {
        'amplitudes': [10.0, 5.0, 2.0],
        'frequencies': [1.0, 1.5, 2.0],
        'phases': [0.0, np.pi/4, np.pi/2],
        'base_supply': 100.0,
        'times': np.linspace(0, 2*np.pi, 50),
    }


@pytest.fixture
def time_array():
    """Fixture providing time array for dynamics."""
    return np.linspace(0, 10, 100)


# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================

@pytest.fixture
def assert_close():
    """Fixture providing custom assertion for floating point comparison."""
    def _assert_close(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        """Assert two values are close."""
        if abs(expected) < abs_tol:
            assert abs(actual - expected) < abs_tol
        else:
            assert abs((actual - expected) / expected) < rel_tol
    
    return _assert_close


@pytest.fixture
def assert_monotonic():
    """Fixture providing assertion for monotonicity."""
    def _assert_monotonic(values, increasing=True):
        """Assert values are monotonic."""
        for i in range(len(values) - 1):
            if increasing:
                assert values[i] <= values[i + 1], \
                    f"Not increasing at {i}: {values[i]} > {values[i+1]}"
            else:
                assert values[i] >= values[i + 1], \
                    f"Not decreasing at {i}: {values[i]} < {values[i+1]}"
    
    return _assert_monotonic


@pytest.fixture
def assert_in_range():
    """Fixture providing range assertion."""
    def _assert_in_range(value, min_val, max_val, name="value"):
        """Assert value is in range."""
        assert min_val <= value <= max_val, \
            f"{name} = {value} not in [{min_val}, {max_val}]"
    
    return _assert_in_range


# ============================================================================
# SESSION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


# ============================================================================
# PARAMETRIZATION HELPERS
# ============================================================================

def pytest_generate_tests(metafunc):
    """Generate parametrized tests dynamically."""
    
    # Example: Add custom parametrization if needed
    if "omega_param" in metafunc.fixturenames:
        metafunc.parametrize("omega_param", [0.0, 0.5, 0.707, 0.9])


# ============================================================================
# LOGGING
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(42)
    yield
    # Cleanup after test if needed


# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """Fixture for timing test execution."""
    import time
    
    class Timer:
        def __init__(self):
            self.start = None
            self.end = None
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
        
        @property
        def elapsed(self):
            return self.end - self.start
    
    return Timer


# ============================================================================
# CUSTOM MARKERS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on module/name patterns."""
    for item in items:
        # Mark tests by module
        if "test_core" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_filter" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_harmonic" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_advanced" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark edge case tests
        if "edge" in item.nodeid or "boundary" in item.nodeid:
            item.add_marker(pytest.mark.edge_case)
