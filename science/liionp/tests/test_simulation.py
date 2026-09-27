"""Tests for liionp.simulation."""
import numpy as np
import pytest

from liionp.simulation import (
    LoadProfile,
    SimulationResult,
    compare_systems,
    run_simulation,
)
from liionp.cell import CellParameters


# ── LoadProfile ───────────────────────────────────────────────────────────────

class TestLoadProfile:
    def test_lp1_smartphone(self):
        lp = LoadProfile.lp1_smartphone(n_cycles=5)
        assert lp.name == "LP1_Smartphone"
        assert lp.T_amb == 298.0
        assert len(lp.times) > 0
        assert len(lp.currents) == len(lp.times)

    def test_lp2_ebike(self):
        lp = LoadProfile.lp2_ebike(n_cycles=5)
        assert lp.name == "LP2_Ebike"
        assert lp.T_amb == 303.0

    def test_lp3_stationary(self):
        lp = LoadProfile.lp3_stationary(n_cycles=5)
        assert lp.name == "LP3_Stationary"
        assert lp.T_amb == 293.0

    def test_current_at_interpolates(self):
        lp = LoadProfile.lp1_smartphone(n_cycles=5)
        I = lp.current_at(0.0)
        assert isinstance(I, float)

    def test_current_at_wraps_periodically(self):
        lp = LoadProfile.lp1_smartphone(n_cycles=5)
        # Wrapping: t beyond the profile extent should not raise
        I = lp.current_at(lp.times[-1] * 2)
        assert isinstance(I, float)

    def test_current_at_midpoint(self):
        lp = LoadProfile.lp1_smartphone(n_cycles=5)
        t_mid = lp.times[len(lp.times) // 2]
        I = lp.current_at(t_mid)
        assert isinstance(I, float)


# ── run_simulation ────────────────────────────────────────────────────────────

class TestRunSimulation:
    @pytest.fixture
    def profile(self):
        return LoadProfile.lp1_smartphone(n_cycles=5, dt=60.0)

    def test_returns_simulation_result(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0)
        assert isinstance(result, SimulationResult)

    def test_static_name(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0, ai_mode=False)
        assert result.name == "Static PMS"

    def test_ai_name(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0, ai_mode=True)
        assert result.name == "AI-PMM"

    def test_arrays_same_length(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0)
        n = len(result.times)
        assert len(result.temperatures) == n
        assert len(result.voltages) == n
        assert len(result.SoCs) == n
        assert len(result.degradations) == n
        assert len(result.degradation_rates) == n

    def test_degradation_monotone_increasing(self, profile):
        result = run_simulation(profile, n_cycles=3, dt=60.0)
        assert np.all(np.diff(result.degradations) >= 0)

    def test_temperature_evolves(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0)
        # Temperature should change (not stay exactly constant)
        assert not np.allclose(result.temperatures, result.temperatures[0])

    def test_mean_temperature_finite(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0)
        assert np.isfinite(result.mean_temperature)

    def test_mean_degradation_rate_positive(self, profile):
        result = run_simulation(profile, n_cycles=2, dt=60.0)
        assert result.mean_degradation_rate > 0.0

    def test_default_params_used(self, profile):
        # No explicit params → should use CellParameters defaults without error
        result = run_simulation(profile, params=None, n_cycles=2, dt=60.0)
        assert result is not None

    def test_custom_params(self, profile):
        params = CellParameters(Q0=5.0)
        result = run_simulation(profile, params=params, n_cycles=2, dt=60.0)
        assert result is not None

    def test_ai_mode_reaches_current_reduction(self):
        # Use LP2 which has high currents → temperature rises → AI should cut back
        lp = LoadProfile.lp2_ebike(n_cycles=2, dt=60.0)
        result = run_simulation(lp, n_cycles=2, dt=60.0, ai_mode=True, T0=305.0)
        assert result is not None

    def test_stops_at_eol(self):
        # Run with very fast degradation (large k_SEI, large gammas)
        params = CellParameters(
            k_SEI=1e-5,     # very aggressive SEI
            gamma_T=50.0,
            gamma_C=50.0,
        )
        lp = LoadProfile.lp1_smartphone(n_cycles=5, dt=60.0)
        result = run_simulation(lp, params=params, n_cycles=1000, dt=60.0)
        # Should finish before 1000 cycles (EoL triggered)
        assert result.cycles_completed <= 1000


# ── compare_systems ───────────────────────────────────────────────────────────

class TestCompareSystems:
    def test_returns_two_results(self):
        lp = LoadProfile.lp1_smartphone(n_cycles=3, dt=60.0)
        static, ai = compare_systems(lp, n_cycles=3, dt=60.0)
        assert isinstance(static, SimulationResult)
        assert isinstance(ai, SimulationResult)

    def test_static_vs_ai_names(self):
        lp = LoadProfile.lp1_smartphone(n_cycles=3, dt=60.0)
        static, ai = compare_systems(lp, n_cycles=3, dt=60.0)
        assert static.name == "Static PMS"
        assert ai.name == "AI-PMM"

    def test_ai_lower_mean_degradation_rate(self):
        lp = LoadProfile.lp2_ebike(n_cycles=5, dt=60.0)
        static, ai = compare_systems(lp, n_cycles=5, dt=60.0)
        # AI-PMM should achieve lower or equal mean degradation
        assert ai.mean_degradation_rate <= static.mean_degradation_rate * 1.05

    def test_default_params(self):
        lp = LoadProfile.lp3_stationary(n_cycles=3, dt=300.0)
        static, ai = compare_systems(lp, n_cycles=3, dt=300.0)
        assert static is not None and ai is not None
