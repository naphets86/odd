"""Tests for liionp.analysis — improvement factors, range model, EV helpers."""
import math

import pytest

from liionp.analysis import (
    cold_capacity,
    ev_cumulative_km,
    ev_lifetime_years,
    improvement_factor_combined,
    improvement_factor_thermal,
    improvement_factor_total,
    improvement_factor_voltage,
    phi_factor_from_rates,
    preheating_efficiency,
    range_loss_ratio,
    range_vs_cycles,
)
from liionp.constants import D_EOL, T_REF_COLD


# ── Thermal improvement factor ────────────────────────────────────────────────

class TestImprovementFactorThermal:
    def test_unity_at_same_temperature(self):
        phi = improvement_factor_thermal(50_000.0, T_AI=298.0, T_static=298.0)
        assert phi == pytest.approx(1.0)

    def test_greater_than_one_when_ai_cooler(self):
        phi = improvement_factor_thermal(50_000.0, T_AI=300.0, T_static=308.0)
        assert phi > 1.0

    def test_less_than_one_when_ai_hotter(self):
        phi = improvement_factor_thermal(50_000.0, T_AI=310.0, T_static=300.0)
        assert phi < 1.0

    def test_approx_1_683_at_8K_difference(self):
        # Φ_T = exp(50000/8.314 * (1/300 - 1/308)) ≈ 1.683
        phi = improvement_factor_thermal(50_000.0, T_AI=300.0, T_static=308.0)
        assert phi == pytest.approx(1.683, rel=0.01)


# ── Voltage improvement factor ────────────────────────────────────────────────

class TestImprovementFactorVoltage:
    def test_unity_at_zero_reduction(self):
        phi = improvement_factor_voltage(eta_V=0.0, gamma_V_fraction=0.35)
        assert phi == pytest.approx(1.0)

    def test_greater_than_one_for_positive_reduction(self):
        phi = improvement_factor_voltage(eta_V=0.30, gamma_V_fraction=0.35)
        assert phi > 1.0

    def test_approx_1_118(self):
        # Proposition 2: Φ_V = 1/(1-0.35*0.30) ≈ 1.118
        phi = improvement_factor_voltage(eta_V=0.30, gamma_V_fraction=0.35)
        assert phi == pytest.approx(1.118, rel=0.001)


# ── Combined improvement factor ───────────────────────────────────────────────

class TestImprovementFactorCombined:
    def test_greater_than_one(self):
        phi = improvement_factor_combined(phi_T=1.307, phi_V=1.118, gamma_T_fraction=0.65)
        assert phi > 1.0

    def test_approx_1_46(self):
        # Φ = 1.683^0.65 * 1.118^0.35 ≈ 1.458 (uses correct Φ_T from formula)
        phi = improvement_factor_combined(phi_T=1.683, phi_V=1.118, gamma_T_fraction=0.65)
        assert phi == pytest.approx(1.458, rel=0.01)

    def test_pure_thermal(self):
        # gamma_T = 1 → gamma_V = 0 → result = phi_T^1 * phi_V^0 = phi_T
        phi = improvement_factor_combined(phi_T=1.5, phi_V=1.2, gamma_T_fraction=1.0)
        assert phi == pytest.approx(1.5)

    def test_pure_voltage(self):
        phi = improvement_factor_combined(phi_T=1.5, phi_V=1.2, gamma_T_fraction=0.0)
        assert phi == pytest.approx(1.2)


# ── Total improvement factor (with mechanical channel) ────────────────────────

class TestImprovementFactorTotal:
    def test_greater_than_base_phi(self):
        phi_base = 1.47
        phi_total = improvement_factor_total(phi=phi_base, rho_M=0.15, eta_M=0.27)
        assert phi_total > phi_base

    def test_approx_1_62(self):
        # Theorem 3: Φ_ges ≈ 1.62
        phi_total = improvement_factor_total(phi=1.47, rho_M=0.15, eta_M=0.27)
        assert phi_total == pytest.approx(1.62, rel=0.01)

    def test_zero_mechanical_returns_phi(self):
        # rho_M = 0 → no mechanical contribution → Φ_ges = Φ
        phi_total = improvement_factor_total(phi=1.47, rho_M=0.0, eta_M=0.27)
        assert phi_total == pytest.approx(1.47, rel=1e-6)


# ── Phi from rates ────────────────────────────────────────────────────────────

class TestPhiFactorFromRates:
    def test_basic(self):
        phi = phi_factor_from_rates(mean_rate_static=2.0e-4, mean_rate_ai=1.0e-4)
        assert phi == pytest.approx(2.0)

    def test_unity_when_equal(self):
        phi = phi_factor_from_rates(1e-4, 1e-4)
        assert phi == pytest.approx(1.0)


# ── Range vs cycles ───────────────────────────────────────────────────────────

class TestRangeVsCycles:
    def test_full_range_at_zero_cycles(self):
        R = range_vs_cycles(R0=580.0, n=0.0, n_EoL=3241.0)
        assert R == pytest.approx(580.0)

    def test_eol_at_n_eol(self):
        R = range_vs_cycles(R0=580.0, n=3241.0, n_EoL=3241.0, D_EoL=0.20)
        # R(n_EoL) = R0 * (1 - 0.20 * 1) = 0.80 * R0
        assert R == pytest.approx(580.0 * 0.80)

    def test_decreases_monotonically(self):
        R100 = range_vs_cycles(580.0, 100.0, 3241.0)
        R500 = range_vs_cycles(580.0, 500.0, 3241.0)
        assert R500 < R100

    def test_custom_D_EoL(self):
        R = range_vs_cycles(R0=500.0, n=400.0, n_EoL=400.0, D_EoL=0.10)
        assert R == pytest.approx(500.0 * 0.90)


# ── Range loss ratio ──────────────────────────────────────────────────────────

class TestRangeLossRatio:
    def test_inverse_of_phi(self):
        assert range_loss_ratio(1.47) == pytest.approx(1.0 / 1.47)

    def test_less_than_one_for_phi_greater_one(self):
        assert range_loss_ratio(2.0) < 1.0


# ── Cold capacity ─────────────────────────────────────────────────────────────

class TestColdCapacity:
    def test_no_derating_above_reference(self):
        # T_amb > T_ref → delta_T = 0 → Q = Q0
        Q = cold_capacity(Q0=3.0, T_amb=300.0, T_ref=293.0)
        assert Q == pytest.approx(3.0)

    def test_no_derating_at_reference(self):
        Q = cold_capacity(Q0=3.0, T_amb=T_REF_COLD)
        assert Q == pytest.approx(3.0)

    def test_derating_below_reference(self):
        # T_amb = 263 K = -10 °C, T_ref = 293 K (20 °C)
        # ΔT = 30 K → Q = 3.0 * (1 - 0.007 * 30) = 3.0 * 0.79 = 2.37
        Q = cold_capacity(Q0=3.0, T_amb=263.0, T_ref=293.0, kappa_T=0.007)
        assert Q == pytest.approx(3.0 * (1.0 - 0.007 * 30.0))

    def test_range_loss_at_minus10(self):
        # kappa_T * ΔT = 0.007 * (293 - 263) = 0.21 (21 % loss at −10 °C)
        Q = cold_capacity(Q0=1.0, T_amb=263.0, T_ref=T_REF_COLD, kappa_T=0.007)
        loss = 1.0 - Q
        assert loss == pytest.approx(0.007 * (T_REF_COLD - 263.0), rel=0.05)


# ── EV lifetime helpers ───────────────────────────────────────────────────────

class TestEvLifetimeYears:
    def test_basic(self):
        years = ev_lifetime_years(cycles_to_eol=2205.0, cycles_per_year=310.0)
        assert years == pytest.approx(2205.0 / 310.0)

    def test_scales_inversely_with_cycles_per_year(self):
        y1 = ev_lifetime_years(1000.0, 200.0)
        y2 = ev_lifetime_years(1000.0, 400.0)
        assert y1 == pytest.approx(2.0 * y2)


class TestEvCumulativeKm:
    def test_basic(self):
        km = ev_cumulative_km(years=7.1, km_per_year=22_000.0)
        assert km == pytest.approx(7.1 * 22_000.0)

    def test_zero_years(self):
        assert ev_cumulative_km(0.0, 22_000.0) == pytest.approx(0.0)


# ── Preheating efficiency ─────────────────────────────────────────────────────

class TestPreheatingEfficiency:
    def test_paper_value(self):
        # η_heat = 142 km / 17.5 kWh ≈ 8.1 km/kWh
        eta = preheating_efficiency(range_gain=142.0, energy_heat=17.5)
        assert eta == pytest.approx(8.114, rel=0.01)

    def test_proportional(self):
        eta1 = preheating_efficiency(100.0, 10.0)
        eta2 = preheating_efficiency(200.0, 10.0)
        assert eta2 == pytest.approx(2.0 * eta1)
