"""Tests for liionp.degradation — all equations from liionp.tex."""
import math

import numpy as np
import pytest

import liionp.degradation as deg
from liionp.constants import R_GAS


# ── Arrhenius ─────────────────────────────────────────────────────────────────

class TestArrheniusRate:
    def test_basic(self):
        A, Ea, T = 1.0, 50_000.0, 298.0
        expected = math.exp(-Ea / (R_GAS * T))
        assert deg.arrhenius_rate(A, Ea, T) == pytest.approx(expected)

    def test_proportional_to_A(self):
        Ea, T = 50_000.0, 298.0
        r1 = deg.arrhenius_rate(1.0, Ea, T)
        r2 = deg.arrhenius_rate(2.0, Ea, T)
        assert r2 == pytest.approx(2.0 * r1)

    def test_higher_T_higher_rate(self):
        Ea = 50_000.0
        r_low = deg.arrhenius_rate(1.0, Ea, 298.0)
        r_high = deg.arrhenius_rate(1.0, Ea, 318.0)
        assert r_high > r_low


class TestArrheniusRatio:
    def test_unity_at_same_temp(self):
        ratio = deg.arrhenius_ratio(50_000.0, 298.0, 298.0)
        assert ratio == pytest.approx(1.0)

    def test_approx_3_558_at_plus20K(self):
        # k(318)/k(298) = exp(50000/8.314 * (1/298 - 1/318)) ≈ 3.558
        ratio = deg.arrhenius_ratio(50_000.0, 298.0, 318.0)
        assert ratio == pytest.approx(3.558, rel=0.01)

    def test_cooling_reduces_rate(self):
        ratio = deg.arrhenius_ratio(50_000.0, 308.0, 300.0)
        assert ratio < 1.0


# ── SEI growth ────────────────────────────────────────────────────────────────

class TestSeiGrowthRate:
    def test_positive(self):
        rate = deg.sei_growth_rate(1e-17, 50_000.0, 298.0, 1e-9)
        assert rate > 0.0

    def test_larger_sei_slower_rate(self):
        r1 = deg.sei_growth_rate(1e-17, 50_000.0, 298.0, 1e-9)
        r2 = deg.sei_growth_rate(1e-17, 50_000.0, 298.0, 2e-9)
        assert r1 > r2

    def test_higher_T_higher_rate(self):
        r_low = deg.sei_growth_rate(1e-17, 50_000.0, 298.0, 1e-9)
        r_high = deg.sei_growth_rate(1e-17, 50_000.0, 318.0, 1e-9)
        assert r_high > r_low


class TestSeiThickness:
    def test_sqrt_time_scaling(self):
        k, Ea, T = 1e-17, 50_000.0, 298.0
        d1 = deg.sei_thickness(k, 1e6, Ea, T)
        d4 = deg.sei_thickness(k, 4e6, Ea, T)
        assert d4 == pytest.approx(2.0 * d1, rel=1e-10)

    def test_zero_time_gives_zero(self):
        assert deg.sei_thickness(1e-17, 0.0, 50_000.0, 298.0) == pytest.approx(0.0)

    def test_higher_T_thicker(self):
        k, t, Ea = 1e-17, 1e6, 50_000.0
        assert deg.sei_thickness(k, t, Ea, 318.0) > deg.sei_thickness(k, t, Ea, 298.0)


class TestCapacityLossSei:
    def test_proportional_to_alpha(self):
        assert deg.capacity_loss_sei(2.0, 1e-6) == pytest.approx(2e-6)

    def test_proportional_to_delta(self):
        assert deg.capacity_loss_sei(1e-4, 2e-6) == pytest.approx(2e-10)


# ── Capacity drift ────────────────────────────────────────────────────────────

class TestCapacityDrift:
    def test_new_cell(self):
        assert deg.capacity_drift(3.0, 3.0) == pytest.approx(0.0)

    def test_eol_at_80_percent(self):
        assert deg.capacity_drift(2.4, 3.0) == pytest.approx(0.20)

    def test_half_capacity(self):
        assert deg.capacity_drift(1.5, 3.0) == pytest.approx(0.50)


# ── Voltage degradation ───────────────────────────────────────────────────────

class TestOvervoltageDegradationRate:
    def test_no_degradation_at_limit(self):
        assert deg.overvoltage_degradation_rate(4.2, 4.2) == 0.0

    def test_no_degradation_below_limit(self):
        assert deg.overvoltage_degradation_rate(4.0, 4.2) == 0.0

    def test_positive_above_limit(self):
        rate = deg.overvoltage_degradation_rate(4.3, 4.2)
        assert rate > 0.0

    def test_exponential_growth(self):
        r1 = deg.overvoltage_degradation_rate(4.3, 4.2, k_OV=0.01, beta_OV=10.0)
        r2 = deg.overvoltage_degradation_rate(4.4, 4.2, k_OV=0.01, beta_OV=10.0)
        assert r2 > r1

    def test_custom_params(self):
        rate = deg.overvoltage_degradation_rate(4.3, 4.2, k_OV=0.05, beta_OV=5.0)
        expected = 0.05 * (math.exp(5.0 * 0.1) - 1.0)
        assert rate == pytest.approx(expected)


class TestUndervoltageDegradationRate:
    def test_no_degradation_at_limit(self):
        assert deg.undervoltage_degradation_rate(2.5, 2.5) == 0.0

    def test_no_degradation_above_limit(self):
        assert deg.undervoltage_degradation_rate(3.0, 2.5) == 0.0

    def test_positive_below_limit(self):
        rate = deg.undervoltage_degradation_rate(2.3, 2.5)
        assert rate > 0.0

    def test_custom_params(self):
        rate = deg.undervoltage_degradation_rate(2.3, 2.5, k_UV=0.02, beta_UV=8.0)
        expected = 0.02 * (math.exp(8.0 * 0.2) - 1.0)
        assert rate == pytest.approx(expected)


class TestVoltageDegradationRate:
    def test_normal_range_zero(self):
        assert deg.voltage_degradation_rate(3.7, 2.5, 4.2) == pytest.approx(0.0)

    def test_overcharge_positive(self):
        assert deg.voltage_degradation_rate(4.3, 2.5, 4.2) > 0.0

    def test_undercharge_positive(self):
        assert deg.voltage_degradation_rate(2.3, 2.5, 4.2) > 0.0


# ── Cyclic degradation ────────────────────────────────────────────────────────

class TestCyclicDegradationRate:
    def test_positive_for_discharge(self):
        assert deg.cyclic_degradation_rate(1.0, 298.0) > 0.0

    def test_positive_for_charge(self):
        assert deg.cyclic_degradation_rate(-1.0, 298.0) > 0.0

    def test_scales_with_current(self):
        r1 = deg.cyclic_degradation_rate(1.0, 298.0)
        r2 = deg.cyclic_degradation_rate(2.0, 298.0)
        assert r2 == pytest.approx(2.0 * r1)

    def test_higher_T_higher_rate(self):
        r_low = deg.cyclic_degradation_rate(1.0, 298.0)
        r_high = deg.cyclic_degradation_rate(1.0, 318.0)
        assert r_high > r_low


# ── Total degradation rate ────────────────────────────────────────────────────

class TestTotalDegradationRate:
    _kwargs = dict(
        T=298.0, V=3.7, I=1.0,
        V_min=2.5, V_max=4.2,
        k_SEI=1e-17, Ea_SEI=50_000.0,
        gamma_T=0.65, gamma_V=0.35, gamma_C=0.10,
    )

    def test_positive_in_normal_range(self):
        rate = deg.total_degradation_rate(**self._kwargs)
        assert rate > 0.0

    def test_increases_with_temperature(self):
        kwargs_hot = {**self._kwargs, "T": 318.0}
        r_hot = deg.total_degradation_rate(**kwargs_hot)
        r_normal = deg.total_degradation_rate(**self._kwargs)
        assert r_hot > r_normal

    def test_increases_with_overvoltage(self):
        kwargs_ov = {**self._kwargs, "V": 4.3}
        r_ov = deg.total_degradation_rate(**kwargs_ov)
        r_normal = deg.total_degradation_rate(**self._kwargs)
        assert r_ov > r_normal

    def test_custom_voltage_params(self):
        rate = deg.total_degradation_rate(
            **self._kwargs,
            k_OV=0.05, beta_OV=5.0, k_UV=0.05, beta_UV=5.0,
        )
        assert rate > 0.0


# ── Integral / lifetime ───────────────────────────────────────────────────────

class TestCumulativeDegradation:
    def test_constant_rate(self):
        rates = np.full(100, 0.001)
        result = deg.cumulative_degradation(rates, dt=1.0)
        assert result == pytest.approx(0.1)

    def test_zero_rates(self):
        assert deg.cumulative_degradation(np.zeros(50), dt=1.0) == pytest.approx(0.0)

    def test_different_dt(self):
        rates = np.ones(10)
        result = deg.cumulative_degradation(rates, dt=2.0)
        assert result == pytest.approx(20.0)


class TestLifetimeFromRate:
    def test_basic(self):
        # L = 0.20 / 1e-4 = 2000
        assert deg.lifetime_from_rate(0.20, 1e-4) == pytest.approx(2000.0)

    def test_halving_rate_doubles_lifetime(self):
        L1 = deg.lifetime_from_rate(0.20, 1e-4)
        L2 = deg.lifetime_from_rate(0.20, 5e-5)
        assert L2 == pytest.approx(2.0 * L1)
