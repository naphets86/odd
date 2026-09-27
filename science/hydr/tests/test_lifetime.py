"""Tests für hydr.lifetime – 100 % Abdeckung."""

import math
import pytest

from hydr.lifetime import (
    oil_lifetime,
    activation_energy_pressure,
    bulk_modulus_aging,
    failure_rate_oxidation,
    failure_rate_friction,
    failure_rate_cavitation,
    total_failure_rate,
    K_B_EV,
)


class TestOilLifetime:
    def test_higher_viscosity_stability_gives_longer_lifetime(self):
        base = dict(
            C_hyd=1000.0,
            eta_0=0.046,
            beta=2.0,
            E_a=0.7,
            T=313.15,
        )
        t_stable = oil_lifetime(delta_eta_irrev=0.001, **base)
        t_unstable = oil_lifetime(delta_eta_irrev=0.01, **base)
        assert t_stable > t_unstable

    def test_higher_temperature_reduces_lifetime(self):
        base = dict(
            C_hyd=1000.0,
            delta_eta_irrev=0.001,
            eta_0=0.046,
            beta=2.0,
            E_a=0.7,
        )
        t_cold = oil_lifetime(T=313.15, **base)
        t_hot = oil_lifetime(T=373.15, **base)
        assert t_hot < t_cold

    def test_formula_manual(self):
        C_hyd = 500.0
        delta_eta = 1e-4
        eta_0 = 0.046
        beta = 2.0
        E_a = 0.7
        T = 323.15
        k_B = K_B_EV
        expected = C_hyd * (delta_eta / eta_0) ** (-beta) * math.exp(E_a / (k_B * T))
        assert oil_lifetime(C_hyd, delta_eta, eta_0, beta, E_a, T) == pytest.approx(expected)

    def test_custom_k_B(self):
        """Gleiche Formel auch mit J-Einheiten."""
        C_hyd = 200.0
        delta_eta = 1e-4
        eta_0 = 0.046
        beta = 2.0
        E_a_J = 0.7 * 1.6022e-19  # eV → J
        T = 323.15
        k_B_J = 1.380649e-23
        result = oil_lifetime(C_hyd, delta_eta, eta_0, beta, E_a_J, T, k_B=k_B_J)
        assert result > 0

    def test_zero_delta_eta_raises(self):
        with pytest.raises(ValueError):
            oil_lifetime(1000.0, 0.0, 0.046, 2.0, 0.7, 313.15)

    def test_negative_delta_eta_raises(self):
        with pytest.raises(ValueError):
            oil_lifetime(1000.0, -0.001, 0.046, 2.0, 0.7, 313.15)

    def test_zero_eta_0_raises(self):
        with pytest.raises(ValueError):
            oil_lifetime(1000.0, 0.001, 0.0, 2.0, 0.7, 313.15)

    def test_negative_eta_0_raises(self):
        with pytest.raises(ValueError):
            oil_lifetime(1000.0, 0.001, -0.046, 2.0, 0.7, 313.15)

    def test_zero_T_raises(self):
        with pytest.raises(ValueError):
            oil_lifetime(1000.0, 0.001, 0.046, 2.0, 0.7, 0.0)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            oil_lifetime(1000.0, 0.001, 0.046, 2.0, 0.7, -1.0)


class TestActivationEnergyPressure:
    def test_at_zero_pressure(self):
        """E_a(0) = E_a0."""
        assert activation_energy_pressure(0.7, 0.0) == pytest.approx(0.7)

    def test_at_p_star_returns_zero(self):
        """E_a(P*) = 0."""
        E_a0 = 0.7
        P_star = 1e8
        result = activation_energy_pressure(E_a0, P=P_star, P_star=P_star)
        assert result == pytest.approx(0.0)

    def test_reduces_with_pressure(self):
        E_a0 = 0.7
        E_low = activation_energy_pressure(E_a0, P=100e5)
        E_high = activation_energy_pressure(E_a0, P=500e5)
        assert E_high < E_low

    def test_zero_P_star_raises(self):
        with pytest.raises(ValueError):
            activation_energy_pressure(0.7, 100e5, P_star=0.0)

    def test_500bar_gives_50_percent_reduction(self):
        """Lt. Paper: P=500 bar (5e7 Pa), P*=1000 bar (1e8 Pa) → 50 % Abnahme."""
        E_a0 = 0.7
        result = activation_energy_pressure(E_a0, P=500e5, P_star=1000e5)
        assert result == pytest.approx(E_a0 * 0.5)


class TestBulkModulusAging:
    def test_zero_deterioration(self):
        """Kein Viskositätsanstieg → K_alt = K0."""
        result = bulk_modulus_aging(1.58e9, 0.0, 0.046)
        assert result == pytest.approx(1.58e9)

    def test_100_percent_viscosity_increase_gives_15_percent_reduction(self):
        """Lt. Paper: Δη/η_0 = 1 (100 %) → ~15 % Rückgang."""
        K0 = 1.58e9
        eta_0 = 0.046
        delta_eta = eta_0  # Δη = η_0 → ratio = 1
        result = bulk_modulus_aging(K0, delta_eta, eta_0, delta_K_coeff=0.15)
        reduction = (K0 - result) / K0
        assert reduction == pytest.approx(0.15, rel=1e-6)

    def test_negative_delta_eta_uses_abs(self):
        """Betrag von Δη, da Viskositätsänderung auch negativ sein kann."""
        K0 = 1.0e9
        result = bulk_modulus_aging(K0, delta_eta=-0.01, eta_0=0.046)
        assert result < K0

    def test_zero_eta_0_raises(self):
        with pytest.raises(ValueError):
            bulk_modulus_aging(1.58e9, 0.01, 0.0)

    def test_negative_eta_0_raises(self):
        with pytest.raises(ValueError):
            bulk_modulus_aging(1.58e9, 0.01, -0.046)


class TestFailureRateOxidation:
    def test_at_T_opt_is_A1(self):
        """Bei T = T_opt: exp(0) = 1 → Ṅ_ox = A1."""
        result = failure_rate_oxidation(T=330.0, T_opt=330.0, T0=10.0, A1=2.5)
        assert result == pytest.approx(2.5)

    def test_above_T_opt_increases(self):
        r_at_Topt = failure_rate_oxidation(330.0, 330.0, 10.0)
        r_above = failure_rate_oxidation(350.0, 330.0, 10.0)
        assert r_above > r_at_Topt

    def test_zero_T0_raises(self):
        with pytest.raises(ValueError):
            failure_rate_oxidation(330.0, 330.0, 0.0)


class TestFailureRateFriction:
    def test_at_T_min_is_A2(self):
        """Bei T = T_min: exp(0) = 1 → Ṅ_Reib = A2."""
        result = failure_rate_friction(T=280.0, T_min=280.0, T0=10.0, A2=1.5)
        assert result == pytest.approx(1.5)

    def test_below_T_min_increases(self):
        r_at_Tmin = failure_rate_friction(280.0, 280.0, 10.0)
        r_below = failure_rate_friction(260.0, 280.0, 10.0)
        assert r_below > r_at_Tmin

    def test_zero_T0_raises(self):
        with pytest.raises(ValueError):
            failure_rate_friction(280.0, 280.0, 0.0)


class TestFailureRateCavitation:
    def test_at_sat_pressure_is_A3(self):
        """Bei P_min = P_Dampf: exp(0) = 1 → Ṅ_Kav = A3."""
        result = failure_rate_cavitation(P_min=1000.0, P_vapor=1000.0, P0=5000.0, A3=3.0)
        assert result == pytest.approx(3.0)

    def test_higher_P_min_reduces_cavitation(self):
        r_low = failure_rate_cavitation(1500.0, 1000.0, 5000.0)
        r_high = failure_rate_cavitation(5000.0, 1000.0, 5000.0)
        assert r_high < r_low

    def test_zero_P0_raises(self):
        with pytest.raises(ValueError):
            failure_rate_cavitation(5000.0, 1000.0, 0.0)


class TestTotalFailureRate:
    def test_sum_of_three_mechanisms(self):
        T = 330.0
        T_opt = 330.0
        T_min = 280.0
        T0 = 10.0
        P_min = 5000.0
        P_vapor = 1000.0
        P0 = 5000.0
        total = total_failure_rate(T, T_opt, T_min, T0, P_min, P_vapor, P0)
        ox = failure_rate_oxidation(T, T_opt, T0)
        fric = failure_rate_friction(T, T_min, T0)
        kav = failure_rate_cavitation(P_min, P_vapor, P0)
        assert total == pytest.approx(ox + fric + kav)

    def test_positive_result(self):
        result = total_failure_rate(
            T=330.0, T_opt=330.0, T_min=280.0, T0=10.0,
            P_min=5000.0, P_vapor=1000.0, P0=5000.0,
            A1=1.0, A2=1.0, A3=1.0,
        )
        assert result > 0
