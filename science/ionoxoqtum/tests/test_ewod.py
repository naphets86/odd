"""
tests/test_ewod.py
==================
Vollständige Testabdeckung für core/ewod.py
"""

import pytest
import numpy as np
from src.ewod import (
    EWODDielectric,
    EWODSystem,
    standard_al2o3_system,
    EPSILON_0,
)


# ── EWODDielectric ────────────────────────────────────────────────────────────

class TestEWODDielectric:

    def test_creation(self):
        diel = EWODDielectric(thickness=10e-9, epsilon_r=9.0)
        assert diel.thickness == pytest.approx(10e-9)
        assert diel.epsilon_r == pytest.approx(9.0)

    def test_capacitance_per_area(self):
        """C/A = ε_0 ε_r / d."""
        diel = EWODDielectric(thickness=10e-9, epsilon_r=9.0)
        expected = EPSILON_0 * 9.0 / 10e-9
        assert diel.capacitance_per_area == pytest.approx(expected, rel=1e-10)

    def test_thinner_dielectric_higher_capacitance(self):
        diel_thin = EWODDielectric(thickness=5e-9, epsilon_r=9.0)
        diel_thick = EWODDielectric(thickness=10e-9, epsilon_r=9.0)
        assert diel_thin.capacitance_per_area > diel_thick.capacitance_per_area

    def test_higher_epsilon_higher_capacitance(self):
        diel_low = EWODDielectric(thickness=10e-9, epsilon_r=4.0)
        diel_hi  = EWODDielectric(thickness=10e-9, epsilon_r=9.0)
        assert diel_hi.capacitance_per_area > diel_low.capacitance_per_area

    def test_sio2_reference(self):
        """SiO₂ bei 10 nm: C/A ≈ 3.45 mF/m²."""
        diel = EWODDielectric(thickness=10e-9, epsilon_r=3.9)
        cap = diel.capacitance_per_area
        assert 2e-3 < cap < 5e-3


# ── EWODSystem – EWOD-Effizienz ───────────────────────────────────────────────

class TestEWODEfficiency:

    def test_al2o3_efficiency_value(self):
        """Al₂O₃ 10nm: η = ε₀ ε_r / (2 γ d) ≈ 0.055 V⁻² (SI-Einheiten)."""
        sys = standard_al2o3_system()
        eta = sys.ewod_efficiency
        # SI: eps0*9/(2*0.072*10e-9) = 7.97e-11/1.44e-9 = 0.0553 V^-2
        assert 0.03 < eta < 0.10

    def test_efficiency_formula(self):
        """η = ε_0 ε_r / (2 γ d)."""
        diel = EWODDielectric(10e-9, 9.0)
        sys = EWODSystem(diel, contact_angle_0=110.0, surface_tension_lg=0.072)
        expected = EPSILON_0 * 9.0 / (2 * 0.072 * 10e-9)
        assert sys.ewod_efficiency == pytest.approx(expected, rel=1e-10)

    def test_thicker_dielectric_lower_efficiency(self):
        diel_thin = EWODDielectric(5e-9, 9.0)
        diel_thick = EWODDielectric(20e-9, 9.0)
        sys_thin  = EWODSystem(diel_thin)
        sys_thick = EWODSystem(diel_thick)
        assert sys_thin.ewod_efficiency > sys_thick.ewod_efficiency


# ── Kontaktwinkel ─────────────────────────────────────────────────────────────

class TestContactAngle:

    def test_zero_voltage_returns_theta0(self):
        sys = standard_al2o3_system(contact_angle_0=110.0)
        theta = sys.contact_angle(0.0)
        assert theta == pytest.approx(110.0, abs=0.01)

    def test_angle_decreases_with_voltage(self):
        sys = standard_al2o3_system()
        theta_0V = sys.contact_angle(0.0)
        theta_1V = sys.contact_angle(1.0)
        assert theta_1V < theta_0V

    def test_angle_never_below_zero(self):
        sys = standard_al2o3_system()
        for V in [0, 0.5, 1.0, 5.0, 100.0]:
            assert sys.contact_angle(V) >= 0.0

    def test_angle_never_above_180(self):
        sys = standard_al2o3_system()
        assert sys.contact_angle(0.0) <= 180.0

    def test_young_lippmann_formula(self):
        """cos θ(V) = cos θ₀ + η V²."""
        diel = EWODDielectric(10e-9, 9.0)
        sys = EWODSystem(diel, contact_angle_0=90.0)
        V = 0.5
        cos_expected = np.cos(np.radians(90.0)) + sys.ewod_efficiency * V**2
        cos_expected = np.clip(cos_expected, -1.0, 1.0)
        theta_expected = np.degrees(np.arccos(cos_expected))
        assert sys.contact_angle(V) == pytest.approx(theta_expected, abs=0.01)

    def test_voltage_sweep_length(self):
        sys = standard_al2o3_system()
        V, theta = sys.voltage_sweep(v_max=2.0, n_points=100)
        assert len(V) == 100
        assert len(theta) == 100

    def test_voltage_sweep_monotone(self):
        """Kontaktwinkel sollte mit steigender Spannung monoton fallen."""
        sys = standard_al2o3_system()
        V, theta = sys.voltage_sweep(v_max=1.0, n_points=50)
        diffs = np.diff(theta)
        # Alle Differenzen ≤ 0 (monoton fallend oder gleich)
        assert np.all(diffs <= 1e-10)


# ── Schaltspannung ────────────────────────────────────────────────────────────

class TestSwitchingVoltage:

    def test_switching_voltage_positive(self):
        sys = standard_al2o3_system(contact_angle_0=110.0)
        V_schalt = sys.switching_voltage(60.0)
        assert V_schalt > 0

    def test_switching_voltage_from_paper(self):
        """V_schalt = sqrt(Δcos θ / η). Mit η=0.055 V⁻² ergibt sich V ≈ 3.9 V."""
        sys = standard_al2o3_system(contact_angle_0=110.0)
        V_schalt = sys.switching_voltage(60.0)
        # delta_cos = cos(60°) - cos(110°) = 0.5 + 0.342 = 0.842
        # V = sqrt(0.842 / 0.0553) ≈ 3.9 V
        assert 1.0 < V_schalt < 10.0

    def test_switching_voltage_nan_if_impossible(self):
        """Wenn Zielwinkel > θ₀ (größer als Anfangswinkel), unmöglich."""
        sys = standard_al2o3_system(contact_angle_0=110.0)
        V = sys.switching_voltage(120.0)  # > 110° nicht erreichbar durch EWOD
        assert np.isnan(V)

    def test_switching_voltage_zero_for_theta0(self):
        """Für θ_ziel = θ₀: V_schalt = 0."""
        sys = standard_al2o3_system(contact_angle_0=110.0)
        V = sys.switching_voltage(110.0)
        assert V == pytest.approx(0.0, abs=1e-8)

    def test_higher_target_angle_needs_less_voltage(self):
        """Kleinere Winkeländerung braucht weniger Spannung."""
        sys = standard_al2o3_system(contact_angle_0=110.0)
        V_large = sys.switching_voltage(50.0)  # größere Änderung
        V_small = sys.switching_voltage(90.0)  # kleinere Änderung
        assert V_large > V_small


# ── Schaltenergie ─────────────────────────────────────────────────────────────

class TestSwitchingEnergy:

    def test_energy_positive(self):
        sys = standard_al2o3_system()
        E = sys.switching_energy(0.123, (100e-9)**2)
        assert E > 0

    def test_energy_formula(self):
        """E = ½ C V²."""
        diel = EWODDielectric(10e-9, 9.0)
        sys = EWODSystem(diel)
        V = 0.123
        A = (100e-9)**2
        C = diel.capacitance_per_area * A
        expected = 0.5 * C * V**2
        assert sys.switching_energy(V, A) == pytest.approx(expected, rel=1e-10)

    def test_energy_in_zeptojoule_range(self):
        """Schaltenergie bei 100nm-Elektrode. E = ½ C V² mit V≈3.9V, C≈8e-21 F."""
        sys = standard_al2o3_system()
        V = sys.switching_voltage(60.0)
        E = sys.switching_energy(V, (100e-9)**2)
        # C = eps0*9/10e-9 * (100e-9)^2 ≈ 7.97e-21 F, V≈3.9V -> E ≈ 6e-20 J
        assert 1e-22 < E < 1e-15

    def test_energy_scales_quadratic_with_voltage(self):
        """E ∝ V²."""
        sys = standard_al2o3_system()
        A = (100e-9)**2
        E1 = sys.switching_energy(0.1, A)
        E2 = sys.switching_energy(0.2, A)
        assert E2 == pytest.approx(4 * E1, rel=1e-10)

    def test_energy_scales_linear_with_area(self):
        """E ∝ A."""
        sys = standard_al2o3_system()
        E1 = sys.switching_energy(0.1, 1e-14)
        E2 = sys.switching_energy(0.1, 2e-14)
        assert E2 == pytest.approx(2 * E1, rel=1e-10)


# ── Standard-System ───────────────────────────────────────────────────────────

class TestStandardSystem:

    def test_standard_al2o3_creates_system(self):
        sys = standard_al2o3_system()
        assert isinstance(sys, EWODSystem)

    def test_standard_al2o3_10nm(self):
        sys = standard_al2o3_system()
        assert sys.dielectric.thickness == pytest.approx(10e-9)

    def test_standard_al2o3_epsilon_9(self):
        sys = standard_al2o3_system()
        assert sys.dielectric.epsilon_r == pytest.approx(9.0)

    def test_repr_string(self):
        sys = standard_al2o3_system()
        s = repr(sys)
        assert "EWODSystem" in s
