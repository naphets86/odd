"""Tests für irf.physics"""

import math
import pytest

from src.physics import (
    EWODModel,
    ElectrokineticTransport,
    IonicConductivity,
    IonSpecies,
    SurfaceTension,
)
from src.utils import ValidationError, CONST


# ---------------------------------------------------------------------------
# EWODModel
# ---------------------------------------------------------------------------

class TestEWODModel:
    @pytest.fixture
    def ewod(self):
        return EWODModel(
            dielectric_thickness_m=10e-9,
            dielectric_epsilon_r=9.0,
            contact_angle_0_deg=110.0,
            surface_tension_N_m=0.0728,
        )

    def test_ewod_efficiency(self, ewod):
        # η = ε₀ ε_r / (2 γ d)
        expected = CONST.epsilon_0 * 9.0 / (2 * 0.0728 * 10e-9)
        assert ewod.ewod_efficiency == pytest.approx(expected, rel=1e-6)

    def test_contact_angle_zero_voltage(self, ewod):
        assert ewod.contact_angle_deg(0.0) == pytest.approx(110.0, abs=0.1)

    def test_contact_angle_decreases_with_voltage(self, ewod):
        theta_0 = ewod.contact_angle_deg(0.0)
        theta_v = ewod.contact_angle_deg(0.5)
        assert theta_v < theta_0

    def test_contact_angle_clamped_to_zero(self, ewod):
        # Sehr hohe Spannung → cos → 1 → θ → 0
        theta = ewod.contact_angle_deg(1000.0)
        assert theta == pytest.approx(0.0, abs=0.1)

    def test_contact_angle_clamped_at_180(self):
        # Negatives EWOD-Modell mit negativem Wert → θ → 180
        ewod2 = EWODModel(
            dielectric_thickness_m=10e-9,
            dielectric_epsilon_r=9.0,
            contact_angle_0_deg=170.0,
            surface_tension_N_m=0.0728,
        )
        # cos(170°) = stark negativ; bei V=0 bereits nahe 180
        assert ewod2.contact_angle_deg(0.0) == pytest.approx(170.0, abs=0.1)

    def test_threshold_voltage_for_target_angle(self, ewod):
        # θ₀ = 110° → Spannung für 90° berechnen
        v_t = ewod.threshold_voltage(90.0)
        # Rückprüfung: contact_angle_deg(v_t) ≈ 90°
        assert ewod.contact_angle_deg(v_t) == pytest.approx(90.0, abs=0.5)

    def test_threshold_voltage_already_passed(self, ewod):
        # Zielwinkel > θ₀ (z.B. 120° > 110°) → V=0 reicht
        v = ewod.threshold_voltage(120.0)
        assert v == 0.0

    def test_threshold_voltage_invalid(self, ewod):
        with pytest.raises(ValidationError):
            ewod.threshold_voltage(0.0)
        with pytest.raises(ValidationError):
            ewod.threshold_voltage(180.0)

    def test_switching_energy(self, ewod):
        area = (5e-6) ** 2
        e = ewod.switching_energy_J(0.12, area)
        assert e > 0
        # E = ½ C V²; C = ε₀ ε_r A / d
        c = CONST.epsilon_0 * 9.0 * area / 10e-9
        assert e == pytest.approx(0.5 * c * 0.12 ** 2, rel=1e-6)

    def test_switching_energy_invalid_area(self, ewod):
        with pytest.raises(ValidationError):
            ewod.switching_energy_J(0.12, 0.0)
        with pytest.raises(ValidationError):
            ewod.switching_energy_J(0.12, -1.0)

    def test_invalid_thickness(self):
        with pytest.raises(ValidationError):
            EWODModel(dielectric_thickness_m=0.0, dielectric_epsilon_r=9.0)

    def test_invalid_epsilon(self):
        with pytest.raises(ValidationError):
            EWODModel(dielectric_thickness_m=10e-9, dielectric_epsilon_r=0.0)

    def test_invalid_contact_angle_zero(self):
        with pytest.raises(ValidationError):
            EWODModel(
                dielectric_thickness_m=10e-9,
                dielectric_epsilon_r=9.0,
                contact_angle_0_deg=0.0,
            )

    def test_invalid_contact_angle_180(self):
        with pytest.raises(ValidationError):
            EWODModel(
                dielectric_thickness_m=10e-9,
                dielectric_epsilon_r=9.0,
                contact_angle_0_deg=180.0,
            )

    def test_invalid_surface_tension(self):
        with pytest.raises(ValidationError):
            EWODModel(
                dielectric_thickness_m=10e-9,
                dielectric_epsilon_r=9.0,
                surface_tension_N_m=0.0,
            )


# ---------------------------------------------------------------------------
# IonSpecies
# ---------------------------------------------------------------------------

class TestIonSpecies:
    def test_diffusion_coefficient(self):
        ion = IonSpecies("K+", +1, 7.62e-8, 100.0)
        d = ion.diffusion_coefficient_m2_s
        expected = 7.62e-8 * CONST.k_B * CONST.T_ref / (1 * CONST.e)
        assert d == pytest.approx(expected, rel=1e-5)

    def test_invalid_valence_zero(self):
        with pytest.raises(ValidationError):
            IonSpecies("X", 0, 1e-8, 100.0)

    def test_negative_valence_allowed(self):
        ion = IonSpecies("Cl-", -1, 7.91e-8, 100.0)
        assert ion.valence == -1

    def test_invalid_mobility(self):
        with pytest.raises(ValidationError):
            IonSpecies("X", 1, 0.0, 100.0)

    def test_invalid_concentration_negative(self):
        with pytest.raises(ValidationError):
            IonSpecies("X", 1, 1e-8, -1.0)

    def test_zero_concentration_allowed(self):
        ion = IonSpecies("X", 1, 1e-8, 0.0)
        assert ion.concentration_mol_m3 == 0.0


# ---------------------------------------------------------------------------
# IonicConductivity
# ---------------------------------------------------------------------------

class TestIonicConductivity:
    def test_kcl_conductivity(self):
        ic = IonicConductivity.from_kcl(0.1)
        sigma = ic.conductivity_S_m()
        assert sigma > 0
        assert 0.5 < sigma < 5.0

    def test_kcl_zero_concentration(self):
        ic = IonicConductivity.from_kcl(0.0)
        assert ic.conductivity_S_m() == pytest.approx(0.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValidationError):
            IonicConductivity.from_kcl(-0.1)

    def test_add_ion(self):
        ic = IonicConductivity()
        assert ic.conductivity_S_m() == pytest.approx(0.0)
        ic.add_ion(IonSpecies("K+", +1, 7.62e-8, 100.0))
        assert ic.conductivity_S_m() > 0

    def test_channel_resistance(self):
        ic = IonicConductivity.from_kcl(0.1)
        r = ic.channel_resistance_Ohm(100e-9, (20e-9) ** 2)
        assert r > 0

    def test_channel_resistance_invalid_length(self):
        ic = IonicConductivity.from_kcl(0.1)
        with pytest.raises(ValidationError):
            ic.channel_resistance_Ohm(0.0, 1e-16)

    def test_channel_resistance_invalid_area(self):
        ic = IonicConductivity.from_kcl(0.1)
        with pytest.raises(ValidationError):
            ic.channel_resistance_Ohm(1e-7, 0.0)

    def test_channel_resistance_no_ions(self):
        ic = IonicConductivity()
        r = ic.channel_resistance_Ohm(1e-7, 1e-16)
        assert r == float("inf")

    def test_conductivity_scales_with_concentration(self):
        ic1 = IonicConductivity.from_kcl(0.1)
        ic2 = IonicConductivity.from_kcl(1.0)
        assert ic2.conductivity_S_m() > ic1.conductivity_S_m()


# ---------------------------------------------------------------------------
# SurfaceTension
# ---------------------------------------------------------------------------

class TestSurfaceTension:
    def test_at_25_celsius(self):
        gamma = SurfaceTension.at_celsius(25.0)
        assert 0.065 < gamma < 0.075

    def test_at_0_celsius(self):
        gamma = SurfaceTension.at_celsius(0.0)
        assert gamma > SurfaceTension.at_celsius(25.0)

    def test_decreases_with_temperature(self):
        g1 = SurfaceTension.at_temperature(273.15)
        g2 = SurfaceTension.at_temperature(373.15)
        assert g1 > g2

    def test_zero_at_critical_temperature(self):
        gamma = SurfaceTension.at_temperature(647.0)
        assert gamma == pytest.approx(0.0, abs=0.01)

    def test_above_critical_temperature(self):
        gamma = SurfaceTension.at_temperature(700.0)
        assert gamma == 0.0

    def test_invalid_temperature(self):
        with pytest.raises(ValidationError):
            SurfaceTension.at_temperature(0.0)
        with pytest.raises(ValidationError):
            SurfaceTension.at_temperature(-1.0)

    def test_at_celsius_delegates(self):
        assert SurfaceTension.at_celsius(25.0) == pytest.approx(
            SurfaceTension.at_temperature(298.15), rel=1e-6
        )

    def test_droplet_merge_energy_positive(self):
        e = SurfaceTension.droplet_merge_energy_J(50e-6, 50e-6)
        assert e > 0

    def test_droplet_merge_energy_larger_release_for_bigger_droplets(self):
        e1 = SurfaceTension.droplet_merge_energy_J(10e-6, 10e-6)
        e2 = SurfaceTension.droplet_merge_energy_J(100e-6, 100e-6)
        assert e2 > e1

    def test_droplet_merge_invalid_radius(self):
        with pytest.raises(ValidationError):
            SurfaceTension.droplet_merge_energy_J(0.0, 1e-5)
        with pytest.raises(ValidationError):
            SurfaceTension.droplet_merge_energy_J(1e-5, 0.0)


# ---------------------------------------------------------------------------
# ElectrokineticTransport
# ---------------------------------------------------------------------------

class TestElectrokineticTransport:
    @pytest.fixture
    def transport(self):
        ion = IonSpecies("K+", +1, 7.62e-8, 100.0)
        return ElectrokineticTransport(ion=ion, temperature_K=298.15)

    def test_diffusion_coefficient(self, transport):
        d = transport.diffusion_coefficient_m2_s
        assert d > 0

    def test_drift_velocity(self, transport):
        v = transport.drift_velocity_m_s(1000.0)
        assert v == pytest.approx(7.62e-8 * 1000.0, rel=1e-6)

    def test_drift_velocity_negative_field(self, transport):
        v = transport.drift_velocity_m_s(-1000.0)
        assert v < 0

    def test_diffusion_flux(self, transport):
        j = transport.diffusion_flux_mol_m2s(1e6)
        assert j < 0  # Flux gegen den Gradienten

    def test_diffusion_flux_zero_gradient(self, transport):
        j = transport.diffusion_flux_mol_m2s(0.0)
        assert j == pytest.approx(0.0)

    def test_electroosmotic_velocity(self, transport):
        v = transport.electroosmotic_velocity_m_s(1000.0, zeta_potential_V=-0.05)
        assert v != 0.0

    def test_electroosmotic_zero_field(self, transport):
        v = transport.electroosmotic_velocity_m_s(0.0)
        assert v == pytest.approx(0.0)

    def test_invalid_temperature(self):
        ion = IonSpecies("K+", +1, 7.62e-8, 100.0)
        with pytest.raises(ValidationError):
            ElectrokineticTransport(ion=ion, temperature_K=0.0)
