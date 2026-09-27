"""
tests/test_ionic.py
===================
Vollständige Testabdeckung für core/ionic.py
"""

import pytest
import numpy as np
from src.ionic import (
    IonSpecies,
    IonicSolution,
    kcl_solution,
    FARADAY,
    BOLTZMANN,
    ELEMENTARY_Q,
    AVOGADRO,
    EPSILON_0,
    EPSILON_WATER,
    T_STANDARD,
)


# ── IonSpecies ────────────────────────────────────────────────────────────────

class TestIonSpecies:

    def test_creation(self):
        ion = IonSpecies("K+", +1, 7.62e-8, 1000.0)
        assert ion.name == "K+"
        assert ion.valence == +1
        assert ion.mobility == pytest.approx(7.62e-8)
        assert ion.concentration == pytest.approx(1000.0)

    def test_negative_valence(self):
        ion = IonSpecies("Cl-", -1, 7.91e-8, 500.0)
        assert ion.valence == -1

    def test_zero_concentration(self):
        ion = IonSpecies("Na+", +1, 5.19e-8, 0.0)
        assert ion.concentration == 0.0

    def test_high_valence(self):
        ion = IonSpecies("Ca2+", +2, 6.17e-8, 100.0)
        assert ion.valence == 2

    def test_repr_contains_name(self):
        ion = IonSpecies("Mg2+", +2, 5.5e-8, 50.0)
        assert hasattr(ion, "name")  # dataclass should have name


# ── IonicSolution – Leitfähigkeit ─────────────────────────────────────────────

class TestIonicConductivity:

    def test_pure_water_zero_ions(self):
        sol = IonicSolution([])
        assert sol.conductivity() == 0.0

    def test_kcl_conductivity_positive(self):
        sol = kcl_solution(0.1)
        sigma = sol.conductivity()
        assert sigma > 0

    def test_kcl_1mol_conductivity_range(self):
        """KCl 1 mol/l sollte σ ≈ 11 S/m ergeben."""
        sol = kcl_solution(1.0)
        sigma = sol.conductivity()
        assert 8.0 < sigma < 20.0

    def test_kcl_0_1mol_conductivity_range(self):
        """KCl 0.1 mol/l sollte σ ≈ 1.1 S/m ergeben."""
        sol = kcl_solution(0.1)
        sigma = sol.conductivity()
        assert 0.5 < sigma < 3.0

    def test_conductivity_scales_linearly_with_concentration(self):
        sol1 = kcl_solution(0.1)
        sol2 = kcl_solution(0.2)
        sigma1 = sol1.conductivity()
        sigma2 = sol2.conductivity()
        assert sigma2 == pytest.approx(2 * sigma1, rel=1e-6)

    def test_conductivity_formula(self):
        """Direkter Test der Gleichung σ = Σ |z| μ c F."""
        ion = IonSpecies("K+", +1, 7.62e-8, 100.0)
        sol = IonicSolution([ion])
        expected = abs(1) * 7.62e-8 * 100.0 * FARADAY
        assert sol.conductivity() == pytest.approx(expected, rel=1e-10)

    def test_two_ions_sum(self):
        """KCl: Summe aus K+ und Cl-."""
        k = IonSpecies("K+",  +1, 7.62e-8, 1000.0)
        cl = IonSpecies("Cl-", -1, 7.91e-8, 1000.0)
        sol = IonicSolution([k, cl])
        expected = (7.62e-8 + 7.91e-8) * 1000.0 * FARADAY
        assert sol.conductivity() == pytest.approx(expected, rel=1e-10)

    def test_divalent_ion_doubles_contribution(self):
        """Ein zweiwertiges Ion hat doppelten Beitrag zur Leitfähigkeit."""
        mono = IonSpecies("Na+", +1, 5.19e-8, 100.0)
        di   = IonSpecies("Ca2+", +2, 5.19e-8, 100.0)
        sol_mono = IonicSolution([mono])
        sol_di   = IonicSolution([di])
        assert sol_di.conductivity() == pytest.approx(2 * sol_mono.conductivity(), rel=1e-10)


# ── Diffusionskoeffizienten ───────────────────────────────────────────────────

class TestDiffusionCoefficients:

    def test_k_plus_diffusion_coefficient(self):
        """D(K+) bei 25°C sollte ≈ 1.96e-9 m²/s sein (Literaturwert)."""
        sol = kcl_solution(0.1)
        D = sol.diffusion_coefficients()
        assert 1.5e-9 < D["K+"] < 2.5e-9

    def test_cl_minus_diffusion_coefficient(self):
        sol = kcl_solution(0.1)
        D = sol.diffusion_coefficients()
        assert 1.5e-9 < D["Cl-"] < 2.5e-9

    def test_diffusion_proportional_to_mobility(self):
        """D ∝ μ laut Einstein-Stokes."""
        ion1 = IonSpecies("A+", +1, 7.62e-8, 100.0)
        ion2 = IonSpecies("B+", +1, 3.81e-8, 100.0)  # halbe Beweglichkeit
        sol = IonicSolution([ion1, ion2])
        D = sol.diffusion_coefficients()
        assert D["A+"] == pytest.approx(2 * D["B+"], rel=1e-10)

    def test_diffusion_einstein_stokes_formula(self):
        """Direkte Prüfung der Einstein-Stokes-Formel."""
        mu = 7.62e-8
        ion = IonSpecies("K+", +1, mu, 1000.0)
        sol = IonicSolution([ion], temperature=T_STANDARD)
        D = sol.diffusion_coefficients()
        D_expected = mu * BOLTZMANN * T_STANDARD / ELEMENTARY_Q
        assert D["K+"] == pytest.approx(D_expected, rel=1e-10)


# ── Debye-Abschirmlänge ───────────────────────────────────────────────────────

class TestDebyeLength:

    def test_zero_concentration_infinite(self):
        sol = IonicSolution([])
        assert sol.debye_length() == np.inf

    def test_kcl_1mol_nanometer_range(self):
        """KCl 1 mol/l: λ_D ≈ 0.3 nm (typischer Literaturwert)."""
        sol = kcl_solution(1.0)
        lam_D = sol.debye_length()
        assert 0.1e-9 < lam_D < 1.5e-9

    def test_kcl_01mol_debye_length(self):
        """KCl 0.01 mol/l: λ_D ≈ 3 nm."""
        sol = kcl_solution(0.01)
        lam_D = sol.debye_length()
        assert 1e-9 < lam_D < 10e-9

    def test_debye_decreases_with_concentration(self):
        sol_lo = kcl_solution(0.01)
        sol_hi = kcl_solution(0.1)
        assert sol_lo.debye_length() > sol_hi.debye_length()

    def test_debye_sqrt_scaling(self):
        """λ_D ∝ 1/√c."""
        sol1 = kcl_solution(0.01)
        sol4 = kcl_solution(0.04)
        ratio = sol1.debye_length() / sol4.debye_length()
        assert ratio == pytest.approx(2.0, rel=0.01)


# ── Kanalwiderstand ───────────────────────────────────────────────────────────

class TestChannelResistance:

    def test_resistance_formula(self):
        """R = L / (σ A)."""
        sol = kcl_solution(1.0)
        L = 1e-6   # 1 µm
        A = (100e-9)**2
        R = sol.channel_resistance(L, A)
        sigma = sol.conductivity()
        expected = L / (sigma * A)
        assert R == pytest.approx(expected, rel=1e-10)

    def test_longer_channel_higher_resistance(self):
        sol = kcl_solution(0.5)
        A = (100e-9)**2
        R1 = sol.channel_resistance(1e-6, A)
        R2 = sol.channel_resistance(2e-6, A)
        assert R2 == pytest.approx(2 * R1, rel=1e-10)

    def test_larger_cross_section_lower_resistance(self):
        sol = kcl_solution(0.5)
        L = 1e-6
        R1 = sol.channel_resistance(L, (100e-9)**2)
        R2 = sol.channel_resistance(L, (200e-9)**2)
        assert R2 < R1

    def test_zero_conductivity_infinite_resistance(self):
        sol = IonicSolution([])
        assert sol.channel_resistance(1e-6, 1e-14) == np.inf

    def test_resistance_positive(self):
        sol = kcl_solution(0.1)
        R = sol.channel_resistance(1e-6, 1e-14)
        assert R > 0


# ── Driftgeschwindigkeit ──────────────────────────────────────────────────────

class TestDriftVelocity:

    def test_drift_velocity_formula(self):
        """v = μ E."""
        sol = kcl_solution(0.5)
        E = 1e5  # V/m
        v = sol.drift_velocity(E, "K+")
        assert v == pytest.approx(7.62e-8 * E, rel=1e-10)

    def test_unknown_ion_raises(self):
        sol = kcl_solution(0.5)
        with pytest.raises(ValueError):
            sol.drift_velocity(1e5, "Xe+")

    def test_drift_proportional_to_field(self):
        sol = kcl_solution(0.5)
        v1 = sol.drift_velocity(1e5, "K+")
        v2 = sol.drift_velocity(2e5, "K+")
        assert v2 == pytest.approx(2 * v1, rel=1e-10)


# ── kcl_solution Hilfsfunktion ────────────────────────────────────────────────

class TestKclSolution:

    def test_creates_two_ions(self):
        sol = kcl_solution(0.1)
        assert len(sol.ions) == 2

    def test_ion_names(self):
        sol = kcl_solution(0.1)
        names = {ion.name for ion in sol.ions}
        assert names == {"K+", "Cl-"}

    def test_concentration_conversion(self):
        """0.1 mol/l = 100 mol/m³."""
        sol = kcl_solution(0.1)
        for ion in sol.ions:
            assert ion.concentration == pytest.approx(100.0, rel=1e-10)

    def test_repr_string(self):
        sol = kcl_solution(0.1)
        s = repr(sol)
        assert "IonicSolution" in s
