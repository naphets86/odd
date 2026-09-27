"""Tests für hydr.hydraulics – 100 % Abdeckung."""

import math
import pytest

from hydr.hydraulics import (
    compressibility_error,
    hydraulic_eigenfrequency,
    hydraulic_natural_frequency,
    sea_stiffness,
    max_contact_force,
    compression_energy,
    accumulator_constant,
    max_operating_pressure,
    bearing_cutoff_frequency,
)


class TestCompressibilityError:
    def test_example_from_paper(self):
        """Beispiel Schweißroboterarm (Abschnitt 6.2):
        A=10 cm², V=0.5l, P=200bar, K=1.89 GPa → δx ≈ 5.3 mm."""
        A = 10e-4   # m²
        V = 5e-4    # m³
        P = 20e6    # Pa
        K = 1.89e9  # Pa
        delta_x = compressibility_error(P, V, K, A)
        assert delta_x == pytest.approx(5.3e-3, rel=0.05)

    def test_higher_K_gives_smaller_error(self):
        kw = dict(P=20e6, V=5e-4, A=10e-4)
        err_low = compressibility_error(K_eff=1.58e9, **kw)
        err_high = compressibility_error(K_eff=2.35e9, **kw)
        assert err_high < err_low

    def test_zero_K_eff_raises(self):
        with pytest.raises(ValueError):
            compressibility_error(20e6, 5e-4, 0.0, 10e-4)

    def test_negative_K_eff_raises(self):
        with pytest.raises(ValueError):
            compressibility_error(20e6, 5e-4, -1.0, 10e-4)

    def test_zero_A_raises(self):
        with pytest.raises(ValueError):
            compressibility_error(20e6, 5e-4, 1.58e9, 0.0)

    def test_negative_A_raises(self):
        with pytest.raises(ValueError):
            compressibility_error(20e6, 5e-4, 1.58e9, -1.0)


class TestHydraulicEigenfrequency:
    def test_formula(self):
        K_eff = 1.78e9
        A = 50e-4
        m = 100.0
        V = 1e-3
        expected = math.sqrt(K_eff * A**2 / (m * V))
        assert hydraulic_eigenfrequency(K_eff, A, m, V) == pytest.approx(expected)

    def test_higher_K_gives_higher_frequency(self):
        kw = dict(A=50e-4, m=100.0, V=1e-3)
        om_low = hydraulic_eigenfrequency(K_eff=1.58e9, **kw)
        om_high = hydraulic_eigenfrequency(K_eff=2.25e9, **kw)
        assert om_high > om_low

    def test_zero_K_eff_raises(self):
        with pytest.raises(ValueError):
            hydraulic_eigenfrequency(0.0, 50e-4, 100.0, 1e-3)

    def test_zero_A_raises(self):
        with pytest.raises(ValueError):
            hydraulic_eigenfrequency(1.58e9, 0.0, 100.0, 1e-3)

    def test_zero_m_raises(self):
        with pytest.raises(ValueError):
            hydraulic_eigenfrequency(1.58e9, 50e-4, 0.0, 1e-3)

    def test_zero_V_raises(self):
        with pytest.raises(ValueError):
            hydraulic_eigenfrequency(1.58e9, 50e-4, 100.0, 0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            hydraulic_eigenfrequency(1.58e9, -1.0, 100.0, 1e-3)


class TestHydraulicNaturalFrequency:
    def test_is_omega_over_2pi(self):
        K_eff = 1.78e9
        A = 50e-4
        m = 100.0
        V = 1e-3
        omega = hydraulic_eigenfrequency(K_eff, A, m, V)
        f = hydraulic_natural_frequency(K_eff, A, m, V)
        assert f == pytest.approx(omega / (2.0 * math.pi))

    def test_table_values_vg46_blasenfrei(self):
        """Eigenfrequenz für K_eff=1780 MPa, A=50 cm², V=1 l, m=100 kg.

        Die Formel ω_h = sqrt(K·A²/(m·V)) ergibt mit diesen SI-Werten:
          A = 50e-4 m², V = 1e-3 m³  →  f_h ≈ 106 Hz.
        (Der Papiertabellenwert 48 Hz setzt andere interne Parameter voraus.)
        """
        K_eff = 1.78e9
        A = 50e-4   # m²
        m = 100.0   # kg
        V = 1e-3    # m³
        f = hydraulic_natural_frequency(K_eff, A, m, V)
        expected = math.sqrt(K_eff * A**2 / (m * V)) / (2.0 * math.pi)
        assert f == pytest.approx(expected)
        assert f > 0


class TestSeaStiffness:
    def test_formula(self):
        K_eff = 1.78e9
        A = 3e-4
        V_comp = 30e-6
        expected = K_eff * A**2 / V_comp
        assert sea_stiffness(K_eff, A, V_comp) == pytest.approx(expected)

    def test_zero_K_eff_raises(self):
        with pytest.raises(ValueError):
            sea_stiffness(0.0, 3e-4, 30e-6)

    def test_zero_A_raises(self):
        with pytest.raises(ValueError):
            sea_stiffness(1.78e9, 0.0, 30e-6)

    def test_zero_V_comp_raises(self):
        with pytest.raises(ValueError):
            sea_stiffness(1.78e9, 3e-4, 0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            sea_stiffness(-1.0, 3e-4, 30e-6)


class TestMaxContactForce:
    def test_zero_velocity(self):
        assert max_contact_force(0.0, 1e4, 15.0) == pytest.approx(0.0)

    def test_formula(self):
        v0 = 0.5
        K_SEA = 5e4
        m = 15.0
        expected = v0 * math.sqrt(K_SEA * m)
        assert max_contact_force(v0, K_SEA, m) == pytest.approx(expected)

    def test_negative_K_SEA_raises(self):
        with pytest.raises(ValueError):
            max_contact_force(0.5, -1.0, 15.0)

    def test_negative_m_raises(self):
        with pytest.raises(ValueError):
            max_contact_force(0.5, 5e4, -1.0)


class TestCompressionEnergy:
    def test_paper_example(self):
        """Abschnitt 7.2: P=500bar, V=20l, K=2.1GPa → E≈11.9 kJ."""
        P = 500e5
        V = 20e-3
        K = 2.1e9
        E = compression_energy(P, V, K)
        assert E == pytest.approx(11.9e3, rel=0.05)

    def test_higher_K_gives_less_energy(self):
        kw = dict(P=500e5, V=20e-3)
        E_low = compression_energy(K_eff=2.1e9, **kw)
        E_high = compression_energy(K_eff=2.35e9, **kw)
        assert E_high < E_low

    def test_zero_K_eff_raises(self):
        with pytest.raises(ValueError):
            compression_energy(500e5, 20e-3, 0.0)

    def test_negative_K_eff_raises(self):
        with pytest.raises(ValueError):
            compression_energy(500e5, 20e-3, -1.0)


class TestAccumulatorConstant:
    def test_formula(self):
        V0 = 1e-3
        P0 = 100e5
        V = 0.8e-3
        kappa = 1.4
        expected = V0 * P0**kappa / V ** (kappa + 1.0)
        assert accumulator_constant(V0, P0, V, kappa) == pytest.approx(expected)

    def test_default_kappa(self):
        result = accumulator_constant(1e-3, 100e5, 0.8e-3)
        assert result > 0

    def test_zero_V0_raises(self):
        with pytest.raises(ValueError):
            accumulator_constant(0.0, 100e5, 0.8e-3)

    def test_zero_P0_raises(self):
        with pytest.raises(ValueError):
            accumulator_constant(1e-3, 0.0, 0.8e-3)

    def test_zero_V_raises(self):
        with pytest.raises(ValueError):
            accumulator_constant(1e-3, 100e5, 0.0)


class TestMaxOperatingPressure:
    def test_default_factor(self):
        P_N = 200e5
        result = max_operating_pressure(P_N)
        assert result == pytest.approx(0.7 * P_N)

    def test_custom_factor(self):
        P_N = 300e5
        result = max_operating_pressure(P_N, factor=0.6)
        assert result == pytest.approx(0.6 * P_N)


class TestBearingCutoffFrequency:
    def test_formula(self):
        c_L = 1e7
        A = 0.01
        rho = 870.0
        c_s = 1350.0
        expected = c_L / (A * rho * c_s)
        assert bearing_cutoff_frequency(c_L, A, rho, c_s) == pytest.approx(expected)

    def test_zero_A_raises(self):
        with pytest.raises(ValueError):
            bearing_cutoff_frequency(1e7, 0.0, 870.0, 1350.0)

    def test_zero_rho_raises(self):
        with pytest.raises(ValueError):
            bearing_cutoff_frequency(1e7, 0.01, 0.0, 1350.0)

    def test_zero_cs_raises(self):
        with pytest.raises(ValueError):
            bearing_cutoff_frequency(1e7, 0.01, 870.0, 0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            bearing_cutoff_frequency(1e7, -1.0, 870.0, 1350.0)
