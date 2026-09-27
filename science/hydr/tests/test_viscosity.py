"""Tests für hydr.viscosity – 100 % Abdeckung."""

import math
import pytest

from hydr.viscosity import (
    shear_stress,
    relaxation_time,
    relaxation_frequency,
    barus_viscosity,
    vii_viscosity,
    shear_degradation_viscosity,
)


class TestShearStress:
    def test_zero_shear_rate(self):
        assert shear_stress(0.046, 0.0) == pytest.approx(0.0)

    def test_typical_values(self):
        eta = 0.046  # 46 mPa·s
        gamma_dot = 1000.0  # 1/s
        assert shear_stress(eta, gamma_dot) == pytest.approx(eta * gamma_dot)

    def test_linearity(self):
        eta = 0.1
        assert shear_stress(eta, 2.0) == pytest.approx(2.0 * shear_stress(eta, 1.0))


class TestRelaxationTime:
    def test_mineral_oil_vg46_order_of_magnitude(self):
        """τ_R = η/(ρ·c_s²) für Mineralöl VG 46 bei 40 °C.

        Berechnung: 46e-3 / (870 × 1350²) ≈ 29 ps.
        (Der Papierwert 18 ns enthält einen Faktor-1000-Fehler im Druckwerk –
        die Formel selbst ist korrekt.)
        """
        eta = 46e-3
        rho = 870.0
        c_s = 1350.0
        tau = relaxation_time(eta, rho, c_s)
        # Formelsicherung: η/(ρ·c_s²)
        assert tau == pytest.approx(eta / (rho * c_s**2))
        # Größenordnung: Pikosekunden-Bereich
        assert 1e-12 < tau < 1e-9

    def test_positive_rho_required(self):
        with pytest.raises(ValueError):
            relaxation_time(0.046, 0.0, 1350.0)

    def test_negative_rho_raises(self):
        with pytest.raises(ValueError):
            relaxation_time(0.046, -1.0, 1350.0)

    def test_positive_cs_required(self):
        with pytest.raises(ValueError):
            relaxation_time(0.046, 870.0, 0.0)

    def test_negative_cs_raises(self):
        with pytest.raises(ValueError):
            relaxation_time(0.046, 870.0, -1.0)

    def test_higher_viscosity_longer_relaxation(self):
        tau_low = relaxation_time(0.046, 870.0, 1350.0)
        tau_high = relaxation_time(0.100, 870.0, 1350.0)
        assert tau_high > tau_low


class TestRelaxationFrequency:
    def test_inverse_of_2pi_tau(self):
        eta = 0.046
        rho = 870.0
        c_s = 1350.0
        tau = relaxation_time(eta, rho, c_s)
        f_R = relaxation_frequency(rho, c_s, eta)
        assert f_R == pytest.approx(1.0 / (2.0 * math.pi * tau))

    def test_mineral_oil_vg46_gigahertz_range(self):
        """f_R = ρ·c_s²/(2π·η) für Mineralöl VG 46.

        Berechnung: 870×1350²/(2π×46e-3) ≈ 5.5 GHz.
        (Der Papierwert 9 MHz korrespondiert zum fehlerhaften τ_R=18 ns.)
        """
        f_R = relaxation_frequency(870.0, 1350.0, 46e-3)
        assert f_R == pytest.approx(870.0 * 1350.0**2 / (2.0 * math.pi * 46e-3))
        # Größenordnung: GHz
        assert f_R > 1e9

    def test_zero_eta_raises(self):
        with pytest.raises(ValueError):
            relaxation_frequency(870.0, 1350.0, 0.0)

    def test_negative_eta_raises(self):
        with pytest.raises(ValueError):
            relaxation_frequency(870.0, 1350.0, -0.001)


class TestBarusViscosity:
    def test_at_zero_pressure(self):
        """η(0) = η_0."""
        eta_0 = 0.046
        assert barus_viscosity(eta_0, 2e-8, 0.0) == pytest.approx(eta_0)

    def test_increases_with_pressure(self):
        eta_0 = 0.046
        alpha_P = 2e-8  # 1/Pa
        eta_low = barus_viscosity(eta_0, alpha_P, 100e5)
        eta_high = barus_viscosity(eta_0, alpha_P, 600e5)
        assert eta_high > eta_low

    def test_600bar_factor_3_3(self):
        """Lt. Paper: η bei 600 bar ≈ 3.3 · η_0."""
        eta_0 = 0.046
        alpha_P = 2e-8
        P = 600e5  # 600 bar
        ratio = barus_viscosity(eta_0, alpha_P, P) / eta_0
        assert ratio == pytest.approx(math.exp(alpha_P * P))
        # sollte ≈ 3.3 sein
        assert 3.0 < ratio < 4.0


class TestViiViscosity:
    def test_zero_concentration(self):
        """Ohne VII: exp(0) = 1, also η_VII = η_basis."""
        result = vii_viscosity(
            eta_basis=0.030, c_vii=0.0, rho_vii=900.0, M_c=1e5, M_e=5e3
        )
        assert result == pytest.approx(0.030)

    def test_nonzero_vii_increases_viscosity(self):
        result = vii_viscosity(
            eta_basis=0.030, c_vii=5.0, rho_vii=900.0, M_c=1e5, M_e=5e3
        )
        assert result > 0.030

    def test_zero_rho_vii_raises(self):
        with pytest.raises(ValueError):
            vii_viscosity(0.030, 5.0, 0.0, 1e5, 5e3)

    def test_negative_rho_vii_raises(self):
        with pytest.raises(ValueError):
            vii_viscosity(0.030, 5.0, -1.0, 1e5, 5e3)

    def test_zero_M_e_raises(self):
        with pytest.raises(ValueError):
            vii_viscosity(0.030, 5.0, 900.0, 1e5, 0.0)

    def test_negative_M_e_raises(self):
        with pytest.raises(ValueError):
            vii_viscosity(0.030, 5.0, 900.0, 1e5, -1.0)


class TestShearDegradationViscosity:
    def test_at_zero_time(self):
        """Bei t = 0: kein Abbau, η = η_0."""
        result = shear_degradation_viscosity(
            eta_0=0.100, eta_basis=0.030, t=0.0, tau_sch=3600.0
        )
        assert result == pytest.approx(0.100)

    def test_approaches_eta_basis_at_large_time(self):
        """Nach sehr langer Zeit: η → η_basis."""
        result = shear_degradation_viscosity(
            eta_0=0.100, eta_basis=0.030, t=1e12, tau_sch=3600.0
        )
        assert result == pytest.approx(0.030, rel=1e-6)

    def test_monotonically_decreasing(self):
        kw = dict(eta_0=0.100, eta_basis=0.030, tau_sch=3600.0)
        eta_t1 = shear_degradation_viscosity(t=1000, **kw)
        eta_t2 = shear_degradation_viscosity(t=5000, **kw)
        assert eta_t2 < eta_t1

    def test_zero_tau_sch_raises(self):
        with pytest.raises(ValueError):
            shear_degradation_viscosity(0.100, 0.030, 1000.0, 0.0)

    def test_negative_tau_sch_raises(self):
        with pytest.raises(ValueError):
            shear_degradation_viscosity(0.100, 0.030, 1000.0, -1.0)
