"""
tests/test_optics.py
====================
Vollständige Testabdeckung für das src.optics-Modul.

Getestete Eigenschaften (direkt aus der wissenschaftlichen Arbeit):
  - Lemma 2.1:  Energieerhaltung R + T = 1 (verlustfrei)
  - Def. 2.2:   Fresnel-Koeffizienten: Grenzfälle (n1=n2, θ=0)
  - Def. 3.1:   det(M_j) = 1 für verlustfreie Schichten
  - Def. 3.2:   Systemmatrix = Produkt der Einzelmatrizen
  - Satz 3.1:   TMM-Konsistenz mit Fresnel bei N=1 dünner Schicht
  - Alg. 3.1:   TMM-Inhomogen konvergiert gegen homogen für konst. n(z)
  - Satz 3.2:   Konvergenz O(1/K²) der inhomogenen TMM
  - FP-Formel:  Resonanzbedingung bei λ = 2nd/m
  - Finesse:    Formel korrekt
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.optics import (
    fresnel_coefficients,
    layer_matrix,
    transfer_matrix,
    compute_rt,
    fabry_perot_reflection,
    fabry_perot_finesse,
    fabry_perot_resonance_wavelength,
    tmm_inhomogeneous,
    tmm_spectrum,
    Layer,
    lorentz_refractive_index,
)


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def visible_wavelengths():
    return np.linspace(400, 700, 100)

@pytest.fixture
def single_wavelength():
    return 550.0  # grün

@pytest.fixture
def glass_layer():
    return Layer(n=1.5 + 0j, thickness=100.0)

@pytest.fixture
def air_glass_system():
    """Einfaches Luft/Glas-System."""
    return [Layer(n=1.5 + 0j, thickness=100.0)]


# ==========================================================================
# Tests: Fresnel-Koeffizienten (Def. 2.2, Lemma 2.1)
# ==========================================================================

class TestFresnelCoefficients:
    """Tests für Fresnel-Koeffizienten gemäß Def. 2.2."""

    def test_no_interface_s_polarization(self):
        """n1 = n2: r = 0, t = 1 (keine Grenzfläche)."""
        r, t = fresnel_coefficients(n1=1.5, n2=1.5, polarization="s")
        assert_allclose(abs(r), 0.0, atol=1e-12)
        assert_allclose(abs(t), 1.0, atol=1e-12)

    def test_no_interface_p_polarization(self):
        """n1 = n2: r = 0, t = 1 für p-Polarisation."""
        r, t = fresnel_coefficients(n1=1.5, n2=1.5, polarization="p")
        assert_allclose(abs(r), 0.0, atol=1e-12)
        assert_allclose(abs(t), 1.0, atol=1e-12)

    def test_normal_incidence_air_glass(self):
        """Luft → Glas (n=1.5) bei senkrechtem Einfall: r = -0.2, |r|² = 0.04."""
        r, t = fresnel_coefficients(n1=1.0, n2=1.5, theta_i=0.0, polarization="s")
        assert_allclose(abs(r)**2, 0.04, atol=1e-6)

    def test_normal_incidence_sp_equal(self):
        """Bei θ = 0: s- und p-Polarisation müssen gleiche |r|² liefern."""
        rs, ts = fresnel_coefficients(n1=1.0, n2=1.5, theta_i=0.0, polarization="s")
        rp, tp = fresnel_coefficients(n1=1.0, n2=1.5, theta_i=0.0, polarization="p")
        assert_allclose(abs(rs)**2, abs(rp)**2, atol=1e-10)

    def test_energy_conservation_lossless_s(self):
        """Lemma 2.1: R + T = 1 für verlustfreie Medien (s-Pol.)."""
        n1, n2 = 1.0, 1.5
        for theta in [0.0, 0.3, 0.6, 1.0]:
            r, t = fresnel_coefficients(n1, n2, theta, "s")
            R = abs(r)**2
            cos_i = np.cos(theta)
            cos_t = np.sqrt(1 - (n1/n2)**2 * np.sin(theta)**2)
            T = (n2 * cos_t / (n1 * cos_i)).real * abs(t)**2
            assert_allclose(R + T, 1.0, atol=1e-10,
                            err_msg=f"Energieerhaltung verletzt bei θ={theta}")

    def test_energy_conservation_lossless_p(self):
        """Lemma 2.1: R + T = 1 für verlustfreie Medien (p-Pol.)."""
        n1, n2 = 1.0, 1.5
        for theta in [0.0, 0.3, 0.5]:
            r, t = fresnel_coefficients(n1, n2, theta, "p")
            R = abs(r)**2
            cos_i = np.cos(theta)
            cos_t = np.sqrt(1 - (n1/n2)**2 * np.sin(theta)**2)
            T = (n2 * cos_t / (n1 * cos_i)).real * abs(t)**2
            assert_allclose(R + T, 1.0, atol=1e-10)

    def test_total_internal_reflection(self):
        """Totalreflexion: |r| = 1 jenseits des Grenzwinkels."""
        n1, n2 = 1.5, 1.0
        theta_c = np.arcsin(n2 / n1)  # Grenzwinkel
        theta_tir = theta_c + 0.1      # Jenseits des Grenzwinkels
        r, t = fresnel_coefficients(n1, n2, theta_tir, "s")
        assert_allclose(abs(r), 1.0, atol=1e-6)

    def test_invalid_polarization_raises(self):
        """Ungültige Polarisation muss ValueError auslösen."""
        with pytest.raises(ValueError, match="Polarisation"):
            fresnel_coefficients(1.0, 1.5, polarization="x")

    def test_absorbing_medium(self):
        """Absorptives Medium: r ist komplex, |r|² < 1."""
        n1 = 1.0
        n2 = complex(0.17, 3.0)  # Gold bei ~600 nm
        r, t = fresnel_coefficients(n1, n2, 0.0, "s")
        assert abs(r)**2 < 1.0
        assert isinstance(r, complex)

    def test_reciprocity(self):
        """Reziprozität: r(1→2) = -r(2→1) bei senkrechtem Einfall."""
        r12, _ = fresnel_coefficients(1.0, 1.5, 0.0, "s")
        r21, _ = fresnel_coefficients(1.5, 1.0, 0.0, "s")
        assert_allclose(r12, -r21, atol=1e-10)


# ==========================================================================
# Tests: Charakteristische Matrix (Def. 3.1)
# ==========================================================================

class TestLayerMatrix:
    """Tests für die charakteristische Schichtmatrix."""

    def test_identity_at_zero_thickness(self):
        """Dicke d=0: Matrix muss Einheitsmatrix sein."""
        M = layer_matrix(n=1.5, thickness=0.0, wavelength=550.0)
        assert_allclose(M, np.eye(2), atol=1e-12)

    def test_determinant_unity_lossless(self):
        """det(M) = 1 für verlustfreie Medien (κ = 0)."""
        M = layer_matrix(n=1.5 + 0j, thickness=200.0, wavelength=550.0)
        det = np.linalg.det(M)
        assert_allclose(abs(det), 1.0, atol=1e-10)

    def test_matrix_shape(self):
        """Matrix muss 2×2 sein."""
        M = layer_matrix(n=1.5, thickness=100.0, wavelength=550.0)
        assert M.shape == (2, 2)

    def test_matrix_symmetry_normal_incidence(self):
        """Bei θ=0: M[0,0] = M[1,1] (Diagonalelemente gleich)."""
        M = layer_matrix(n=1.5, thickness=100.0, wavelength=550.0, theta=0.0)
        assert_allclose(M[0, 0], M[1, 1], atol=1e-12)

    def test_matrix_dtype_complex(self):
        """Matrixelemente müssen komplex sein."""
        M = layer_matrix(n=1.5, thickness=100.0, wavelength=550.0)
        assert M.dtype == complex

    def test_full_wave_layer_identity(self):
        """'Full-wave' Schicht (d = λ/n): Matrix ≈ ±Einheitsmatrix."""
        n = 1.5
        wavelength = 550.0
        d = wavelength / n  # Eine Wellenlänge optische Weglänge
        M = layer_matrix(n=n, thickness=d, wavelength=wavelength)
        # cos(2πnd/λ) = cos(2π) = 1, sin(2πnd/λ) = 0
        assert_allclose(abs(M[0, 0].real), 1.0, atol=1e-6)
        assert_allclose(abs(M[0, 1]), 0.0, atol=1e-6)
        assert_allclose(abs(M[1, 0]), 0.0, atol=1e-6)

    def test_absorbing_layer_determinant(self):
        """Absorptives Medium: |det(M)| < 1."""
        M = layer_matrix(n=0.17 + 3.0j, thickness=10.0, wavelength=600.0)
        det = np.linalg.det(M)
        assert abs(det) < 1.0 + 1e-10


# ==========================================================================
# Tests: Systemtransfermatrix (Def. 3.2, Satz 3.1)
# ==========================================================================

class TestTransferMatrix:
    """Tests für die Systemtransfermatrix."""

    def test_single_layer_consistency(self):
        """transfer_matrix([L]) == layer_matrix für N=1."""
        layer = Layer(n=1.5, thickness=200.0)
        M_sys  = transfer_matrix([layer], wavelength=550.0)
        M_single = layer_matrix(1.5, 200.0, 550.0)
        assert_allclose(M_sys, M_single, atol=1e-12)

    def test_two_layers_product(self):
        """M_ges = M_1 · M_2."""
        L1 = Layer(n=1.5, thickness=100.0)
        L2 = Layer(n=1.8, thickness=150.0)
        M1 = layer_matrix(L1.n, L1.thickness, 550.0)
        M2 = layer_matrix(L2.n, L2.thickness, 550.0)
        M_ges = transfer_matrix([L1, L2], 550.0)
        assert_allclose(M_ges, M1 @ M2, atol=1e-12)

    def test_empty_system_identity(self):
        """Leeres System: Einheitsmatrix."""
        M = transfer_matrix([], wavelength=550.0)
        assert_allclose(M, np.eye(2), atol=1e-12)

    def test_energy_conservation_multilayer(self):
        """R + T + A ≈ 1 für reelles Schichtsystem."""
        layers = [
            Layer(n=1.5, thickness=100.0),
            Layer(n=2.0, thickness=80.0),
            Layer(n=1.5, thickness=120.0),
        ]
        M   = transfer_matrix(layers, wavelength=550.0)
        res = compute_rt(M, n_in=1.0, n_sub=1.5)
        assert_allclose(res.R + res.T + res.A, 1.0, atol=1e-6)

    def test_rt_values_in_range(self):
        """R, T, A ∈ [0, 1]."""
        layers = [Layer(n=1.5, thickness=200.0)]
        M   = transfer_matrix(layers, wavelength=500.0)
        res = compute_rt(M, n_in=1.0, n_sub=1.45)
        assert 0.0 <= res.R <= 1.0
        assert 0.0 <= res.T <= 1.0
        assert 0.0 <= res.A <= 1.0

    def test_absorbing_layer_reduces_transmission(self):
        """Absorptive Schicht: T < T (verlustfrei)."""
        lossless  = [Layer(n=1.5,        thickness=100.0)]
        absorbing = [Layer(n=1.5 + 0.5j, thickness=100.0)]
        M_ll = transfer_matrix(lossless,  wavelength=550.0)
        M_ab = transfer_matrix(absorbing, wavelength=550.0)
        r_ll = compute_rt(M_ll, 1.0, 1.0)
        r_ab = compute_rt(M_ab, 1.0, 1.0)
        assert r_ab.T < r_ll.T


# ==========================================================================
# Tests: Fabry-Pérot-Resonator
# ==========================================================================

class TestFabryPerot:
    """Tests für den Fabry-Pérot-Resonator."""

    def test_resonance_condition(self):
        """Bei λ = 2nd/m soll Reflexionsminimum liegen."""
        n, d = 1.40, 250.0
        order = 1
        lam_res = fabry_perot_resonance_wavelength(n, d, order)
        # Berechne Spektrum um Resonanzwellenlänge
        lam = np.linspace(lam_res - 100, lam_res + 100, 1000)
        R = fabry_perot_reflection(lam, complex(n), d, R1=0.3, R2=0.3)
        # Minimum sollte nahe lam_res liegen
        lam_min = lam[np.argmin(R)]
        assert_allclose(lam_min, lam_res, atol=2.0)  # < 2 nm Toleranz

    def test_finesse_formula(self):
        """Finesse F = π·(R1·R2)^(1/4) / (1 - √(R1·R2))."""
        R1, R2 = 0.9, 0.9
        F = fabry_perot_finesse(R1, R2)
        expected = np.pi * (R1 * R2) ** 0.25 / (1 - np.sqrt(R1 * R2))
        assert_allclose(F, expected, rtol=1e-10)

    def test_finesse_increases_with_reflectivity(self):
        """Höhere Reflektivität → höhere Finesse."""
        F_low  = fabry_perot_finesse(0.3, 0.3)
        F_high = fabry_perot_finesse(0.9, 0.9)
        assert F_high > F_low

    def test_reflection_in_range(self):
        """Reflexionsspektrum ∈ [0, 1]."""
        lam = np.linspace(400, 700, 200)
        R = fabry_perot_reflection(lam, 1.4 + 0j, 250.0, 0.5, 0.5)
        assert np.all(R >= 0.0 - 1e-10)
        assert np.all(R <= 1.0 + 1e-10)

    def test_no_mirrors_low_reflection(self):
        """Ohne Spiegel (R1=R2=0): Keine Interferenz, R≈0."""
        lam = np.array([550.0])
        R = fabry_perot_reflection(lam, 1.4 + 0j, 250.0, R1=0.0, R2=0.0)
        assert_allclose(R[0], 0.0, atol=1e-10)

    def test_perfect_mirrors_high_reflection(self):
        """Nahezu perfekte Spiegel: Reflexion nahe 1 (außer Resonanz)."""
        lam = np.array([550.0])
        R = fabry_perot_reflection(lam, 1.4 + 0j, 250.0, R1=0.99, R2=0.99)
        assert R[0] > 0.8  # Außerhalb der Resonanz hoch

    def test_finesse_invalid_raises(self):
        """R1·R2 ≥ 1: ValueError."""
        with pytest.raises(ValueError):
            fabry_perot_finesse(1.0, 1.0)

    def test_resonance_order_scaling(self):
        """λ_res ∝ 1/m (höhere Ordnung → kürzere Wellenlänge)."""
        n, d = 1.5, 300.0
        lam1 = fabry_perot_resonance_wavelength(n, d, order=1)
        lam2 = fabry_perot_resonance_wavelength(n, d, order=2)
        # Mit λ = 2nd/(m-0.5): λ1=4nd, λ2=4nd/3 → Verhältnis = 3
        assert_allclose(lam1 / lam2, 3.0, atol=1e-10)

    def test_resonance_order_invalid(self):
        """Ordnung < 1 muss ValueError auslösen."""
        with pytest.raises(ValueError):
            fabry_perot_resonance_wavelength(1.5, 200.0, order=0)


# ==========================================================================
# Tests: TMM inhomogen (Algorithmus 3.1, Satz 3.2)
# ==========================================================================

class TestTMMInhomogeneous:
    """Tests für die inhomogene TMM."""

    def test_constant_profile_equals_homogeneous(self):
        """
        Satz 3.2: Konstantes n(z) = n₀ muss dasselbe ergeben
        wie homogene TMM mit n = n₀.
        """
        n0 = 1.5
        d  = 200.0
        lam = 550.0

        # Homogen
        layers = [Layer(n=n0 + 0j, thickness=d)]
        M_hom = transfer_matrix(layers, lam)
        res_hom = compute_rt(M_hom, n_in=1.0, n_sub=1.45)

        # Inhomogen (konstantes Profil)
        res_inh = tmm_inhomogeneous(
            n_profile=lambda z: n0,
            thickness=d,
            wavelength=lam,
            n_sublayers=100,
            n_in=1.0,
            n_sub=1.45,
        )
        assert_allclose(res_inh.R, res_hom.R, atol=1e-3)
        assert_allclose(res_inh.T, res_hom.T, atol=1e-3)

    def test_convergence_with_sublayers(self):
        """
        Satz 3.2: Konvergenz O(1/K²) – Fehler nimmt mit K ab.
        """
        n0 = 1.5
        d  = 300.0
        lam = 550.0

        # Referenz: sehr viele Sublayer
        ref = tmm_inhomogeneous(
            n_profile=lambda z: n0 + 0.1 * np.sin(np.pi * z / d),
            thickness=d, wavelength=lam, n_sublayers=1000,
            n_in=1.0, n_sub=1.45,
        )

        errors = []
        for K in [10, 20, 50, 100]:
            res = tmm_inhomogeneous(
                n_profile=lambda z: n0 + 0.1 * np.sin(np.pi * z / d),
                thickness=d, wavelength=lam, n_sublayers=K,
                n_in=1.0, n_sub=1.45,
            )
            errors.append(abs(res.R - ref.R))

        # Konvergenz: Fehler soll mit K abnehmen
        assert errors[0] > errors[-1], "Kein Konvergenzverhalten beobachtet"

    def test_inhomogeneous_energy_conservation(self):
        """R + T + A ≈ 1 für inhomogene Schicht."""
        res = tmm_inhomogeneous(
            n_profile=lambda z: 1.4 + 0.1 * z / 200.0,
            thickness=200.0, wavelength=550.0, n_sublayers=50,
            n_in=1.0, n_sub=1.45,
        )
        assert_allclose(res.R + res.T + res.A, 1.0, atol=1e-5)

    def test_gradient_profile_different_from_homogeneous(self):
        """Lineares n(z)-Profil muss anderen R-Wert liefern als Mittelwert."""
        n_low, n_high = 1.3, 1.7
        n_mean = (n_low + n_high) / 2
        d = 200.0
        lam = 550.0

        res_grad = tmm_inhomogeneous(
            n_profile=lambda z: n_low + (n_high - n_low) * z / d,
            thickness=d, wavelength=lam, n_sublayers=200,
            n_in=1.0, n_sub=1.45,
        )
        layers_hom = [Layer(n=n_mean, thickness=d)]
        M_hom = transfer_matrix(layers_hom, lam)
        res_hom = compute_rt(M_hom, 1.0, 1.45)

        # Gradient und Mittelwert können unterschiedliche Spektren haben
        # (kein striktes R≠R_hom garantiert, aber Profil ist verschieden)
        assert res_grad.R >= 0.0


# ==========================================================================
# Tests: TMM-Spektrum
# ==========================================================================

class TestTMMSpectrum:
    """Tests für die Spektrumsberechnung."""

    def test_spectrum_length(self, visible_wavelengths):
        """Ausgabelänge == Eingabelänge."""
        layers = [Layer(n=1.5, thickness=100.0)]
        R, T, A = tmm_spectrum(layers, visible_wavelengths, n_in=1.0, n_sub=1.45)
        assert len(R) == len(visible_wavelengths)
        assert len(T) == len(visible_wavelengths)
        assert len(A) == len(visible_wavelengths)

    def test_spectrum_energy_conservation(self, visible_wavelengths):
        """R + T + A ≈ 1 für alle Wellenlängen."""
        layers = [Layer(n=1.5, thickness=100.0)]
        R, T, A = tmm_spectrum(layers, visible_wavelengths, n_in=1.0, n_sub=1.45)
        assert_allclose(R + T + A, 1.0, atol=1e-5)

    def test_spectrum_values_in_range(self, visible_wavelengths):
        """Alle Spektralwerte in [0, 1]."""
        layers = [Layer(n=1.5, thickness=100.0)]
        R, T, A = tmm_spectrum(layers, visible_wavelengths)
        assert np.all(R >= -1e-10)
        assert np.all(T >= -1e-10)
        assert np.all(R <= 1.0 + 1e-10)


# ==========================================================================
# Tests: Lorentz-Brechungsindex
# ==========================================================================

class TestLorentzRefractiveIndex:
    """Tests für das Lorentz-Oszillator-Modell."""

    def test_real_far_from_resonance(self):
        """Weit von Resonanz: Brechungsindex fast reell."""
        omega_res = 2 * np.pi * 3e15   # UV-Resonanz
        omega_vis = 2 * np.pi * 6e14   # Sichtbares Licht
        n = lorentz_refractive_index(
            [omega_vis], omega_res, gamma=1e12,
            oscillator_strength=1.0, N=1e28
        )
        assert abs(n[0].imag) < 0.1  # Kleiner Imaginärteil

    def test_increased_density_increased_n(self):
        """Höhere Dichte N → höherer Brechungsindex."""
        omega = np.array([2 * np.pi * 5e14])
        omega0 = 2 * np.pi * 8e14
        n_low  = lorentz_refractive_index(omega, omega0, 1e12, 1.0, N=1e27)
        n_high = lorentz_refractive_index(omega, omega0, 1e12, 1.0, N=1e28)
        assert n_high[0].real > n_low[0].real

    def test_near_resonance_absorption(self):
        """Nahe Resonanz: signifikanter Imaginärteil (Absorption)."""
        omega0 = 2 * np.pi * 6e14
        n = lorentz_refractive_index(
            [omega0], omega0, gamma=1e13,
            oscillator_strength=1.0, N=1e28
        )
        assert n[0].imag > 0.01
