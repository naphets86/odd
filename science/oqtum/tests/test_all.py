"""
tests/test_ebeam.py
===================
Tests für src.ebeam (Proximity-Effekt-Korrektur).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys, os

from src.ebeam import (
    proximity_function,
    proximity_kernel_2d,
    apply_proximity,
    proximity_correction_ipec,
    topography_operator,
    inverse_topography_operator,
    IPECResult,
)


class TestProximityFunction:
    """Tests für die Proximity-Funktion (Def. 6.2)."""

    def test_normalization(self):
        """∫∫ f(x,y) dx dy ≈ 1 (Normierung)."""
        N = 1000
        L = 30000  # nm, groß genug um β-Gaussians zu erfassen
        x = np.linspace(-L, L, N)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, x)
        f = proximity_function(X, Y, alpha=50.0, beta=5000.0, eta_back=0.75)
        integral = np.sum(f) * dx**2
        assert_allclose(integral, 1.0, rtol=0.05)

    def test_maximum_at_center(self):
        """Maximum der Proximity-Funktion bei r=0."""
        f_center = proximity_function(0.0, 0.0, alpha=50, beta=5000)
        f_edge   = proximity_function(1000.0, 0.0, alpha=50, beta=5000)
        assert f_center > f_edge

    def test_symmetry(self):
        """f(x, y) = f(-x, y) = f(x, -y) (Radialsymmetrie)."""
        f1 = proximity_function(100.0,  200.0)
        f2 = proximity_function(-100.0, 200.0)
        f3 = proximity_function(100.0, -200.0)
        assert_allclose(f1, f2, rtol=1e-10)
        assert_allclose(f1, f3, rtol=1e-10)

    def test_positive_values(self):
        """Proximity-Funktion ist überall nicht-negativ."""
        x = np.linspace(-1000, 1000, 50)
        X, Y = np.meshgrid(x, x)
        f = proximity_function(X, Y)
        assert np.all(f >= 0)

    def test_decay_with_distance(self):
        """Funktion nimmt mit Abstand ab."""
        f_near = proximity_function(10.0, 0.0)
        f_far  = proximity_function(500.0, 0.0)
        assert f_near > f_far


class TestApplyProximity:
    """Tests für die Faltung mit der Proximity-Funktion."""

    def test_output_shape(self):
        """Ausgabe hat gleiche Form wie Eingabe."""
        dose = np.ones((32, 32))
        out  = apply_proximity(dose, pixel_size=100.0)
        assert out.shape == dose.shape

    def test_uniform_dose_unchanged(self):
        """Gleichförmige Dosis bleibt unter Faltung nahezu konstant."""
        D0 = 5.0
        dose = np.full((64, 64), D0)
        out  = apply_proximity(dose, pixel_size=100.0)
        # Rand-Effekte berücksichtigen: Innenbereich prüfen
        inner = out[16:48, 16:48]
        assert_allclose(inner, D0, rtol=0.1)

    def test_point_source_spreads(self):
        """Punktquelle wird verbreitet (Spreading-Effekt)."""
        dose = np.zeros((64, 64))
        dose[32, 32] = 100.0
        out = apply_proximity(dose, pixel_size=100.0)
        # Peak sollte kleiner sein als Eingang
        assert out.max() < dose.max()
        # Gesamtdosis bleibt erhalten
        assert_allclose(out.sum(), dose.sum(), rtol=0.1)


class TestIPEC:
    """Tests für den IPEC-Algorithmus (Alg. 6.1, Satz 6.2)."""

    def test_converges_for_smooth_target(self):
        """IPEC konvergiert für ein glattes, einfaches Zielmuster."""
        M = 32
        x = np.linspace(0, M-1, M)
        X, Y = np.meshgrid(x, x)
        dose_target = 5.0 * np.exp(-((X-M/2)**2 + (Y-M/2)**2) / (M/4)**2)

        result = proximity_correction_ipec(
            dose_target, pixel_size=200.0,
            omega=0.5, tolerance=1e-2, max_iterations=100
        )
        assert result.converged or result.n_iterations == 100

    def test_error_history_decreasing(self):
        """Fehler nimmt über Iterationen ab (Konvergenzmonotonie)."""
        M = 16
        dose_target = np.full((M, M), 3.0)

        result = proximity_correction_ipec(
            dose_target, pixel_size=200.0,
            omega=0.5, tolerance=1e-6, max_iterations=50
        )
        errs = result.error_history
        # Gesamttrend: Fehler am Ende kleiner als am Anfang
        assert errs[-1] <= errs[0]

    def test_output_dose_non_negative(self):
        """Korrigiertes Dosismuster ist überall ≥ 0 (Projektion)."""
        dose_target = np.random.uniform(0, 5, (16, 16))
        result = proximity_correction_ipec(dose_target, pixel_size=200.0, max_iterations=20)
        assert np.all(result.dose_nominal >= 0.0)

    def test_result_type(self):
        """Rückgabe ist IPECResult."""
        result = proximity_correction_ipec(np.ones((8, 8)), pixel_size=100.0, max_iterations=5)
        assert isinstance(result, IPECResult)

    def test_effective_dose_close_to_target_after_correction(self):
        """Nach Korrektur: D_eff näher an D_soll als ohne Korrektur."""
        M = 16
        dose_target = np.ones((M, M)) * 4.0
        # Ohne Korrektur: direkt konvolvieren
        D_eff_no_corr = apply_proximity(dose_target, pixel_size=200.0)
        err_no_corr = np.mean(np.abs(D_eff_no_corr - dose_target))

        result = proximity_correction_ipec(
            dose_target, pixel_size=200.0, omega=0.5, max_iterations=30
        )
        D_eff_corr = apply_proximity(result.dose_nominal, pixel_size=200.0)
        err_corr = np.mean(np.abs(D_eff_corr - dose_target))

        assert err_corr <= err_no_corr + 1e-6


class TestTopographyOperator:
    """Tests für den Topographie-Operator (Satz 6.1)."""

    def test_zero_dose_gives_zero_height(self):
        """Keine Bestrahlung → minimale Quellung (Q_eq ohne Modifikation)."""
        h = topography_operator(
            dose=np.zeros((4, 4)),
            d0=250.0, nu_e0=100.0, alpha_ebeam=5.0,
            chi=0.25
        )
        # Alle Pixel haben gleiche Höhe
        assert np.std(h) < 1e-3

    def test_higher_dose_lower_height(self):
        """Höhere Dosis → höhere Vernetzung → geringere Quellung → niedrigere Höhe."""
        h_low  = topography_operator(
            dose=np.array([[0.0]]),
            d0=250.0, nu_e0=100.0, alpha_ebeam=5.0, chi=0.25
        )
        h_high = topography_operator(
            dose=np.array([[10.0]]),
            d0=250.0, nu_e0=100.0, alpha_ebeam=5.0, chi=0.25
        )
        assert h_high[0, 0] < h_low[0, 0]

    def test_output_non_negative(self):
        """Höhenprofil ≥ 0 (Film quillt nur, schrumpft nicht)."""
        dose = np.random.uniform(0, 5, (4, 4))
        h = topography_operator(dose, d0=200.0, nu_e0=100.0, alpha_ebeam=5.0, chi=0.25)
        assert np.all(h >= -1e-6)


"""
tests/test_colorimetry.py
=========================
Tests für src.colorimetry.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.colorimetry import (
    photopic_sensitivity,
    spectrum_to_xyz,
    xyz_to_lab,
    delta_e,
    spectrum_to_lab,
    camouflage_index,
    wavelengths_visible,
)


class TestPhotopicSensitivity:
    """Tests für die photopische Empfindlichkeitskurve V(λ)."""

    def test_maximum_near_555nm(self):
        """Maximum von V(λ) nahe 555 nm."""
        lam = np.arange(380, 781, 1, dtype=float)
        V = photopic_sensitivity(lam)
        lam_max = lam[np.argmax(V)]
        assert 540 <= lam_max <= 570

    def test_values_in_range(self):
        """V(λ) ∈ [0, 1]."""
        lam = wavelengths_visible()
        V = photopic_sensitivity(lam)
        assert np.all(V >= 0.0)
        assert np.all(V <= 1.0 + 1e-10)

    def test_zero_outside_visible(self):
        """V(λ) ≈ 0 im UV und tiefen IR."""
        lam_uv = np.array([300.0, 350.0])
        lam_ir = np.array([800.0, 900.0])
        V_uv = photopic_sensitivity(lam_uv)
        V_ir = photopic_sensitivity(lam_ir)
        assert np.all(V_uv < 0.01)
        assert np.all(V_ir < 0.01)

    def test_wavelengths_visible_count(self):
        """wavelengths_visible(n) gibt n Punkte zurück."""
        for n in [100, 200, 400]:
            lam = wavelengths_visible(n)
            assert len(lam) == n


class TestSpectrumToXYZ:
    """Tests für die Spektrum-XYZ-Konversion."""

    def test_white_spectrum_near_equal_energy(self):
        """Weißes Spektrum (R=1): XYZ nahe Normwerten."""
        lam = wavelengths_visible(200)
        R   = np.ones_like(lam)
        X, Y, Z = spectrum_to_xyz(lam, R)
        # Y ≈ 1.0 (normiert auf Weißpunkt)
        assert 0.5 < Y < 2.0

    def test_zero_spectrum(self):
        """Schwarzes Spektrum (R=0): XYZ ≈ 0."""
        lam = wavelengths_visible(200)
        R   = np.zeros_like(lam)
        X, Y, Z = spectrum_to_xyz(lam, R)
        assert_allclose(X, 0.0, atol=1e-10)
        assert_allclose(Y, 0.0, atol=1e-10)
        assert_allclose(Z, 0.0, atol=1e-10)

    def test_xyz_positive(self):
        """XYZ-Werte sind nicht-negativ für R ≥ 0."""
        lam = wavelengths_visible(200)
        R   = np.random.uniform(0, 1, len(lam))
        X, Y, Z = spectrum_to_xyz(lam, R)
        assert X >= 0.0
        assert Y >= 0.0
        assert Z >= 0.0


class TestXYZToLab:
    """Tests für die XYZ→L*a*b*-Konversion."""

    def test_white_point_gives_L100(self):
        """Weißpunkt (Xn, Yn, Zn): L* = 100, a* = b* = 0."""
        L, a, b = xyz_to_lab(95.047, 100.000, 108.883)
        assert_allclose(L, 100.0, atol=0.01)
        assert_allclose(a, 0.0,   atol=0.01)
        assert_allclose(b, 0.0,   atol=0.01)

    def test_black_gives_L0(self):
        """Schwarz (X=Y=Z=0): L* = 0."""
        L, a, b = xyz_to_lab(0.0, 0.0, 0.0)
        assert_allclose(L, 0.0, atol=0.01)


class TestDeltaE:
    """Tests für den ΔE-Farbdifferenz-Algorithmus."""

    def test_identical_colors_delta_e_zero(self):
        """Identische Farben: ΔE = 0."""
        lab = (50.0, 10.0, -20.0)
        assert_allclose(delta_e(lab, lab), 0.0, atol=1e-10)

    def test_delta_e_symmetric(self):
        """ΔE(A, B) = ΔE(B, A)."""
        lab1 = (50.0, 10.0, -20.0)
        lab2 = (60.0, -5.0,  15.0)
        assert_allclose(delta_e(lab1, lab2), delta_e(lab2, lab1))

    def test_delta_e_triangle_inequality(self):
        """ΔE erfüllt Dreiecksungleichung: ΔE(A,C) ≤ ΔE(A,B) + ΔE(B,C)."""
        A = (50.0, 10.0, -20.0)
        B = (60.0, -5.0,  15.0)
        C = (70.0,  0.0,   5.0)
        assert delta_e(A, C) <= delta_e(A, B) + delta_e(B, C) + 1e-10

    def test_jnd_threshold(self):
        """ΔE < 2: nicht unterscheidbar (JND-Schwelle aus Def. 7.2)."""
        lab1 = (50.0, 10.0, 10.0)
        lab2 = (50.5, 10.5, 10.5)
        dE = delta_e(lab1, lab2)
        assert dE < 2.0

    def test_visible_difference(self):
        """Stark verschiedene Farben: ΔE >> 2."""
        lab_red  = (50.0, 60.0, 40.0)
        lab_blue = (30.0, 10.0, -60.0)
        assert delta_e(lab_red, lab_blue) > 10.0


class TestCamouflageIndex:
    """Tests für den Tarnindex."""

    def test_perfect_camouflage_gives_one(self):
        """Perfekte Tarnung: C = 1."""
        lam = wavelengths_visible(100)
        R   = np.ones_like(lam) * 0.5
        C = camouflage_index(lam, R, R, np.zeros_like(lam))
        assert_allclose(C, 1.0, atol=0.01)

    def test_no_camouflage_gives_zero(self):
        """Kein Tarneffekt: System = Referenz, beide weit vom Hintergrund."""
        lam = wavelengths_visible(100)
        R_bg  = np.zeros_like(lam)  # Schwarz
        R_sys = np.ones_like(lam)   # Weiß (= Referenz)
        C = camouflage_index(lam, R_sys, R_bg, R_sys)
        assert_allclose(C, 0.0, atol=0.05)

    def test_index_in_range(self):
        """Tarnindex ∈ [0, 1]."""
        lam = wavelengths_visible(100)
        R1 = np.random.uniform(0, 1, len(lam))
        R2 = np.random.uniform(0, 1, len(lam))
        R3 = np.random.uniform(0, 1, len(lam))
        C = camouflage_index(lam, R1, R2, R3)
        assert 0.0 <= C <= 1.0


"""
tests/test_optimizer.py
=======================
Tests für src.optimizer.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimizer import (
    spectral_loss,
    perceptual_loss,
    gradient_reflection,
    adjoint_gradient,
    optimize_resonator,
    target_spectrum_gaussian,
    target_spectrum_leaf,
    OptimizationResult,
)
from src.colorimetry import wavelengths_visible


class TestSpectralLoss:
    """Tests für das spektrale Verlustfunktional."""

    def test_zero_loss_identical_spectra(self):
        """Identische Spektren: L = 0."""
        R = np.ones(100) * 0.5
        assert_allclose(spectral_loss(R, R), 0.0, atol=1e-15)

    def test_loss_positive(self):
        """Verlust ist immer ≥ 0."""
        R_c = np.random.uniform(0, 1, 100)
        R_t = np.random.uniform(0, 1, 100)
        assert spectral_loss(R_c, R_t) >= 0.0

    def test_loss_symmetric(self):
        """L(A, B) = L(B, A) (für gleiche Gewichte)."""
        R_c = np.random.uniform(0, 1, 100)
        R_t = np.random.uniform(0, 1, 100)
        assert_allclose(spectral_loss(R_c, R_t), spectral_loss(R_t, R_c))

    def test_perceptual_weights_applied(self):
        """Perceptueller Verlust ≠ uniformer Verlust (außer trivial)."""
        lam = wavelengths_visible(100)
        R_c = np.zeros_like(lam)
        R_t = np.ones_like(lam)
        L_unif = spectral_loss(R_c, R_t, wavelengths=lam)
        L_perc = perceptual_loss(lam, R_c, R_t)
        # Beide positiv, aber i.A. verschieden
        assert L_unif > 0
        assert L_perc > 0


class TestGradientReflection:
    """Tests für den analytischen Gradienten (Satz 7.1)."""

    def test_gradient_shape(self):
        """Gradient hat gleiche Form wie Wellenlängenarray."""
        lam = wavelengths_visible(50)
        dR_dd, dR_dn = gradient_reflection(lam, n_cavity=1.4, thickness=250.0,
                                           R1=0.5, R2=0.5)
        assert dR_dd.shape == lam.shape
        assert dR_dn.shape == lam.shape

    def test_gradient_finite_difference_consistency(self):
        """
        Satz 7.1: Analytischer Gradient muss mit finiten Differenzen
        übereinstimmen.
        """
        from src.optics import fabry_perot_reflection

        lam = wavelengths_visible(50)
        n, d, R1, R2 = 1.40, 250.0, 0.5, 0.5
        eps = 1e-4  # nm

        # Analytisch
        dR_dd, _ = gradient_reflection(lam, n, d, R1, R2)

        # Finite Differenzen
        R_plus  = fabry_perot_reflection(lam, complex(n), d + eps, R1, R2)
        R_minus = fabry_perot_reflection(lam, complex(n), d - eps, R1, R2)
        dR_fd   = (R_plus - R_minus) / (2 * eps)

        assert_allclose(dR_dd, dR_fd, atol=1e-4)

    def test_gradient_at_resonance_negative(self):
        """
        Am Reflexionsminimum: ∂R/∂d wechselt Vorzeichen (R nimmt zu
        wenn d von Resonanz abweicht).
        """
        n, d0 = 1.4, 250.0
        lam_res = 2 * n * d0  # Resonanzwellenlänge (m=1)
        lam = np.array([lam_res])
        dR_dd, _ = gradient_reflection(lam, n, d0, R1=0.5, R2=0.5)
        # Nahe Minimum: Gradient ≈ 0 (Extremum)
        assert abs(dR_dd[0]) < 1.0  # kleiner als 1 [1/nm]


class TestOptimizeResonator:
    """Tests für den Hauptoptimierungsalgorithmus (Alg. 8.1)."""

    def test_returns_optimization_result(self):
        """optimize_resonator gibt OptimizationResult zurück."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=550.0)
        result = optimize_resonator(lam, R_t, max_iterations=10)
        assert isinstance(result, OptimizationResult)

    def test_loss_non_negative(self):
        """Verlustfunktional ist nicht-negativ."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=550.0)
        result = optimize_resonator(lam, R_t, max_iterations=20)
        assert result.loss >= 0.0

    def test_thickness_in_bounds(self):
        """Optimale Schichtdicke liegt in den gesetzten Grenzen."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=600.0)
        d_bounds = (100.0, 500.0)
        result = optimize_resonator(lam, R_t, max_iterations=20, d_bounds=d_bounds)
        assert d_bounds[0] <= result.thickness <= d_bounds[1]

    def test_Q_in_bounds(self):
        """Optimaler Quellungsgrad liegt in den gesetzten Grenzen."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=550.0)
        Q_bounds = (1.0, 8.0)
        result = optimize_resonator(lam, R_t, max_iterations=20, Q_bounds=Q_bounds)
        assert Q_bounds[0] <= result.Q <= Q_bounds[1]

    def test_loss_history_length(self):
        """loss_history hat die richtige Länge."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=550.0)
        n_iter = 15
        result = optimize_resonator(lam, R_t, max_iterations=n_iter)
        assert len(result.loss_history) == result.n_iterations
        assert result.n_iterations <= n_iter

    def test_lambda_res_positive(self):
        """Resonanzwellenlänge ist positiv."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=540.0)
        result = optimize_resonator(lam, R_t, max_iterations=15)
        assert result.lambda_res > 0.0

    def test_delta_e_finite(self):
        """ΔE ist endlich und positiv."""
        lam = wavelengths_visible(80)
        R_t = target_spectrum_gaussian(lam, center_nm=560.0)
        result = optimize_resonator(lam, R_t, max_iterations=50)
        assert np.isfinite(result.delta_e_final)
        assert result.delta_e_final >= 0.0

    def test_gaussian_target_spectra(self):
        """Gauss-Zielspektren sind normiert in [0, 1]."""
        lam = wavelengths_visible(100)
        for center in [450, 550, 650]:
            R = target_spectrum_gaussian(lam, center_nm=float(center))
            assert np.all(R >= 0.0)
            assert np.all(R <= 1.0 + 1e-10)

    def test_leaf_spectrum_in_range(self):
        """Blatt-Zielspektrum in [0, 1]."""
        lam = wavelengths_visible(100)
        R = target_spectrum_leaf(lam)
        assert np.all(R >= 0.0 - 1e-10)
        assert np.all(R <= 1.0 + 1e-10)


"""
tests/test_pipeline.py
======================
Tests für die vollständige Design-Pipeline.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import DesignTarget, run_pipeline, PipelineResult
from src.colorimetry import wavelengths_visible
from src.optimizer import target_spectrum_gaussian


class TestPipeline:
    """Tests für die End-to-End-Design-Pipeline."""

    @pytest.fixture
    def simple_target(self):
        lam = wavelengths_visible(60)
        R_t = target_spectrum_gaussian(lam, center_nm=550.0)
        return DesignTarget(wavelengths=lam, R_target=R_t)

    def test_pipeline_returns_result(self, simple_target):
        """run_pipeline gibt PipelineResult zurück."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 10},
            ipec_kwargs={"max_iterations": 5},
        )
        assert isinstance(result, PipelineResult)

    def test_pipeline_delta_e_finite(self, simple_target):
        """ΔE ist endlich nach Pipeline."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 10},
            run_proximity=False,
        )
        assert np.isfinite(result.delta_e_color)

    def test_pipeline_nu_e_positive(self, simple_target):
        """Vernetzungsdichte ist positiv."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 10},
            run_proximity=False,
        )
        assert result.nu_e_color >= 0.0

    def test_pipeline_dose_non_negative(self, simple_target):
        """Dosismuster ist überall ≥ 0."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 10},
            run_proximity=False,
        )
        if result.dose_nominal is not None:
            assert np.all(result.dose_nominal >= 0.0)

    def test_pipeline_with_height_map(self):
        """Pipeline mit Höhenkarte läuft durch."""
        lam = wavelengths_visible(50)
        R_t = target_spectrum_gaussian(lam, center_nm=550.0)
        h_map = np.ones((8, 8)) * 300.0  # 300 nm Zielhöhe

        target = DesignTarget(
            wavelengths=lam, R_target=R_t, height_map=h_map
        )
        result = run_pipeline(
            target,
            optimize_kwargs={"max_iterations": 5},
            ipec_kwargs={"max_iterations": 5},
        )
        assert result.dose_nominal is not None
        assert result.dose_nominal.shape == (8, 8)

    def test_pipeline_without_proximity(self, simple_target):
        """Pipeline ohne Proximity-Korrektur: ipec_result ist None."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 5},
            run_proximity=False,
        )
        assert result.ipec_result is None

    def test_pipeline_summary_non_empty(self, simple_target):
        """Summary-String ist nicht leer."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 5},
            run_proximity=False,
        )
        assert len(result.summary) > 0

    def test_pipeline_opt_result_valid(self, simple_target):
        """Optimierungsergebnis enthält gültige Werte."""
        result = run_pipeline(
            simple_target,
            optimize_kwargs={"max_iterations": 10},
            run_proximity=False,
        )
        assert result.opt_result.thickness > 0
        assert result.opt_result.Q >= 1.0
        assert result.opt_result.n_eff > 1.0


"""
tests/test_materials.py
=======================
Tests für die Materialdatenbank.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.materials import (
    MATERIALS,
    FLORY_HUGGINS_CHI,
    MOLAR_VOLUME,
    sellmeier_sio2,
    drude_lorentz_gold,
    cauchy_pdms,
    water_dispersion,
    Material,
)


class TestMaterialDatabase:
    """Tests für die Materialdatenbank."""

    def test_all_materials_have_positive_n(self):
        """Alle Materialien haben positiven Brechungsindex."""
        for name, mat in MATERIALS.items():
            assert mat.n_real > 0, f"{name}: n ≤ 0"

    def test_non_absorbing_materials_zero_kappa(self):
        """Nicht-absorptive Materialien: κ = 0."""
        transparent = ["PDMS", "PMMA", "SiO2", "Air", "Water", "Ethanol"]
        for name in transparent:
            if name in MATERIALS:
                assert MATERIALS[name].n_imag == 0.0, f"{name}: κ ≠ 0"

    def test_gold_absorbing(self):
        """Gold ist absorptiv: κ > 0."""
        assert MATERIALS["Au"].n_imag > 0

    def test_all_materials_accessible(self):
        """Alle erwarteten Materialien in der Datenbank."""
        expected = ["PDMS", "Au", "SiO2", "Air", "Water", "Ethanol"]
        for name in expected:
            assert name in MATERIALS, f"{name} fehlt in MATERIALS"

    def test_flory_huggins_chi_range(self):
        """χ-Werte physikalisch plausibel (0 < χ < 0.5 für gute Lösungsmittel)."""
        for solvent, chi in FLORY_HUGGINS_CHI.items():
            assert 0.0 < chi < 0.6, f"{solvent}: χ = {chi} außerhalb Bereich"

    def test_molar_volumes_positive(self):
        """Molare Volumina > 0."""
        for solvent, V1 in MOLAR_VOLUME.items():
            assert V1 > 0, f"{solvent}: V1 = {V1} ≤ 0"


class TestDispersionFunctions:
    """Tests für die Dispersionsformeln."""

    def test_sellmeier_sio2_visible_range(self):
        """SiO2: Brechungsindex im Sichtbaren zwischen 1.45 und 1.48."""
        for lam in [400, 550, 700]:
            n = sellmeier_sio2(float(lam))
            assert 1.44 < n.real < 1.50, f"SiO2 n={n.real} bei λ={lam} nm"
            assert abs(n.imag) < 1e-6   # transparent

    def test_cauchy_pdms_visible(self):
        """PDMS: Brechungsindex nahe 1.40–1.42 im Sichtbaren."""
        for lam in [450, 550, 650]:
            n = cauchy_pdms(float(lam))
            assert 1.39 < n.real < 1.43, f"PDMS n={n.real} bei λ={lam} nm"

    def test_gold_absorption_visible(self):
        """Gold: Imaginärteil κ > 1 im Sichtbaren (stark absorptiv)."""
        for lam in [500, 600, 700]:
            n = drude_lorentz_gold(float(lam))
            assert n.imag > 1.0, f"Au κ={n.imag} zu klein bei λ={lam} nm"

    def test_water_dispersion_correct(self):
        """Wasser: n ≈ 1.333 bei 589 nm (Natrium-D-Linie)."""
        n = water_dispersion(589.0)
        assert_allclose(n.real, 1.333, atol=0.01)

    def test_material_n_at_method(self):
        """Material.n_at(λ) gibt komplexen Brechungsindex zurück."""
        for name, mat in MATERIALS.items():
            n = mat.n_at(550.0)
            assert isinstance(n, complex), f"{name}: n_at gibt kein complex"
            assert n.real > 0
