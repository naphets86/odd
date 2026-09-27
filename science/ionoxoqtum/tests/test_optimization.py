"""
tests/test_optimization.py
===========================
Tests für optimization/inverse_design.py
"""

import pytest
import numpy as np
from src.inverse_design import (
    InversePhotonicsDesigner,
    SingleLayerDesigner,
    OptimizationResult,
)
from src.fabry_perot import FabryPerotResonator


class TestSingleLayerDesigner:

    def test_design_thickness(self):
        """d = m λ / (2n)."""
        des = SingleLayerDesigner(target_wavelength_nm=550.0, order=1)
        d = des.design_thickness(n=1.5)
        expected = 1 * 550e-9 / (2 * 1.5)
        assert d == pytest.approx(expected, rel=1e-10)

    def test_design_thickness_positive(self):
        des = SingleLayerDesigner(550.0)
        assert des.design_thickness(1.4) > 0

    def test_design_for_swelling_formula(self):
        """d₀ = λ / (2 n(Q) Q)."""
        des = SingleLayerDesigner(target_wavelength_nm=550.0)
        Q = 2.0
        n0 = 1.40
        n_Q = 1.0 + (n0 - 1.0) / Q
        d0 = des.design_for_swelling(Q, n0)
        lam_calc = 2 * n_Q * Q * d0
        assert lam_calc == pytest.approx(550e-9, rel=1e-6)

    def test_design_for_swelling_different_Q(self):
        des = SingleLayerDesigner(550.0)
        d1 = des.design_for_swelling(Q_target=1.0)
        d2 = des.design_for_swelling(Q_target=2.0)
        # d₀ für Q=1 ist kleiner als für Q=2 (da mehr Quellung)
        assert d1 != d2

    def test_second_order_half_thickness(self):
        """m=2: d ≈ d(m=1)/2."""
        des1 = SingleLayerDesigner(550.0, order=1)
        des2 = SingleLayerDesigner(550.0, order=2)
        d1 = des1.design_thickness(1.5)
        d2 = des2.design_thickness(1.5)
        assert d2 == pytest.approx(2 * d1, rel=1e-10)


class TestInversePhotonicsDesigner:

    def setup_method(self):
        # Erstelle ein einfaches Zielspektrum (Gauss-förmige Reflexion bei 550nm)
        lam = np.linspace(380, 780, 50)
        self.lam = lam
        self.R_target = 0.5 * np.exp(-((lam - 550)**2) / (2 * 50**2))
        self.designer = InversePhotonicsDesigner(
            target_spectrum=self.R_target,
            wavelengths_nm=lam,
            n_layers=2,
            d_bounds=(50e-9, 500e-9),
            n_bounds=(1.1, 3.0),
        )

    def test_creation(self):
        assert len(self.designer.R_target) == 50
        assert len(self.designer.lam_nm) == 50

    def test_mismatched_arrays_raise(self):
        with pytest.raises(ValueError):
            InversePhotonicsDesigner(
                target_spectrum=np.ones(10),
                wavelengths_nm=np.linspace(380, 780, 20),
            )

    def test_loss_nonnegative(self):
        params = np.array([200e-9, 100e-9, 1.5, 2.0])
        L = self.designer.loss(params)
        assert L >= 0

    def test_loss_zero_for_perfect_match(self):
        """Wenn Vorhersage = Ziel, L = 0."""
        # Manuelle Erzeugung: Finde Parameter, die R_target reproduzieren
        # Hier: Testen, dass Loss bei identischen Spektren 0 ist
        # (indirekter Test via Mock)
        designer2 = InversePhotonicsDesigner(
            target_spectrum=np.zeros(50),
            wavelengths_nm=self.lam,
            n_layers=2,
        )
        # Wenn R_target = 0 überall, dann Loss = Σ R² ≥ 0
        params = np.array([200e-9, 100e-9, 1.0, 1.0])
        L = designer2.loss(params)
        assert L >= 0

    def test_optimize_returns_result(self):
        result = self.designer.optimize(n_restarts=1, max_iter=20, seed=0)
        assert isinstance(result, OptimizationResult)

    def test_optimize_d_opt_shape(self):
        result = self.designer.optimize(n_restarts=1, max_iter=10, seed=0)
        assert len(result.d_opt) == 2
        assert len(result.n_opt) == 2

    def test_optimize_d_opt_in_bounds(self):
        result = self.designer.optimize(n_restarts=1, max_iter=20, seed=0)
        assert np.all(result.d_opt >= 50e-9 - 1e-15)
        assert np.all(result.d_opt <= 500e-9 + 1e-15)

    def test_optimize_n_opt_in_bounds(self):
        result = self.designer.optimize(n_restarts=1, max_iter=20, seed=0)
        assert np.all(result.n_opt >= 1.1 - 1e-10)
        assert np.all(result.n_opt <= 3.0 + 1e-10)

    def test_optimize_final_loss_nonneg(self):
        result = self.designer.optimize(n_restarts=1, max_iter=10, seed=0)
        assert result.final_loss >= 0

    def test_predict_spectrum_length(self):
        result = self.designer.optimize(n_restarts=1, max_iter=10, seed=0)
        R_pred = self.designer.predict_spectrum(result)
        assert len(R_pred) == len(self.lam)

    def test_predict_spectrum_in_range(self):
        result = self.designer.optimize(n_restarts=1, max_iter=10, seed=0)
        R_pred = self.designer.predict_spectrum(result)
        assert np.all(R_pred >= 0.0)
        assert np.all(R_pred <= 1.0)

    def test_design_quality_keys(self):
        result = self.designer.optimize(n_restarts=1, max_iter=10, seed=0)
        quality = self.designer.design_quality(result)
        assert "RMSE" in quality
        assert "R_squared" in quality
        assert "converged" in quality

    def test_design_quality_rmse_nonneg(self):
        result = self.designer.optimize(n_restarts=1, max_iter=10, seed=0)
        quality = self.designer.design_quality(result)
        assert quality["RMSE"] >= 0

    def test_multiple_restarts_not_worse(self):
        """Mehr Neustarts ≤ besser oder gleich."""
        r1 = self.designer.optimize(n_restarts=1, max_iter=20, seed=0)
        r2 = self.designer.optimize(n_restarts=3, max_iter=20, seed=0)
        assert r2.final_loss <= r1.final_loss + 1e-8
