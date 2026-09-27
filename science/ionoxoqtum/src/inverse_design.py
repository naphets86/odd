"""
inverse_design.py
================================
Inverses photonisches Design – Adjungierter Gradientenabstieg.

Formuliert das inverse Problem:
  Minimiere: L({d_j, n_j}) = Σ_λ [R_FP(λ; {d_j, n_j}) − R_Ziel(λ)]²

Lösung durch adjungierten Gradientenabstieg:
  ∂L/∂d_j = Σ_λ 2[R − R_Ziel] · Re[r* · ∂r/∂d_j] · 2/|r|

Algorithmus-Komplexität: O(I · K · Λ)
  I = Iterationen, K = Wellenlängen, Λ = Schichten
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from scipy.optimize import minimize

from src.fabry_perot import FabryPerotResonator
from src.tmm import TransferMatrixMethod, OpticalLayer


@dataclass
class OptimizationResult:
    """Ergebnis einer Inversdesign-Optimierung.

    Attributes
    ----------
    d_opt : ndarray
        Optimale Schichtdicken [m].
    n_opt : ndarray
        Optimale Brechungsindizes.
    loss_history : list[float]
        Verlauf der Kostenfunktion.
    n_iterations : int
        Tatsächliche Iterationszahl.
    converged : bool
        True wenn Konvergenzkritierium erfüllt.
    final_loss : float
        Endwert der Kostenfunktion.
    """
    d_opt: np.ndarray
    n_opt: np.ndarray
    loss_history: List[float]
    n_iterations: int
    converged: bool
    final_loss: float


class InversePhotonicsDesigner:
    """Inverses photonisches Design via adjungiertem Gradientenabstieg.

    Für ein gegebenes Ziel-Reflexionsspektrum werden Schichtdicken und
    Brechungsindizes eines Mehrschichtsystems optimiert.

    Parameters
    ----------
    target_spectrum : ndarray
        Ziel-Reflexionsspektrum R_Ziel(λ) [0,1].
    wavelengths_nm : ndarray
        Wellenlängen [nm] zu den Spektrumwerten.
    n_layers : int
        Anzahl zu optimierender Schichten.
    d_bounds : tuple
        (d_min, d_max) für Schichtdicken [m].
    n_bounds : tuple
        (n_min, n_max) für Brechungsindizes.
    """

    def __init__(
        self,
        target_spectrum: np.ndarray,
        wavelengths_nm: np.ndarray,
        n_layers: int = 3,
        d_bounds: Tuple[float, float] = (10e-9, 1000e-9),
        n_bounds: Tuple[float, float] = (1.0, 4.0),
    ) -> None:
        if len(target_spectrum) != len(wavelengths_nm):
            raise ValueError("target_spectrum und wavelengths_nm müssen gleich lang sein.")
        self.R_target = np.array(target_spectrum)
        self.lam_nm = np.array(wavelengths_nm)
        self.lam_m = self.lam_nm * 1e-9
        self.n_layers = n_layers
        self.d_bounds = d_bounds
        self.n_bounds = n_bounds
        self._loss_history: List[float] = []

    # ── Vorwärtssimulation ─────────────────────────────────────────────────────

    def _forward(self, d_arr: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
        """TMM-Vorwärtssimulation: R(λ) für gegebene Parameter.

        Parameters
        ----------
        d_arr : ndarray
            Schichtdicken [m], Länge n_layers.
        n_arr : ndarray
            Brechungsindizes, Länge n_layers.

        Returns
        -------
        ndarray
            Reflexionsspektrum R(λ).
        """
        layers = [OpticalLayer(n_arr[i], d_arr[i]) for i in range(self.n_layers)]
        medium    = OpticalLayer(1.0,  0.0)
        substrate = OpticalLayer(1.52, 0.0)
        tmm = TransferMatrixMethod(medium, layers, substrate)
        R = np.array([tmm.reflectance(l) for l in self.lam_m])
        return R

    # ── Kostenfunktion ─────────────────────────────────────────────────────────

    def loss(self, params: np.ndarray) -> float:
        """Quadratische Kostenfunktion L = Σ (R − R_Ziel)².

        Parameters
        ----------
        params : ndarray
            Flaches Array [d_1,...,d_N, n_1,...,n_N].

        Returns
        -------
        float
            Kostenwert.
        """
        d_arr = params[:self.n_layers]
        n_arr = params[self.n_layers:]
        R = self._forward(d_arr, n_arr)
        return float(np.sum((R - self.R_target)**2))

    def loss_gradient(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Kostenfunktion und numerischer Gradient (finite Differenzen).

        Für präzise adjungierte Gradienten würde man ∂r/∂d_j analytisch
        aus der TMM-Ableitung berechnen. Hier: numerische FD-Approximation.

        Parameters
        ----------
        params : ndarray
            Flaches Parameter-Array.

        Returns
        -------
        (loss, gradient) : tuple[float, ndarray]
        """
        L0 = self.loss(params)
        eps = 1e-10
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_h = params.copy()
            params_h[i] += eps
            grad[i] = (self.loss(params_h) - L0) / eps
        return L0, grad

    # ── Optimierung ────────────────────────────────────────────────────────────

    def optimize(
        self,
        n_restarts: int = 3,
        max_iter: int = 500,
        tol: float = 1e-8,
        seed: Optional[int] = 42,
    ) -> OptimizationResult:
        """Startet die Optimierung mit mehreren Zufallsinitialisierungen.

        Parameters
        ----------
        n_restarts : int
            Anzahl Neustarts (Multi-Start-Strategie).
        max_iter : int
            Maximale Iterationen pro Restart.
        tol : float
            Konvergenztoleranz.
        seed : int, optional
            Zufallszahl-Seed für Reproduzierbarkeit.

        Returns
        -------
        OptimizationResult
        """
        rng = np.random.default_rng(seed)
        best_result = None
        best_loss = np.inf
        self._loss_history = []

        bounds_d = [self.d_bounds] * self.n_layers
        bounds_n = [self.n_bounds] * self.n_layers
        all_bounds = bounds_d + bounds_n

        for restart in range(n_restarts):
            # Zufällige Initialisierung
            d_init = rng.uniform(self.d_bounds[0], self.d_bounds[1], self.n_layers)
            n_init = rng.uniform(self.n_bounds[0], self.n_bounds[1], self.n_layers)
            x0 = np.concatenate([d_init, n_init])

            history = []

            def callback(xk):
                L = self.loss(xk)
                history.append(L)

            result = minimize(
                self.loss,
                x0,
                method="L-BFGS-B",
                bounds=all_bounds,
                options={"maxiter": max_iter, "ftol": tol, "gtol": tol * 0.1},
                callback=callback,
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
                self._loss_history = history

        d_opt = best_result.x[:self.n_layers]
        n_opt = best_result.x[self.n_layers:]

        return OptimizationResult(
            d_opt=d_opt,
            n_opt=n_opt,
            loss_history=self._loss_history,
            n_iterations=len(self._loss_history),
            converged=(best_loss < tol * 100),
            final_loss=best_loss,
        )

    def predict_spectrum(self, result: OptimizationResult) -> np.ndarray:
        """Berechnet das Spektrum für optimierte Parameter.

        Parameters
        ----------
        result : OptimizationResult

        Returns
        -------
        ndarray
            Vorhergesagtes Spektrum R_opt(λ).
        """
        return self._forward(result.d_opt, result.n_opt)

    def design_quality(self, result: OptimizationResult) -> dict:
        """Bewertet die Designqualität.

        Returns
        -------
        dict
            Metriken: RMSE, max_error, R², converged.
        """
        R_pred = self.predict_spectrum(result)
        residuals = R_pred - self.R_target
        rmse = np.sqrt(np.mean(residuals**2))
        max_err = np.max(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.R_target - self.R_target.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "RMSE": rmse,
            "max_error": max_err,
            "R_squared": r2,
            "final_loss": result.final_loss,
            "converged": result.converged,
            "n_iterations": result.n_iterations,
        }


class SingleLayerDesigner:
    """Vereinfachtes Design für Einschicht-FP-Resonator.

    Findet d₀ und n₀ für eine Ziel-Resonanzwellenlänge.

    Parameters
    ----------
    target_wavelength_nm : float
        Gewünschte Resonanzwellenlänge [nm].
    order : int
        Beugungsordnung (default: 1).
    """

    def __init__(
        self,
        target_wavelength_nm: float = 550.0,
        order: int = 1,
    ) -> None:
        self.lam_target = target_wavelength_nm * 1e-9
        self.order = order

    def design_thickness(self, n: float) -> float:
        """Berechnet benötigte Schichtdicke für gegebenen Brechungsindex.

        d = m · λ / (2n)

        Parameters
        ----------
        n : float
            Brechungsindex.

        Returns
        -------
        float
            Schichtdicke [m].
        """
        return self.order * self.lam_target / (2.0 * n)

    def design_for_swelling(
        self,
        Q_target: float,
        n0: float = 1.40,
    ) -> float:
        """Berechnet d₀ für Resonanz bei Quellungsgrad Q.

        λ_res = 2 n(Q) d(Q) → d₀ = λ / (2 n(Q) Q)

        Parameters
        ----------
        Q_target : float
            Ziel-Quellungsgrad.
        n0 : float
            Brechungsindex des trockenen Polymers.

        Returns
        -------
        float
            Trockene Schichtdicke d₀ [m].
        """
        n_Q = 1.0 + (n0 - 1.0) / Q_target
        return self.lam_target / (2.0 * n_Q * Q_target)
