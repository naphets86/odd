"""
polymer.py
=====================
Flory-Rehner-Quellungsmodell für vernetzte Polymernetzwerke.

Gleichungen:
  - Flory-Rehner-Gleichgewicht:
      ln(1−φ) + φ + χ·φ² + (Vm/Vc)·(φ^(1/3) − φ/2) = 0

  - Quellung-Brechungsindex-Kopplung:
      n(Q) ≈ 1 + (n₀ − 1) / Q

  - Resonanzwellenlänge:
      λ_res(Q) ≈ 2·d₀·Q  (für n₀ ≈ 1.4)

  - Quellungskinetik:
      ∂Q/∂t = D_eff ∇²Q − (Q − Q_eq)/τ_Q

  - Relaxationszeit:
      τ_Q ≈ h² / (π² D_eff)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PolymerParameters:
    """Materialparameter für ein vernetztes Polymernetzwerk.

    Parameters
    ----------
    chi : float
        Flory-Huggins-Wechselwirkungsparameter χ (dimensionslos).
        Typisch: 0.3 – 0.6 für PDMS/Lösungsmittel.
    Vm : float
        Molarvolumen des Lösungsmittels [m³/mol].
        Wasser: 1.8e-5 m³/mol; Ethanol: 5.8e-5 m³/mol.
    Vc : float
        Molares Kettenvolumen zwischen Vernetzungspunkten [m³/mol].
        Typisch: 5e-4 m³/mol (Mc≈600 g/mol, ρ≈1200 kg/m³).
        Steuert direkt den Quellungsgrad: größeres Vc → größeres Q_eq.
    n0 : float
        Brechungsindex des trockenen Polymers.
    D_eff : float
        Effektiver Diffusionskoeffizient [m²/s].
    """
    chi: float = 0.45
    Vm: float = 1.8e-5       # m³/mol (Wasser)
    Vc: float = 5.0e-4       # m³/mol  (Vc_molar = Mc/ρ, z.B. Mc=600 g/mol, ρ=1200 kg/m³)
    n0: float = 1.40
    D_eff: float = 1.0e-11   # m²/s


class FloryRehnerModel:
    """Flory-Rehner-Quellungsmodell.

    Parameters
    ----------
    params : PolymerParameters
        Materialparameter des Polymernetzwerks.
    """

    def __init__(self, params: Optional[PolymerParameters] = None, **kwargs) -> None:
        if params is not None:
            self.p = params
        else:
            # Ermöglicht direkte Keyword-Argumente
            self.p = PolymerParameters(**kwargs) if kwargs else PolymerParameters()

    # ── Gleichgewichtsquellung ─────────────────────────────────────────────────

    def _fr_equation(self, Q: float) -> float:
        """Flory-Rehner-Gleichung (Nullstelle gesucht).

        f(Q) = ln(1−φ) + φ + χ·φ² + (Vm/Vc)·(φ^(1/3) − φ/2) = 0
        mit φ = 1/Q.
        """
        if Q <= 1.0:
            return np.inf
        phi = 1.0 / Q
        ln_term = np.log(1.0 - phi) + phi + self.p.chi * phi**2
        elastic_term = (self.p.Vm / self.p.Vc) * (phi ** (1.0 / 3.0) - phi / 2.0)
        return ln_term + elastic_term

    def equilibrium_swelling(
        self,
        Q_min: float = 1.001,
        Q_max: float = 100.0,
    ) -> float:
        """Löst die Flory-Rehner-Gleichung numerisch nach Q_eq.

        Parameters
        ----------
        Q_min, Q_max : float
            Suchintervall für den Quellungsgrad.

        Returns
        -------
        float
            Gleichgewichts-Quellungsgrad Q_eq.

        Raises
        ------
        ValueError
            Wenn keine Lösung im Intervall gefunden wird.
        """
        f_min = self._fr_equation(Q_min)
        f_max = self._fr_equation(Q_max)
        if np.sign(f_min) == np.sign(f_max):
            # Kein Vorzeichenwechsel: Suche verfeinern
            for Q_try in np.linspace(Q_min, Q_max, 1000):
                if abs(self._fr_equation(Q_try)) < 1e-10:
                    return Q_try
            raise ValueError(
                f"Keine Lösung der FR-Gleichung in [{Q_min}, {Q_max}]. "
                f"f({Q_min})={f_min:.4f}, f({Q_max})={f_max:.4f}"
            )
        return brentq(self._fr_equation, Q_min, Q_max, xtol=1e-8)

    def swelling_vs_chi(
        self,
        chi_range: Tuple[float, float] = (0.1, 0.8),
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gleichgewichtsquellung als Funktion von χ.

        Parameters
        ----------
        chi_range : tuple
            (χ_min, χ_max).
        n_points : int
            Anzahl Stützstellen.

        Returns
        -------
        (chi_arr, Q_arr) : tuple[ndarray, ndarray]
        """
        chi_arr = np.linspace(chi_range[0], chi_range[1], n_points)
        Q_arr = np.zeros(n_points)
        for i, chi in enumerate(chi_arr):
            original_chi = self.p.chi
            self.p.chi = chi
            try:
                Q_arr[i] = self.equilibrium_swelling()
            except ValueError:
                Q_arr[i] = np.nan
            self.p.chi = original_chi
        return chi_arr, Q_arr

    # ── Optische Kopplung ──────────────────────────────────────────────────────

    def refractive_index(self, Q: float) -> float:
        """Brechungsindex n(Q) ≈ 1 + (n₀ − 1) / Q.

        Gleichung (n-quellung) der Arbeit.
        """
        return 1.0 + (self.p.n0 - 1.0) / Q

    def resonance_wavelength(self, d0: float, Q: float, order: int = 1) -> float:
        """Resonanzwellenlänge λ_res = 2 n(Q) d(Q) / order  [m].

        Parameters
        ----------
        d0 : float
            Trockene Schichtdicke [m].
        Q : float
            Quellungsgrad.
        order : int
            Beugungsordnung.

        Returns
        -------
        float
            Resonanzwellenlänge [m].
        """
        n = self.refractive_index(Q)
        d = Q * d0
        return 2.0 * n * d / order

    def tunable_range_nm(
        self,
        d0: float,
        Q_min: float = 1.0,
        Q_max: float = 4.0,
        order: int = 1,
    ) -> Tuple[float, float]:
        """Durchstimmbarer Wellenlängenbereich [nm].

        Returns
        -------
        (λ_min_nm, λ_max_nm) : tuple[float, float]
        """
        lam_min = self.resonance_wavelength(d0, Q_min, order) * 1e9
        lam_max = self.resonance_wavelength(d0, Q_max, order) * 1e9
        return min(lam_min, lam_max), max(lam_min, lam_max)

    # ── Quellungskinetik ───────────────────────────────────────────────────────

    def relaxation_time(self, h: float) -> float:
        """Relaxationszeit τ_Q ≈ h² / (π² D_eff)  [s].

        Parameters
        ----------
        h : float
            Schichtdicke [m].

        Returns
        -------
        float
            Relaxationszeit in Sekunden.
        """
        return h**2 / (np.pi**2 * self.p.D_eff)

    def kinetic_swelling(
        self,
        h: float,
        Q_init: float,
        Q_eq: float,
        t_max: Optional[float] = None,
        n_times: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zeitverlauf der Quellung Q(t) (monoexponentiell).

        Q(t) = Q_eq + (Q_init − Q_eq) · exp(−t / τ_Q)

        Parameters
        ----------
        h : float
            Schichtdicke [m].
        Q_init : float
            Anfangs-Quellungsgrad.
        Q_eq : float
            Gleichgewichts-Quellungsgrad.
        t_max : float, optional
            Maximalzeit [s]. Default: 5 τ_Q.
        n_times : int
            Anzahl Zeitstützpunkte.

        Returns
        -------
        (t, Q(t)) : tuple[ndarray, ndarray]
        """
        tau = self.relaxation_time(h)
        if t_max is None:
            t_max = 5.0 * tau
        t = np.linspace(0, t_max, n_times)
        Q_t = Q_eq + (Q_init - Q_eq) * np.exp(-t / tau)
        return t, Q_t

    def switching_time_ms(self, h: float) -> float:
        """Schaltzeit (1/e-Relaxationszeit) in Millisekunden.

        Parameters
        ----------
        h : float
            Schichtdicke [m].

        Returns
        -------
        float
            Schaltzeit in ms.
        """
        return self.relaxation_time(h) * 1e3

    # ── Lorentz-Lorenz ─────────────────────────────────────────────────────────

    def lorentz_lorenz_n(self, Q: float) -> float:
        """Brechungsindex via Lorentz-Lorenz-Relation.

        (n²−1)/(n²+2) = N·α / (3ε₀)  mit N ∝ 1/Q.
        Näherung für moderate Quellung.
        """
        # Clausius-Mossotti-Parameter aus Trockenzustand extrahieren
        n0 = self.p.n0
        CM_0 = (n0**2 - 1.0) / (n0**2 + 2.0)
        CM_Q = CM_0 / Q
        n_sq = (1.0 + 2.0 * CM_Q) / (1.0 - CM_Q)
        if n_sq < 0:
            return 1.0
        return np.sqrt(n_sq)

    def __repr__(self) -> str:
        try:
            Q_eq = self.equilibrium_swelling()
        except (ValueError, Exception):
            Q_eq = float("nan")
        return (
            f"FloryRehnerModel(χ={self.p.chi:.2f}, "
            f"Q_eq≈{Q_eq:.2f}, n(Q_eq)≈{self.refractive_index(Q_eq):.3f})"
        )
