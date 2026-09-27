"""Regler für das KI-Power-Management-Modul.

Implementiert:
  - PIDController (Gl. 22 aus liionp.tex) mit Integrator-Aufwickelschutz
    (Anti-Windup) und Lyapunov-Stabilitätsnachweis (Satz 2).
  - SimpleMPC – ein rasterbasierter Modellprädiktiver Regler (MPC)
    zur Spannungsregelung (Gl. 23).
  - PlateDistanceMPCController – Referenztrajektoriengenerator für den
    Plattenabstandskanal (Gl. 53, Abschnitt 6.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .mechanics import swelling_strain, PlateDistanceMPCController  # noqa: F401 (re-export)
from .constants import D_SEP_0, D_SEP_MIN, D_SEP_MAX


@dataclass
class PIDController:
    """PID-Regler mit Integrator-Aufwickelschutz (Anti-Windup) (Gl. 22).

    u(t) = K_p e(t) + K_i ∫ e dτ + K_d ė(t) + u_FF

    Attributes:
        Kp:             Proportionalverstärkung.
        Ki:             Integralverstärkung.
        Kd:             Differenzialverstärkung.
        output_min:     Unterer Sättigungswert.
        output_max:     Oberer Sättigungswert.
        integral_limit: Anti-Windup-Begrenzung des Integratorzustands.
    """

    Kp: float
    Ki: float
    Kd: float
    output_min: float = -np.inf
    output_max: float = np.inf
    integral_limit: float = 1e6

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        """Setzt Integrator und Ableitungsspeicher zurück."""
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint: float, measured: float, dt: float) -> float:
        """Berechnet den PID-Regelausgang.

        Args:
            setpoint: Sollwert.
            measured: Gemessener bzw. geschätzter Istwert.
            dt:       Zeitschritt [s]. Bei 0 wird der Differenzialanteil übersprungen.

        Returns:
            Gesättigter Regelausgang.
        """
        error = setpoint - measured
        self._integral = float(
            np.clip(
                self._integral + error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        derivative = (error - self._prev_error) / dt if dt > 0.0 else 0.0
        self._prev_error = error

        raw = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return float(np.clip(raw, self.output_min, self.output_max))

    def lyapunov_function(self, e: float, xi: float) -> float:
        """Lyapunov-Kandidat V_L(e, ξ) für den geschlossenen PI-Teilkreis.

        V_L = ½ e²  +  K_i / (2 K_p) · ξ²

        Verwendet im asymptotischen Stabilitätsbeweis (Satz 2 in liionp.tex).

        Args:
            e:  Regler-Regelabweichung e_T = T_set − T.
            xi: Integratorzustand ξ = ∫₀ᵗ e dτ.

        Returns:
            V_L ≥ 0.
        """
        return 0.5 * e**2 + (self.Ki / (2.0 * self.Kp)) * xi**2


@dataclass
class SimpleMPC:
    """Vereinfachter MPC zur Spannungs-/Stromregelung (Gl. 23).

    Die vollständige Optimierung (Gl. 23) wird durch eine grobe Rastersuche
    über den zulässigen Strombereich approximiert. Dies hält den Regler
    deterministisch und abhängigkeitsfrei, erfasst dabei aber den wesentlichen
    Zielkonflikt zwischen Spannungsführung, Stromregelung und Degradation.

    Attributes:
        N_p:      Prädiktionshorizont (informativ; Rastersuche verwendet 1 Schritt).
        q_V:      Gewichtung des Spannungsregelungsfehlers.
        r_I:      Gewichtung der Stromabweichung vom Referenzstrom.
        p_D:      Gewichtung der momentanen Degradationsrate.
        I_min:    Minimaler zulässiger Strom [A].
        I_max:    Maximaler zulässiger Strom [A].
        V_min:    Harte untere Spannungsgrenze [V].
        V_max:    Harte obere Spannungsgrenze [V].
        V_target: Komfortabler Betriebsspannungswert [V].
        I_ref:    Referenz-(Wunsch-)Strom [A].
        n_grid:   Anzahl der Kandidatenströme in der Rastersuche.
    """

    N_p: int = 10
    q_V: float = 1.0
    r_I: float = 0.1
    p_D: float = 0.5
    I_min: float = -5.0
    I_max: float = 5.0
    V_min: float = 2.5
    V_max: float = 4.2
    V_target: float = 4.0
    I_ref: float = 1.0
    n_grid: int = 21

    def compute_current(
        self,
        SoC: float,
        R_int: float,
        V_ocv_func: Callable[[float], float],
        deg_rate_func: Callable[[float, float], float],
        reduced_limit: bool = False,
    ) -> float:
        """Berechnet den optimalen Strom für den nächsten Zeitschritt.

        Args:
            SoC:           Aktueller Ladezustand ∈ [0, 1].
            R_int:         Innenwiderstand [Ω].
            V_ocv_func:    OCV(SoC) → Spannung [V].
            deg_rate_func: (V, I) → Degradationsrate.
            reduced_limit: KI-Modus — obere Spannungsgrenze um 50 mV verringern.

        Returns:
            Optimaler Strom [A], begrenzt auf [I_min, I_max].
        """
        V_max_eff = self.V_max - 0.05 if reduced_limit else self.V_max

        best_I = self.I_ref
        best_cost = np.inf

        for I_test in np.linspace(self.I_min, self.I_max, self.n_grid):
            ocv = V_ocv_func(SoC)
            V_pred = ocv - I_test * R_int

            cost = (
                self.q_V * (V_pred - self.V_target) ** 2
                + self.r_I * (I_test - self.I_ref) ** 2
                + self.p_D * deg_rate_func(V_pred, I_test)
            )
            if V_pred > V_max_eff or V_pred < self.V_min:
                cost += 1e6

            if cost < best_cost:
                best_cost = cost
                best_I = I_test

        return float(np.clip(best_I, self.I_min, self.I_max))
