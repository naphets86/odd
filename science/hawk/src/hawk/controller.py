"""BioFalcon-Zielverfolgungsregler — Algorithmus 1 der wissenschaftlichen Arbeit.

Dieses Modul implementiert den vollständigen Regelungsalgorithmus des
bioinspirierten Sturzangriffssystems:

- **Phase** (Definition 2.3): Fünf biomechanisch distinkte Sturzflugphasen.
- **BGSGController** (Definition 5.3): Bioinspiriertes Gleitflächen-Steuergesetz
  mit global asymptotischer Stabilitätsgarantie (Hauptlemma 4.2).
- **proportional_navigation** (Abschnitt 5.5): Proportionalnavigation für den
  Fangschlag (Phase P5).
- **BioFalconTracker** (Algorithmus 1): Vollständiger Tracking-Algorithmus
  analog zur Augenbewegungsstrategie des Wanderfalken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from hawk.aerodynamics import air_density, reference_area


# ---------------------------------------------------------------------------
# Flight phases  (Definition 2.3)
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    """Fünf biomechanisch distinkte Phasen des Wanderfalken-Sturzfluges (Definition 2.3).

    Attributes
    ----------
    P1 : int
        Aufwärtspositionierung — der Falke gewinnt an Höhe und positioniert sich
        oberhalb des Ziels.
    P2 : int
        Sturzeinleitung — Übergang in die Sturzposition, Flügel werden teilweise
        angelegt.
    P3 : int
        Freier Sturzflug — maximale Beschleunigung mit vollständig gefalteten
        Flügeln (β_fold = 0).
    P4 : int
        Pullout-Manöver — graduelles Ausfahren der Flügel zur Flugbahnkorrektur.
    P5 : int
        Fangschlag — Endanflug mit Proportionalnavigation auf das Ziel.
    """

    P1 = 1  # Aufwärtspositionierung
    P2 = 2  # Sturzeinleitung
    P3 = 3  # Freier Sturzflug
    P4 = 4  # Pullout-Manöver
    P5 = 5  # Fangschlag


# ---------------------------------------------------------------------------
# Proportional Navigation  (Section 5.5)
# ---------------------------------------------------------------------------

def proportional_navigation(
    p_target: np.ndarray,
    p_self: np.ndarray,
    v_rel: np.ndarray,
    nav_gain: float = 4.0,
) -> np.ndarray:
    """Berechne den Proportionalnavigations-Steuervektor für Phase P5 (Fangschlag).

    Proportionalnavigation (PN) liefert den kommandierten Beschleunigungsvektor:

        n_c = N' · v_c · (ω_λ × los_hat)

    wobei v_c die Schließgeschwindigkeit und ω_λ die Sichtlinienrotationsrate ist.
    Das PN-Gesetz garantiert Zielerfassung, wenn N' ≥ 2·(v_target/v_drone) + 1
    (Satz 5.1). Für Optimalität gilt 3 ≤ N' ≤ 5.

    Parameters
    ----------
    p_target : array-like, shape (3,)
        Zielposition [m].
    p_self : array-like, shape (3,)
        Eigene Drohnenposition [m].
    v_rel : array-like, shape (3,)
        Relativgeschwindigkeit v_target − v_self [m/s].
    nav_gain : float
        Navigationskonstante N' (Standard: 4.0).

    Returns
    -------
    np.ndarray, shape (3,)
        Kommandierter lateraler Beschleunigungsvektor n_c [m/s²].
    """
    p_target = np.asarray(p_target, dtype=float)
    p_self = np.asarray(p_self, dtype=float)
    v_rel = np.asarray(v_rel, dtype=float)

    los = p_target - p_self
    los_norm = float(np.linalg.norm(los))
    if los_norm < 1e-12:        # already at target — no command
        return np.zeros(3)

    los_hat = los / los_norm

    # Closing speed: positive when range is decreasing
    v_closing = -float(np.dot(v_rel, los_hat))

    # LOS rotation rate vector  ω_λ = (los × v_rel) / |los|²
    omega_los = np.cross(los, v_rel) / (los_norm ** 2)

    return nav_gain * v_closing * np.cross(omega_los, los_hat)


# ---------------------------------------------------------------------------
# BGSG controller  (Definition 5.3 + Hauptlemma 4.2)
# ---------------------------------------------------------------------------

@dataclass
class BGSGController:
    """Bioinspiriertes Gleitflächen-Steuergesetz (BGSG) — Regler (Definition 5.3).

    Der BGSG-Regler kombiniert einen Gleitflächen-Schaltanteil mit einem
    PID-Äquivalentanteil:

        s(t)       = ė(t) + Λ·e(t)                   (Gleitfläche)
        u_sw(t)    = −K_sw · sign(s(t))               (Schaltanteil)
        u_eq(t)    = −K_p·e − K_d·ė − K_i·∫e dτ     (PID-Äquivalent)
        u_BGSG(t)  = u_eq(t) + u_sw(t) [+ u_ff]      (Gesamtsteuerung)

    Globale asymptotische Stabilität ist für alle Anfangsbedingungen garantiert,
    wenn die Verstärkungsbedingung K_p·K_d > K_i·I erfüllt ist (Hauptlemma 4.2).

    Attributes
    ----------
    k_p : float
        Proportionalverstärkung K_p (Standard: 4.0).
    k_d : float
        Differentialverstärkung K_d (Standard: 2.0).
    k_i : float
        Integralverstärkung K_i (Standard: 0.5).
    k_sw : float
        Schaltverstärkung K_sw (Standard: 1.0).
    lambda_ : float
        Gleitflächenneigung Λ (Standard: 2.0).
    """

    k_p: float = 4.0
    k_d: float = 2.0
    k_i: float = 0.5
    k_sw: float = 1.0
    lambda_: float = 2.0

    _integral: np.ndarray = field(
        default_factory=lambda: np.zeros(3),
        init=False,
        repr=False,
    )

    def reset(self) -> None:
        """Setze den Integratorzustand auf null zurück."""
        self._integral = np.zeros_like(self._integral)

    def sliding_surface(
        self,
        error: np.ndarray,
        error_dot: np.ndarray,
    ) -> np.ndarray:
        """Berechne den Gleitflächenvektor s = ė + Λ·e.

        Parameters
        ----------
        error : array-like, shape (n,)
            Lagefehler e = r_soll − r_ist [m].
        error_dot : array-like, shape (n,)
            Geschwindigkeitsfehler ė = v_soll − v_ist [m/s].

        Returns
        -------
        np.ndarray, shape (n,)
            Gleitflächenvektor s(t) [m/s].
        """
        return np.asarray(error_dot, dtype=float) + self.lambda_ * np.asarray(error, dtype=float)

    def compute(
        self,
        error: np.ndarray,
        error_dot: np.ndarray,
        dt: float,
        feedforward: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Berechne den BGSG-Steuerausgang u_BGSG [m/s²].

        Führt einen Regelschritt durch: Integralakkumulation, Gleitflächen-
        berechnung, Schalt- und Äquivalentanteil sowie optionaler Vorsteuerung.

        Parameters
        ----------
        error : array-like, shape (n,)
            Lagefehler e = r_soll − r_ist [m].
        error_dot : array-like, shape (n,)
            Geschwindigkeitsfehler ė = v_soll − v_ist [m/s].
        dt : float
            Integrationsschrittweite Δt [s].
        feedforward : array-like, shape (n,), optional
            Externer Vorsteuerungsterm u_ff [m/s²], z. B. adaptive
            Widerstandskompensation.

        Returns
        -------
        np.ndarray, shape (n,)
            Steuerbeschleunigungsvektor u_BGSG(t) [m/s²].
        """
        error = np.asarray(error, dtype=float)
        error_dot = np.asarray(error_dot, dtype=float)

        # Lazy re-initialise integrator if dimension changes
        if self._integral.shape != error.shape:
            self._integral = np.zeros_like(error)

        # Accumulate integral  ξ += e·Δt
        self._integral = self._integral + error * dt

        # Sliding surface  s = ė + Λ·e
        s = self.sliding_surface(error, error_dot)

        # Switching term (discontinuous)
        u_sw = -self.k_sw * np.sign(s)

        # PID equivalent term
        u_eq = -self.k_p * error - self.k_d * error_dot - self.k_i * self._integral

        u = u_eq + u_sw
        if feedforward is not None:
            u = u + np.asarray(feedforward, dtype=float)
        return u


# ---------------------------------------------------------------------------
# Algorithm 1: BioFalcon-Tracking-Algorithmus
# ---------------------------------------------------------------------------

@dataclass
class BioFalconTracker:
    """Algorithmus 1: BioFalcon-Tracking-Algorithmus.

    Implementiert den vollständigen Zielverfolgungsalgorithmus analog zur
    Augenbewegungsstrategie des Wanderfalken. Pro Zeitschritt werden
    Eingangsgrössen verarbeitet und der optimale Steuervektor bestimmt:

    **Eingabe** je Zeitschritt:
        p_target(t), p_self(t), v(t), phase, t

    **Ausgabe** je Zeitschritt:
        u*(t) — optimaler Steuerbeschleunigungsvektor [m/s²]
        β_fold — Flügelfaltungsgrad ∈ [0, 1]

    **Phasenlogik** (Algorithmus 1, Schritte 5–13):

    - **Phase P3** (Freier Sturzflug): Flügel vollständig gefaltet (β_fold = 0),
      BGSG-Regler mit adaptiver Widerstandsvorsteuerung.
    - **Phase P4** (Pullout): Graduelles Flügelausfahren
      β_fold = min(1, (t − t_po) / τ_po), BGSG mit Auftriebsvorsteuerung.
    - **Phase P5** (Fangschlag): Flügel vollständig geöffnet (β_fold = 1),
      Proportionalnavigation.
    - **Phase P1/P2**: Kein aktiver Regeleingriff (u* = 0).

    Attributes
    ----------
    mass : float
        Drohnenmasse [kg] (Standard: 1.2).
    tau_pullout : float
        Pullout-Zeitkonstante τ_po [s] (Standard: 1.2).
    nav_gain : float
        Proportionalnavigationskonstante N' (Standard: 4.0).
    controller : BGSGController
        BGSG-Reglerinstanz.
    """

    mass: float = 1.2
    tau_pullout: float = 1.2
    nav_gain: float = 4.0
    controller: BGSGController = field(default_factory=BGSGController)

    # ------------------------------------------------------------------
    # Sub-steps exposed for direct testing
    # ------------------------------------------------------------------

    def compute_relative_vector(
        self,
        p_target: np.ndarray,
        p_self: np.ndarray,
    ) -> np.ndarray:
        """Algorithmus 1, Schritt 3: Berechne Relativvektor Δp ← p_target − p_self.

        Parameters
        ----------
        p_target : array-like, shape (3,)
            Zielposition [m].
        p_self : array-like, shape (3,)
            Eigene Drohnenposition [m].

        Returns
        -------
        np.ndarray, shape (3,)
            Relativvektor Δp [m].
        """
        return np.asarray(p_target, dtype=float) - np.asarray(p_self, dtype=float)

    def compute_log_spiral_param(
        self,
        delta_p: np.ndarray,
        v_mag: float,
        t_go: float,
    ) -> float:
        """Algorithmus 1, Schritt 4: Berechne Log-Spiralparameter κ.

            κ = arctan(|Δp| / (v · t_go))

        Dieser Parameter beschreibt die Krümmung der optimalen Annäherungs-
        kurve analog zur logarithmischen Spiraltrajektorie des Wanderfalken.
        Gibt π/2 zurück, wenn der Nenner null ist (|Δp|/0 → ∞).

        Parameters
        ----------
        delta_p : array-like, shape (3,)
            Relativvektor Δp [m].
        v_mag : float
            Betrag der Eigengeschwindigkeit |v| [m/s].
        t_go : float
            Geschätzte Restflugzeit bis zum Zielpunkt [s].

        Returns
        -------
        float
            Log-Spiralparameter κ [rad].
        """
        denom = v_mag * t_go
        if abs(denom) < 1e-12:
            return math.pi / 2.0
        return math.atan(float(np.linalg.norm(delta_p)) / denom)

    # ------------------------------------------------------------------
    # Internal feedforward helper
    # ------------------------------------------------------------------

    def _feedforward(
        self,
        c_d_hat: float,
        v: np.ndarray,
        h: float,
        beta_fold: float,
    ) -> np.ndarray:
        """Berechne die adaptive Widerstandsvorsteuerung entlang der Geschwindigkeitsrichtung.

            u_ff = Ĉ_D · (ρ · v² · A_ref) / (2m) · v̂

        Gibt den Nullvektor zurück, wenn die Geschwindigkeit vernachlässigbar ist.

        Parameters
        ----------
        c_d_hat : float
            Adaptiver Schätzwert des Widerstandsbeiwertes Ĉ_D.
        v : array-like, shape (3,)
            Eigengeschwindigkeitsvektor [m/s].
        h : float
            Aktuelle Höhe über NN [m].
        beta_fold : float
            Flügelfaltungsgrad β_fold ∈ [0, 1].

        Returns
        -------
        np.ndarray, shape (3,)
            Vorsteuerungsbeschleunigung u_ff [m/s²].
        """
        v_mag = float(np.linalg.norm(v))
        if v_mag < 1e-12:
            return np.zeros(3)
        rho = air_density(h)
        a_ref = reference_area(beta_fold)
        mag = c_d_hat * (rho * v_mag ** 2 * a_ref) / (2.0 * self.mass)
        return mag * (v / v_mag)

    # ------------------------------------------------------------------
    # Main step  (Algorithm 1)
    # ------------------------------------------------------------------

    def step(
        self,
        p_target: np.ndarray,
        p_self: np.ndarray,
        v: np.ndarray,
        v_target: np.ndarray,
        phase: Phase,
        t: float,
        t_pullout_start: float,
        dt: float,
        h: float = 0.0,
        c_d_hat: float = 0.032,
        t_go: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Führe einen Zeitschritt von Algorithmus 1 aus.

        Verarbeitet die aktuellen Zustandsgrößen und liefert phasenabhängig
        den optimalen Steuervektor u*(t) sowie den Flügelfaltungsgrad β_fold.

        Parameters
        ----------
        p_target : array-like, shape (3,)
            Zielposition p_target(t) [m].
        p_self : array-like, shape (3,)
            Eigene Drohnenposition p_self(t) [m].
        v : array-like, shape (3,)
            Eigener Geschwindigkeitsvektor v(t) [m/s].
        v_target : array-like, shape (3,)
            Geschwindigkeitsvektor des Ziels [m/s].
        phase : Phase
            Aktuelle Sturzflugphase (P3, P4 oder P5; P1/P2 → Nullausgabe).
        t : float
            Aktuelle Simulationszeit [s].
        t_pullout_start : float
            Startzeitpunkt der Phase P4 (Pullout-Beginn) [s].
        dt : float
            Integrationsschrittweite Δt [s].
        h : float
            Aktuelle Höhe über NN [m].
        c_d_hat : float
            Adaptiver Schätzwert des Widerstandsbeiwertes Ĉ_D.
        t_go : float
            Geschätzte Restflugzeit bis zum Zielpunkt [s].

        Returns
        -------
        u_star : np.ndarray, shape (3,)
            Optimaler Steuerbeschleunigungsvektor u*(t) [m/s²].
        beta_fold : float
            Flügelfaltungsgrad β_fold ∈ [0, 1].
        """
        p_target = np.asarray(p_target, dtype=float)
        p_self = np.asarray(p_self, dtype=float)
        v = np.asarray(v, dtype=float)
        v_target = np.asarray(v_target, dtype=float)

        # --- Step 3: relative vector ---
        delta_p = self.compute_relative_vector(p_target, p_self)

        # --- Step 4: log-spiral parameter (κ retained per algorithm) ---
        v_mag = float(np.linalg.norm(v))
        _kappa = self.compute_log_spiral_param(delta_p, v_mag, t_go)  # noqa: F841

        # Error terms for BGSG
        error = delta_p
        error_dot = v_target - v

        # --- Steps 5-13: phase-dependent control ---
        if phase == Phase.P3:
            # Freier Sturzflug — wings fully folded
            beta_fold = 0.0
            u_ff = self._feedforward(c_d_hat, v, h, beta_fold)
            u_star = self.controller.compute(error, error_dot, dt, feedforward=u_ff)

        elif phase == Phase.P4:
            # Pullout — gradual wing extension
            beta_fold = min(1.0, (t - t_pullout_start) / self.tau_pullout)
            # Lift feedforward opposes drag direction
            u_ff_lift = -self._feedforward(c_d_hat, v, h, beta_fold)
            u_star = self.controller.compute(error, error_dot, dt, feedforward=u_ff_lift)

        elif phase == Phase.P5:
            # Fangschlag — proportional navigation
            beta_fold = 1.0
            v_rel = v_target - v
            u_star = proportional_navigation(p_target, p_self, v_rel, self.nav_gain)

        else:
            # Phases P1 / P2 — no active dive control
            beta_fold = 1.0
            u_star = np.zeros(3)

        return u_star, beta_fold
