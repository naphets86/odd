"""
uml_marte_stability.py

Erweitertes UML-MARTE-Profil mit Stabilitätsannotationen fuer Software-Systeme.
Implementiert Stereotypen, Tagged Values und Stabilitaetskriterien gemaess
der in Kapitel 9 beschriebenen Erweiterung.

Autor: Stephan Epp (Erweiterung)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Enumerationen fuer Stabilitaets-Annotationen
# ---------------------------------------------------------------------------

class StabilityMode(Enum):
    """Art des zu pruefenden Stabilitaetskriteriums."""
    LYAPUNOV_ASYMPTOTIC = auto()  # Asymptotisch stabil (Lyapunov)
    LYAPUNOV_MARGINAL    = auto()  # Marginal stabil
    BIBO                 = auto()  # Bounded-Input-Bounded-Output stabil
    SPECTRAL             = auto()  # Spektralradius < 1 (diskrete Systeme)
    QUEUE_UTILIZATION    = auto()  # Auslastung rho < 1 (Warteschlange)


class StabilityStatus(Enum):
    """Ergebnis einer Stabilitaetspruefung."""
    STABLE   = "stabil"
    UNSTABLE = "instabil"
    MARGINAL = "marginal"
    UNKNOWN  = "unbekannt"


# ---------------------------------------------------------------------------
# 2. Basis-Stereotypen fuer MARTE-Erweiterung
# ---------------------------------------------------------------------------

@dataclass
class MarteStereotype:
    """Basisklasse fuer alle MARTE-Stereotypen."""
    name: str
    base_metaclass: str  # z.B. 'Class', 'Operation', 'Component'


@dataclass
class TimingConstraint:
    """MARTE TimingConstraint – unveraenderter Bestandteil des Originals."""
    min_duration: float
    max_duration: float
    typical_duration: float
    unit: str = "ms"
    timeout: Optional[float] = None
    deadline: Optional[float] = None


# ---------------------------------------------------------------------------
# 3. Neue Stabilitaets-Stereotypen (Erweiterung von MARTE)
# ---------------------------------------------------------------------------

@dataclass
class StabilityAnnotation(MarteStereotype):
    """
    <<StabilityAnnotation>> – Stereotyp fuer MARTE-Klassen oder -Komponenten.

    Annotiert ein UML-Element mit dem geforderten Stabilitaetsmodus und
    den zugehoerigen quantitativen Schwellenwerten.

    Tagged Values:
        stability_mode     – welches Kriterium ist relevant
        lyapunov_margin    – erforderlicher Abstand zum Stabilitaetsrand
        decay_rate_min     – minimale Zerfallsrate alpha (nur Lyapunov)
        spectral_radius_ub – obere Schranke fuer Spektralradius
        rho_ub             – obere Schranke fuer Auslastung (Queue)
        monitoring_period  – Pruefintervall in ms
    """
    stability_mode: StabilityMode = StabilityMode.LYAPUNOV_ASYMPTOTIC
    lyapunov_margin: float = 0.05        # Sicherheitsabstand zum Rand
    decay_rate_min: float = 0.01         # Mindest-Zerfallsrate alpha > 0
    spectral_radius_ub: float = 0.99     # < 1 fuer diskrete Systeme
    rho_ub: float = 0.90                 # < 1 fuer Warteschlangen
    monitoring_period: float = 1000.0    # in ms


@dataclass
class DecayRateAnnotation(MarteStereotype):
    """
    <<DecayRateAnnotation>> – Stereotyp fuer MARTE-Operationen.

    Spezifiziert die erwartete exponentielle Zerfallsrate einer Operation,
    d.h. das System muss nach dem Aufruf exponentiell zum Gleichgewicht
    konvergieren: ||x(t)|| <= C * exp(-alpha * t) * ||x(0)||
    """
    alpha_min: float = 0.0   # Mindest-Zerfallsrate (alpha > 0 => asymptotisch stabil)
    alpha_max: float = float('inf')  # Maximale Zerfallsrate
    convergence_constant_ub: float = 10.0  # C in obiger Ungleichung


@dataclass
class LyapunovFunctionAnnotation(MarteStereotype):
    """
    <<LyapunovFunctionAnnotation>> – Stereotyp fuer MARTE-Klassen.

    Gibt an, welche Lyapunov-Funktion V(x) fuer den Stabilitaetsnachweis
    des annotierten Subsystems verwendet wird.

    Tagged Values:
        function_type – 'quadratic' (V=x'Px), 'entropy', 'custom'
        positive_definite_verified – wurde V > 0 formal nachgewiesen?
        derivative_negative – wurde dV/dt < 0 formal nachgewiesen?
    """
    function_type: str = "quadratic"
    positive_definite_verified: bool = False
    derivative_negative: bool = False
    custom_description: str = ""


@dataclass
class EquilibriumAnnotation(MarteStereotype):
    """
    <<EquilibriumAnnotation>> – Stereotyp fuer MARTE-Komponenten.

    Unterscheidet (wie in Kap. 3 formalisiert) zwischen
    Gleichgewichtsaussagen (algebraisch, kein e-Funktionen) und
    dynamischen Stabilitaetsaussagen.
    """
    is_equilibrium_only: bool = True   # True => nur Gleichgewicht (z.B. Little's Law)
    supports_stability_proof: bool = False  # True => dynamische Stabilitaet beweisbar
    equilibrium_law: str = ""          # z.B. "Little's Law: L = lambda * W"


# ---------------------------------------------------------------------------
# 4. Annotiertes UML-Element (Modell-Element mit mehreren Stereotypen)
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedElement:
    """
    Repraesentiert ein UML-Modellelement, das mit MARTE-Stereotypen (inkl.
    der neuen Stabilitaets-Stereotypen) annotiert ist.
    """
    element_name: str
    element_type: str  # 'Class', 'Component', 'Operation', ...
    timing: Optional[TimingConstraint] = None
    stability: Optional[StabilityAnnotation] = None
    decay: Optional[DecayRateAnnotation] = None
    lyapunov: Optional[LyapunovFunctionAnnotation] = None
    equilibrium: Optional[EquilibriumAnnotation] = None


# ---------------------------------------------------------------------------
# 5. Stabilitaetsanalyse-Engine
# ---------------------------------------------------------------------------

class StabilityAnalyzer:
    """
    Kernklasse zur Laufzeit-Pruefung der Stabilitaets-Annotationen.

    Unterstuetzt:
    - Lyapunov-Stabilitaet ueber Eigenwertanalyse kontinuierlicher Systeme
    - Spektralradius diskreter Systeme (Markov-Ketten)
    - Auslastungs-Stabilitaet fuer Warteschlangen (rho < 1)
    - BIBO-Stabilitaet (vereinfacht via Eigenwert-Realteile)
    """

    # -- 5a. Kontinuierliches System: A-Matrix (Zustandsraumdarstellung) ----

    @staticmethod
    def lyapunov_continuous(
        A: np.ndarray,
        annotation: StabilityAnnotation,
    ) -> Tuple[StabilityStatus, Dict]:
        """
        Prueft asymptotische Stabilitaet eines linearen kontinuierlichen
        Systems x' = A*x anhand der Eigenwerte von A.

        Stabil <=> alle Eigenwerte haben negativen Realteil.
        Decay-Rate alpha = -max(Re(eigenvalues)).
        """
        eigenvalues = np.linalg.eigvals(A)
        real_parts  = np.real(eigenvalues)
        max_real    = float(np.max(real_parts))
        alpha       = -max_real  # positiv => stabil

        if alpha > annotation.decay_rate_min + annotation.lyapunov_margin:
            status = StabilityStatus.STABLE
        elif alpha > annotation.decay_rate_min:
            status = StabilityStatus.MARGINAL
        else:
            status = StabilityStatus.UNSTABLE

        return status, {
            "eigenvalues": eigenvalues,
            "max_real_part": max_real,
            "decay_rate_alpha": alpha,
            "required_alpha_min": annotation.decay_rate_min,
            "margin": annotation.lyapunov_margin,
        }

    # -- 5b. Diskretes System: Uebergangsmatrix P (Markov-Kette) -----------

    @staticmethod
    def spectral_stability(
        P: np.ndarray,
        annotation: StabilityAnnotation,
    ) -> Tuple[StabilityStatus, Dict]:
        """
        Prueft Stabilitaet einer diskreten Markov-Kette P anhand des
        Spektralradius rho(P) = max |Eigenwert|.

        Stabil <=> rho(P) < 1  (exkl. des trivialen Eigenwerts 1 bei
        irreduziblen stochastischen Matrizen).
        """
        eigenvalues = np.linalg.eigvals(P)
        # Ignoriere den dominanten Eigenwert nahe 1 bei stochastischen Matrizen
        abs_eigs = np.abs(eigenvalues)
        abs_eigs_filtered = np.sort(abs_eigs)
        # Zweitgroesster Betrag bestimmt Konvergenzrate
        spectral_radius = float(abs_eigs_filtered[-2]) if len(abs_eigs_filtered) > 1 else float(abs_eigs_filtered[-1])
        alpha = -math.log(spectral_radius) if spectral_radius > 0 else float('inf')

        if spectral_radius < annotation.spectral_radius_ub:
            status = StabilityStatus.STABLE
        elif math.isclose(spectral_radius, 1.0, abs_tol=1e-6):
            status = StabilityStatus.MARGINAL
        else:
            status = StabilityStatus.UNSTABLE

        return status, {
            "eigenvalues": eigenvalues,
            "spectral_radius_sub": spectral_radius,
            "required_ub": annotation.spectral_radius_ub,
            "implied_decay_rate": alpha,
        }

    # -- 5c. Warteschlangen-Stabilitaet ------------------------------------

    @staticmethod
    def queue_stability(
        arrival_rate: float,
        service_rate: float,
        annotation: StabilityAnnotation,
    ) -> Tuple[StabilityStatus, Dict]:
        """
        Prueft Stabilitaet eines M/M/1-Systems: rho = lambda/mu < 1.

        Stabil <=> rho < annotation.rho_ub  (Sicherheitsabstand inklusive).
        """
        if service_rate <= 0:
            raise ValueError("service_rate muss positiv sein.")
        rho = arrival_rate / service_rate

        if rho < annotation.rho_ub:
            status = StabilityStatus.STABLE
        elif rho < 1.0:
            status = StabilityStatus.MARGINAL
        else:
            status = StabilityStatus.UNSTABLE

        return status, {
            "arrival_rate_lambda": arrival_rate,
            "service_rate_mu": service_rate,
            "utilization_rho": rho,
            "required_rho_ub": annotation.rho_ub,
            "margin_to_instability": 1.0 - rho,
        }

    # -- 5d. Generischer Dispatcher ----------------------------------------

    @classmethod
    def analyze(
        cls,
        element: AnnotatedElement,
        *,
        A: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        arrival_rate: Optional[float] = None,
        service_rate: Optional[float] = None,
    ) -> Dict:
        """
        Analysiert ein annotiertes UML-Element und gibt ein strukturiertes
        Stabilitaetsergebnis zurueck.
        """
        if element.stability is None:
            return {"status": StabilityStatus.UNKNOWN, "reason": "Keine StabilityAnnotation vorhanden."}

        ann = element.stability
        result = {"element": element.element_name, "mode": ann.stability_mode}

        if ann.stability_mode == StabilityMode.LYAPUNOV_ASYMPTOTIC:
            if A is None:
                raise ValueError("Systemmatrix A erforderlich fuer LYAPUNOV_ASYMPTOTIC.")
            status, details = cls.lyapunov_continuous(A, ann)

        elif ann.stability_mode == StabilityMode.SPECTRAL:
            if P is None:
                raise ValueError("Uebergangsmatrix P erforderlich fuer SPECTRAL.")
            status, details = cls.spectral_stability(P, ann)

        elif ann.stability_mode == StabilityMode.QUEUE_UTILIZATION:
            if arrival_rate is None or service_rate is None:
                raise ValueError("arrival_rate und service_rate erforderlich fuer QUEUE_UTILIZATION.")
            status, details = cls.queue_stability(arrival_rate, service_rate, ann)

        else:
            status, details = StabilityStatus.UNKNOWN, {"reason": "Modus nicht unterstuetzt."}

        result["status"]  = status
        result["details"] = details

        # Timing-Constraint-Pruefung
        if element.timing is not None:
            tc = element.timing
            if tc.deadline is not None:
                result["timing_deadline_ok"] = tc.typical_duration <= tc.deadline

        return result


# ---------------------------------------------------------------------------
# 6. Code-Generator fuer annotierte UML-Elemente
# ---------------------------------------------------------------------------

class StabilityAwareCodeGenerator:
    """
    Erweitert den in Kap. 4 beschriebenen zweistufigen Code-Generator um
    Stabilitaetspruefungen, die zur Laufzeit eingebettet werden.

    Erzeugt Python-Code-Fragmente fuer jedes annotierte Element.
    """

    @staticmethod
    def generate_class(element: AnnotatedElement) -> str:
        """Generiert eine Python-Klasse mit eingebetteter Stabilitaetspruefung."""
        lines: List[str] = []
        lines.append(f"class {element.element_name}:")
        lines.append(f'    """Generiert aus UML-Element <<{element.element_type}>>."""')
        lines.append("")

        # Konstruktor mit Stabilitaets-Prameter
        lines.append("    def __init__(self):")
        if element.stability:
            ann = element.stability
            lines.append(f"        self._stability_mode = StabilityMode.{ann.stability_mode.name}")
            lines.append(f"        self._rho_ub = {ann.rho_ub}")
            lines.append(f"        self._decay_rate_min = {ann.decay_rate_min}")
        if element.timing:
            tc = element.timing
            lines.append(f"        self._min_duration = {tc.min_duration}  # {tc.unit}")
            lines.append(f"        self._max_duration = {tc.max_duration}  # {tc.unit}")
            if tc.deadline is not None:
                lines.append(f"        self._deadline = {tc.deadline}  # {tc.unit}")
        lines.append("")

        # Stabilitaets-Check-Methode
        if element.stability:
            lines.append("    def check_stability(self, analyzer: StabilityAnalyzer, **kwargs) -> dict:")
            lines.append(f'        """Laufzeit-Stabilitaetspruefung fuer {element.element_name}."""')
            lines.append("        return StabilityAnalyzer.analyze(self._element, **kwargs)")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_stability_check_code(element: AnnotatedElement) -> str:
        """Generiert Inline-Stabilitaetspruefung als Code-String."""
        if element.stability is None:
            return "# Kein Stabilitaets-Stereotype vorhanden"
        ann = element.stability
        code_lines = [
            f"# <<StabilityAnnotation>> fuer {element.element_name}",
            f"# Modus: {ann.stability_mode.name}",
        ]
        if ann.stability_mode == StabilityMode.QUEUE_UTILIZATION:
            code_lines += [
                "rho = arrival_rate / service_rate",
                f"assert rho < {ann.rho_ub}, f'Stabilitaet verletzt: rho={{rho:.3f}} >= {ann.rho_ub}'",
            ]
        elif ann.stability_mode == StabilityMode.LYAPUNOV_ASYMPTOTIC:
            code_lines += [
                "eigenvalues = np.linalg.eigvals(A)",
                "alpha = -float(np.max(np.real(eigenvalues)))",
                f"assert alpha > {ann.decay_rate_min}, f'Zerfallsrate unzureichend: alpha={{alpha:.4f}}'",
            ]
        elif ann.stability_mode == StabilityMode.SPECTRAL:
            code_lines += [
                "spectral_radius = float(np.max(np.abs(np.linalg.eigvals(P))))",
                f"assert spectral_radius < {ann.spectral_radius_ub}, 'Spektralradius zu gross'",
            ]
        return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# 7. Demonstrations-Szenarien (direkt ausfuehrbar)
# ---------------------------------------------------------------------------

def demo_elevator_door_stability() -> None:
    """
    Erweiterte Aufzugstuer-Fallstudie mit Stabilitaets-Annotationen.
    Demonstriert alle drei Stabilitaetsmodi.
    """
    print("=" * 65)
    print("DEMO: Aufzugstuer-Steuerung mit Stabilitaets-Annotationen")
    print("=" * 65)

    # --- 7a. Warteschlangen-Stabilitaet ---
    queue_element = AnnotatedElement(
        element_name="ElevatorDoorController",
        element_type="Component",
        timing=TimingConstraint(
            min_duration=2500, max_duration=4000, typical_duration=3000,
            unit="ms", deadline=4000,
        ),
        stability=StabilityAnnotation(
            name="ElevatorDoorController_StabilityAnnotation",
            base_metaclass="Component",
            stability_mode=StabilityMode.QUEUE_UTILIZATION,
            rho_ub=0.90,
        ),
    )

    result = StabilityAnalyzer.analyze(
        queue_element, arrival_rate=0.2, service_rate=0.3
    )
    print(f"\n[1] {result['element']} – Modus: QUEUE_UTILIZATION")
    print(f"    Status : {result['status'].value}")
    d = result["details"]
    print(f"    rho    : {d['utilization_rho']:.3f}  (Grenze: {d['required_rho_ub']})")
    print(f"    Abstand: {d['margin_to_instability']:.3f}")
    print(f"    Deadline OK: {result.get('timing_deadline_ok', 'n/a')}")

    # --- 7b. Lyapunov-Stabilitaet (kontinuierliches Subsystem) ---
    lyap_element = AnnotatedElement(
        element_name="MotorDynamicsSubsystem",
        element_type="Class",
        stability=StabilityAnnotation(
            name="MotorDynamics_StabilityAnnotation",
            base_metaclass="Class",
            stability_mode=StabilityMode.LYAPUNOV_ASYMPTOTIC,
            decay_rate_min=0.5,
            lyapunov_margin=0.1,
        ),
        lyapunov=LyapunovFunctionAnnotation(
            name="MotorDynamics_LyapunovAnnotation",
            base_metaclass="Class",
            function_type="quadratic",
            positive_definite_verified=True,
            derivative_negative=True,
        ),
    )

    # Zustandsraummatrix: stabiles System mit Eigenwerten -1 und -2
    A = np.array([[-1.5,  0.3],
                  [ 0.1, -2.0]])
    result2 = StabilityAnalyzer.analyze(lyap_element, A=A)
    print(f"\n[2] {result2['element']} – Modus: LYAPUNOV_ASYMPTOTIC")
    print(f"    Status       : {result2['status'].value}")
    d2 = result2["details"]
    print(f"    Zerfallsrate : alpha = {d2['decay_rate_alpha']:.4f}")
    print(f"    Eigenwerte   : {np.round(d2['eigenvalues'], 4)}")
    print(f"    Lyapunov-Typ : {lyap_element.lyapunov.function_type}")

    # --- 7c. Markov-Kette (Spektral-Stabilitaet) ---
    markov_element = AnnotatedElement(
        element_name="DoorStateMarkovChain",
        element_type="Class",
        stability=StabilityAnnotation(
            name="DoorMarkov_StabilityAnnotation",
            base_metaclass="Class",
            stability_mode=StabilityMode.SPECTRAL,
            spectral_radius_ub=0.99,
        ),
        equilibrium=EquilibriumAnnotation(
            name="DoorMarkov_EquilibriumAnnotation",
            base_metaclass="Class",
            is_equilibrium_only=False,
            supports_stability_proof=True,
            equilibrium_law="pi * P = pi  (stationaere Verteilung)",
        ),
    )

    P = np.array([
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.0, 0.6, 0.4],
        [0.9, 0.0, 0.0, 0.1],
    ])
    result3 = StabilityAnalyzer.analyze(markov_element, P=P)
    print(f"\n[3] {result3['element']} – Modus: SPECTRAL")
    print(f"    Status          : {result3['status'].value}")
    d3 = result3["details"]
    print(f"    Spektralradius  : {d3['spectral_radius_sub']:.4f}  (Grenze: {d3['required_ub']})")
    print(f"    Zerfallsrate    : alpha = {d3['implied_decay_rate']:.4f}")
    print(f"    Gleichgewichtsg.: {markov_element.equilibrium.equilibrium_law}")

    # --- 7d. Code-Generierung ---
    print("\n" + "=" * 65)
    print("GENERIERTER STABILITAETS-CHECK-CODE (Queue):")
    print("=" * 65)
    print(StabilityAwareCodeGenerator.generate_stability_check_code(queue_element))

    print("\n" + "=" * 65)
    print("GENERIERTER STABILITAETS-CHECK-CODE (Lyapunov):")
    print("=" * 65)
    print(StabilityAwareCodeGenerator.generate_stability_check_code(lyap_element))


def demo_equilibrium_vs_stability() -> None:
    """
    Demonstriert die in Kap. 3 gezeigte Dichotomie:
    Gleichgewichtsaussagen (Little's Law) vs. Stabilitaetsaussagen.
    """
    print("\n" + "=" * 65)
    print("DEMO: Gleichgewicht vs. Stabilitaet")
    print("=" * 65)

    stable_ann = StabilityAnnotation(
        name="StableQueueAnnotation",
        base_metaclass="Component",
        stability_mode=StabilityMode.QUEUE_UTILIZATION,
        rho_ub=1.0,
    )
    eq_ann = EquilibriumAnnotation(
        name="LittlesLawAnnotation",
        base_metaclass="Component",
        is_equilibrium_only=True,
        supports_stability_proof=False,
        equilibrium_law="L = lambda * W  (Little's Law)",
    )

    scenarios = [
        ("Stabiles System",    0.7, 1.0),
        ("Instabiles System",  1.2, 1.0),
    ]

    for label, lam, mu in scenarios:
        rho = lam / mu
        stable_element = AnnotatedElement(
            element_name=f"MM1Queue_{label.replace(' ', '_')}",
            element_type="Component",
            stability=stable_ann,
            equilibrium=eq_ann,
        )
        res = StabilityAnalyzer.analyze(stable_element, arrival_rate=lam, service_rate=mu)
        print(f"\n  {label}: lambda={lam}, mu={mu}, rho={rho:.2f}")
        print(f"  Stabilitaetsstatus : {res['status'].value}")
        if rho < 1.0:
            W = 1.0 / (mu - lam)
            L = lam * W
            print(f"  Little's Law (W)   : {W:.4f} s")
            print(f"  Little's Law (L)   : {L:.4f} Kunden")
        else:
            print("  Little's Law       : nicht anwendbar (System instabil)")
        print(f"  Gleichgewicht only : {eq_ann.is_equilibrium_only}")
        print(f"  Stabilitaetsbeweis : {eq_ann.supports_stability_proof}")


if __name__ == "__main__":
    demo_elevator_door_stability()
    demo_equilibrium_vs_stability()
