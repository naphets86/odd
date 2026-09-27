"""
Wahrscheinlichkeitsverschiebungs-Analyse: Formale Analyse der Verschiebung
von Wahrscheinlichkeitsverteilungen bei der Lösung von SAT-Problemen.

Implementiert die drei Phasen:
- Phase 1: Logarithmische Laufzeit O(log n)
- Phase 2: Übergansphase mit superlinearer Konvergenz
- Phase 3: Annäherung an Worst-Case O(n^3)
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class ProbabilityDistribution:
    """Beschreibt die Wahrscheinlichkeitsverteilung der Laufzeit."""
    
    mean: float
    variance: float
    mode1: float = None  # Für bimodale Verteilungen
    mode2: float = None
    density_function = None
    
    def standard_deviation(self) -> float:
        """Berechnet die Standardabweichung."""
        return math.sqrt(self.variance)


class ProbabilityShiftAnalyzer:
    """
    Analysiert die Verschiebung der Wahrscheinlichkeitsverteilung
    über den Versuchsverlauf.
    """
    
    # Parameter aus der Arbeit
    LAMBDA = 0.7  # Konvergenz-Parameter
    ALPHA = 1.5   # Übergangs-Exponent
    MU = 0.5      # Konvergenz-Konstante
    
    def __init__(self, n: int, formula_count: int = None):
        """
        Initialisiert den Analyzer.
        
        Args:
            n: Anzahl Variablen
            formula_count: Anzahl zu lösender Formeln (standard: 2^n)
        """
        self.n = n
        self.max_formulas = formula_count or (2 ** n)
        self.pyramid_size = 2 ** n
        
        # Phasengrenzen
        self.phase1_end = math.sqrt(self.pyramid_size)  # Erste sqrt(2^n) Formeln
        self.phase2_end = self.pyramid_size / 2
    
    def get_phase(self, k: int) -> int:
        """
        Bestimmt die aktuelle Phase basierend auf Formel-Index k.
        
        Args:
            k: Formel-Index (1-basiert)
            
        Returns:
            Phasen-Nummer: 1, 2 oder 3
        """
        if k <= self.phase1_end:
            return 1
        elif k <= self.phase2_end:
            return 2
        else:
            return 3
    
    def phase1_expected_runtime(self, k: int) -> float:
        """
        Erwartete Laufzeit in Phase 1 (Logarithmisch).
        
        Satz: E[T_k] ∈ O(log n) für k ≤ sqrt(2^n)
        
        Args:
            k: Formel-Index
            
        Returns:
            Erwartete Laufzeit in Schritten
        """
        # Logarithmische Laufzeit mit kleiner Konstante
        return math.log(self.n, 3) + np.random.normal(0, 0.1)
    
    def phase2_expected_runtime(self, k: int, base_runtime: float = None) -> float:
        """
        Erwartete Laufzeit in Phase 2 (Übergansphase).
        
        Satz: E[T_k] = E[T_1] * (1 + λ * k/sqrt(2^n))^α
        
        Args:
            k: Formel-Index
            base_runtime: Baseline-Laufzeit aus Phase 1
            
        Returns:
            Erwartete Laufzeit
        """
        if base_runtime is None:
            base_runtime = math.log(self.n, 3)
        
        normalized_k = k / self.phase1_end
        factor = (1 + self.LAMBDA * normalized_k) ** self.ALPHA
        
        return base_runtime * factor
    
    def phase3_expected_runtime(self, k: int) -> float:
        """
        Erwartete Laufzeit in Phase 3 (Asymptotische Annäherung).
        
        Satz: E[T_k] = n^3 - O(n^3 * e^(-μ(2^n - k)))
        
        Args:
            k: Formel-Index
            
        Returns:
            Erwartete Laufzeit in Schritten
        """
        worst_case = self.n ** 3
        
        # Exponentieller Abfall der Abweichung
        remaining = self.pyramid_size - k
        convergence_term = worst_case * np.exp(-self.MU * remaining / self.pyramid_size)
        
        return worst_case - convergence_term
    
    def expected_runtime(self, k: int) -> float:
        """
        Berechnet die erwartete Laufzeit für die k-te Formel.
        
        Args:
            k: Formel-Index (1-basiert)
            
        Returns:
            Erwartete Laufzeit
        """
        phase = self.get_phase(k)
        
        if phase == 1:
            return self.phase1_expected_runtime(k)
        elif phase == 2:
            return self.phase2_expected_runtime(k)
        else:
            return self.phase3_expected_runtime(k)
    
    def bimodal_distribution(self, k: int) -> ProbabilityDistribution:
        """
        Beschreibt die bimodale Wahrscheinlichkeitsverteilung in Phase 1.
        
        Modus 1: Um log(n) - schnelle Lösungen
        Modus 2: Um n^3 - Worst-Case-Lösungen
        
        Args:
            k: Formel-Index
            
        Returns:
            Wahrscheinlichkeitsverteilung mit zwei Modi
        """
        if self.get_phase(k) != 1:
            # Nicht bimodal in anderen Phasen
            mean = self.expected_runtime(k)
            return ProbabilityDistribution(mean=mean, variance=mean * 0.1)
        
        # Phase 1: Bimodale Verteilung
        mode1 = math.log(self.n, 3)
        mode2 = self.n ** 3
        
        # Gewichte
        weight_fast = 0.7  # Meisten Probleme sind schnell
        weight_slow = 0.3  # Einige sind langsam
        
        mean = weight_fast * mode1 + weight_slow * mode2
        
        # Varianz bei Bimodalität hoch
        variance = weight_fast * (mode1 - mean) ** 2 + weight_slow * (mode2 - mean) ** 2
        
        dist = ProbabilityDistribution(mean=mean, variance=variance, mode1=mode1, mode2=mode2)
        return dist
    
    def convergence_rate(self, k: int) -> float:
        """
        Berechnet die Konvergenzgeschwindigkeit zur Worst-Case-Komplexität.
        
        Lemma: |E[T_k] - n^3| ≤ K * e^(-β * k / 2^n)
        
        Args:
            k: Formel-Index
            
        Returns:
            Konvergenzgeschwindigkeit (höher = schneller)
        """
        worst_case = self.n ** 3
        current = self.expected_runtime(k)
        
        if current >= worst_case:
            return 0.0
        
        # Abweichung
        error = worst_case - current
        
        # Konvergenzgeschwindigkeit
        if k > 0:
            normalized_k = k / self.pyramid_size
            rate = self.MU * normalized_k
            return rate
        return 0.0
    
    def phase_transition_points(self) -> Tuple[int, int]:
        """
        Findet die Indizes der Phasengrenzen.
        
        Returns:
            Tupel (Phase1/2-Grenze, Phase2/3-Grenze)
        """
        return (int(self.phase1_end), int(self.phase2_end))
    
    def runtime_trajectory(self, n_samples: int = 100) -> List[float]:
        """
        Generiert eine Stichprobe von erwarteten Laufzeiten.
        
        Args:
            n_samples: Anzahl der Stichproben
            
        Returns:
            Liste von erwarteten Laufzeiten
        """
        trajectory = []
        for k in range(1, min(n_samples, self.max_formulas)):
            trajectory.append(self.expected_runtime(k))
        return trajectory


class RuntimeAnalysis:
    """
    Detaillierte Laufzeit-Analyse mit statistischen Methoden.
    """
    
    @staticmethod
    def estimate_phase_duration(n: int, phase: int) -> int:
        """
        Schätzt die Dauer einer Phase in Formel-Indizes.
        
        Args:
            n: Anzahl Variablen
            phase: Phase-Nummer (1, 2 oder 3)
            
        Returns:
            Schätzung der Phasendauer
        """
        pyramid_size = 2 ** n
        
        if phase == 1:
            return int(np.sqrt(pyramid_size))
        elif phase == 2:
            return pyramid_size // 2 - int(np.sqrt(pyramid_size))
        else:
            return pyramid_size - pyramid_size // 2
    
    @staticmethod
    def convergence_error_bound(n: int, k: int, K: float = 1.0, beta: float = 0.5) -> float:
        """
        Berechnet die obere Schranke für Konvergenzfehler.
        
        Lemma: |E[T_k] - n^3| ≤ K * e^(-β * k / 2^n)
        
        Args:
            n: Anzahl Variablen
            k: Formel-Index
            K: Konstante K
            beta: Konvergenzparameter β
            
        Returns:
            Obere Schranke für Fehler
        """
        pyramid_size = 2 ** n
        exponent = -beta * k / pyramid_size
        return K * np.exp(exponent)
    
    @staticmethod
    def expected_first_success_time(success_prob: float = 1/3) -> float:
        """
        Berechnet die erwartete Zeit bis zur ersten erfolgreichen Lösung.
        
        Geometrische Verteilung: E[X] = 1/p
        
        Args:
            success_prob: Erfolgswahrscheinlichkeit pro Versuch
            
        Returns:
            Erwartete Anzahl Versuche
        """
        if success_prob <= 0:
            return float('inf')
        return 1.0 / success_prob
    
    @staticmethod
    def variance_in_phase(n: int, phase: int) -> float:
        """
        Schätzt die Varianz der Laufzeit in einer Phase.
        
        Args:
            n: Anzahl Variablen
            phase: Phase-Nummer
            
        Returns:
            Geschätzte Varianz
        """
        if phase == 1:
            # Hohe Varianz in Phase 1 (bimodal)
            base = math.log(n, 3)
            return base ** 2 * 10
        elif phase == 2:
            # Mittlere Varianz in Phase 2
            base = n ** 2
            return base * 5
        else:
            # Geringe relative Varianz in Phase 3
            base = n ** 3
            return base * 0.01


class ThreeWayPartition:
    """
    Modelliert die charakteristische 1/3-Dreiteilung der Kombinationspyramide.
    """
    
    def __init__(self, p_left: float = 1/3, p_right: float = 1/3, p_undefined: float = 1/3):
        """
        Initialisiert die Dreiteilung.
        
        Args:
            p_left: Wahrscheinlichkeit für Erfüllung
            p_right: Wahrscheinlichkeit für Unerfüllbarkeit
            p_undefined: Wahrscheinlichkeit für undefiniert
        """
        total = p_left + p_right + p_undefined
        if not np.isclose(total, 1.0):
            raise ValueError(f"Wahrscheinlichkeiten müssen sich zu 1 addieren: {total}")
        
        self.p_left = p_left
        self.p_right = p_right
        self.p_undefined = p_undefined
    
    def decision_outcome(self) -> int:
        """
        Simuliert einen Entscheidungsausgang in der Pyramide.
        
        Returns:
            0 für links (Erfüllung)
            1 für rechts (Unerfüllbarkeit)
            2 für undefiniert
        """
        r = np.random.random()
        if r < self.p_left:
            return 0  # Links
        elif r < self.p_left + self.p_right:
            return 1  # Rechts
        else:
            return 2  # Undefiniert
    
    def simulate_path_depth(self, max_depth: int = None) -> int:
        """
        Simuliert die Tiefe eines Pfads in der Pyramide bis zur Entscheidung.
        
        Args:
            max_depth: Maximale Pfadtiefe (default: 30)
            
        Returns:
            Tiefe des Pfads
        """
        if max_depth is None:
            max_depth = 30
        
        depth = 0
        while depth < max_depth:
            outcome = self.decision_outcome()
            depth += 1
            
            # Frühe Phasen: Entscheidung mit p_left + p_right
            if outcome in [0, 1]:
                return depth
        
        return depth
