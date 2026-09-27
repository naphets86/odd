"""
Metaverteilungs-Theorie: Wahrscheinlichkeitsverteilungen von Wahrscheinlichkeitsverteilungen.

Diese Klasse implementiert die in der Arbeit beschriebene mathematische Struktur,
die Wahrscheinlichkeitsverteilungen auf Verteilungsräumen definiert.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple


class ProbabilitySpace:
    """
    Klassischer Wahrscheinlichkeitsraum (Ω, F, P).
    
    Attributes:
        sample_space: Die Menge aller möglichen Ausgänge (Ω)
        measure: Die Wahrscheinlichkeitsfunktion P: F → [0,1]
    """
    
    def __init__(self, n_outcomes: int, measure: np.ndarray = None):
        """
        Initialisiert einen Wahrscheinlichkeitsraum.
        
        Args:
            n_outcomes: Größe des Stichprobenraums |Ω|
            measure: Wahrscheinlichkeitsfunktion (default: gleichverteilt)
        """
        self.n_outcomes = n_outcomes
        
        if measure is None:
            # Gleichverteilung
            self.measure = np.ones(n_outcomes) / n_outcomes
        else:
            measure = np.asarray(measure, dtype=float)
            
            # 1. Prüfe auf negative Werte
            if np.any(measure < 0):
                raise ValueError("Wahrscheinlichkeiten müssen nicht-negativ sein")
            
            # 2. Prüfe auf Nullsumme (verhindert Division durch Null)
            total = np.sum(measure)
            if np.isclose(total, 0.0):
                raise ValueError("Die Summe des Maßes darf nicht null sein")

            # 3. Normalisiere, falls die Summe nicht 1.0 ist
            if not np.allclose(total, 1.0):
                measure = measure / total
                
            self.measure = measure
    
    def entropy(self) -> float:
        """
        Berechnet die Shannon-Entropie der Verteilung.
        
        Returns:
            H(P) = -Σ p_i * log(p_i)
        """
        # Ignoriere Nullen (0 * log(0) = 0)
        nonzero = self.measure[self.measure > 0]
        return -np.sum(nonzero * np.log2(nonzero))
    
    def kl_divergence(self, other: 'ProbabilitySpace') -> float:
        """
        Kullback-Leibler Divergenz zu einer anderen Verteilung.
        
        Args:
            other: Andere Wahrscheinlichkeitsverteilung Q
            
        Returns:
            D_KL(P || Q) = Σ p_i * log(p_i / q_i)
        """
        if self.n_outcomes != other.n_outcomes:
            raise ValueError("Räume müssen gleiche Größe haben")
        
        # Berechne KL-Divergenz unter Beachtung von Grenzfällen
        kl = 0.0
        for i in range(self.n_outcomes):
            if self.measure[i] > 0:
                if other.measure[i] == 0:
                    return np.inf
                kl += self.measure[i] * np.log2(self.measure[i] / other.measure[i])
        return kl
    
    def wasserstein_distance(self, other: 'ProbabilitySpace') -> float:
        """
        Wasserstein-Distanz (Earth Mover's Distance) zu anderer Verteilung.
        
        Args:
            other: Andere Wahrscheinlichkeitsverteilung
            
        Returns:
            Wasserstein-Distanz
        """
        if self.n_outcomes != other.n_outcomes:
            raise ValueError("Räume müssen gleiche Größe haben")
        
        # 1D Wasserstein: Integral der Differenz der CDF
        sorted_p = np.sort(self.measure)
        sorted_q = np.sort(other.measure)
        return np.sum(np.abs(np.cumsum(sorted_p) - np.cumsum(sorted_q)))


class DistributionSpace:
    """
    Verteilungsraum D = {P: F → [0,1] | P ist Wahrscheinlichkeitsmaß}.
    
    Dies ist der Raum aller möglichen Wahrscheinlichkeitsverteilungen
    auf einem fixen Stichprobenraum Ω.
    """
    
    def __init__(self, n_outcomes: int, n_distributions: int = None):
        """
        Initialisiert einen Verteilungsraum.
        
        Args:
            n_outcomes: Größe des Stichprobenraums |Ω|
            n_distributions: Anzahl Distributionen zum Speichern (optional)
        """
        self.n_outcomes = n_outcomes
        self.distributions: List[ProbabilitySpace] = []
        self._capacity = n_distributions
    
    def add_distribution(self, measure: np.ndarray):
        """Fügt eine neue Verteilung zum Raum hinzu."""
        dist = ProbabilitySpace(self.n_outcomes, measure)
        self.distributions.append(dist)
    
    def dimension(self) -> int:
        """
        Dimension des Verteilungsraums als konvexer Polytop.
        
        Der Standardsimplex für n Outcomes hat Dimension n-1.
        
        Returns:
            dim(D) = |Ω| - 1 = n_outcomes - 1
        """
        return self.n_outcomes - 1
    
    def number_of_trajectories(self) -> int:
        """
        Anzahl der möglichen Verschiebungstrajektorien.
        
        Returns:
            Anzahl der eindeutigen Abfolgen von Verteilungen
        """
        # Dies ist die Anzahl der Permutationen der Distributionen
        from math import factorial
        return factorial(len(self.distributions))


class MetaDistribution:
    """
    Metaverteilung: Eine Wahrscheinlichkeitsverteilung auf dem Verteilungsraum D.
    
    Sie beschreibt die Wahrscheinlichkeit, dass das System eine bestimmte
    Verschiebung der Wahrscheinlichkeitsverteilung durchläuft.
    """
    
    def __init__(self, distribution_space: DistributionSpace):
        """
        Initialisiert eine Metaverteilung.
        
        Args:
            distribution_space: Der zugrundeliegende Verteilungsraum D
        """
        self.distribution_space = distribution_space
        self.trajectory_probs: Dict[Tuple, float] = {}
        self._total_prob = 0.0
    
    def add_trajectory(self, trajectory: Tuple[int, ...], probability: float):
        """
        Fügt eine Verschiebungstrajektorie mit ihrer Wahrscheinlichkeit hinzu.
        
        Args:
            trajectory: Tupel von Indizes der Distributionen
            probability: Wahrscheinlichkeit dieser Trajektorie
        """
        if not 0 <= probability <= 1:
            raise ValueError("Wahrscheinlichkeit muss in [0,1] sein")
        
        self.trajectory_probs[trajectory] = probability
        self._total_prob = sum(self.trajectory_probs.values())
    
    def entropy(self) -> float:
        """
        Berechnet die Entropie der Metaverteilung.
        
        Satz (Entropie von Metaverteilungen):
        H(μ) ≥ E_μ[H(P)]
        
        Returns:
            Entropie der Metaverteilung
        """
        if not self.trajectory_probs:
            return 0.0
        
        # Normalisiere Wahrscheinlichkeiten
        normalized = {k: v / self._total_prob 
                     for k, v in self.trajectory_probs.items()}
        
        # Shannon-Entropie
        entropy = 0.0
        for prob in normalized.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def average_distribution_entropy(self) -> float:
        """
        Berechnet den Erwartungswert der Entropie der einzelnen Distributionen.
        
        E_μ[H(P)] = Σ_trajectory μ(trajectory) * H(P_trajectory)
        
        Returns:
            Durchschnittliche Entropie
        """
        if not self.trajectory_probs:
            return 0.0
        
        avg_entropy = 0.0
        for trajectory, prob in self.trajectory_probs.items():
            # Berechne mittlere Entropie der Distributionen in dieser Trajektorie
            trajectory_entropies = []
            for idx in trajectory:
                if idx < len(self.distribution_space.distributions):
                    dist = self.distribution_space.distributions[idx]
                    trajectory_entropies.append(dist.entropy())
            
            if trajectory_entropies:
                avg_entropy += (prob / self._total_prob) * np.mean(trajectory_entropies)
        
        return avg_entropy
    
    def shift_degree(self) -> float:
        """
        Misst das Ausmaß der Wahrscheinlichkeitsverschiebung.
        
        Höher = stärkere Verschiebung zwischen Verteilungen
        
        Returns:
            Durchschnittliche paarweise Entfernung zwischen Distributionen
        """
        if len(self.distribution_space.distributions) < 2:
            return 0.0
        
        total_distance = 0.0
        n_pairs = 0
        
        for i in range(len(self.distribution_space.distributions)):
            for j in range(i + 1, len(self.distribution_space.distributions)):
                dist_i = self.distribution_space.distributions[i]
                dist_j = self.distribution_space.distributions[j]
                
                total_distance += dist_i.wasserstein_distance(dist_j)
                n_pairs += 1
        
        return total_distance / n_pairs if n_pairs > 0 else 0.0


class ShiftTrajectory:
    """
    Eine Verschiebungstrajektorie durch den Verteilungsraum.
    
    Beschreibt die zeitliche Entwicklung der Wahrscheinlichkeitsverteilung
    bei der Lösung von SAT-Problemen.
    """
    
    def __init__(self, time_steps: List[ProbabilitySpace]):
        """
        Initialisiert eine Verschiebungstrajektorie.
        
        Args:
            time_steps: Liste von Verteilungen über Zeit
        """
        self.time_steps = time_steps
    
    def convergence_rate(self, target: ProbabilitySpace) -> float:
        """
        Misst die Konvergenzgeschwindigkeit zur Zielverteilung.
        
        Args:
            target: Die Zielverteilung (z.B. Worst-Case-Verteilung)
            
        Returns:
            Konvergenzrate (höher = schneller)
        """
        if len(self.time_steps) < 2:
            return 0.0
        
        distances = []
        for dist in self.time_steps:
            distances.append(dist.kl_divergence(target))
        
        # Exponentielle Konvergenzrate: (Δ_t - Δ_{t+1}) / Δ_t
        if distances[0] == 0:
            return 0.0
        
        rate = 1.0 - (distances[-1] / distances[0])
        return max(0.0, rate)
    
    def phase_transition_points(self, threshold: float = 0.5) -> List[int]:
        """
        Findet Punkte, an denen sich die Verteilung signifikant ändert.
        
        Args:
            threshold: Schwelle für signifikante Änderung
            
        Returns:
            Indizes der Übergangspunkte
        """
        transitions = []
        
        for t in range(1, len(self.time_steps)):
            curr = self.time_steps[t]
            prev = self.time_steps[t - 1]
            
            # Wasserstein-Distanz als Maß für Änderung
            change = curr.wasserstein_distance(prev)
            
            if change > threshold:
                transitions.append(t)
        
        return transitions
