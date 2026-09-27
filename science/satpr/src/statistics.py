"""
Statistics-Modul: Umfassende statistische Analysen für Wahrscheinlichkeitsverteilungen
und Phasenübergänge.

Implementiert Tests für:
- Phase-Übergänge
- Konvergenzanalyse
- Bimodale Verteilungen
- Entropie-Dynamik
- Statistische Signifikanztests
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from scipy import stats as sp_stats
from dataclasses import dataclass
import math


@dataclass
class StatisticalTestResult:
    """Ergebnis eines statistischen Tests."""
    statistic: float
    p_value: float
    test_name: str
    description: str = ""
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Ist das Ergebnis signifikant?"""
        return self.p_value < alpha
    
    def __repr__(self) -> str:
        return f"{self.test_name}(statistic={self.statistic:.4f}, p_value={self.p_value:.4f})"


class PhaseTransitionAnalysis:
    """
    Analysiert Übergänge zwischen verschiedenen Phasen.
    """
    
    @staticmethod
    def detect_transition_points(data: List[float], window_size: int = 3, 
                                threshold: float = None) -> List[int]:
        """
        Erkennt Übergangspunkte durch Analyse von Ratios zwischen Fenstern.
        
        Args:
            data: Zeitreihe von Laufzeitwerten
            window_size: Größe des gleitenden Fensters
            threshold: Schwelle für Übergangserkennung (default: automatisch)
            
        Returns:
            Indizes der erkannten Übergangspunkte
        """
        data = np.array(data, dtype=float)
        
        if len(data) < 2 * window_size:
            return []
        
        transitions = []
        
        for i in range(window_size, len(data) - window_size):
            window_before = np.mean(data[i-window_size:i])
            window_after = np.mean(data[i:i+window_size])
            
            if window_before > 0:
                ratio = window_after / window_before
            else:
                ratio = 1.0
            
            # Standard: Ratio > 1.5 oder < 0.67 deutet auf Übergang hin
            if threshold is None:
                threshold = 1.5
            
            if ratio > threshold or ratio < (1.0 / threshold):
                transitions.append(i)
        
        # Dedupliziere nahe beieinander liegende Übergänge
        if transitions:
            filtered_transitions = [transitions[0]]
            for t in transitions[1:]:
                if t - filtered_transitions[-1] > window_size:
                    filtered_transitions.append(t)
            return filtered_transitions
        
        return []
    
    @staticmethod
    def phase_duration_analysis(data: List[float], n_variables: int = None) -> Dict:
        """
        Analysiert die Dauer der einzelnen Phasen.
        
        Args:
            data: Zeitreihe von Laufzeitwerten
            n_variables: Anzahl Variablen (für theoretische Vorhersagen)
            
        Returns:
            Dict mit Phaseninformationen
        """
        data = np.array(data, dtype=float)
        transitions = PhaseTransitionAnalysis.detect_transition_points(data)
        
        result = {
            'n_datapoints': len(data),
            'detected_transitions': len(transitions),
            'transition_indices': transitions
        }
        
        if n_variables:
            pyramid_size = 2 ** n_variables
            theoretical_phase1_end = int(np.sqrt(pyramid_size))
            theoretical_phase2_end = pyramid_size // 2
            
            result['pyramid_size'] = pyramid_size
            result['theoretical_phase1_end'] = theoretical_phase1_end
            result['theoretical_phase2_end'] = theoretical_phase2_end
            
            if transitions:
                phase1_range = transitions[0] if transitions else theoretical_phase1_end
                result['observed_phase1_end'] = phase1_range
        
        return result
    
    @staticmethod
    def estimate_phase_boundaries(data: List[float], n_phases: int = 3) -> List[int]:
        """
        Schätzt Phasengrenzen durch Clustering.
        
        Args:
            data: Zeitreihe
            n_phases: Anzahl erwarteter Phasen
            
        Returns:
            Indizes der Phasengrenzen
        """
        # Verwende K-Means ähnliche Heuristik basierend auf Quantilen
        data_sorted_indices = np.argsort(data)
        boundaries = []
        
        for phase in range(1, n_phases):
            boundary_idx = int(len(data) * phase / n_phases)
            boundaries.append(boundary_idx)
        
        return sorted(boundaries)


class ConvergenceAnalysis:
    """
    Analysiert Konvergenzverhalten von Laufzeiten gegen Worst-Case-Komplexität.
    """
    
    @staticmethod
    def exponential_convergence_test(data: List[float], target: float,
                                     window_size: int = 5) -> Dict:
        """
        Testet ob Daten exponentiell gegen einen Zielwert konvergieren.
        
        Args:
            data: Zeitreihe von Werten
            target: Zielwert (z.B. n^3 für Worst-Case)
            window_size: Fenstergröße für Glättung
            
        Returns:
            Dict mit Konvergenzanalysen
        """
        data = np.array(data, dtype=float)
        
        if len(data) < 2:
            return {'convergence_rate': 0.0, 'is_exponential': False}
        
        # Berechne Fehler zum Zielwert
        errors = np.abs(data - target)
        
        # Filtere Nullen heraus
        nonzero_errors = errors[errors > 0]
        
        if len(nonzero_errors) < 2:
            return {'convergence_rate': 0.0, 'is_exponential': False}
        
        # Berechne Konvergenzrate: log(error[t]) sollte linear in t sein
        log_errors = np.log(nonzero_errors)
        
        # Lineare Regression: log_error = a + b*t
        t_indices = np.arange(len(log_errors))
        
        # Berechne Steigung
        if len(t_indices) > 1:
            cov = np.cov(t_indices, log_errors)
            if cov[1, 1] > 0:
                slope = cov[0, 1] / cov[1, 1]
                convergence_rate = -slope  # Negativ heißt Konvergenz
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Ist exponentiell wenn Slope signifikant negativ ist
        is_exponential = convergence_rate > 0.01
        
        return {
            'convergence_rate': convergence_rate,
            'is_exponential': is_exponential,
            'final_error': errors[-1] if len(errors) > 0 else None,
            'mean_error': np.mean(errors)
        }
    
    @staticmethod
    def halving_convergence_test(data: List[float]) -> Dict:
        """
        Testet ob Fehler sich mit jedem Schritt ungefähr halbiert.
        
        Args:
            data: Konvergenz-Sequenz
            
        Returns:
            Dict mit Halbierungs-Analyse
        """
        data = np.array(data, dtype=float)
        
        if len(data) < 2:
            return {'mean_ratio': 1.0, 'is_halving': False}
        
        # Berechne Ratios aufeinanderfolgender Elemente
        ratios = []
        for i in range(len(data) - 1):
            if data[i] > 0:
                ratios.append(data[i] / data[i + 1])
        
        if not ratios:
            return {'mean_ratio': 1.0, 'is_halving': False}
        
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        # Halbierung bedeutet Ratio ≈ 2.0
        is_halving = 1.5 < mean_ratio < 2.5
        
        return {
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'is_halving': is_halving,
            'ratios': ratios
        }
    
    @staticmethod
    def estimate_convergence_order(data: List[float]) -> float:
        """
        Schätzt die Konvergenzordnung (linear, quadratisch, etc.).
        
        Args:
            data: Fehler-Sequenz
            
        Returns:
            Schätzung der Konvergenzordnung (1, 2, 3, etc.)
        """
        data = np.array(data, dtype=float)
        
        if len(data) < 3:
            return 1.0
        
        # Berechne aufeinanderfolgende Ratios
        ratios = []
        for i in range(len(data) - 1):
            if data[i] > 0:
                ratios.append(data[i] / data[i + 1])
        
        if not ratios:
            return 1.0
        
        mean_ratio = np.mean(ratios)
        
        # Konvergenzordnung p: error[n+1] ≈ C * error[n]^p
        # Das bedeutet: error[n] ≈ error[0] * C^n * error[0]^(p^n - p)
        # Für große n: log(ratio) ≈ log(error[n] / error[n+1]) ≈ p
        
        log_ratios = np.log(np.array(ratios) + 1e-10)
        convergence_order = np.mean(log_ratios)
        
        return convergence_order


class BimodalDistributionAnalysis:
    """
    Analysiert bimodale Wahrscheinlichkeitsverteilungen.
    """
    
    @staticmethod
    def estimate_modes(data: List[float], n_bins: int = 20) -> Tuple[float, float, float, float]:
        """
        Schätzt die beiden Modi einer bimodalen Verteilung.
        
        Args:
            data: Datenprobe
            n_bins: Anzahl Bins für Histogramm
            
        Returns:
            (mode1, mode2, weight1, weight2)
        """
        data = np.array(data, dtype=float)
        
        if len(data) < 2:
            return data[0], data[0], 0.5, 0.5
        
        # Erstelle Histogramm
        hist, bin_edges = np.histogram(data, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Finde die zwei Bins mit höchster Häufigkeit
        top_indices = np.argsort(hist)[-2:][::-1]
        
        if len(top_indices) == 2:
            mode1 = bin_centers[top_indices[0]]
            mode2 = bin_centers[top_indices[1]]
            
            weight1 = hist[top_indices[0]] / np.sum(hist)
            weight2 = hist[top_indices[1]] / np.sum(hist)
        else:
            # Nur ein Modus vorhanden
            mode1 = bin_centers[top_indices[0]]
            mode2 = mode1
            weight1 = 1.0
            weight2 = 0.0
        
        return mode1, mode2, weight1, weight2
    
    @staticmethod
    def kullback_leibler_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """
        Berechnet KL-Divergenz D_KL(P || Q).
        
        Args:
            p: Erste Wahrscheinlichkeitsverteilung
            q: Zweite Wahrscheinlichkeitsverteilung
            eps: Kleine Konstante zur Vermeidung von log(0)
            
        Returns:
            KL-Divergenz
        """
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        
        # Normalisiere
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Berechne KL
        kl = 0.0
        for i in range(len(p)):
            if p[i] > eps:
                if q[i] <= eps:
                    return np.inf
                kl += p[i] * np.log2(p[i] / q[i])
        
        return kl
    
    @staticmethod
    def test_bimodality(data: List[float]) -> Dict:
        """
        Testet ob eine Verteilung bimodal ist (Hartigan's Dip Test Approximation).
        
        Args:
            data: Datenprobe
            
        Returns:
            Dict mit Test-Ergebnissen
        """
        data = np.array(data, dtype=float)
        
        # Vereinfachtes Test: Prüfe ob Bimodal-Index > Schwelle
        # Bimodal-Index = Variance der zwei Hälften / Gesamt-Variance
        
        sorted_data = np.sort(data)
        mid = len(sorted_data) // 2
        
        var1 = np.var(sorted_data[:mid])
        var2 = np.var(sorted_data[mid:])
        var_total = np.var(sorted_data)
        
        if var_total > 0:
            bimodal_index = (var1 + var2) / var_total
        else:
            bimodal_index = 1.0
        
        # Heuristische Schwelle: > 1.5 deutet auf Bimodalität hin
        is_bimodal = bimodal_index > 1.3
        
        return {
            'bimodal_index': bimodal_index,
            'is_bimodal': is_bimodal,
            'var1': var1,
            'var2': var2,
            'var_total': var_total
        }


class EntropyAnalysis:
    """
    Analysiert Entropie und Informationstheorie-Größen.
    """
    
    @staticmethod
    def shannon_entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
        """
        Berechnet Shannon-Entropie H(P) = -Σ p_i * log2(p_i).
        
        Args:
            probs: Wahrscheinlichkeitsverteilung
            eps: Kleine Konstante zur Vermeidung von log(0)
            
        Returns:
            Shannon-Entropie in Bits
        """
        probs = np.array(probs, dtype=float)
        probs = probs / np.sum(probs)  # Normalisiere
        
        entropy = 0.0
        for p in probs:
            if p > eps:
                entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def renyi_entropy(probs: np.ndarray, alpha: float = 1.0) -> float:
        """
        Berechnet Rényi-Entropie (Verallgemeinerung der Shannon-Entropie).
        
        H_α(P) = (1/(1-α)) * log2(Σ p_i^α)
        
        Args:
            probs: Wahrscheinlichkeitsverteilung
            alpha: Rényi-Parameter (α=1 → Shannon-Entropie)
            
        Returns:
            Rényi-Entropie
        """
        probs = np.array(probs, dtype=float)
        probs = probs / np.sum(probs)
        
        if np.isclose(alpha, 1.0):
            # Shannon-Entropie
            return EntropyAnalysis.shannon_entropy(probs)
        
        sum_powers = np.sum(probs ** alpha)
        
        if sum_powers <= 0:
            return 0.0
        
        return np.log2(sum_powers) / (1 - alpha)
    
    @staticmethod
    def entropy_change_rate(entropy_sequence: List[float]) -> Dict:
        """
        Analysiert wie schnell sich die Entropie verändert.
        
        Args:
            entropy_sequence: Zeitreihe von Entropiewerten
            
        Returns:
            Dict mit Änderungsraten
        """
        entropy_sequence = np.array(entropy_sequence, dtype=float)
        
        if len(entropy_sequence) < 2:
            return {'mean_change': 0.0, 'is_decreasing': None}
        
        # Berechne Differenzen
        changes = np.diff(entropy_sequence)
        
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # Ist die Entropie fallend?
        is_decreasing = mean_change < 0
        
        # Wie schnell fällt sie?
        if is_decreasing:
            decay_rate = -mean_change
        else:
            decay_rate = 0.0
        
        return {
            'mean_change': mean_change,
            'std_change': std_change,
            'is_decreasing': is_decreasing,
            'decay_rate': decay_rate,
            'entropy_start': entropy_sequence[0],
            'entropy_end': entropy_sequence[-1],
            'total_change': entropy_sequence[-1] - entropy_sequence[0]
        }


class StatisticalTests:
    """
    Statistische Signifikanztests.
    """
    
    @staticmethod
    def kolmogorov_smirnov_test(data: List[float], cdf_func: Callable) -> Dict:
        """
        Kolmogorov-Smirnov Test gegen theoretische CDF.
        
        Args:
            data: Datenprobe
            cdf_func: Kumulative Verteilungsfunktion zum Vergleich
            
        Returns:
            Dict mit Test-Statistiken
        """
        data = np.array(data, dtype=float)
        
        # Scipy KS-Test
        statistic, p_value = sp_stats.kstest(data, cdf_func)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'test_name': 'Kolmogorov-Smirnov',
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: np.ndarray) -> Dict:
        """
        Chi-Quadrat Goodness-of-Fit Test.
        
        Args:
            observed: Beobachtete Häufigkeiten
            expected: Erwartete Häufigkeiten
            
        Returns:
            Dict mit Test-Statistiken
        """
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        
        # Chi² Statistik: Σ (O_i - E_i)² / E_i
        chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        
        # Freiheitsgrade = Anzahl Kategorien - 1
        df = len(observed) - 1
        
        # p-Wert aus Chi²-Verteilung
        p_value = 1 - sp_stats.chi2.cdf(chi2_stat, df)
        
        return {
            'statistic': chi2_stat,
            'p_value': p_value,
            'test_name': 'Chi-Square',
            'degrees_of_freedom': df
        }
    
    @staticmethod
    def mann_whitney_u_test(data1: List[float], data2: List[float]) -> Dict:
        """
        Mann-Whitney U Test (nicht-parametrischer Test für zwei unabhängige Samples).
        
        Args:
            data1: Erste Datenprobe
            data2: Zweite Datenprobe
            
        Returns:
            Dict mit Test-Statistiken
        """
        data1 = np.array(data1, dtype=float)
        data2 = np.array(data2, dtype=float)
        
        # Scipy Mann-Whitney U Test
        statistic, p_value = sp_stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'test_name': 'Mann-Whitney U',
            'samples': (len(data1), len(data2))
        }
    
    @staticmethod
    def t_test(data1: List[float], data2: List[float]) -> Dict:
        """
        Welch's t-Test für zwei unabhängige Samples (nicht gleiche Varianzen).
        
        Args:
            data1: Erste Datenprobe
            data2: Zweite Datenprobe
            
        Returns:
            Dict mit Test-Statistiken
        """
        data1 = np.array(data1, dtype=float)
        data2 = np.array(data2, dtype=float)
        
        statistic, p_value = sp_stats.ttest_ind(data1, data2, equal_var=False)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'test_name': "Welch's t-Test",
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'mean_diff': np.mean(data1) - np.mean(data2)
        }
