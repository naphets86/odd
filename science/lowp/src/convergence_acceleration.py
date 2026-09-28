"""
Beschleunigungsmethoden für die Leibniz-Reihe

Implementierung von:
- Euler-Transformation (O(2^-N) statt O(1/N))
- Richardson-Extrapolation (O(1/N³) und besser statt O(1/N))
- Shanks-Transformation
- Verbesserte Konvergenzraten

Basierend auf "Neue Leibniz Ergebnisse - Beschleunigungsmethoden"
"""

import math
from fractions import Fraction

import numpy as np
from typing import Callable, List, Tuple, Optional


class LeibnizSeries:
    """Grundlegende Leibniz-Reihe: π/4 = Σ((-1)^n / (2n+1))"""
    
    PI_OVER_4 = np.pi / 4
    
    def __init__(self):
        self.partial_sums = []
    
    def compute_partial_sum(self, N: int) -> float:
        """
        Berechne die Partialsumme S_N = Σ_{n=0}^{N} (-1)^n / (2n+1)
        (Index N inklusive, also N+1 Terme; S_0 = 1).
        
        Args:
            N: Index der Partialsumme
            
        Returns:
            Partialsumme S_N
        """
        result = 0.0
        for n in range(N + 1):
            result += ((-1)**n) / (2*n + 1)
        return result
    
    def partial_sum_vector(self, max_n: int) -> np.ndarray:
        """
        Generiere Vektor von Partialsummen S_0, S_1, ..., S_max_n
        
        Args:
            max_n: Maximum Index
            
        Returns:
            Array mit allen Partialsummen
        """
        sums = np.zeros(max_n + 1)
        current_sum = 0.0
        for n in range(max_n + 1):
            current_sum += ((-1)**n) / (2*n + 1)
            sums[n] = current_sum
        self.partial_sums = sums
        return sums
    
    def error_naive(self, n_terms: int) -> float:
        """
        Fehler der naiven Leibniz-Reihe nach n Termen: O(1/N)
        
        |S_N - π/4| ≤ 1/(2N+3) (Leibniz-Kriterium)
        
        Args:
            n_terms: Anzahl der Terme
            
        Returns:
            Fehlergrenze
        """
        return 1.0 / (2*n_terms + 3)
    
    def convergence_rate_naive(self, n_values: List[int]) -> dict:
        """
        Analysiere Konvergenzrate O(1/N)
        
        Args:
            n_values: Liste von Term-Anzahlen
            
        Returns:
            Dictionary mit Summen und Fehlern
        """
        results = {}
        for n in n_values:
            partial_sum = self.compute_partial_sum(n)
            error = abs(partial_sum - self.PI_OVER_4)
            results[n] = {
                'sum': partial_sum,
                'error': error,
                'error_estimate': self.error_naive(n),
                'convergence': error * n  # Sollte konstant sein für O(1/N)
            }
        return results


class EulerTransformation:
    """
    Euler-Transformation für alternierende Reihen.
    
    Für S = Σ(-1)^k a_k gilt:
    S = Σ_{n≥0} 1/2^(n+1) Σ_{k=0}^n (-1)^k C(n,k) a_k
    
    (a_k, NICHT a_{n-k}: mit a_{n-k} fehlt der Faktor (-1)^n und die
    Reihe konvergiert gegen einen falschen Wert.)
    Für a_k = 1/(2k+1) ist die innere Summe (2n)!!/(2n+1)!!.
    Dies gibt exponentielle Konvergenz O(2^-N).
    """
    
    def __init__(self, leibniz: Optional[LeibnizSeries] = None):
        self.leibniz = leibniz or LeibnizSeries()
    
    def binomial_coefficient(self, n: int, k: int) -> int:
        """Berechne Binomialkoeffizient C(n,k) (exakt, ganzzahlig)"""
        return math.comb(n, k)
    
    def euler_transform_term(self, n: int) -> float:
        """
        Berechne den n-ten Term der Euler-transformierten Reihe.
        
        Term_n = 1/2^(n+1) * Σ_{k=0}^n (-1)^k C(n,k) / (2k+1)
        
        Die innere Summe alterniert mit riesigen Summanden (C(100,50) ~ 1e29)
        und wird deshalb exakt mit Brüchen berechnet (Gleitkomma würde durch
        Auslöschung alle Stellen verlieren).
        
        Args:
            n: Index des Terms
            
        Returns:
            Term_n
        """
        inner_sum = Fraction(0)
        for k in range(n + 1):
            inner_sum += Fraction((-1)**k * self.binomial_coefficient(n, k), 2*k + 1)
        return float(inner_sum / 2**(n + 1))
    
    def euler_transform_sum(self, n_terms: int) -> float:
        """
        Berechne die Euler-transformierte Summe (n_terms Terme).
        
        S_euler = Σ_{n=0}^{N-1} 1/2^(n+1) Σ_{k=0}^n (-1)^k C(n,k) / (2k+1)
        
        Args:
            n_terms: Anzahl der Terme der transformierten Reihe
            
        Returns:
            Näherung von π/4 via Euler-Transformation
        """
        result = 0.0
        for n in range(n_terms):
            result += self.euler_transform_term(n)
        return result
    
    def euler_error_estimate(self, n_terms: int) -> float:
        """
        Obere Schranke für den Fehler nach n_terms Termen: O(2^-N).
        
        Der n-te Term ist 2^-(n+1) (2n)!!/(2n+1)!! ≤ 2^-(n+1)/√(n+1);
        die Restsumme ab n = N ist damit ≤ 2^-N/√(N+1).
        
        Args:
            n_terms: Anzahl der Terme
            
        Returns:
            Fehlerschranke
        """
        return 2.0**(-n_terms) / math.sqrt(n_terms + 1)
    
    def convergence_analysis(self, n_values: List[int]) -> dict:
        """
        Analysiere Konvergenzverbesserung durch Euler-Transformation
        
        Args:
            n_values: Liste von Term-Anzahlen
            
        Returns:
            Dictionary mit Vergleich naive vs. Euler
        """
        results = {}
        for n in n_values:
            # Naive Leibniz
            leibniz_sum = self.leibniz.compute_partial_sum(n)
            leibniz_error = abs(leibniz_sum - LeibnizSeries.PI_OVER_4)
            
            # Euler-Transformation
            euler_sum = self.euler_transform_sum(n)
            euler_error = abs(euler_sum - LeibnizSeries.PI_OVER_4)
            
            # Verbesserungsfaktor
            improvement = leibniz_error / (euler_error + 1e-15)
            
            results[n] = {
                'leibniz_sum': leibniz_sum,
                'leibniz_error': leibniz_error,
                'euler_sum': euler_sum,
                'euler_error': euler_error,
                'improvement_factor': improvement,
            }
        return results


class RichardsonExtrapolation:
    """
    Richardson-Extrapolation für die Leibniz-Reihe.
    
    Der Fehler alterniert im Vorzeichen:
    S_N - π/4 = (-1)^N * [1/(4(N+1)) + O(1/N³)]
    
    Die Formel ((2N+3) S_N - (2N+1) S_{N+1})/2 aus der Arbeit ignoriert das
    Vorzeichen und liefert einen Fehler von ≈ 1/2. Richtig ist eine
    Kombination mit gleichen Vorzeichen der Gewichte:
    
        Ŝ_N = ((N+1) S_N + (N+2) S_{N+1}) / (2N+3)
    
    Sie eliminiert die führenden Fehlerterme (Fehler O(1/N⁴)).
    """
    
    def __init__(self, leibniz: Optional[LeibnizSeries] = None):
        self.leibniz = leibniz or LeibnizSeries()
    
    def richardson_extrapolation_2terms(self, S_N: float, S_N1: float, N: int) -> float:
        """
        Richardson-Extrapolation mit zwei aufeinanderfolgenden Partialsummen.
        
        Ŝ_N = ((N+1) S_N + (N+2) S_{N+1}) / (2N+3)
        
        Die Gewichte summieren sich zu 1 und heben die alternierenden
        Fehlerterme 1/(4(N+1)) von S_N und S_{N+1} gegeneinander auf.
        
        Args:
            S_N: Partialsumme S_N
            S_N1: Partialsumme S_{N+1}
            N: Index N
            
        Returns:
            Verbesserte Approximation
        """
        return ((N + 1) * S_N + (N + 2) * S_N1) / (2*N + 3)
    
    def richardson_error_estimate(self, N: int) -> float:
        """
        Fehlerschätzung nach Richardson-Extrapolation: höchstens O(1/N³)
        (tatsächlich O(1/N⁴), numerisch geprüft).
        
        Args:
            N: Index
            
        Returns:
            Fehlergrenze
        """
        return 1.0 / ((2*N + 1)**3)
    
    def richardson_cascade(self, n_base: int) -> dict:
        """
        Wende Richardson-Extrapolation mehrfach an (Richardson-Kaskade).
        
        Args:
            n_base: Basis-Partialsum Index
            
        Returns:
            Dictionary mit allen Extrapolationsstufen
        """
        # Berechne mehrere Partialsummen
        S = {}
        for k in range(n_base, n_base + 10):
            S[k] = self.leibniz.compute_partial_sum(k)
        
        results = {
            'base_sums': S,
            'first_richardson': {},
            'second_richardson': {},
        }
        
        # Erste Richardson-Extrapolation
        for k in range(n_base, n_base + 9):
            extrap = self.richardson_extrapolation_2terms(S[k], S[k+1], k)
            results['first_richardson'][k] = extrap
        
        # Zweite Richardson-Extrapolation (falls möglich)
        for k in range(n_base, n_base + 8):
            extrap1_k = results['first_richardson'][k]
            extrap1_k1 = results['first_richardson'][k+1]
            # Zweite Stufe (einfach erneut anwenden)
            extrap2 = (extrap1_k + extrap1_k1) / 2.0
            results['second_richardson'][k] = extrap2
        
        return results
    
    def convergence_analysis(self, n_values: List[int]) -> dict:
        """
        Analysiere Richardson-Extrapolation vs. naive Leibniz
        
        Args:
            n_values: Liste von Index-Werten
            
        Returns:
            Dictionary mit Vergleichen
        """
        results = {}
        for n in n_values:
            S_n = self.leibniz.compute_partial_sum(n)
            S_n1 = self.leibniz.compute_partial_sum(n + 1)
            
            richardson = self.richardson_extrapolation_2terms(S_n, S_n1, n)
            
            naive_error = abs(S_n - LeibnizSeries.PI_OVER_4)
            richardson_error = abs(richardson - LeibnizSeries.PI_OVER_4)
            
            results[n] = {
                'naive_sum': S_n,
                'naive_error': naive_error,
                'richardson_sum': richardson,
                'richardson_error': richardson_error,
                'error_reduction': naive_error / (richardson_error + 1e-15),
                'richardson_estimate': self.richardson_error_estimate(n),
            }
        
        return results


class ShanksTransformation:
    """
    Shanks-Transformation für Reihenkonvergenz.
    
    Für eine Folge S_n definiere:
    e_n^(1) = (S_{n+1} S_{n-1} - S_n²) / (S_{n+1} - 2S_n + S_{n-1})
    """
    
    def __init__(self, leibniz: Optional[LeibnizSeries] = None):
        self.leibniz = leibniz or LeibnizSeries()
    
    def shanks_transform_single(self, S_prev: float, S_curr: float, S_next: float) -> float:
        """
        Wende Shanks-Transformation auf drei aufeinanderfolgende Terme an.
        
        e^(1) = (S_{n+1} * S_{n-1} - S_n²) / (S_{n+1} - 2*S_n + S_{n-1})
        
        Args:
            S_prev: S_{n-1}
            S_curr: S_n (aktuell)
            S_next: S_{n+1}
            
        Returns:
            Transformierter Wert
        """
        numerator = S_next * S_prev - S_curr**2
        denominator = S_next - 2*S_curr + S_prev
        
        if abs(denominator) < 1e-15:
            return S_curr  # Fallback wenn Nenner zu klein
        
        return numerator / denominator
    
    def shanks_vector(self, n_max: int) -> np.ndarray:
        """
        Wende Shanks-Transformation auf eine Folge von Partialsummen an.
        
        Args:
            n_max: Maximale Index
            
        Returns:
            Array mit transformierten Werten
        """
        S = self.leibniz.partial_sum_vector(n_max + 2)
        
        transformed = np.zeros(n_max + 1)
        for n in range(1, n_max + 1):
            transformed[n] = self.shanks_transform_single(S[n-1], S[n], S[n+1])
        
        return transformed
    
    def convergence_analysis(self, n_values: List[int]) -> dict:
        """
        Analysiere Shanks-Transformation
        
        Args:
            n_values: Liste von Indizes
            
        Returns:
            Dictionary mit Vergleichen
        """
        results = {}
        S_all = self.leibniz.partial_sum_vector(max(n_values) + 2)
        
        for n in n_values:
            if n < 1:
                continue
            
            S_n_minus_1 = S_all[n-1]
            S_n = S_all[n]
            S_n_plus_1 = S_all[n+1]
            
            shanks = self.shanks_transform_single(S_n_minus_1, S_n, S_n_plus_1)
            
            naive_error = abs(S_n - LeibnizSeries.PI_OVER_4)
            shanks_error = abs(shanks - LeibnizSeries.PI_OVER_4)
            
            results[n] = {
                'naive_sum': S_n,
                'naive_error': naive_error,
                'shanks_sum': shanks,
                'shanks_error': shanks_error,
                'improvement': naive_error / (shanks_error + 1e-15),
            }
        
        return results


class ConvergenceComparison:
    """Vergleiche alle Beschleunigungsmethoden."""
    
    def __init__(self):
        self.leibniz = LeibnizSeries()
        self.euler = EulerTransformation(self.leibniz)
        self.richardson = RichardsonExtrapolation(self.leibniz)
        self.shanks = ShanksTransformation(self.leibniz)
    
    def compare_methods(self, n_values: List[int]) -> dict:
        """
        Vergleiche alle Methoden auf gleichen Eingabewerten
        
        Args:
            n_values: Liste von Term-Anzahlen
            
        Returns:
            Umfassender Vergleich aller Methoden
        """
        results = {
            'naive': self.leibniz.convergence_rate_naive(n_values),
            'euler': self.euler.convergence_analysis(n_values),
            'richardson': self.richardson.convergence_analysis(n_values),
            'shanks': self.shanks.convergence_analysis(n_values),
        }
        
        return results
    
    def theoretical_reference(self) -> float:
        """Gebe π/4 als Referenzwert"""
        return LeibnizSeries.PI_OVER_4
