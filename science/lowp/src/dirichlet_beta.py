"""
Dirichlet Beta-Funktion und Catalan-Konstante

Implementierung der Dirichlet Beta-Funktion:
    β(s) := Σ((-1)^n / (2n+1)^s) für Re(s) > 0

Mit analytischer Fortsetzung, Residuentheorie und Catalan-Konstante.
Basierend auf "Neue Leibniz Ergebnisse zur Dirichlet Beta-Funktion"
"""

import numpy as np
from scipy import special
from scipy.integrate import quad
import warnings
from typing import Union, Tuple, Optional


class DirichletBeta:
    """Dirichlet Beta-Funktion und verwandte spezielle Funktionen."""
    
    # Catalan-Konstante (Referenzwert)
    CATALAN_CONSTANT = 0.9159655941772190150546035149323841107741493742816721342664306
    
    # Mascheroni-Konstante
    EULER_GAMMA = 0.5772156649015328606065120900824024310421593359399235988057672
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialisierung der Dirichlet Beta-Funktion.
        
        Args:
            tolerance: Konvergenztoleranz für numerische Berechnungen
        """
        self.tolerance = tolerance
        self.max_iterations = 10000
    
    def dirichlet_beta_series(self, s: complex, n_terms: int = 1000) -> complex:
        """
        Berechnung von β(s) über die definierende Reihe für Re(s) > 0.
        
        β(s) = Σ((-1)^n / (2n+1)^s) für n=0 bis ∞
        
        Args:
            s: Komplexer Parameter
            n_terms: Anzahl der Reihenglieder
            
        Returns:
            Näherung von β(s)
            
        Raises:
            ValueError: Wenn Re(s) ≤ 0
        """
        if np.real(s) <= 0:
            raise ValueError(f"Reihe konvergiert nur für Re(s) > 0, erhalten: Re(s) = {np.real(s)}")
        
        result = 0.0j if isinstance(s, complex) else 0.0
        for n in range(n_terms):
            term = ((-1)**n) / ((2*n + 1)**s)
            result += term
            
            # Abbruch bei Konvergenz
            if abs(term) < self.tolerance:
                break
        
        return result
    
    def catalan_constant_series(self, n_terms: int = 10000) -> float:
        """
        Berechnung der Catalan-Konstante G = β(2).
        
        G = Σ((-1)^n / (2n+1)^2) = 1 - 1/9 + 1/25 - 1/49 + ...
        
        Args:
            n_terms: Anzahl der Reihenglieder
            
        Returns:
            Näherung der Catalan-Konstante
        """
        result = 0.0
        for n in range(n_terms):
            term = ((-1)**n) / ((2*n + 1)**2)
            result += term
            
            if abs(term) < self.tolerance:
                break
        
        return result
    
    def catalan_arctan_integral(self, n_quad: int = 500) -> float:
        """
        Berechnung der Catalan-Konstante via Arcustangens-Integral.
        
        G = ∫₀¹ arctan(x)/x dx
        
        Args:
            n_quad: Anzahl der Quadraturpunkte
            
        Returns:
            Näherung der Catalan-Konstante
        """
        def integrand(x):
            if x == 0:
                return 1.0  # lim_{x→0} arctan(x)/x = 1
            return np.arctan(x) / x
        
        result, _ = quad(integrand, 0, 1, limit=n_quad)
        return result
    
    def catalan_logarithmic_integral(self, n_quad: int = 500) -> float:
        """
        Berechnung der Catalan-Konstante via logarithmisches Integral.
        
        G = -∫₀¹ ln(x)/(1+x²) dx
        
        (Hinweis: -∫₀¹ ln(1+x²)/(2x) dx = -π²/48 ist NICHT G.)
        
        Args:
            n_quad: Anzahl der Quadraturpunkte
            
        Returns:
            Näherung der Catalan-Konstante
        """
        def integrand(x):
            if x == 0:
                return 0.0  # Randpunkt (von quad nicht ausgewertet)
            return -np.log(x) / (1 + x**2)
        
        result, _ = quad(integrand, 0, 1, limit=n_quad)
        return result
    
    def catalan_sine_integral(self, n_quad: int = 500) -> float:
        """
        Berechnung der Catalan-Konstante via Sinus-Integral.
        
        G = ½ ∫₀^(π/2) x/sin(x) dx
        
        (Hinweis: ∫₀^(π/2) x/sin(x) dx = 2G, der Faktor ½ ist nötig.)
        
        Args:
            n_quad: Anzahl der Quadraturpunkte
            
        Returns:
            Näherung der Catalan-Konstante
        """
        def integrand(x):
            if x == 0:
                return 1.0  # lim_{x→0} x/sin(x) = 1
            return x / np.sin(x)
        
        result, _ = quad(integrand, 0, np.pi/2, limit=n_quad)
        return 0.5 * result
    
    def beta_functional_equation(self, s: complex) -> complex:
        """
        Anwendung der funktionalen Gleichung der Dirichlet Beta-Funktion.
        
        β(s) = (2/π)^(1-s) cos(πs/2) Γ(1-s) β(1-s)
        
        (Die Form 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s) gilt für die
        Riemann-Zeta-Funktion, nicht für β.)
        
        Dies erlaubt analytische Fortsetzung außerhalb Re(s) > 0.
        An s = 1, 2, 3, ... ist die Formel numerisch singulär (0·∞).
        
        Args:
            s: Komplexer Parameter
            
        Returns:
            β(s) via funktionale Gleichung
        """
        factor1 = (2.0 / np.pi)**(1 - s)
        factor2 = np.cos(np.pi * s / 2)
        factor3 = special.gamma(1 - s)
        
        # β(1-s) über die Reihe, falls Re(1-s) > 0, sonst über Fortsetzung
        if np.real(1 - s) > 0:
            beta_1_minus_s = self.dirichlet_beta_series(1 - s)
        else:
            beta_1_minus_s = self._beta_analytic_continuation(1 - s)
        
        return factor1 * factor2 * factor3 * beta_1_minus_s
    
    def _beta_analytic_continuation(self, s: complex) -> complex:
        """
        Analytische Fortsetzung: Reihe für Re(s) > 0, sonst funktionale Gleichung.
        
        Args:
            s: Komplexer Parameter
            
        Returns:
            β(s)
        """
        if np.real(s) > 0:
            return self.dirichlet_beta_series(s)
        # Re(s) ≤ 0  =>  Re(1-s) ≥ 1 > 0, die Gleichung terminiert
        return self.beta_functional_equation(s)
    
    def poles_and_residues(self) -> dict:
        """
        Pole und Residuen der Dirichlet Beta-Funktion.
        
        β ist eine GANZE Funktion, sie hat keine Pole. Die Punkte
        s = -(2k+1) sind die trivialen Nullstellen (β(-k) = E_k/2 mit
        Euler-Zahlen E_k, E_ungerade = 0). Das Residuum ist dort 0.
        
        Returns:
            Dictionary {s = -(2k+1): Residuum = 0.0} für die ersten 10 Punkte
        """
        return {-(2*k + 1): 0.0 for k in range(10)}
    
    def zeta_eta_relationship(self, s: complex, n_terms: int = 1000) -> Tuple[complex, complex]:
        """
        ζ(s) (Partialsumme) und β(s) (Reihe) für Re(s) > 0.
        
        β lässt sich NICHT allein aus ζ und η darstellen: die frühere Formel
        β = (ζ+η)/2^(s+1) ist falsch. Exakt gilt über Hurwitz-Zeta:
        β(s) = 4^(-s) [ζ(s,1/4) - ζ(s,3/4)].
        
        Args:
            s: Komplexer Parameter
            n_terms: Anzahl Reihenglieder
            
        Returns:
            (β(s), ζ(s))
        """
        if np.real(s) <= 0:
            raise ValueError("Diese Methode erfordert Re(s) > 0")
        
        # Riemann Zeta-Funktion (Partialsumme)
        zeta_s = 0.0j if isinstance(s, complex) else 0.0
        for n in range(1, n_terms):
            zeta_s += 1.0 / (n**s)
        
        # Dirichlet Beta direkt aus der definierenden Reihe
        beta_s = self.dirichlet_beta_series(s, n_terms)
        
        return beta_s, zeta_s
    
    def special_values(self) -> dict:
        """
        Berechnung spezieller Werte der Dirichlet Beta-Funktion.
        
        Returns:
            Dictionary mit speziellen Werten
        """
        special_vals = {
            'β(1) (Leibniz)': self.dirichlet_beta_series(1, n_terms=10000),
            'β(2) (Catalan)': self.catalan_constant_series(),
            'β(1) theoretical': np.pi / 4,
            'β(2) theoretical': self.CATALAN_CONSTANT,
        }
        
        # Zusätzliche Werte
        for n in range(3, 6):
            special_vals[f'β({n})'] = self.dirichlet_beta_series(n, n_terms=5000)
        
        return special_vals
    
    def error_estimate(self, s: float, n_terms: int) -> float:
        """
        Fehlerschätzung für Reihenabbruch bei Re(s) > 0.
        
        Der Fehler ist höchstens das erste nicht berechnete Glied.
        |Fehler| ≤ 1/(2n+1)^s für das n-te Glied
        
        Args:
            s: Reeller Parameter (Re(s) > 0)
            n_terms: Anzahl der berechneten Glieder
            
        Returns:
            Obere Schranke des Fehlers
        """
        if s <= 0:
            raise ValueError("Erfordert s > 0")
        
        # Das erste nicht berechnete Glied (n-ter Glied)
        return 1.0 / ((2*n_terms + 1)**s)


class CatalanConstant:
    """Spezialisierte Klasse für die Catalan-Konstante."""
    
    def __init__(self):
        self.beta = DirichletBeta()
        self.value = 0.9159655941772190150546035149323841107741493742816721342664306
    
    def compute_via_series(self, n_terms: int = 10000) -> float:
        """G = Σ((-1)^n / (2n+1)²)"""
        return self.beta.catalan_constant_series(n_terms)
    
    def compute_via_arctan(self) -> float:
        """G = ∫₀¹ arctan(x)/x dx"""
        return self.beta.catalan_arctan_integral()
    
    def compute_via_logarithm(self) -> float:
        """G = -∫₀¹ ln(1+x²)/(2x) dx"""
        return self.beta.catalan_logarithmic_integral()
    
    def compute_via_sine(self) -> float:
        """G = ∫₀^(π/2) x/sin(x) dx"""
        return self.beta.catalan_sine_integral()
    
    def verify_representations(self) -> dict:
        """Vergleich aller Darstellungen der Catalan-Konstante."""
        results = {
            'Series': self.compute_via_series(),
            'Arctan': self.compute_via_arctan(),
            'Logarithm': self.compute_via_logarithm(),
            'Sine': self.compute_via_sine(),
            'Reference': self.value,
        }
        return results
    
    def error_to_reference(self, computed_value: float) -> float:
        """Berechne Fehler zu Referenzwert."""
        return abs(computed_value - self.value)
