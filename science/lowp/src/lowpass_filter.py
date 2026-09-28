"""
Low-Pass Filter (RC-Filter) und die e-Funktion

Implementierung der Differentialgleichung:
    τ dV_out/dt + V_out = V_in

Mit analytischen Lösungen und Frequenzantwort.
Basierend auf "Der Tiefpassfilter und die e-Funktion" (lowp.tex)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Union
from scipy.integrate import odeint
from scipy.optimize import fsolve


class ExponentialFunction:
    """Die Exponentialfunktion und ihre Reproduzierbarkeit unter Differentiation."""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialisierung.
        
        Args:
            tolerance: Konvergenztoleranz
        """
        self.tolerance = tolerance
    
    def taylor_series(self, s: complex, t: float, n_terms: int = 100) -> complex:
        """
        Berechne e^(st) via Taylor-Reihe.
        
        e^(st) = Σ(st)^n / n! für n=0 bis ∞
        
        Args:
            s: Parameter s
            t: Zeit t
            n_terms: Anzahl der Reihenglieder
            
        Returns:
            Näherung von e^(st)
        """
        result = 0.0j if isinstance(s, complex) or isinstance(t, complex) else 0.0
        
        factorial = 1.0
        power = 1.0
        
        for n in range(n_terms):
            if n > 0:
                factorial *= n
                power *= (s * t)
            
            term = power / factorial
            result += term
            
            if abs(term) < self.tolerance:
                break
        
        return result
    
    def derivative_taylor(self, s: complex, t: float, n_terms: int = 100) -> complex:
        """
        Berechne d/dt[e^(st)] = s*e^(st) via Taylor-Reihe.
        
        Dies demonstriert die Reproduzierbarkeit: Die Ableitung ergibt
        wieder die Exponentialfunktion multipliziert mit s.
        
        Args:
            s: Parameter s
            t: Zeit t
            n_terms: Anzahl der Reihenglieder
            
        Returns:
            Näherung von s*e^(st)
        """
        # Direkt: d/dt[e^(st)] = s*e^(st)
        return s * self.taylor_series(s, t, n_terms)
    
    def taylor_term_analysis(self, s: float, t: float, n_max: int = 20) -> dict:
        """
        Analysiere, wie sich die Reihenglieder bei der Ableitung verschieben.
        
        Dies zeigt die "Verschiebung um ein Glied" bei der Differentiation,
        die in Beobachtung 49 beschrieben wird.
        
        Args:
            s: Parameter s (reell)
            t: Zeit (reell)
            n_max: Maximale Anzahl Glieder
            
        Returns:
            Dictionary mit Original- und abgeleiteten Gliedern
        """
        result = {
            'original_terms': {},
            'derivative_terms': {},
            'derivative_from_original': {},
        }
        
        # Originalglieder: (st)^n / n!
        for n in range(n_max):
            factorial = 1
            for k in range(1, n + 1):
                factorial *= k
            
            value = (s * t)**n / factorial
            result['original_terms'][n] = value
            
            # Ableitung des n-ten Gliedes
            if n == 0:
                deriv = 0  # d/dt[1] = 0
            else:
                factorial_n_minus_1 = 1
                for k in range(1, n):
                    factorial_n_minus_1 *= k
                deriv = s * (s * t)**(n-1) / factorial_n_minus_1
            
            result['derivative_terms'][n] = deriv
        
        # Zusammenfassung: Ableitungen ergeben verschobene Glieder
        result['shift_analysis'] = {
            'original_index_n': 'Glied n der Original-Reihe',
            'after_derivative_becomes': f'Glied (n-1) multipliziert mit s',
            'explanation': 'Die Reihe wird um ein Glied "nach links" verschoben'
        }
        
        return result
    
    def reproduction_property(self, s: complex, t_values: np.ndarray, 
                            n_terms: int = 100) -> dict:
        """
        Demonstriere die Reproduzierbarkeit: f'(t) = s*f(t)
        
        Args:
            s: Parameter
            t_values: Zeitwerte
            n_terms: Reihenglieder
            
        Returns:
            Dictionary mit e^(st) und ihrer Ableitung
        """
        results = {
            'time': t_values,
            'exp_st': np.array([self.taylor_series(s, t, n_terms) for t in t_values]),
            'derivative': np.array([s * self.taylor_series(s, t, n_terms) for t in t_values]),
            's_parameter': s,
        }
        
        return results


class RCLowpassFilter:
    """
    RC-Tiefpassfilter: τ dV_out/dt + V_out = V_in
    
    Eigenschaften:
    - Zeitkonstante τ = RC
    - Grenzfrequenz f_c = 1/(2πτ)
    - Phasenversatz bei verschiedenen Frequenzen
    """
    
    def __init__(self, tau: float = 1.0):
        """
        Initialisierung des RC-Filters.
        
        Args:
            tau: Zeitkonstante τ = RC [Sekunden]
        """
        self.tau = tau
        self.exp_func = ExponentialFunction()
    
    @property
    def cutoff_frequency(self) -> float:
        """Grenzfrequenz f_c = 1/(2πτ) [Hz]"""
        return 1.0 / (2 * np.pi * self.tau)
    
    @property
    def omega_cutoff(self) -> float:
        """Grenzfrequenz ω_c = 1/τ [rad/s]"""
        return 1.0 / self.tau
    
    def differential_equation(self, V_out: float, t: float, V_in_func: Callable) -> float:
        """
        Differentialgleichung des RC-Filters.
        
        τ dV_out/dt + V_out = V_in(t)
        dV_out/dt = (V_in(t) - V_out) / τ
        
        Args:
            V_out: Ausgangsspannung
            t: Zeit
            V_in_func: Eingangsspannungsfunktion V_in(t)
            
        Returns:
            dV_out/dt
        """
        V_in = V_in_func(t)
        return (V_in - V_out) / self.tau
    
    def solve_ode(self, V_in_func: Callable, t_span: np.ndarray, 
                  V_out_0: float = 0.0) -> np.ndarray:
        """
        Löse die Differentialgleichung numerisch mittels ODE-Solver.
        
        Args:
            V_in_func: Eingangsspannungsfunktion V_in(t)
            t_span: Zeitpunkte
            V_out_0: Anfangsbedingung V_out(0)
            
        Returns:
            Ausgangsspannungen an den Zeitpunkten
        """
        def dydt(y, t):
            return self.differential_equation(y, t, V_in_func)
        
        solution = odeint(dydt, V_out_0, t_span)
        return solution.flatten()
    
    def analytical_solution_step(self, V_in: float, t: np.ndarray, 
                                V_out_0: float = 0.0) -> np.ndarray:
        """
        Analytische Lösung für Sprungantwort.
        
        Eingabe: V_in(t) = V_in (konstant für t ≥ 0)
        Lösung: V_out(t) = V_in * (1 - e^(-t/τ)) + V_out_0 * e^(-t/τ)
        
        Args:
            V_in: Eingangsspannung (konstant)
            t: Zeitpunkte
            V_out_0: Anfangsbedingung
            
        Returns:
            Ausgangsspannungen
        """
        return V_in * (1 - np.exp(-t / self.tau)) + V_out_0 * np.exp(-t / self.tau)
    
    def analytical_solution_ramp(self, slope: float, t: np.ndarray) -> np.ndarray:
        """
        Analytische Lösung für Rampeneingang.
        
        Eingabe: V_in(t) = slope * t
        
        Args:
            slope: Steigung der Rampe
            t: Zeitpunkte
            
        Returns:
            Ausgangsspannungen
        """
        return slope * (t - self.tau * (1 - np.exp(-t / self.tau)))
    
    def frequency_response(self, omega: Union[float, np.ndarray]) -> complex:
        """
        Frequenzgang H(jω) des Tiefpassfilters.
        
        H(jω) = 1 / (1 + jωτ)
        
        Args:
            omega: Kreisfrequenz (rad/s)
            
        Returns:
            Komplexer Frequenzgang
        """
        if isinstance(omega, np.ndarray):
            return 1.0 / (1.0 + 1j * omega * self.tau)
        else:
            return 1.0 / (1.0 + 1j * omega * self.tau)
    
    def magnitude_response(self, omega: Union[float, np.ndarray]) -> float:
        """
        Betrag des Frequenzgangs |H(jω)|.
        
        |H(jω)| = 1 / √(1 + (ωτ)²)
        
        Args:
            omega: Kreisfrequenz (rad/s)
            
        Returns:
            Betrag |H(jω)|
        """
        if isinstance(omega, np.ndarray):
            return 1.0 / np.sqrt(1.0 + (omega * self.tau)**2)
        else:
            return 1.0 / np.sqrt(1.0 + (omega * self.tau)**2)
    
    def phase_response(self, omega: Union[float, np.ndarray]) -> float:
        """
        Phase des Frequenzgangs ∠H(jω).
        
        ∠H(jω) = -arctan(ωτ)
        
        Args:
            omega: Kreisfrequenz (rad/s)
            
        Returns:
            Phase in Radiant
        """
        if isinstance(omega, np.ndarray):
            return -np.arctan(omega * self.tau)
        else:
            return -np.arctan(omega * self.tau)
    
    def attenuation_db(self, omega: Union[float, np.ndarray]) -> float:
        """
        Dämpfung in dB.
        
        A(ω) [dB] = 20 * log10(|H(jω)|) = -10 * log10(1 + (ωτ)²)
        
        Geschlossene Form ohne Epsilon-Offset: |H| > 0 für alle endlichen ω,
        daher ist kein Schutz vor log(0) nötig und A(0) = 0 dB exakt.
        
        Args:
            omega: Kreisfrequenz (rad/s)
            
        Returns:
            Dämpfung in dB (≤ 0)
        """
        return -10.0 * np.log10(1.0 + (omega * self.tau)**2)
    
    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """
        Impulsantwort (Stoßantwort) des Filters.
        
        h(t) = (1/τ) * e^(-t/τ) für t ≥ 0
        
        Args:
            t: Zeitpunkte
            
        Returns:
            Impulsantwort h(t)
        """
        return (1.0 / self.tau) * np.exp(-t / self.tau)
    
    def step_response(self, t: np.ndarray) -> np.ndarray:
        """
        Sprungantwort des Filters.
        
        s(t) = 1 - e^(-t/τ) für t ≥ 0
        
        Args:
            t: Zeitpunkte
            
        Returns:
            Sprungantwort s(t)
        """
        return 1 - np.exp(-t / self.tau)
    
    def settling_time_95(self) -> float:
        """
        Zeit, bis die Ausgabe auf 95% des Endwertes erreicht: t_95 ≈ 3τ
        
        Returns:
            Einschwingzeit (95%)
        """
        return 3.0 * self.tau
    
    def settling_time_99(self) -> float:
        """
        Zeit, bis die Ausgabe auf 99% des Endwertes erreicht: t_99 ≈ 4.6τ
        
        Returns:
            Einschwingzeit (99%)
        """
        return np.log(100) * self.tau


class RLFilter:
    """RL-Filter für Stromkreisanalyse."""
    
    def __init__(self, tau: float = 1.0):
        """
        Initialisierung des RL-Filters (L/R).
        
        Args:
            tau: Zeitkonstante τ = L/R
        """
        self.tau = tau
    
    def current_step_response(self, V_in: float, t: np.ndarray) -> np.ndarray:
        """
        Stromverlauf für Spannungssprung.
        
        I(t) = (V_in/R) * (1 - e^(-Rt/L))
        
        Args:
            V_in: Eingangsspannung
            t: Zeitpunkte
            
        Returns:
            Strom I(t)
        """
        return V_in * (1.0 - np.exp(-t / self.tau))
    
    def energy_stored(self, I: float, L: float) -> float:
        """
        Energie im Induktor: E = (1/2) * L * I²
        
        Args:
            I: Strom
            L: Induktivität
            
        Returns:
            Gespeicherte Energie
        """
        return 0.5 * L * I**2


class RLCFilter:
    """RLC-Filter für gedämpfte Schwingungen."""
    
    def __init__(self, L: float, R: float, C: float):
        """
        Initialisierung des RLC-Filters.
        
        Args:
            L: Induktivität
            R: Widerstand
            C: Kapazität
        """
        self.L = L
        self.R = R
        self.C = C
        self.omega_0 = 1.0 / np.sqrt(L * C)  # Eigenfrequenz
        self.zeta = R / 2.0 * np.sqrt(C / L)  # Dämpfungsverhältnis
    
    @property
    def damping_ratio(self) -> float:
        """Dämpfungsverhältnis ζ"""
        return self.zeta
    
    @property
    def natural_frequency(self) -> float:
        """Eigenfrequenz ω₀"""
        return self.omega_0
    
    @property
    def is_underdamped(self) -> bool:
        """Ist das System unterdämpft? (0 < ζ < 1)"""
        return 0 < self.zeta < 1
    
    @property
    def is_critically_damped(self) -> bool:
        """Ist das System kritisch gedämpft? (ζ = 1)"""
        return abs(self.zeta - 1) < 1e-10
    
    @property
    def is_overdamped(self) -> bool:
        """Ist das System überdämpft? (ζ > 1)"""
        return self.zeta > 1
    
    def step_response_underdamped(self, t: np.ndarray) -> np.ndarray:
        """Sprungantwort für unterdämpften Fall (ζ < 1)"""
        omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
        
        exp_term = np.exp(-self.zeta * self.omega_0 * t)
        cos_term = np.cos(omega_d * t)
        sin_term = (self.zeta / np.sqrt(1 - self.zeta**2)) * np.sin(omega_d * t)
        
        return 1 - exp_term * (cos_term + sin_term)
    
    def step_response_critically_damped(self, t: np.ndarray) -> np.ndarray:
        """Sprungantwort für kritisch gedämpften Fall (ζ = 1)"""
        return 1 - (1 + self.omega_0 * t) * np.exp(-self.omega_0 * t)
    
    def step_response_overdamped(self, t: np.ndarray) -> np.ndarray:
        """Sprungantwort für überdämpften Fall (ζ > 1)"""
        sqrt_term = np.sqrt(self.zeta**2 - 1)
        r1 = -self.omega_0 * (self.zeta - sqrt_term)
        r2 = -self.omega_0 * (self.zeta + sqrt_term)
        
        c1 = r2 / (r2 - r1)
        c2 = -r1 / (r2 - r1)
        
        return 1 - (c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t))
