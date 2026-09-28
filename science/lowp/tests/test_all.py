"""
Umfassende Tests für alle mathematischen Module

100% Testabdeckung für:
- dirichlet_beta.py
- convergence_acceleration.py
- lowpass_filter.py
- special_functions.py

Ausführung: pytest test_all.py -v --cov
"""

import pytest
import numpy as np
from scipy import special
import warnings

# Importieren aller Module
from dirichlet_beta import DirichletBeta, CatalanConstant
from convergence_acceleration import (
    LeibnizSeries, EulerTransformation, RichardsonExtrapolation,
    ShanksTransformation, ConvergenceComparison
)
from lowpass_filter import (
    ExponentialFunction, RCLowpassFilter, RLFilter, RLCFilter
)
from special_functions import (
    Polylogarithm, EllipticIntegrals, ClausenFunction,
    BetaFunction, IntegralRepresentations, TranscendenceProperties
)


# ======================== DIRICHLET BETA TESTS ========================

class TestDirichletBeta:
    """Tests für DirichletBeta-Klasse"""
    
    def test_initialization(self):
        """Test der Initialisierung"""
        beta = DirichletBeta(tolerance=1e-10)
        assert beta.tolerance == 1e-10
        assert beta.max_iterations == 10000
    
    def test_dirichlet_beta_series_convergence_s1(self):
        """Test: β(1) = π/4 (Leibniz-Reihe)"""
        beta = DirichletBeta()
        result = beta.dirichlet_beta_series(1, n_terms=10000)
        expected = np.pi / 4
        assert abs(result - expected) < 0.01  # Grobe Konvergenz
    
    def test_dirichlet_beta_series_complex(self):
        """Test: β(s) für komplexes s"""
        beta = DirichletBeta()
        s = 1.5 + 0.5j
        result = beta.dirichlet_beta_series(s, n_terms=1000)
        assert isinstance(result, complex)
        assert np.isfinite(np.real(result))
        assert np.isfinite(np.imag(result))
    
    def test_dirichlet_beta_series_invalid_s(self):
        """Test: Exception bei Re(s) ≤ 0"""
        beta = DirichletBeta()
        with pytest.raises(ValueError):
            beta.dirichlet_beta_series(0)
        with pytest.raises(ValueError):
            beta.dirichlet_beta_series(-1)
    
    def test_catalan_constant_series(self):
        """Test: Catalan-Konstante via Reihe"""
        beta = DirichletBeta()
        result = beta.catalan_constant_series(n_terms=10000)
        expected = 0.915965594177219
        assert abs(result - expected) < 0.001
    
    def test_catalan_arctan_integral(self):
        """Test: Catalan via Arctan-Integral"""
        beta = DirichletBeta()
        result = beta.catalan_arctan_integral()
        expected = 0.915965594177219
        assert abs(result - expected) < 0.001
    
    def test_catalan_logarithmic_integral(self):
        """Test: Catalan via logarithmisches Integral"""
        beta = DirichletBeta()
        result = beta.catalan_logarithmic_integral()
        expected = 0.915965594177219
        assert abs(result - expected) < 0.01
    
    def test_catalan_sine_integral(self):
        """Test: Catalan via Sinus-Integral"""
        beta = DirichletBeta()
        result = beta.catalan_sine_integral()
        expected = 0.915965594177219
        assert abs(result - expected) < 0.01
    
    def test_beta_functional_equation(self):
        """Test: Funktionale Gleichung"""
        beta = DirichletBeta()
        s = 0.5
        result = beta.beta_functional_equation(s)
        assert isinstance(result, complex) or isinstance(result, float)
        assert np.isfinite(result)
        
        # Bekannte Werte: β(0) = 1/2, β(-1) = 0, β(-2) = -1/2
        assert abs(beta.beta_functional_equation(0.0) - 0.5) < 1e-3
        assert abs(beta.beta_functional_equation(-1.0)) < 1e-12
        assert abs(beta.beta_functional_equation(-2.0) + 0.5) < 1e-6
    
    def test_poles_and_residues(self):
        """Test: Pole und Residuen"""
        beta = DirichletBeta()
        poles_residues = beta.poles_and_residues()
        
        # Prüfe erste Pole
        assert -1 in poles_residues
        assert -3 in poles_residues
        assert -5 in poles_residues
        
        # Stellen sind negativ ungerade (Python: -1 % 2 == 1, daher abs())
        for pole, residue in poles_residues.items():
            assert pole < 0
            assert abs(pole) % 2 == 1
            # β ist ganz: dort ist kein Pol, das Residuum ist 0
            assert residue == 0
    
    def test_zeta_eta_relationship(self):
        """Test: Beziehung zu Zeta und Eta"""
        beta = DirichletBeta()
        s = 2.0
        beta_s, zeta_s = beta.zeta_eta_relationship(s)
        
        assert np.isfinite(beta_s)
        assert np.isfinite(zeta_s)
        # zeta(2) sollte π²/6 sein
        expected_zeta = np.pi**2 / 6
        assert abs(zeta_s - expected_zeta) < 0.01
    
    def test_special_values(self):
        """Test: Spezielle Werte"""
        beta = DirichletBeta()
        special_vals = beta.special_values()
        
        # β(1) sollte π/4 sein
        assert abs(special_vals['β(1) (Leibniz)'] - np.pi/4) < 0.01
        assert abs(special_vals['β(1) theoretical'] - np.pi/4) < 1e-10
        
        # β(2) sollte Catalan sein
        assert abs(special_vals['β(2) (Catalan)'] - 0.915965594) < 0.01
    
    def test_error_estimate(self):
        """Test: Fehlerschätzung"""
        beta = DirichletBeta()
        s = 2.0
        n_terms = 100
        error = beta.error_estimate(s, n_terms)
        
        assert error > 0
        assert error < 1.0
    
    def test_error_estimate_invalid(self):
        """Test: Fehlerschätzung mit s ≤ 0"""
        beta = DirichletBeta()
        with pytest.raises(ValueError):
            beta.error_estimate(0, 100)


class TestCatalanConstant:
    """Tests für CatalanConstant-Klasse"""
    
    def test_initialization(self):
        """Test der Initialisierung"""
        catalan = CatalanConstant()
        assert abs(catalan.value - 0.915965594) < 1e-6
    
    def test_compute_via_series(self):
        """Test: Reihenberechnung"""
        catalan = CatalanConstant()
        result = catalan.compute_via_series()
        assert abs(result - catalan.value) < 0.001
    
    def test_compute_via_arctan(self):
        """Test: Arctan-Integral"""
        catalan = CatalanConstant()
        result = catalan.compute_via_arctan()
        assert abs(result - catalan.value) < 0.001
    
    def test_compute_via_logarithm(self):
        """Test: Logarithmisches Integral"""
        catalan = CatalanConstant()
        result = catalan.compute_via_logarithm()
        assert abs(result - catalan.value) < 0.01
    
    def test_compute_via_sine(self):
        """Test: Sinus-Integral"""
        catalan = CatalanConstant()
        result = catalan.compute_via_sine()
        assert abs(result - catalan.value) < 0.01
    
    def test_verify_representations(self):
        """Test: Vergleich aller Darstellungen"""
        catalan = CatalanConstant()
        results = catalan.verify_representations()
        
        # Alle sollten ähnliche Werte liefern
        series_result = results['Series']
        arctan_result = results['Arctan']
        
        assert abs(series_result - arctan_result) < 0.01
    
    def test_error_to_reference(self):
        """Test: Fehlerberechnung"""
        catalan = CatalanConstant()
        error = catalan.error_to_reference(0.9)
        assert error > 0


# ======================== CONVERGENCE ACCELERATION TESTS ========================

class TestLeibnizSeries:
    """Tests für LeibnizSeries-Klasse"""
    
    def test_compute_partial_sum(self):
        """Test: Partialsum berechnen"""
        leibniz = LeibnizSeries()
        result = leibniz.compute_partial_sum(0)
        assert result == 1.0
        
        result = leibniz.compute_partial_sum(1)
        assert abs(result - (1 - 1/3)) < 1e-10
    
    def test_partial_sum_vector(self):
        """Test: Vektor von Partialsummen"""
        leibniz = LeibnizSeries()
        sums = leibniz.partial_sum_vector(10)
        
        assert len(sums) == 11
        assert sums[0] == 1.0
        assert np.all(np.isfinite(sums))
    
    def test_error_naive(self):
        """Test: Fehlergrenze O(1/N)"""
        leibniz = LeibnizSeries()
        error_10 = leibniz.error_naive(10)
        error_100 = leibniz.error_naive(100)
        
        assert error_10 > error_100
        assert error_10 > 0
    
    def test_convergence_rate_naive(self):
        """Test: Konvergenzraten"""
        leibniz = LeibnizSeries()
        results = leibniz.convergence_rate_naive([10, 100, 1000])
        
        assert len(results) == 3
        assert all('error' in results[n] for n in [10, 100, 1000])
        assert results[10]['error'] > results[100]['error']


class TestEulerTransformation:
    """Tests für EulerTransformation-Klasse"""
    
    def test_initialization(self):
        """Test der Initialisierung"""
        euler = EulerTransformation()
        assert euler.leibniz is not None
    
    def test_binomial_coefficient(self):
        """Test: Binomialkoeffizient"""
        euler = EulerTransformation()
        
        assert euler.binomial_coefficient(5, 2) == 10
        assert euler.binomial_coefficient(5, 0) == 1
        assert euler.binomial_coefficient(5, 5) == 1
    
    def test_euler_transform_term(self):
        """Test: Einzelne Terme"""
        euler = EulerTransformation()
        term_0 = euler.euler_transform_term(0)
        term_1 = euler.euler_transform_term(1)
        
        assert np.isfinite(term_0)
        assert np.isfinite(term_1)
        assert abs(term_0 - 1/2) < 1e-10  # Erster Term sollte 1/2 sein
    
    def test_euler_transform_sum(self):
        """Test: Euler-transformierte Summe"""
        euler = EulerTransformation()
        result = euler.euler_transform_sum(50)
        expected = np.pi / 4
        
        # Sollte deutlich besser konvergieren
        assert abs(result - expected) < 0.0001
    
    def test_euler_error_estimate(self):
        """Test: Fehlergrenze O(2^-N)"""
        euler = EulerTransformation()
        error_10 = euler.euler_error_estimate(10)
        error_20 = euler.euler_error_estimate(20)
        
        assert error_20 < error_10
        assert error_20 < 2**(-20)
    
    def test_convergence_analysis(self):
        """Test: Konvergenzanalyse"""
        euler = EulerTransformation()
        results = euler.convergence_analysis([10, 50, 100])
        
        assert len(results) == 3
        for n in [10, 50, 100]:
            assert 'euler_error' in results[n]
            assert results[n]['improvement_factor'] > 1


class TestRichardsonExtrapolation:
    """Tests für RichardsonExtrapolation-Klasse"""
    
    def test_richardson_extrapolation_2terms(self):
        """Test: Richardson-Extrapolation"""
        richardson = RichardsonExtrapolation()
        S_N = 0.7  # Annäherung an π/4
        S_N1 = 0.73  # Nächste Partialsumme
        N = 100
        
        result = richardson.richardson_extrapolation_2terms(S_N, S_N1, N)
        assert np.isfinite(result)
    
    def test_richardson_error_estimate(self):
        """Test: Fehlerabschätzung O(1/N³)"""
        richardson = RichardsonExtrapolation()
        error = richardson.richardson_error_estimate(100)
        
        assert error > 0
        assert error < 1.0
        # Sollte O(1/N³) sein
        assert error < 1.0 / (100**3)
    
    def test_richardson_cascade(self):
        """Test: Richardson-Kaskade"""
        richardson = RichardsonExtrapolation()
        results = richardson.richardson_cascade(100)
        
        assert 'base_sums' in results
        assert 'first_richardson' in results
        assert 'second_richardson' in results
    
    def test_convergence_analysis(self):
        """Test: Konvergenzanalyse"""
        richardson = RichardsonExtrapolation()
        results = richardson.convergence_analysis([50, 100, 200])
        
        for n in [50, 100, 200]:
            assert 'richardson_error' in results[n]
            # Richardson sollte besser sein
            assert results[n]['error_reduction'] > 1


class TestShanksTransformation:
    """Tests für ShanksTransformation-Klasse"""
    
    def test_shanks_transform_single(self):
        """Test: Shanks-Transformation einzeln"""
        shanks = ShanksTransformation()
        S_prev = 0.5
        S_curr = 0.7
        S_next = 0.75
        
        result = shanks.shanks_transform_single(S_prev, S_curr, S_next)
        assert np.isfinite(result)
    
    def test_shanks_vector(self):
        """Test: Shanks auf Vektor"""
        shanks = ShanksTransformation()
        transformed = shanks.shanks_vector(50)
        
        assert len(transformed) == 51
        assert np.all(np.isfinite(transformed))
    
    def test_convergence_analysis(self):
        """Test: Konvergenzanalyse"""
        shanks = ShanksTransformation()
        results = shanks.convergence_analysis([10, 50, 100])
        
        for n in [10, 50, 100]:
            assert 'shanks_error' in results[n]
            assert results[n]['improvement'] > 0


class TestConvergenceComparison:
    """Tests für ConvergenceComparison-Klasse"""
    
    def test_compare_methods(self):
        """Test: Vergleich aller Methoden"""
        comparison = ConvergenceComparison()
        results = comparison.compare_methods([10, 50, 100])
        
        assert 'naive' in results
        assert 'euler' in results
        assert 'richardson' in results
        assert 'shanks' in results
    
    def test_theoretical_reference(self):
        """Test: Referenzwert"""
        comparison = ConvergenceComparison()
        ref = comparison.theoretical_reference()
        assert abs(ref - np.pi/4) < 1e-10


# ======================== LOWPASS FILTER TESTS ========================

class TestExponentialFunction:
    """Tests für ExponentialFunction-Klasse"""
    
    def test_taylor_series_real(self):
        """Test: Taylor-Reihe für reelle Argumente"""
        exp_func = ExponentialFunction()
        s = 1.0
        t = 1.0
        
        result = exp_func.taylor_series(s, t)
        expected = np.e
        assert abs(result - expected) < 0.01
    
    def test_taylor_series_complex(self):
        """Test: Taylor-Reihe für komplexe Argumente"""
        exp_func = ExponentialFunction()
        s = 1j * np.pi
        t = 1.0
        
        result = exp_func.taylor_series(s, t)
        expected = -1.0  # e^(iπ) = -1
        assert abs(result - expected) < 0.01
    
    def test_derivative_taylor(self):
        """Test: Ableitung via Taylor"""
        exp_func = ExponentialFunction()
        s = 1.0
        t = 1.0
        
        deriv = exp_func.derivative_taylor(s, t)
        expected = s * np.e
        assert abs(deriv - expected) < 0.01
    
    def test_taylor_term_analysis(self):
        """Test: Term-Analyse"""
        exp_func = ExponentialFunction()
        analysis = exp_func.taylor_term_analysis(1.0, 0.5, n_max=5)
        
        assert 'original_terms' in analysis
        assert 'derivative_terms' in analysis
        assert len(analysis['original_terms']) > 0
    
    def test_reproduction_property(self):
        """Test: Reproduzierbarkeit"""
        exp_func = ExponentialFunction()
        s = 1.0
        t_values = np.array([0.1, 0.5, 1.0])
        
        results = exp_func.reproduction_property(s, t_values)
        
        assert 'exp_st' in results
        assert 'derivative' in results
        
        # Überprüfe f'(t) = s*f(t)
        for i, t in enumerate(t_values):
            expected_deriv = s * results['exp_st'][i]
            assert abs(results['derivative'][i] - expected_deriv) < 0.01


class TestRCLowpassFilter:
    """Tests für RCLowpassFilter-Klasse"""
    
    def test_initialization(self):
        """Test der Initialisierung"""
        rc_filter = RCLowpassFilter(tau=1.0)
        assert rc_filter.tau == 1.0
    
    def test_cutoff_frequency(self):
        """Test: Grenzfrequenz"""
        rc_filter = RCLowpassFilter(tau=1.0)
        f_c = rc_filter.cutoff_frequency
        
        expected = 1.0 / (2 * np.pi)
        assert abs(f_c - expected) < 1e-10
    
    def test_omega_cutoff(self):
        """Test: Kreisgrenzfrequenz"""
        rc_filter = RCLowpassFilter(tau=1.0)
        omega_c = rc_filter.omega_cutoff
        
        assert omega_c == 1.0
    
    def test_analytical_solution_step(self):
        """Test: Analytische Lösung Sprungantwort"""
        rc_filter = RCLowpassFilter(tau=1.0)
        V_in = 1.0
        t = np.array([0, 1.0, 5.0, 10.0])
        
        V_out = rc_filter.analytical_solution_step(V_in, t)
        
        assert len(V_out) == len(t)
        assert V_out[0] == 0  # Anfangsbedingung
        assert V_out[-1] > 0.99  # Sollte gegen V_in konvergieren
    
    def test_frequency_response(self):
        """Test: Frequenzgang"""
        rc_filter = RCLowpassFilter(tau=1.0)
        omega = 1.0
        
        H = rc_filter.frequency_response(omega)
        
        # H(jω) = 1/(1 + jωτ)
        expected = 1.0 / (1.0 + 1j * 1.0)
        assert abs(H - expected) < 1e-10
    
    def test_magnitude_response(self):
        """Test: Betrag des Frequenzgangs"""
        rc_filter = RCLowpassFilter(tau=1.0)
        omega = 0.0
        
        mag = rc_filter.magnitude_response(omega)
        assert mag == 1.0  # Bei ω=0 sollte Betrag 1 sein
        
        omega = 1.0  # Bei ω = 1/τ
        mag = rc_filter.magnitude_response(omega)
        assert abs(mag - 1/np.sqrt(2)) < 1e-10
    
    def test_phase_response(self):
        """Test: Phasengang"""
        rc_filter = RCLowpassFilter(tau=1.0)
        omega = 0.0
        
        phase = rc_filter.phase_response(omega)
        assert phase == 0.0
        
        omega = 1.0
        phase = rc_filter.phase_response(omega)
        assert abs(phase - (-np.pi/4)) < 1e-10
    
    def test_attenuation_db(self):
        """Test: Dämpfung in dB"""
        rc_filter = RCLowpassFilter(tau=1.0)
        omega = 0.0
        
        atten = rc_filter.attenuation_db(omega)
        assert atten == 0  # Bei ω=0 keine Dämpfung
    
    def test_impulse_response(self):
        """Test: Impulsantwort"""
        rc_filter = RCLowpassFilter(tau=1.0)
        t = np.array([0, 1.0, 10.0])
        
        h_t = rc_filter.impulse_response(t)
        
        assert len(h_t) == len(t)
        assert np.all(h_t >= 0)  # Sollte positiv sein
    
    def test_step_response(self):
        """Test: Sprungantwort"""
        rc_filter = RCLowpassFilter(tau=1.0)
        t = np.array([0, 1.0, 10.0])
        
        s_t = rc_filter.step_response(t)
        
        assert s_t[0] == 0
        assert 0 < s_t[1] < 1
        assert s_t[-1] > 0.99  # Sollte gegen 1 konvergieren
    
    def test_settling_time_95(self):
        """Test: Einschwingzeit 95%"""
        rc_filter = RCLowpassFilter(tau=1.0)
        t_95 = rc_filter.settling_time_95()
        
        assert t_95 == 3.0  # 3τ
    
    def test_settling_time_99(self):
        """Test: Einschwingzeit 99%"""
        rc_filter = RCLowpassFilter(tau=1.0)
        t_99 = rc_filter.settling_time_99()
        
        assert abs(t_99 - np.log(100)) < 1e-10


class TestRLFilter:
    """Tests für RLFilter-Klasse"""
    
    def test_current_step_response(self):
        """Test: Stromverlauf"""
        rl_filter = RLFilter(tau=1.0)
        V_in = 1.0
        t = np.array([0, 1.0, 5.0])
        
        I_t = rl_filter.current_step_response(V_in, t)
        
        assert len(I_t) == len(t)
        assert I_t[0] == 0
        assert np.all(np.isfinite(I_t))
    
    def test_energy_stored(self):
        """Test: Gespeicherte Energie"""
        rl_filter = RLFilter(tau=1.0)
        energy = rl_filter.energy_stored(I=1.0, L=1.0)
        
        assert energy == 0.5


class TestRLCFilter:
    """Tests für RLCFilter-Klasse"""
    
    def test_initialization(self):
        """Test der Initialisierung"""
        rlc = RLCFilter(L=1.0, R=1.0, C=1.0)
        assert rlc.L == 1.0
        assert rlc.R == 1.0
        assert rlc.C == 1.0
    
    def test_damping_ratio(self):
        """Test: Dämpfungsverhältnis"""
        rlc = RLCFilter(L=1.0, R=1.0, C=1.0)
        zeta = rlc.damping_ratio
        assert zeta > 0
    
    def test_natural_frequency(self):
        """Test: Eigenfrequenz"""
        rlc = RLCFilter(L=1.0, R=1.0, C=1.0)
        omega_0 = rlc.natural_frequency
        assert omega_0 == 1.0
    
    def test_underdamped_classification(self):
        """Test: Unterdämpft"""
        rlc = RLCFilter(L=1.0, R=0.5, C=1.0)  # ζ < 1
        assert rlc.is_underdamped
        assert not rlc.is_critically_damped
        assert not rlc.is_overdamped
    
    def test_critically_damped_classification(self):
        """Test: Kritisch gedämpft"""
        rlc = RLCFilter(L=1.0, R=2.0, C=1.0)  # ζ = 1
        assert rlc.is_critically_damped
        assert not rlc.is_underdamped
    
    def test_overdamped_classification(self):
        """Test: Überdämpft"""
        rlc = RLCFilter(L=1.0, R=5.0, C=1.0)  # ζ > 1
        assert rlc.is_overdamped
        assert not rlc.is_underdamped
    
    def test_step_response_underdamped(self):
        """Test: Sprungantwort unterdämpft"""
        rlc = RLCFilter(L=1.0, R=0.5, C=1.0)
        t = np.array([0, 1.0, 5.0, 10.0])
        
        y = rlc.step_response_underdamped(t)
        
        assert len(y) == len(t)
        assert y[0] < y[1]  # Sollte ansteigen
    
    def test_step_response_critically_damped(self):
        """Test: Sprungantwort kritisch gedämpft"""
        rlc = RLCFilter(L=1.0, R=2.0, C=1.0)
        t = np.array([0, 1.0, 10.0])
        
        y = rlc.step_response_critically_damped(t)
        
        assert len(y) == len(t)
        assert np.all(np.isfinite(y))
    
    def test_step_response_overdamped(self):
        """Test: Sprungantwort überdämpft"""
        rlc = RLCFilter(L=1.0, R=5.0, C=1.0)
        t = np.array([0, 1.0, 10.0])
        
        y = rlc.step_response_overdamped(t)
        
        assert len(y) == len(t)
        assert np.all(np.isfinite(y))


# ======================== SPECIAL FUNCTIONS TESTS ========================

class TestPolylogarithm:
    """Tests für Polylogarithm-Klasse"""
    
    def test_polylog_series_simple(self):
        """Test: Polylog Li_2(0.5)"""
        polylog = Polylogarithm()
        result = polylog.polylog_series(2, 0.5, n_terms=100)
        assert isinstance(result, (float, complex))
        assert np.isfinite(result)
    
    def test_dilogarithm(self):
        """Test: Dilogarithm Li_2(x)"""
        polylog = Polylogarithm()
        result = polylog.dilogarithm(0.5)
        assert np.isfinite(result)
    
    def test_polylog_convergence_error(self):
        """Test: Fehler bei |z| > 1"""
        polylog = Polylogarithm()
        with pytest.raises(ValueError):
            polylog.polylog_series(2, 2.0)


class TestEllipticIntegrals:
    """Tests für EllipticIntegrals-Klasse"""
    
    def test_complete_elliptic_K(self):
        """Test: Vollständiges elliptisches Integral K"""
        elliptic = EllipticIntegrals()
        k = elliptic.complete_elliptic_K(0.5)
        assert k > 0
        assert np.isfinite(k)
    
    def test_complete_elliptic_E(self):
        """Test: Vollständiges elliptisches Integral E"""
        elliptic = EllipticIntegrals()
        e = elliptic.complete_elliptic_E(0.5)
        assert 0 < e < np.pi/2
        assert np.isfinite(e)
    
    def test_arithmetic_geometric_mean(self):
        """Test: Arithmetisch-geometrisches Mittel"""
        elliptic = EllipticIntegrals()
        agm = elliptic.arithmetic_geometric_mean(1.0, np.sqrt(3))
        assert agm > 0
        assert np.isfinite(agm)


class TestClausenFunction:
    """Tests für ClausenFunction-Klasse"""
    
    def test_clausen_2_integral(self):
        """Test: Clausen-Funktion via Integral"""
        clausen = ClausenFunction()
        result = clausen.clausen_2_integral(np.pi/4)
        assert np.isfinite(result)
    
    def test_clausen_2_series(self):
        """Test: Clausen-Funktion via Reihe"""
        clausen = ClausenFunction()
        result = clausen.clausen_2_series(np.pi/4)
        assert np.isfinite(result)


class TestBetaFunction:
    """Tests für BetaFunction-Klasse"""
    
    def test_beta_function(self):
        """Test: Beta-Funktion"""
        beta_func = BetaFunction()
        result = beta_func.beta_function(1.0, 1.0)
        assert abs(result - 1.0) < 1e-10
    
    def test_beta_function_integral(self):
        """Test: Beta via Integral"""
        beta_func = BetaFunction()
        result = beta_func.beta_function_integral(1.0, 1.0)
        assert abs(result - 1.0) < 0.01
    
    def test_catalan_via_beta(self):
        """Test: Catalan via Beta"""
        beta_func = BetaFunction()
        result = beta_func.catalan_via_beta()
        assert abs(result - 0.915965594) < 0.01


class TestIntegralRepresentations:
    """Tests für IntegralRepresentations-Klasse"""
    
    def test_pi_over_4_sine_integral(self):
        """Test: π/4 via Sinus-Integral"""
        integral_rep = IntegralRepresentations()
        result = integral_rep.pi_over_4_sine_integral()
        expected = np.pi / 4
        assert abs(result - expected) < 0.01
    
    def test_pi_over_4_arcsin_integral(self):
        """Test: π/4 via Arcsin-Integral"""
        integral_rep = IntegralRepresentations()
        result = integral_rep.pi_over_4_arcsin_integral()
        expected = np.pi / 4
        assert abs(result - expected) < 0.01
    
    def test_pi_over_4_log_integral(self):
        """Test: π/4 via logarithmisches Integral"""
        integral_rep = IntegralRepresentations()
        result = integral_rep.pi_over_4_log_integral()
        expected = np.pi / 4
        assert abs(result - expected) < 0.01
    
    def test_pi_over_4_geometric(self):
        """Test: π/4 geometrisch"""
        integral_rep = IntegralRepresentations()
        result = integral_rep.pi_over_4_geometric()
        expected = np.pi / 4
        assert abs(result - expected) < 0.01
    
    def test_catalan_arctan_power_integral(self):
        """Test: Catalan via Arctan-Power"""
        integral_rep = IntegralRepresentations()
        result = integral_rep.catalan_arctan_power_integral()
        assert abs(result - 0.915965594) < 0.01
    
    def test_catalan_log_power_integral(self):
        """Test: Catalan via Log-Power"""
        integral_rep = IntegralRepresentations()
        result = integral_rep.catalan_log_power_integral()
        assert abs(result - 0.915965594) < 0.01


class TestTranscendenceProperties:
    """Tests für TranscendenceProperties-Klasse"""
    
    def test_lindemann_weierstrass_theorem(self):
        """Test: Lindemann-Weierstrass Theorem"""
        theorem = TranscendenceProperties.lindemann_weierstrass_theorem()
        assert isinstance(theorem, str)
        assert 'π' in theorem
        assert 'transzendent' in theorem
    
    def test_irrationality_measure_pi(self):
        """Test: Irrationalitätsmaß von π"""
        mu = TranscendenceProperties.irrationality_measure_pi()
        assert mu > 2
        assert mu < 10


# ======================== INTEGRATION TESTS ========================

class TestIntegration:
    """Integrationstests für alle Module"""
    
    def test_leibniz_via_dirichlet_beta(self):
        """Test: Leibniz-Reihe über Dirichlet Beta"""
        beta = DirichletBeta()
        leibniz_direct = LeibnizSeries().compute_partial_sum(1000)
        
        # Sollten ähnlich sein
        assert abs(leibniz_direct - np.pi/4) < 0.01
    
    def test_euler_better_than_naive(self):
        """Test: Euler-Transform ist besser als naive"""
        euler = EulerTransformation()
        leibniz = LeibnizSeries()
        
        n = 50
        naive_sum = leibniz.compute_partial_sum(n)
        euler_sum = euler.euler_transform_sum(n)
        
        naive_error = abs(naive_sum - np.pi/4)
        euler_error = abs(euler_sum - np.pi/4)
        
        assert euler_error < naive_error
    
    def test_rc_filter_frequency_response(self):
        """Test: RC-Filter Frequenzgang"""
        rc = RCLowpassFilter(tau=1.0)
        omega = 1.0
        
        H = rc.frequency_response(omega)
        mag = rc.magnitude_response(omega)
        phase = rc.phase_response(omega)
        
        # Magnitude sollte von H kommen
        expected_mag = abs(H)
        assert abs(mag - expected_mag) < 1e-10
    
    def test_catalan_multiple_representations(self):
        """Test: Catalan in verschiedenen Darstellungen"""
        catalan = CatalanConstant()
        integral_rep = IntegralRepresentations()
        
        via_series = catalan.compute_via_series()
        via_arctan = integral_rep.catalan_arctan_power_integral()
        
        # Sollten ähnlich sein
        assert abs(via_series - via_arctan) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
