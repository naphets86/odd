"""
test_uml_marte_stability.py

Vollstaendige Test-Suite fuer das Modul uml_marte_stability.py.
Ziel: 100 % Abdeckung aller Klassen, Methoden und Zweige.

Ausfuehren:
    python -m pytest test_uml_marte_stability.py -v          (mit pytest)
    python -m unittest test_uml_marte_stability -v           (ohne pytest)
    python test_uml_marte_stability.py                       (direkt)
"""

import io
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src.uml_marte_stability import (
    # Enumerationen
    StabilityMode,
    StabilityStatus,
    # Datenklassen / Stereotypen
    MarteStereotype,
    TimingConstraint,
    StabilityAnnotation,
    DecayRateAnnotation,
    LyapunovFunctionAnnotation,
    EquilibriumAnnotation,
    AnnotatedElement,
    # Analyse-Engine
    StabilityAnalyzer,
    # Code-Generator
    StabilityAwareCodeGenerator,
    # Demo-Funktionen
    demo_elevator_door_stability,
    demo_equilibrium_vs_stability,
)


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================

def _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION, **kwargs):
    """Erzeugt eine StabilityAnnotation mit Standardwerten."""
    defaults = dict(
        name="TestAnnotation",
        base_metaclass="Component",
        stability_mode=mode,
        lyapunov_margin=0.05,
        decay_rate_min=0.01,
        spectral_radius_ub=0.99,
        rho_ub=0.90,
        monitoring_period=1000.0,
    )
    defaults.update(kwargs)
    return StabilityAnnotation(**defaults)


def _make_element(name="TestElement", etype="Class", stability=None,
                  timing=None, decay=None, lyapunov=None, equilibrium=None):
    return AnnotatedElement(
        element_name=name,
        element_type=etype,
        stability=stability,
        timing=timing,
        decay=decay,
        lyapunov=lyapunov,
        equilibrium=equilibrium,
    )


# ===========================================================================
# 1. Enumerationen
# ===========================================================================

class TestStabilityMode(unittest.TestCase):

    def test_all_members_exist(self):
        members = {m.name for m in StabilityMode}
        self.assertIn("LYAPUNOV_ASYMPTOTIC", members)
        self.assertIn("LYAPUNOV_MARGINAL", members)
        self.assertIn("BIBO", members)
        self.assertIn("SPECTRAL", members)
        self.assertIn("QUEUE_UTILIZATION", members)

    def test_members_are_unique(self):
        values = [m.value for m in StabilityMode]
        self.assertEqual(len(values), len(set(values)))


class TestStabilityStatus(unittest.TestCase):

    def test_string_values(self):
        self.assertEqual(StabilityStatus.STABLE.value,   "stabil")
        self.assertEqual(StabilityStatus.UNSTABLE.value, "instabil")
        self.assertEqual(StabilityStatus.MARGINAL.value, "marginal")
        self.assertEqual(StabilityStatus.UNKNOWN.value,  "unbekannt")

    def test_all_four_members(self):
        self.assertEqual(len(StabilityStatus), 4)


# ===========================================================================
# 2. Datenklassen / Stereotypen
# ===========================================================================

class TestMarteStereotype(unittest.TestCase):

    def test_creation(self):
        s = MarteStereotype(name="MyStereotype", base_metaclass="Class")
        self.assertEqual(s.name, "MyStereotype")
        self.assertEqual(s.base_metaclass, "Class")


class TestTimingConstraint(unittest.TestCase):

    def test_defaults(self):
        tc = TimingConstraint(min_duration=10, max_duration=100,
                              typical_duration=50)
        self.assertEqual(tc.unit, "ms")
        self.assertIsNone(tc.timeout)
        self.assertIsNone(tc.deadline)

    def test_full_construction(self):
        tc = TimingConstraint(
            min_duration=1, max_duration=5, typical_duration=3,
            unit="s", timeout=4.0, deadline=5.0,
        )
        self.assertEqual(tc.unit, "s")
        self.assertEqual(tc.timeout, 4.0)
        self.assertEqual(tc.deadline, 5.0)


class TestStabilityAnnotation(unittest.TestCase):

    def test_defaults(self):
        ann = StabilityAnnotation(name="A", base_metaclass="Class")
        self.assertEqual(ann.stability_mode, StabilityMode.LYAPUNOV_ASYMPTOTIC)
        self.assertAlmostEqual(ann.lyapunov_margin, 0.05)
        self.assertAlmostEqual(ann.decay_rate_min, 0.01)
        self.assertAlmostEqual(ann.spectral_radius_ub, 0.99)
        self.assertAlmostEqual(ann.rho_ub, 0.90)
        self.assertAlmostEqual(ann.monitoring_period, 1000.0)

    def test_custom_values(self):
        ann = _make_stability_ann(
            mode=StabilityMode.SPECTRAL,
            spectral_radius_ub=0.80,
            monitoring_period=500.0,
        )
        self.assertEqual(ann.stability_mode, StabilityMode.SPECTRAL)
        self.assertAlmostEqual(ann.spectral_radius_ub, 0.80)
        self.assertAlmostEqual(ann.monitoring_period, 500.0)


class TestDecayRateAnnotation(unittest.TestCase):

    def test_defaults(self):
        d = DecayRateAnnotation(name="D", base_metaclass="Operation")
        self.assertEqual(d.alpha_min, 0.0)
        self.assertEqual(d.alpha_max, float('inf'))
        self.assertAlmostEqual(d.convergence_constant_ub, 10.0)

    def test_custom(self):
        d = DecayRateAnnotation(name="D", base_metaclass="Operation",
                                alpha_min=0.5, alpha_max=5.0,
                                convergence_constant_ub=2.0)
        self.assertAlmostEqual(d.alpha_min, 0.5)
        self.assertAlmostEqual(d.alpha_max, 5.0)
        self.assertAlmostEqual(d.convergence_constant_ub, 2.0)


class TestLyapunovFunctionAnnotation(unittest.TestCase):

    def test_defaults(self):
        l = LyapunovFunctionAnnotation(name="L", base_metaclass="Class")
        self.assertEqual(l.function_type, "quadratic")
        self.assertFalse(l.positive_definite_verified)
        self.assertFalse(l.derivative_negative)
        self.assertEqual(l.custom_description, "")

    def test_custom(self):
        l = LyapunovFunctionAnnotation(
            name="L", base_metaclass="Class",
            function_type="entropy",
            positive_definite_verified=True,
            derivative_negative=True,
            custom_description="V = -sum(p*log(p))",
        )
        self.assertEqual(l.function_type, "entropy")
        self.assertTrue(l.positive_definite_verified)
        self.assertTrue(l.derivative_negative)
        self.assertEqual(l.custom_description, "V = -sum(p*log(p))")


class TestEquilibriumAnnotation(unittest.TestCase):

    def test_defaults(self):
        e = EquilibriumAnnotation(name="E", base_metaclass="Component")
        self.assertTrue(e.is_equilibrium_only)
        self.assertFalse(e.supports_stability_proof)
        self.assertEqual(e.equilibrium_law, "")

    def test_custom(self):
        e = EquilibriumAnnotation(
            name="E", base_metaclass="Component",
            is_equilibrium_only=False,
            supports_stability_proof=True,
            equilibrium_law="L = lambda * W",
        )
        self.assertFalse(e.is_equilibrium_only)
        self.assertTrue(e.supports_stability_proof)
        self.assertEqual(e.equilibrium_law, "L = lambda * W")


class TestAnnotatedElement(unittest.TestCase):

    def test_minimal_construction(self):
        el = AnnotatedElement(element_name="Foo", element_type="Class")
        self.assertEqual(el.element_name, "Foo")
        self.assertEqual(el.element_type, "Class")
        self.assertIsNone(el.timing)
        self.assertIsNone(el.stability)
        self.assertIsNone(el.decay)
        self.assertIsNone(el.lyapunov)
        self.assertIsNone(el.equilibrium)

    def test_full_construction(self):
        ann = _make_stability_ann()
        tc  = TimingConstraint(1, 10, 5, deadline=10)
        dr  = DecayRateAnnotation(name="D", base_metaclass="Op")
        lf  = LyapunovFunctionAnnotation(name="L", base_metaclass="Class")
        eq  = EquilibriumAnnotation(name="Eq", base_metaclass="Component")
        el  = _make_element(stability=ann, timing=tc, decay=dr,
                            lyapunov=lf, equilibrium=eq)
        self.assertIsNotNone(el.timing)
        self.assertIsNotNone(el.stability)
        self.assertIsNotNone(el.decay)
        self.assertIsNotNone(el.lyapunov)
        self.assertIsNotNone(el.equilibrium)


# ===========================================================================
# 3. StabilityAnalyzer – lyapunov_continuous
# ===========================================================================

class TestLyapunovContinuous(unittest.TestCase):

    def _ann(self, decay_rate_min=0.5, lyapunov_margin=0.1):
        return _make_stability_ann(
            mode=StabilityMode.LYAPUNOV_ASYMPTOTIC,
            decay_rate_min=decay_rate_min,
            lyapunov_margin=lyapunov_margin,
        )

    def test_stable_system(self):
        """Eigenwerte -1.5 und -2.0 => alpha > decay_rate_min + margin => STABLE."""
        A = np.array([[-1.5, 0.0],
                      [0.0, -2.0]])
        status, details = StabilityAnalyzer.lyapunov_continuous(A, self._ann())
        self.assertEqual(status, StabilityStatus.STABLE)
        self.assertAlmostEqual(details["decay_rate_alpha"], 1.5, places=5)
        self.assertAlmostEqual(details["max_real_part"],   -1.5, places=5)
        self.assertIn("eigenvalues", details)
        self.assertIn("required_alpha_min", details)
        self.assertIn("margin", details)

    def test_unstable_system(self):
        """Eigenwert +1 => alpha negativ => UNSTABLE."""
        A = np.array([[1.0, 0.0],
                      [0.0, -2.0]])
        status, details = StabilityAnalyzer.lyapunov_continuous(A, self._ann())
        self.assertEqual(status, StabilityStatus.UNSTABLE)
        self.assertLess(details["decay_rate_alpha"], 0)

    def test_marginal_system(self):
        """alpha knapp ueber decay_rate_min aber unter decay_rate_min + margin => MARGINAL."""
        # decay_rate_min=0.5, margin=0.1 => STABLE threshold = 0.6
        # Eigenwert bei -0.55 => alpha=0.55 => 0.5 < 0.55 < 0.6 => MARGINAL
        A = np.array([[-0.55, 0.0],
                      [0.0, -1.0]])
        ann = self._ann(decay_rate_min=0.5, lyapunov_margin=0.1)
        status, _ = StabilityAnalyzer.lyapunov_continuous(A, ann)
        self.assertEqual(status, StabilityStatus.MARGINAL)

    def test_coupled_system(self):
        """Nicht-diagonale Matrix mit Eigenwerten beide negativ => STABLE."""
        A = np.array([[-1.5,  0.3],
                      [ 0.1, -2.0]])
        status, details = StabilityAnalyzer.lyapunov_continuous(A, self._ann())
        self.assertEqual(status, StabilityStatus.STABLE)
        self.assertGreater(details["decay_rate_alpha"], 0)

    def test_details_keys_complete(self):
        A = np.diag([-1.0, -2.0])
        _, details = StabilityAnalyzer.lyapunov_continuous(A, self._ann())
        for key in ("eigenvalues", "max_real_part", "decay_rate_alpha",
                    "required_alpha_min", "margin"):
            self.assertIn(key, details)


# ===========================================================================
# 4. StabilityAnalyzer – spectral_stability
# ===========================================================================

class TestSpectralStability(unittest.TestCase):

    def _ann(self, spectral_radius_ub=0.99):
        return _make_stability_ann(
            mode=StabilityMode.SPECTRAL,
            spectral_radius_ub=spectral_radius_ub,
        )

    def _elevator_P(self):
        return np.array([
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0],
            [0.0, 0.0, 0.6, 0.4],
            [0.9, 0.0, 0.0, 0.1],
        ])

    def test_stable_markov_chain(self):
        P = self._elevator_P()
        status, details = StabilityAnalyzer.spectral_stability(P, self._ann())
        self.assertEqual(status, StabilityStatus.STABLE)
        self.assertLess(details["spectral_radius_sub"], 0.99)
        self.assertGreater(details["implied_decay_rate"], 0)
        self.assertIn("eigenvalues", details)
        self.assertIn("required_ub", details)

    def test_unstable_spectral(self):
        """Spektralradius > spectral_radius_ub aber != 1 => UNSTABLE."""
        # Konstruiere Matrix, deren zweitgroesster EW-Betrag ~ 0.95 ist
        # und setze ub=0.50 => UNSTABLE
        P = self._elevator_P()
        ann = self._ann(spectral_radius_ub=0.50)
        status, _ = StabilityAnalyzer.spectral_stability(P, ann)
        self.assertEqual(status, StabilityStatus.UNSTABLE)

    def test_marginal_spectral(self):
        """Zweitgroesster EW-Betrag sehr nah an 1.0 => MARGINAL."""
        # Einfache 2x2 stochastische Matrix mit zweitgroesstem EW nahe 1
        # P = [[1-eps, eps],[eps, 1-eps]] hat EW 1 und 1-2*eps
        eps = 1e-8
        P = np.array([[1 - eps, eps],
                      [eps,     1 - eps]])
        ann = self._ann(spectral_radius_ub=0.50)  # ub < sub-EW
        status, details = StabilityAnalyzer.spectral_stability(P, ann)
        self.assertEqual(status, StabilityStatus.MARGINAL)

    def test_spectral_radius_zero_implies_inf_decay(self):
        """Wenn spektral_radius == 0 => implied_decay_rate = inf."""
        # 1x1 Matrix => nur ein EW => abs_eigs_filtered[-2] nicht vorhanden
        # => greife auf letzten zurueck; EW = 0 => inf
        P = np.array([[0.0]])
        ann = self._ann(spectral_radius_ub=0.99)
        status, details = StabilityAnalyzer.spectral_stability(P, ann)
        self.assertEqual(details["implied_decay_rate"], float('inf'))

    def test_details_keys_complete(self):
        P = self._elevator_P()
        _, details = StabilityAnalyzer.spectral_stability(P, self._ann())
        for key in ("eigenvalues", "spectral_radius_sub",
                    "required_ub", "implied_decay_rate"):
            self.assertIn(key, details)


# ===========================================================================
# 5. StabilityAnalyzer – queue_stability
# ===========================================================================

class TestQueueStability(unittest.TestCase):

    def _ann(self, rho_ub=0.90):
        return _make_stability_ann(
            mode=StabilityMode.QUEUE_UTILIZATION,
            rho_ub=rho_ub,
        )

    def test_stable_queue(self):
        """rho = 0.667 < rho_ub = 0.90 => STABLE."""
        status, details = StabilityAnalyzer.queue_stability(0.2, 0.3, self._ann())
        self.assertEqual(status, StabilityStatus.STABLE)
        self.assertAlmostEqual(details["utilization_rho"], 0.2 / 0.3, places=5)
        self.assertAlmostEqual(details["arrival_rate_lambda"], 0.2)
        self.assertAlmostEqual(details["service_rate_mu"], 0.3)
        self.assertAlmostEqual(details["required_rho_ub"], 0.90)
        self.assertGreater(details["margin_to_instability"], 0)

    def test_marginal_queue(self):
        """rho_ub=0.50, rho=0.70 => rho_ub < rho < 1.0 => MARGINAL."""
        status, _ = StabilityAnalyzer.queue_stability(0.7, 1.0, self._ann(rho_ub=0.50))
        self.assertEqual(status, StabilityStatus.MARGINAL)

    def test_unstable_queue(self):
        """rho = 1.2 >= 1.0 => UNSTABLE."""
        status, details = StabilityAnalyzer.queue_stability(1.2, 1.0, self._ann())
        self.assertEqual(status, StabilityStatus.UNSTABLE)
        self.assertAlmostEqual(details["utilization_rho"], 1.2)
        self.assertAlmostEqual(details["margin_to_instability"], -0.2, places=5)

    def test_invalid_service_rate_raises(self):
        """service_rate <= 0 muss ValueError ausloesen."""
        with self.assertRaises(ValueError):
            StabilityAnalyzer.queue_stability(0.5, 0.0, self._ann())

    def test_negative_service_rate_raises(self):
        with self.assertRaises(ValueError):
            StabilityAnalyzer.queue_stability(0.5, -1.0, self._ann())

    def test_details_keys_complete(self):
        _, details = StabilityAnalyzer.queue_stability(0.2, 0.3, self._ann())
        for key in ("arrival_rate_lambda", "service_rate_mu",
                    "utilization_rho", "required_rho_ub",
                    "margin_to_instability"):
            self.assertIn(key, details)


# ===========================================================================
# 6. StabilityAnalyzer – analyze (Dispatcher)
# ===========================================================================

class TestAnalyzeDispatcher(unittest.TestCase):

    # -- 6a. Kein Stability-Stereotyp --------------------------------------

    def test_no_stability_annotation_returns_unknown(self):
        el = _make_element()
        result = StabilityAnalyzer.analyze(el)
        self.assertEqual(result["status"], StabilityStatus.UNKNOWN)
        self.assertIn("reason", result)

    # -- 6b. LYAPUNOV_ASYMPTOTIC -------------------------------------------

    def test_dispatch_lyapunov(self):
        ann = _make_stability_ann(
            mode=StabilityMode.LYAPUNOV_ASYMPTOTIC,
            decay_rate_min=0.5, lyapunov_margin=0.1,
        )
        el = _make_element(stability=ann)
        A = np.diag([-2.0, -3.0])
        result = StabilityAnalyzer.analyze(el, A=A)
        self.assertEqual(result["status"], StabilityStatus.STABLE)
        self.assertEqual(result["element"], "TestElement")
        self.assertEqual(result["mode"], StabilityMode.LYAPUNOV_ASYMPTOTIC)

    def test_lyapunov_missing_A_raises(self):
        ann = _make_stability_ann(mode=StabilityMode.LYAPUNOV_ASYMPTOTIC)
        el = _make_element(stability=ann)
        with self.assertRaises(ValueError):
            StabilityAnalyzer.analyze(el)

    # -- 6c. SPECTRAL ------------------------------------------------------

    def test_dispatch_spectral(self):
        ann = _make_stability_ann(mode=StabilityMode.SPECTRAL)
        el = _make_element(stability=ann)
        P = np.array([[0.7, 0.3],
                      [0.4, 0.6]])
        result = StabilityAnalyzer.analyze(el, P=P)
        self.assertEqual(result["status"], StabilityStatus.STABLE)

    def test_spectral_missing_P_raises(self):
        ann = _make_stability_ann(mode=StabilityMode.SPECTRAL)
        el = _make_element(stability=ann)
        with self.assertRaises(ValueError):
            StabilityAnalyzer.analyze(el)

    # -- 6d. QUEUE_UTILIZATION ---------------------------------------------

    def test_dispatch_queue(self):
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann)
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.3, service_rate=0.5)
        self.assertEqual(result["status"], StabilityStatus.STABLE)

    def test_queue_missing_rates_raises(self):
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann)
        with self.assertRaises(ValueError):
            StabilityAnalyzer.analyze(el, arrival_rate=0.3)

    def test_queue_missing_arrival_rate_raises(self):
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann)
        with self.assertRaises(ValueError):
            StabilityAnalyzer.analyze(el, service_rate=0.5)

    # -- 6e. Unbekannter Modus (BIBO / LYAPUNOV_MARGINAL) ------------------

    def test_dispatch_unknown_mode_bibo(self):
        ann = _make_stability_ann(mode=StabilityMode.BIBO)
        el = _make_element(stability=ann)
        result = StabilityAnalyzer.analyze(el)
        self.assertEqual(result["status"], StabilityStatus.UNKNOWN)
        self.assertIn("reason", result["details"])

    def test_dispatch_unknown_mode_lyapunov_marginal(self):
        ann = _make_stability_ann(mode=StabilityMode.LYAPUNOV_MARGINAL)
        el = _make_element(stability=ann)
        result = StabilityAnalyzer.analyze(el)
        self.assertEqual(result["status"], StabilityStatus.UNKNOWN)

    # -- 6f. Timing-Constraint-Pruefung ------------------------------------

    def test_timing_deadline_ok_true(self):
        """typical_duration <= deadline => timing_deadline_ok = True."""
        tc = TimingConstraint(min_duration=100, max_duration=500,
                              typical_duration=300, deadline=400)
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann, timing=tc)
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.2, service_rate=0.5)
        self.assertTrue(result["timing_deadline_ok"])

    def test_timing_deadline_ok_false(self):
        """typical_duration > deadline => timing_deadline_ok = False."""
        tc = TimingConstraint(min_duration=100, max_duration=500,
                              typical_duration=450, deadline=400)
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann, timing=tc)
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.2, service_rate=0.5)
        self.assertFalse(result["timing_deadline_ok"])

    def test_timing_without_deadline_no_key(self):
        """Kein deadline => kein 'timing_deadline_ok' im Ergebnis."""
        tc = TimingConstraint(min_duration=100, max_duration=500,
                              typical_duration=300)
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann, timing=tc)
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.2, service_rate=0.5)
        self.assertNotIn("timing_deadline_ok", result)

    def test_no_timing_no_key(self):
        """Kein TimingConstraint => kein 'timing_deadline_ok'."""
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(stability=ann)
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.2, service_rate=0.5)
        self.assertNotIn("timing_deadline_ok", result)


# ===========================================================================
# 7. StabilityAwareCodeGenerator – generate_class
# ===========================================================================

class TestGenerateClass(unittest.TestCase):

    def test_class_header_generated(self):
        el = _make_element(name="MyController", etype="Component")
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertIn("class MyController:", code)
        self.assertIn("Component", code)

    def test_stability_fields_included(self):
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION, rho_ub=0.85)
        el = _make_element(name="QueueComp", stability=ann)
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertIn("QUEUE_UTILIZATION", code)
        self.assertIn("0.85", code)
        self.assertIn("check_stability", code)

    def test_no_stability_no_check_method(self):
        el = _make_element(name="SimpleClass")
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertNotIn("check_stability", code)

    def test_timing_fields_included(self):
        tc = TimingConstraint(min_duration=100, max_duration=500,
                              typical_duration=300, unit="ms", deadline=400)
        el = _make_element(name="TimedClass", timing=tc)
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertIn("100", code)
        self.assertIn("500", code)
        self.assertIn("400", code)
        self.assertIn("ms", code)

    def test_timing_without_deadline(self):
        tc = TimingConstraint(min_duration=10, max_duration=50,
                              typical_duration=30)
        el = _make_element(name="NoDeadlineClass", timing=tc)
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertIn("10", code)
        self.assertNotIn("_deadline", code)

    def test_no_timing_no_duration_fields(self):
        el = _make_element(name="NoTimingClass")
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertNotIn("_min_duration", code)

    def test_return_type_is_str(self):
        el = _make_element()
        self.assertIsInstance(StabilityAwareCodeGenerator.generate_class(el), str)


# ===========================================================================
# 8. StabilityAwareCodeGenerator – generate_stability_check_code
# ===========================================================================

class TestGenerateStabilityCheckCode(unittest.TestCase):

    def test_no_stability_returns_comment(self):
        el = _make_element()
        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("Kein Stabilitaets-Stereotype", code)

    def test_queue_utilization_code(self):
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION, rho_ub=0.85)
        el = _make_element(name="QueueEl", stability=ann)
        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("rho = arrival_rate / service_rate", code)
        self.assertIn("0.85", code)
        self.assertIn("assert rho <", code)
        self.assertIn("QUEUE_UTILIZATION", code)

    def test_lyapunov_asymptotic_code(self):
        ann = _make_stability_ann(mode=StabilityMode.LYAPUNOV_ASYMPTOTIC, decay_rate_min=0.3)
        el = _make_element(name="LyapEl", stability=ann)
        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("np.linalg.eigvals(A)", code)
        self.assertIn("0.3", code)
        self.assertIn("assert alpha >", code)
        self.assertIn("LYAPUNOV_ASYMPTOTIC", code)

    def test_spectral_code(self):
        ann = _make_stability_ann(mode=StabilityMode.SPECTRAL, spectral_radius_ub=0.95)
        el = _make_element(name="SpectEl", stability=ann)
        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("np.linalg.eigvals(P)", code)
        self.assertIn("0.95", code)
        self.assertIn("assert spectral_radius <", code)
        self.assertIn("SPECTRAL", code)

    def test_other_mode_returns_header_only(self):
        """Modus BIBO => nur Header-Kommentare, kein assert-Code."""
        ann = _make_stability_ann(mode=StabilityMode.BIBO)
        el = _make_element(stability=ann)
        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("BIBO", code)
        self.assertNotIn("assert", code)

    def test_return_type_is_str(self):
        el = _make_element()
        self.assertIsInstance(
            StabilityAwareCodeGenerator.generate_stability_check_code(el), str
        )

    def test_element_name_in_header(self):
        ann = _make_stability_ann(mode=StabilityMode.QUEUE_UTILIZATION)
        el = _make_element(name="UniqueControllerName", stability=ann)
        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("UniqueControllerName", code)


# ===========================================================================
# 9. Demo-Funktionen (Smoke-Tests – pruefen Ausfuehrbarkeit + Output)
# ===========================================================================

class TestDemoFunctions(unittest.TestCase):

    def _capture(self, func):
        """Faengt stdout ab und gibt es als String zurueck."""
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            func()
        return buf.getvalue()

    def test_demo_elevator_door_stability_runs(self):
        output = self._capture(demo_elevator_door_stability)
        self.assertIn("ElevatorDoorController", output)
        self.assertIn("MotorDynamicsSubsystem", output)
        self.assertIn("DoorStateMarkovChain", output)

    def test_demo_elevator_prints_queue_status(self):
        output = self._capture(demo_elevator_door_stability)
        self.assertIn("QUEUE_UTILIZATION", output)
        self.assertIn("stabil", output)

    def test_demo_elevator_prints_lyapunov_info(self):
        output = self._capture(demo_elevator_door_stability)
        self.assertIn("LYAPUNOV_ASYMPTOTIC", output)
        self.assertIn("alpha", output)

    def test_demo_elevator_prints_spectral_info(self):
        output = self._capture(demo_elevator_door_stability)
        self.assertIn("SPECTRAL", output)
        self.assertIn("Spektralradius", output)

    def test_demo_elevator_prints_generated_code(self):
        output = self._capture(demo_elevator_door_stability)
        self.assertIn("GENERIERTER STABILITAETS-CHECK-CODE", output)
        self.assertIn("rho = arrival_rate / service_rate", output)

    def test_demo_equilibrium_vs_stability_runs(self):
        output = self._capture(demo_equilibrium_vs_stability)
        self.assertIn("Gleichgewicht vs. Stabilitaet", output)

    def test_demo_equilibrium_stable_branch(self):
        """Stabiles System => Little's Law wird ausgegeben."""
        output = self._capture(demo_equilibrium_vs_stability)
        self.assertIn("Little's Law (W)", output)
        self.assertIn("Little's Law (L)", output)

    def test_demo_equilibrium_unstable_branch(self):
        """Instabiles System => Hinweis 'nicht anwendbar'."""
        output = self._capture(demo_equilibrium_vs_stability)
        self.assertIn("nicht anwendbar", output)

    def test_demo_equilibrium_prints_status(self):
        output = self._capture(demo_equilibrium_vs_stability)
        self.assertIn("Stabilitaetsstatus", output)

    def test_main_block_executes_both_demos(self):
        """Simuliert 'python uml_marte_stability.py' via __main__-Block."""
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            import src.uml_marte_stability as mod
            mod.demo_elevator_door_stability()
            mod.demo_equilibrium_vs_stability()
        output = buf.getvalue()
        self.assertIn("ElevatorDoorController", output)
        self.assertIn("Gleichgewicht vs. Stabilitaet", output)


# ===========================================================================
# 10. Integrationstests – vollstaendige Szenarien end-to-end
# ===========================================================================

class TestIntegration(unittest.TestCase):

    def test_full_elevator_queue_scenario(self):
        """Vollstaendiges Aufzugstuer-Szenario: Annotation -> Analyse -> Code."""
        tc = TimingConstraint(
            min_duration=2500, max_duration=4000, typical_duration=3000,
            unit="ms", deadline=4000,
        )
        ann = StabilityAnnotation(
            name="ElevatorDoorController_SA",
            base_metaclass="Component",
            stability_mode=StabilityMode.QUEUE_UTILIZATION,
            rho_ub=0.90,
        )
        el = AnnotatedElement(
            element_name="ElevatorDoorController",
            element_type="Component",
            timing=tc,
            stability=ann,
        )
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.2, service_rate=0.3)
        self.assertEqual(result["status"], StabilityStatus.STABLE)
        self.assertTrue(result["timing_deadline_ok"])

        code = StabilityAwareCodeGenerator.generate_stability_check_code(el)
        self.assertIn("assert rho < 0.9", code)

        class_code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertIn("class ElevatorDoorController:", class_code)
        self.assertIn("check_stability", class_code)

    def test_full_lyapunov_scenario(self):
        """Vollstaendiges Lyapunov-Szenario mit LyapunovFunctionAnnotation."""
        ann = StabilityAnnotation(
            name="Motor_SA",
            base_metaclass="Class",
            stability_mode=StabilityMode.LYAPUNOV_ASYMPTOTIC,
            decay_rate_min=0.5,
            lyapunov_margin=0.1,
        )
        lyap = LyapunovFunctionAnnotation(
            name="Motor_LFA",
            base_metaclass="Class",
            function_type="quadratic",
            positive_definite_verified=True,
            derivative_negative=True,
        )
        el = AnnotatedElement(
            element_name="MotorDynamics",
            element_type="Class",
            stability=ann,
            lyapunov=lyap,
        )
        A = np.array([[-1.5, 0.3], [0.1, -2.0]])
        result = StabilityAnalyzer.analyze(el, A=A)
        self.assertEqual(result["status"], StabilityStatus.STABLE)
        self.assertGreater(result["details"]["decay_rate_alpha"], 0.5)

    def test_full_markov_scenario(self):
        """Vollstaendiges Markov-Szenario mit EquilibriumAnnotation."""
        ann = StabilityAnnotation(
            name="Markov_SA",
            base_metaclass="Class",
            stability_mode=StabilityMode.SPECTRAL,
            spectral_radius_ub=0.99,
        )
        eq = EquilibriumAnnotation(
            name="Markov_EA",
            base_metaclass="Class",
            is_equilibrium_only=False,
            supports_stability_proof=True,
            equilibrium_law="pi * P = pi",
        )
        el = AnnotatedElement(
            element_name="DoorMarkov",
            element_type="Class",
            stability=ann,
            equilibrium=eq,
        )
        P = np.array([
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0],
            [0.0, 0.0, 0.6, 0.4],
            [0.9, 0.0, 0.0, 0.1],
        ])
        result = StabilityAnalyzer.analyze(el, P=P)
        self.assertEqual(result["status"], StabilityStatus.STABLE)
        self.assertLess(result["details"]["spectral_radius_sub"], 0.99)

    def test_little_law_only_equilibrium(self):
        """Little's Law Element: equilibrium_only=True, kein Stabilitaetsbeweis."""
        ann = StabilityAnnotation(
            name="LittlesLaw_SA",
            base_metaclass="Component",
            stability_mode=StabilityMode.QUEUE_UTILIZATION,
            rho_ub=1.0,
        )
        eq = EquilibriumAnnotation(
            name="LittlesLaw_EA",
            base_metaclass="Component",
            is_equilibrium_only=True,
            supports_stability_proof=False,
            equilibrium_law="L = lambda * W",
        )
        el = AnnotatedElement(
            element_name="MM1Queue",
            element_type="Component",
            stability=ann,
            equilibrium=eq,
        )
        # Stabiles System => rho < 1
        result = StabilityAnalyzer.analyze(el, arrival_rate=0.7, service_rate=1.0)
        self.assertEqual(result["status"], StabilityStatus.STABLE)
        self.assertFalse(el.equilibrium.supports_stability_proof)

        # Instabiles System => rho >= 1
        result2 = StabilityAnalyzer.analyze(el, arrival_rate=1.2, service_rate=1.0)
        self.assertEqual(result2["status"], StabilityStatus.UNSTABLE)

    def test_decay_rate_annotation_fields(self):
        """DecayRateAnnotation-Felder werden korrekt gespeichert."""
        dr = DecayRateAnnotation(
            name="TestDecay",
            base_metaclass="Operation",
            alpha_min=1.0,
            alpha_max=5.0,
            convergence_constant_ub=3.0,
        )
        el = AnnotatedElement(
            element_name="OperationEl",
            element_type="Operation",
            decay=dr,
        )
        self.assertEqual(el.decay.alpha_min, 1.0)
        self.assertEqual(el.decay.alpha_max, 5.0)
        self.assertEqual(el.decay.convergence_constant_ub, 3.0)

    def test_generate_class_with_all_fields(self):
        """generate_class mit Timing + Stability erzeugt vollstaendigen Code."""
        tc = TimingConstraint(10, 100, 50, unit="ms", deadline=90)
        ann = _make_stability_ann(mode=StabilityMode.LYAPUNOV_ASYMPTOTIC)
        el = _make_element(name="FullElement", timing=tc, stability=ann)
        code = StabilityAwareCodeGenerator.generate_class(el)
        self.assertIn("class FullElement:", code)
        self.assertIn("LYAPUNOV_ASYMPTOTIC", code)
        self.assertIn("_min_duration", code)
        self.assertIn("_max_duration", code)
        self.assertIn("_deadline", code)
        self.assertIn("check_stability", code)


# ===========================================================================
# Einstiegspunkt
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
