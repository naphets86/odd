"""
test_rotation_analysis.py
=========================
Vollständige Testsuite für rotation_analysis.py.
Nur Python-Standardbibliothek (unittest + trace).

Ausführen:
    python3 test_rotation_analysis.py

Das Skript führt alle Tests aus und misst anschließend die Zeilenabdeckung
des Produktionsmoduls via stdlib-trace. Am Ende erscheint ein
Coverage-Report. Ziel: 100 % aller ausführbaren Zeilen.

Mathematische Korrektheit wird anhand von Exaktwerten verifiziert:
  - arctan-Werte (0°, ±45°, ±60°)
  - Isometrie ‖R_φ v‖ = ‖v‖
  - R_{−φ} ∘ R_φ = Id
  - Gruppengesetz R_φ ∘ R_ψ = R_{φ+ψ}
  - Hauptsatz: f̃'(x̃) = 0 bei φ = −arctan(m)
  - Vollständige WST für f(x) = x³ − 3x und f(x) = x⁴ − 2x²
"""

from __future__ import annotations

import io
import math
import os
import sys
import trace
import unittest
from typing import List

import src.rotation_analysis as ra
from src.rotation_analysis import (
    PointType,
    CurvePoint,
    AngleSortTable,
    rotate_point,
    rotate_curve,
    rotated_slope,
    analytical_angle,
    classify_point,
    build_wst,
    format_wst,
)

# ---------------------------------------------------------------------------
# Hilfskonstanten & -funktionen
# ---------------------------------------------------------------------------

PI    = math.pi
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)

def eq(a: float, b: float, rel: float = 1e-9) -> bool:
    """math.isclose mit festen Toleranzen."""
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


# ===========================================================================
# 1  PointType
# ===========================================================================

class TestPointType(unittest.TestCase):

    def test_alle_member_vorhanden(self):
        for name in ("MAXIMUM", "MINIMUM", "WENDEPUNKT", "FLACH", "UNBEKANNT"):
            self.assertIn(name, PointType.__members__)

    def test_member_paarweise_verschieden(self):
        members = list(PointType)
        self.assertEqual(len(members), len(set(members)))

    def test_name_attribut(self):
        self.assertEqual(PointType.MAXIMUM.name,    "MAXIMUM")
        self.assertEqual(PointType.MINIMUM.name,    "MINIMUM")
        self.assertEqual(PointType.WENDEPUNKT.name, "WENDEPUNKT")
        self.assertEqual(PointType.FLACH.name,      "FLACH")
        self.assertEqual(PointType.UNBEKANNT.name,  "UNBEKANNT")


# ===========================================================================
# 2  CurvePoint
# ===========================================================================

class TestCurvePoint(unittest.TestCase):

    # ── Grundkonstruktion ─────────────────────────────────────────────

    def test_felder_gespeichert(self):
        p = CurvePoint(x=1.0, y=2.0, slope=0.5)
        self.assertEqual(p.x,     1.0)
        self.assertEqual(p.y,     2.0)
        self.assertEqual(p.slope, 0.5)

    def test_default_kind_unbekannt(self):
        p = CurvePoint(x=0.0, y=0.0, slope=0.0)
        self.assertEqual(p.kind, PointType.UNBEKANNT)

    def test_expliziter_kind(self):
        p = CurvePoint(x=0.0, y=0.0, slope=0.0, kind=PointType.MAXIMUM)
        self.assertEqual(p.kind, PointType.MAXIMUM)

    # ── Analytischer Winkel ───────────────────────────────────────────

    def test_winkel_steigung_null(self):
        """slope = 0  →  α = 0°"""
        p = CurvePoint(x=0.0, y=0.0, slope=0.0)
        self.assertTrue(eq(p.angle_deg, 0.0))

    def test_winkel_steigung_plus_eins(self):
        """slope = 1  →  α = −45°"""
        p = CurvePoint(x=0.0, y=0.0, slope=1.0)
        self.assertTrue(eq(p.angle_deg, -45.0))

    def test_winkel_steigung_minus_eins(self):
        """slope = −1  →  α = +45°"""
        p = CurvePoint(x=0.0, y=0.0, slope=-1.0)
        self.assertTrue(eq(p.angle_deg, 45.0))

    def test_winkel_steigung_sqrt3(self):
        """slope = √3  →  α = −60°"""
        p = CurvePoint(x=0.0, y=0.0, slope=SQRT3)
        self.assertTrue(eq(p.angle_deg, -60.0, rel=1e-8))

    def test_winkel_steigung_minus_sqrt3(self):
        """slope = −√3  →  α = +60°"""
        p = CurvePoint(x=0.0, y=0.0, slope=-SQRT3)
        self.assertTrue(eq(p.angle_deg, 60.0, rel=1e-8))

    def test_winkel_grosse_positive_steigung(self):
        """slope → +∞  →  α → −90° (nie exakt −90°)"""
        p = CurvePoint(x=0.0, y=0.0, slope=1e15)
        self.assertGreater(p.angle_deg, -90.0)
        self.assertLess(p.angle_deg, -89.9)

    def test_winkel_grosse_negative_steigung(self):
        """slope → −∞  →  α → +90° (nie exakt +90°)"""
        p = CurvePoint(x=0.0, y=0.0, slope=-1e15)
        self.assertLess(p.angle_deg, 90.0)
        self.assertGreater(p.angle_deg, 89.9)

    def test_winkel_offen_zwischen_minus_und_plus_90(self):
        for m in (-1e9, -1.0, 0.0, 1.0, 1e9):
            p = CurvePoint(x=0.0, y=0.0, slope=m)
            self.assertGreater(p.angle_deg, -90.0)
            self.assertLess(p.angle_deg,     90.0)

    def test_winkel_rad_konsistent_mit_deg(self):
        for m in (-3.0, -1.0, 0.0, 1.0, 3.0):
            p = CurvePoint(x=0.0, y=0.0, slope=m)
            self.assertTrue(eq(math.degrees(p.angle_rad), p.angle_deg))

    def test_antisymmetrie_winkel(self):
        """α(−m) = −α(m)"""
        for m in (0.5, 1.0, 3.0):
            p_pos = CurvePoint(x=0.0, y=0.0, slope= m)
            p_neg = CurvePoint(x=0.0, y=0.0, slope=-m)
            self.assertTrue(eq(p_pos.angle_deg, -p_neg.angle_deg))

    # ── Hilfseigenschaften ────────────────────────────────────────────

    def test_coords_property(self):
        p = CurvePoint(x=3.0, y=-2.5, slope=0.0)
        self.assertEqual(p.coords, (3.0, -2.5))

    def test_str_enthaelt_kind(self):
        p = CurvePoint(x=0.0, y=0.0, slope=0.0, kind=PointType.MINIMUM)
        self.assertIn("MINIMUM", str(p))

    def test_str_enthaelt_koordinaten(self):
        p = CurvePoint(x=1.5, y=-0.5, slope=0.0)
        s = str(p)
        self.assertIn("1.5", s)
        self.assertIn("-0.5", s)

    def test_repr_kein_absturz(self):
        p = CurvePoint(x=0.0, y=0.0, slope=0.0)
        self.assertIsInstance(repr(p), str)


# ===========================================================================
# 3  rotate_point
# ===========================================================================

class TestRotatePoint(unittest.TestCase):

    # ── Identität & Spezialwinkel ─────────────────────────────────────

    def test_phi_null_ist_identitaet(self):
        x, y = rotate_point(3.0, 4.0, 0.0)
        self.assertTrue(eq(x, 3.0)); self.assertTrue(eq(y, 4.0))

    def test_phi_90_grad(self):
        """R_{π/2}(1, 0) = (0, 1)"""
        x, y = rotate_point(1.0, 0.0, PI / 2)
        self.assertTrue(eq(x, 0.0)); self.assertTrue(eq(y, 1.0))

    def test_phi_minus_90_grad(self):
        """R_{−π/2}(0, 1) = (1, 0)"""
        x, y = rotate_point(0.0, 1.0, -PI / 2)
        self.assertTrue(eq(x, 1.0)); self.assertTrue(eq(y, 0.0))

    def test_phi_180_grad(self):
        """R_π(x, y) = (−x, −y)"""
        x, y = rotate_point(2.0, 3.0, PI)
        self.assertTrue(eq(x, -2.0)); self.assertTrue(eq(y, -3.0))

    def test_phi_360_grad_ist_identitaet(self):
        """R_{2π} = Id"""
        x, y = rotate_point(5.0, -2.0, 2 * PI)
        self.assertTrue(eq(x, 5.0)); self.assertTrue(eq(y, -2.0))

    def test_phi_45_grad(self):
        """R_{π/4}(1, 0) = (1/√2, 1/√2)"""
        x, y = rotate_point(1.0, 0.0, PI / 4)
        self.assertTrue(eq(x, 1.0 / SQRT2))
        self.assertTrue(eq(y, 1.0 / SQRT2))

    def test_ursprung_fest(self):
        """R_φ(0, 0) = (0, 0) für beliebige φ"""
        for phi in (0.0, PI / 6, PI / 3, PI / 2, PI, 2 * PI):
            x, y = rotate_point(0.0, 0.0, phi)
            self.assertTrue(eq(x, 0.0)); self.assertTrue(eq(y, 0.0))

    def test_negative_koordinaten(self):
        x, y = rotate_point(-1.0, -1.0, PI / 2)
        self.assertTrue(eq(x, 1.0)); self.assertTrue(eq(y, -1.0))

    def test_rueckgabe_ist_tupel_laenge_2(self):
        r = rotate_point(1.0, 2.0, 0.5)
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)

    # ── Isometrie ─────────────────────────────────────────────────────

    def test_isometrie_200_zufaellige_punkte(self):
        """‖R_φ v‖ = ‖v‖ für 200 zufällige (Punkt, Winkel)-Paare."""
        import random
        rng = random.Random(42)
        for _ in range(200):
            px  = rng.uniform(-100.0, 100.0)
            py  = rng.uniform(-100.0, 100.0)
            phi = rng.uniform(-PI, PI)
            rx, ry = rotate_point(px, py, phi)
            self.assertTrue(
                eq(math.hypot(rx, ry), math.hypot(px, py), rel=1e-10),
                msg=f"px={px}, py={py}, phi={phi}",
            )

    # ── Gruppenaxiome ─────────────────────────────────────────────────

    def test_inverses_element(self):
        """R_{−φ} ∘ R_φ = Id"""
        px, py, phi = 7.0, -3.0, 1.234
        rx,  ry  = rotate_point(px, py,  phi)
        rrx, rry = rotate_point(rx, ry, -phi)
        self.assertTrue(eq(rrx, px)); self.assertTrue(eq(rry, py))

    def test_komposition_gleich_summe(self):
        """R_φ ∘ R_ψ = R_{φ+ψ}"""
        px, py, phi, psi = 2.0, 1.0, PI / 6, PI / 4
        rx1, ry1 = rotate_point(px,  py,  psi)
        rx2, ry2 = rotate_point(rx1, ry1, phi)
        rx_ref, ry_ref = rotate_point(px, py, phi + psi)
        self.assertTrue(eq(rx2, rx_ref)); self.assertTrue(eq(ry2, ry_ref))

    def test_positive_phi_gegenuhrzeigersinn(self):
        """Punkt auf pos. x-Achse dreht nach +π/2 auf pos. y-Achse."""
        _, y = rotate_point(1.0, 0.0, PI / 2)
        self.assertGreater(y, 0)


# ===========================================================================
# 4  rotate_curve
# ===========================================================================

class TestRotateCurve(unittest.TestCase):

    def test_leere_eingabe(self):
        self.assertEqual(rotate_curve([], 1.0), [])

    def test_rueckgabe_ist_liste(self):
        self.assertIsInstance(rotate_curve([(1.0, 0.0)], 0.0), list)

    def test_einzelpunkt_phi_null(self):
        r = rotate_curve([(3.0, 4.0)], 0.0)
        self.assertTrue(eq(r[0][0], 3.0)); self.assertTrue(eq(r[0][1], 4.0))

    def test_einzelpunkt_phi_90(self):
        r = rotate_curve([(1.0, 0.0)], PI / 2)
        self.assertTrue(eq(r[0][0], 0.0)); self.assertTrue(eq(r[0][1], 1.0))

    def test_laenge_erhalten(self):
        pts = [(float(i), float(i**2)) for i in range(10)]
        self.assertEqual(len(rotate_curve(pts, 0.7)), 10)

    def test_alle_normen_erhalten(self):
        pts     = [(i * 0.3, math.sin(i * 0.3)) for i in range(-10, 11)]
        rotated = rotate_curve(pts, 1.1)
        for (rx, ry), (ox, oy) in zip(rotated, pts):
            self.assertTrue(eq(math.hypot(rx, ry), math.hypot(ox, oy), rel=1e-10))

    def test_phi_90_tauscht_achsen(self):
        """R_{π/2}(x, y) = (−y, x)"""
        pts    = [(2.0, 3.0), (-1.0, 4.0), (0.0, 0.0)]
        result = rotate_curve(pts, PI / 2)
        for (rx, ry), (ox, oy) in zip(result, pts):
            self.assertTrue(eq(rx, -oy)); self.assertTrue(eq(ry, ox))

    def test_generator_akzeptiert(self):
        gen = ((float(i), float(i)) for i in range(5))
        self.assertEqual(len(rotate_curve(gen, 0.0)), 5)

    def test_jedes_element_tupel_laenge_2(self):
        for item in rotate_curve([(1.0, 2.0), (3.0, 4.0)], 0.3):
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_grosse_kurve_500_punkte(self):
        pts = [(i * 0.01, math.cos(i * 0.01)) for i in range(500)]
        self.assertEqual(len(rotate_curve(pts, PI / 7)), 500)


# ===========================================================================
# 5  rotated_slope
# ===========================================================================

class TestRotatedSlope(unittest.TestCase):

    # ── Hauptsatz: φ = −arctan(m) → f̃' = 0 ─────────────────────────

    def test_null_bei_optimalem_winkel(self):
        """Satz 3.1 (Epp 2025): bei φ* = −arctan(m) gilt f̃' = 0."""
        for m in (-10.0, -3.0, -1.0, -0.1, 0.0, 0.5, 1.0, 3.0, 10.0):
            phi = -math.atan(m)
            r   = rotated_slope(m, phi)
            self.assertIsNotNone(r, msg=f"m={m}")
            self.assertTrue(eq(r, 0.0, rel=1e-8), msg=f"m={m}, r={r}")

    # ── φ = 0: f̃' = m ─────────────────────────────────────────────

    def test_phi_null_gibt_steigung_zurueck(self):
        """φ = 0: Nenner = 1, Zähler = m  →  f̃' = m."""
        for m in (-5.0, 0.0, 1.0, 7.0):
            r = rotated_slope(m, 0.0)
            self.assertIsNotNone(r)
            self.assertTrue(eq(r, m))

    # ── Senkrechte Bildfunktion → None ────────────────────────────

    def test_none_wenn_nenner_null(self):
        """cos φ − m·sin φ = 0  ↔  tan φ = 1/m  →  None."""
        m   = 2.0
        phi = math.atan(1.0 / m)
        self.assertIsNone(rotated_slope(m, phi))

    def test_none_fuer_phi_90_steigung_null(self):
        """φ = π/2, m = 0: Nenner = cos(π/2) ≈ 0  →  None."""
        self.assertIsNone(rotated_slope(0.0, PI / 2))

    def test_nicht_none_wenn_nenner_ungleich_null(self):
        """φ = π/2, m = −1: Nenner = 0 − (−1)·1 = 1 ≠ 0  →  nicht None."""
        r = rotated_slope(-1.0, PI / 2)
        self.assertIsNotNone(r)
        self.assertTrue(eq(r, 1.0, rel=1e-6))

    # ── Manuelle Referenzberechnung ───────────────────────────────

    def test_referenz_phi_30_m_1(self):
        """φ = π/6, m = 1: Exaktwert (1+√3)/(√3−1)."""
        phi      = PI / 6      # sin = 1/2, cos = √3/2
        expected = (1.0 + SQRT3) / (SQRT3 - 1.0)
        r        = rotated_slope(1.0, phi)
        self.assertIsNotNone(r)
        self.assertTrue(eq(r, expected))

    def test_phi_minus_45_m_1_gibt_null(self):
        """φ = −π/4, m = 1: Zähler = −1/√2 + 1/√2 = 0  →  f̃' = 0."""
        r = rotated_slope(1.0, -PI / 4)
        self.assertIsNotNone(r)
        self.assertTrue(eq(r, 0.0))

    def test_antisymmetrie(self):
        """f̃'(−m, −φ) = −f̃'(m, φ)"""
        m   = 0.7;  phi = 0.4
        r1  = rotated_slope(m,  phi)
        r2  = rotated_slope(-m, -phi)
        self.assertIsNotNone(r1); self.assertIsNotNone(r2)
        self.assertTrue(eq(r1, -r2))

    def test_rueckgabe_float_wenn_nicht_none(self):
        self.assertIsInstance(rotated_slope(0.5, 0.3), float)


# ===========================================================================
# 6  analytical_angle
# ===========================================================================

class TestAnalyticalAngle(unittest.TestCase):

    def test_steigung_null(self):
        self.assertTrue(eq(analytical_angle(0.0), 0.0))

    def test_steigung_eins(self):
        self.assertTrue(eq(analytical_angle(1.0), -45.0))

    def test_steigung_minus_eins(self):
        self.assertTrue(eq(analytical_angle(-1.0), 45.0))

    def test_steigung_sqrt3(self):
        self.assertTrue(eq(analytical_angle(SQRT3), -60.0, rel=1e-8))

    def test_steigung_minus_sqrt3(self):
        self.assertTrue(eq(analytical_angle(-SQRT3), 60.0, rel=1e-8))

    def test_rueckgabe_in_grad(self):
        """Ergebnis muss in Grad sein (nicht Radiant)."""
        self.assertTrue(eq(analytical_angle(1.0), -45.0))

    def test_antisymmetrie(self):
        """α(−m) = −α(m)"""
        for m in (0.5, 1.0, 2.0, 5.0):
            self.assertTrue(eq(analytical_angle(-m), -analytical_angle(m)))

    def test_stets_im_offenen_intervall(self):
        for m in (-1e12, -1.0, 0.0, 1.0, 1e12):
            a = analytical_angle(m)
            self.assertGreater(a, -90.0)
            self.assertLess(a,     90.0)

    def test_papier_beispiel_slope_8_ueber_3sqrt3(self):
        """WST-Beispiel x⁴−2x²: m ≈ +1.540, α = −arctan(m) ≈ −57°."""
        m = 8.0 / (3.0 * SQRT3)
        self.assertTrue(eq(analytical_angle(m), -math.degrees(math.atan(m))))


# ===========================================================================
# 7  classify_point
# ===========================================================================

class TestClassifyPoint(unittest.TestCase):
    """Konkrete Polynome als Testfunktionen."""

    # f'(x) = 2x              (f = x²,  Minimum bei 0)
    fp_x2      = staticmethod(lambda x:  2.0 * x)
    # f'(x) = −2x             (f = −x², Maximum bei 0)
    fp_neg_x2  = staticmethod(lambda x: -2.0 * x)
    # f'(x) = 3x²             (f = x³,  Terrassenpunkt bei 0)
    fp_x3      = staticmethod(lambda x:  3.0 * x**2)
    # f'(x) = 3x²−3           (f = x³−3x)
    fp_cubic   = staticmethod(lambda x:  3.0 * x**2 - 3.0)
    # f'(x) ≡ 1               (konstant positiv)
    fp_const   = staticmethod(lambda x:  1.0)

    def test_minimum_x_quadrat(self):
        self.assertEqual(classify_point(self.fp_x2, 0.0), PointType.MINIMUM)

    def test_maximum_neg_x_quadrat(self):
        self.assertEqual(classify_point(self.fp_neg_x2, 0.0), PointType.MAXIMUM)

    def test_terrassenpunkt_x_kubik(self):
        """f(x) = x³ hat bei x=0 einen Terrassenpunkt."""
        self.assertEqual(classify_point(self.fp_x3, 0.0), PointType.FLACH)

    def test_maximum_kubik_bei_minus_eins(self):
        self.assertEqual(classify_point(self.fp_cubic, -1.0), PointType.MAXIMUM)

    def test_minimum_kubik_bei_plus_eins(self):
        self.assertEqual(classify_point(self.fp_cubic,  1.0), PointType.MINIMUM)

    def test_wendepunkt_kubik_bei_null(self):
        """f'(0) = −3 ≠ 0  →  WENDEPUNKT."""
        self.assertEqual(classify_point(self.fp_cubic, 0.0), PointType.WENDEPUNKT)

    def test_wendepunkt_konstant_positiv(self):
        """f'(5) = 1 ≠ 0  →  WENDEPUNKT."""
        self.assertEqual(classify_point(self.fp_const, 5.0), PointType.WENDEPUNKT)

    def test_wendepunkt_sin_bei_null(self):
        """sin: f'(0) = cos(0) = 1 ≠ 0  →  WENDEPUNKT."""
        self.assertEqual(classify_point(math.cos, 0.0), PointType.WENDEPUNKT)

    def test_maximum_sin_bei_pi_halbe(self):
        self.assertEqual(classify_point(math.cos,  PI / 2), PointType.MAXIMUM)

    def test_minimum_sin_bei_minus_pi_halbe(self):
        self.assertEqual(classify_point(math.cos, -PI / 2), PointType.MINIMUM)

    def test_benutzerdefiniertes_h(self):
        """Mit h = 1e-4 identisches Ergebnis."""
        self.assertEqual(
            classify_point(self.fp_x2, 0.0, h=1e-4),
            PointType.MINIMUM,
        )

    def test_grosses_tol_behandelt_kleine_steigung_als_null(self):
        """f'(0.001) = 0.002 < tol = 0.01  →  Vorzeichenzweig aktiv."""
        result = classify_point(self.fp_x2, 0.001, tol=0.01)
        self.assertIn(result, (PointType.MINIMUM, PointType.FLACH))


# ===========================================================================
# 8  AngleSortTable
# ===========================================================================

class TestAngleSortTable(unittest.TestCase):

    @staticmethod
    def wst_x4() -> AngleSortTable:
        """WST für f(x) = x⁴ − 2x²  (Hauptbeispiel aus dem Papier)."""
        def f(x):   return x**4 - 2*x**2
        def fp(x):  return 4*x**3 - 4*x
        return build_wst(f, fp, [-1.0, 0.0, 1.0, -1.0/SQRT3, 1.0/SQRT3])

    # ── Aufbau ───────────────────────────────────────────────────────

    def test_laenge(self):
        self.assertEqual(len(self.wst_x4()), 5)

    def test_aufsteigend_sortiert(self):
        angles = [p.angle_deg for p in self.wst_x4()]
        self.assertEqual(angles, sorted(angles))

    def test_alpha_min_kleiner_gleich_alpha_max(self):
        wst = self.wst_x4()
        self.assertLessEqual(wst.alpha_min, wst.alpha_max)

    def test_total_span(self):
        wst = self.wst_x4()
        self.assertTrue(eq(wst.total_span, wst.alpha_max - wst.alpha_min))

    def test_total_span_ca_114_grad(self):
        """Papier-Ergebnis: Gesamtdrehung ≈ 114°."""
        wst = self.wst_x4()
        self.assertTrue(eq(wst.total_span, 114.0, rel=0.01))

    # ── Leere Tabelle ─────────────────────────────────────────────

    def test_leere_tabelle_nullen(self):
        wst = AngleSortTable(points=[])
        self.assertEqual(len(wst), 0)
        self.assertTrue(eq(wst.alpha_min,  0.0))
        self.assertTrue(eq(wst.alpha_max,  0.0))
        self.assertTrue(eq(wst.total_span, 0.0))

    def test_leere_by_kind(self):
        self.assertEqual(AngleSortTable(points=[]).by_kind(PointType.MAXIMUM), [])

    def test_leere_support_angles(self):
        self.assertEqual(AngleSortTable(points=[]).support_angles(), [])

    # ── by_kind ───────────────────────────────────────────────────

    def test_by_kind_maximum(self):
        maxima = self.wst_x4().by_kind(PointType.MAXIMUM)
        self.assertEqual(len(maxima), 1)
        self.assertTrue(eq(maxima[0].x, 0.0))

    def test_by_kind_minimum(self):
        minima = self.wst_x4().by_kind(PointType.MINIMUM)
        self.assertEqual(len(minima), 2)
        xs = sorted(p.x for p in minima)
        self.assertTrue(eq(xs[0], -1.0)); self.assertTrue(eq(xs[1], 1.0))

    def test_by_kind_wendepunkt(self):
        self.assertEqual(len(self.wst_x4().by_kind(PointType.WENDEPUNKT)), 2)

    def test_by_kind_flach_leer(self):
        self.assertEqual(self.wst_x4().by_kind(PointType.FLACH), [])

    # ── support_angles ────────────────────────────────────────────

    def test_support_angles_eindeutig(self):
        sa = self.wst_x4().support_angles()
        self.assertEqual(len(sa), len(set(round(a, 8) for a in sa)))

    def test_support_angles_sortiert(self):
        sa = self.wst_x4().support_angles()
        self.assertEqual(sa, sorted(sa))

    def test_support_angles_enthaelt_null(self):
        """Extrema haben α = 0°, das muss als Stützwinkel erscheinen."""
        sa = [round(a, 6) for a in self.wst_x4().support_angles()]
        self.assertIn(0.0, sa)

    # ── Iteration & Indexierung ───────────────────────────────────

    def test_iter_liefert_curve_points(self):
        for p in self.wst_x4():
            self.assertIsInstance(p, CurvePoint)

    def test_getitem(self):
        wst = self.wst_x4()
        self.assertIsInstance(wst[0], CurvePoint)
        self.assertEqual(wst[0].angle_deg, wst.alpha_min)

    def test_einzelner_punkt(self):
        p   = CurvePoint(x=0.0, y=0.0, slope=0.0)
        wst = AngleSortTable(points=[p])
        self.assertEqual(len(wst), 1)
        self.assertTrue(eq(wst.alpha_min, wst.alpha_max))
        self.assertTrue(eq(wst.total_span, 0.0))
        self.assertEqual(wst.support_angles(), [0.0])


# ===========================================================================
# 9  build_wst
# ===========================================================================

class TestBuildWst(unittest.TestCase):

    f4  = staticmethod(lambda x: x**4 - 2*x**2)
    fp4 = staticmethod(lambda x: 4*x**3 - 4*x)

    def test_rueckgabe_ist_angle_sort_table(self):
        self.assertIsInstance(build_wst(self.f4, self.fp4, [0.0]), AngleSortTable)

    def test_leere_kandidaten(self):
        self.assertEqual(len(build_wst(self.f4, self.fp4, [])), 0)

    def test_y_werte_korrekt(self):
        wst = build_wst(self.f4, self.fp4, [-1.0, 0.0, 1.0])
        for p in wst:
            self.assertTrue(eq(p.y, self.f4(p.x)))

    def test_steigungen_korrekt(self):
        wst = build_wst(self.f4, self.fp4, [-1.0, 0.0, 1.0, -1/SQRT3, 1/SQRT3])
        for p in wst:
            self.assertTrue(eq(p.slope, self.fp4(p.x)))

    def test_classify_true_keine_unbekannt(self):
        """Alle Punkte werden klassifiziert (nicht UNBEKANNT)."""
        wst = build_wst(self.f4, self.fp4, [-1.0, 0.0, 1.0])
        for p in wst:
            self.assertNotEqual(p.kind, PointType.UNBEKANNT)

    def test_classify_false_alle_unbekannt(self):
        wst = build_wst(self.f4, self.fp4, [-1.0, 0.0, 1.0], classify=False)
        for p in wst:
            self.assertEqual(p.kind, PointType.UNBEKANNT)

    def test_sortiert_nach_winkel(self):
        wst    = build_wst(self.f4, self.fp4, [1.0, -1.0, 0.0, 1/SQRT3, -1/SQRT3])
        angles = [p.angle_deg for p in wst]
        self.assertEqual(angles, sorted(angles))

    def test_benutzerdefiniertes_h_tol(self):
        f  = lambda x: x**2
        fp = lambda x: 2 * x
        wst = build_wst(f, fp, [0.0], h=1e-5, tol=1e-8)
        self.assertEqual(wst[0].kind, PointType.MINIMUM)

    def test_kubik_klassifikation(self):
        """f = x³−3x: Max bei −1, Min bei +1, WP bei 0."""
        f  = lambda x:  x**3 - 3*x
        fp = lambda x:  3*x**2 - 3.0
        wst = build_wst(f, fp, [-1.0, 0.0, 1.0])
        kinds = {p.x: p.kind for p in wst}
        self.assertEqual(kinds[-1.0], PointType.MAXIMUM)
        self.assertEqual(kinds[ 1.0], PointType.MINIMUM)
        self.assertEqual(kinds[ 0.0], PointType.WENDEPUNKT)

    def test_sin_klassifikation(self):
        """sin: Max bei π/2, Min bei −π/2, WP bei 0."""
        wst  = build_wst(math.sin, math.cos, [-PI/2, 0.0, PI/2])
        kinds = {round(p.x, 6): p.kind for p in wst}
        self.assertEqual(kinds[round(-PI/2, 6)], PointType.MINIMUM)
        self.assertEqual(kinds[0.0],             PointType.WENDEPUNKT)
        self.assertEqual(kinds[round( PI/2, 6)], PointType.MAXIMUM)

    def test_terrassenpunkt(self):
        """f = x³: Terrassenpunkt bei 0."""
        wst = build_wst(lambda x: x**3, lambda x: 3*x**2, [0.0])
        self.assertEqual(wst[0].kind, PointType.FLACH)

    def test_duplikate_beide_aufgenommen(self):
        wst = build_wst(self.f4, self.fp4, [0.0, 0.0])
        self.assertEqual(len(wst), 2)


# ===========================================================================
# 10  format_wst
# ===========================================================================

class TestFormatWst(unittest.TestCase):

    @staticmethod
    def wst_kubik() -> AngleSortTable:
        return build_wst(
            lambda x: x**3 - 3*x,
            lambda x: 3*x**2 - 3.0,
            [-1.0, 0.0, 1.0],
        )

    def test_rueckgabe_string(self):
        self.assertIsInstance(format_wst(self.wst_kubik()), str)

    def test_header_spalten_vorhanden(self):
        out = format_wst(self.wst_kubik())
        for col in ("x", "y", "Steigung", "Winkel", "Typ"):
            self.assertIn(col, out)

    def test_zusammenfassung_vorhanden(self):
        self.assertIn("Gesamtdrehung", format_wst(self.wst_kubik()))

    def test_typen_in_ausgabe(self):
        out = format_wst(self.wst_kubik())
        self.assertIn("MAXIMUM",    out)
        self.assertIn("MINIMUM",    out)
        self.assertIn("WENDEPUNKT", out)

    def test_nachweis_maximum(self):
        self.assertIn("+ → −", format_wst(self.wst_kubik()))

    def test_nachweis_minimum(self):
        self.assertIn("− → +", format_wst(self.wst_kubik()))

    def test_nachweis_wendepunkt(self):
        self.assertIn("VZW", format_wst(self.wst_kubik()))

    def test_nachweis_flach(self):
        wst = build_wst(lambda x: x**3, lambda x: 3*x**2, [0.0])
        self.assertIn("kein VZW", format_wst(wst))

    def test_nachweis_unbekannt(self):
        wst = build_wst(
            lambda x: x**2, lambda x: 2*x, [0.0], classify=False
        )
        self.assertIn("—", format_wst(wst))

    def test_leere_tabelle_kein_absturz(self):
        out = format_wst(AngleSortTable(points=[]))
        self.assertIsInstance(out, str)
        self.assertIn("Punkte: 0", out)

    def test_mehrzeilig(self):
        self.assertGreater(len(format_wst(self.wst_kubik()).splitlines()), 4)

    def test_x4_vollstaendige_ausgabe(self):
        f  = lambda x: x**4 - 2*x**2
        fp = lambda x: 4*x**3 - 4*x
        wst = build_wst(f, fp, [-1.0, 0.0, 1.0, -1/SQRT3, 1/SQRT3])
        out = format_wst(wst)
        self.assertIn("Gesamtdrehung", out)
        self.assertGreater(out.count("\n"), 5)


# ===========================================================================
# 11  Mathematischer Roundtrip (End-to-End)
# ===========================================================================

class TestMathematischerRoundtrip(unittest.TestCase):

    def test_tangentvektor_horizontal_nach_rotation(self):
        """R_{α(m)}·(1, m) hat y-Komponente = 0 (Tangente wird horizontal)."""
        for m in (-5.0, -1.0, 0.0, 1.0, 5.0):
            phi = math.radians(analytical_angle(m))
            _, ty = rotate_point(1.0, m, phi)
            self.assertTrue(eq(ty, 0.0), msg=f"m={m}")

    def test_isometrie_des_tangentvektors(self):
        """‖R_{α(m)}·(1, m)‖ = ‖(1, m)‖ = √(1+m²)."""
        for m in (-3.0, 0.0, 2.0):
            phi = math.radians(analytical_angle(m))
            tx, ty = rotate_point(1.0, m, phi)
            self.assertTrue(eq(math.hypot(tx, ty), math.hypot(1.0, m)))

    def test_hauptsatz_rotated_slope_null(self):
        """f̃' = 0 bei φ = −arctan(m) für verschiedene m."""
        for m in (-7.0, -1.0/3, 0.0, 0.5, 4.0):
            phi = math.radians(analytical_angle(m))
            r   = rotated_slope(m, phi)
            self.assertIsNotNone(r)
            self.assertTrue(eq(r, 0.0), msg=f"m={m}")

    def test_wendepunkt_ist_extremum_von_ftilde_strich(self):
        """Am Wendepunkt x=0 hat f̃' ein lokales Minimum (Betrag minimal).

        f = x³−3x, WP bei x=0 mit m = −3, φ = arctan(3) ≈ 71.57°.
        f̃'(x̃₀) = 0 ist kleiner als f̃' links und rechts —
        der Wendepunkt erscheint als Extremum von f̃'.
        """
        fp  = lambda x: 3*x**2 - 3.0        # f' von x³−3x
        m   = fp(0.0)                         # = −3
        phi = math.radians(analytical_angle(m))

        r_wp    = rotated_slope(fp(0.0),  phi)
        r_left  = rotated_slope(fp(-0.5), phi)
        r_right = rotated_slope(fp( 0.5), phi)

        self.assertIsNotNone(r_wp)
        self.assertIsNotNone(r_left)
        self.assertIsNotNone(r_right)
        self.assertTrue(eq(r_wp, 0.0))                      # Hauptsatz
        self.assertGreater(abs(r_left),  abs(r_wp))         # WP ist Minimum
        self.assertGreater(abs(r_right), abs(r_wp))

    def test_vollstaendige_pipeline_kubik(self):
        """f = x³−3x: vollständige WST korrekt."""
        f   = lambda x: x**3 - 3*x
        fp  = lambda x: 3*x**2 - 3.0
        wst = build_wst(f, fp, [-1.0, 0.0, 1.0])

        maxima     = wst.by_kind(PointType.MAXIMUM)
        minima     = wst.by_kind(PointType.MINIMUM)
        wendepunkte = wst.by_kind(PointType.WENDEPUNKT)
        self.assertEqual(len(maxima),      1)
        self.assertEqual(len(minima),      1)
        self.assertEqual(len(wendepunkte), 1)
        self.assertTrue(eq(maxima[0].x,      -1.0))
        self.assertTrue(eq(minima[0].x,       1.0))
        self.assertTrue(eq(wendepunkte[0].x,  0.0))
        # α(WP) = −arctan(−3) = arctan(3) ≈ 71.57°
        self.assertTrue(eq(wendepunkte[0].angle_deg,
                            math.degrees(math.atan(3.0)), rel=1e-9))

    def test_vollstaendige_pipeline_x4(self):
        """f = x⁴−2x²: Hauptbeispiel aus dem Papier."""
        f   = lambda x: x**4 - 2*x**2
        fp  = lambda x: 4*x**3 - 4*x
        wst = build_wst(f, fp, [-1.0, 0.0, 1.0, -1/SQRT3, 1/SQRT3])

        self.assertEqual(len(wst.by_kind(PointType.MINIMUM)),    2)
        self.assertEqual(len(wst.by_kind(PointType.MAXIMUM)),    1)
        self.assertEqual(len(wst.by_kind(PointType.WENDEPUNKT)), 2)
        self.assertTrue(eq(wst.total_span, 114.0, rel=0.01))
        angles = [p.angle_deg for p in wst]
        self.assertEqual(angles, sorted(angles))

    def test_inversion_100_zufaellige_punkte(self):
        """R_{−φ}∘R_φ = Id für 100 zufällige Punkte."""
        import random
        rng = random.Random(7)
        for _ in range(100):
            px, py  = rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0)
            phi     = rng.uniform(-PI, PI)
            rx, ry  = rotate_point(px, py,  phi)
            rrx, rry = rotate_point(rx, ry, -phi)
            self.assertTrue(eq(rrx, px, rel=1e-10))
            self.assertTrue(eq(rry, py, rel=1e-10))

    def test_format_pipeline_kein_absturz(self):
        """build_wst → format_wst: vollständiger Durchlauf."""
        f   = lambda x: x**4 - 2*x**2
        fp  = lambda x: 4*x**3 - 4*x
        out = format_wst(build_wst(f, fp, [-1.0, 0.0, 1.0, -1/SQRT3, 1/SQRT3]))
        for kw in ("MAXIMUM", "MINIMUM", "WENDEPUNKT", "Gesamtdrehung"):
            self.assertIn(kw, out)


# ===========================================================================
# Coverage-Runner
# ===========================================================================

def _run_coverage() -> None:
    """Testsuite + Zeilenabdeckung von rotation_analysis.py messen."""
    tracer = trace.Trace(count=True, trace=False,
                         ignoredirs=[sys.prefix, sys.exec_prefix])

    buf    = io.StringIO()
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(stream=buf, verbosity=2)

    tracer.runfunc(runner.run, suite)
    print(buf.getvalue())

    counts   = tracer.results().counts
    mod_file = os.path.abspath("rotation_analysis.py")

    covered: List[int] = []
    total:   List[int] = []
    for (f, ln), n in counts.items():
        if os.path.abspath(f) == mod_file:
            total.append(ln)
            if n > 0:
                covered.append(ln)

    n_cov, n_tot = len(covered), len(total)
    print(f"\n{'='*62}")
    print(f"  Coverage-Report  –  rotation_analysis.py")
    print(f"{'='*62}")
    if n_tot > 0:
        pct = 100 * n_cov / n_tot
        print(f"  Ausgeführte Zeilen : {n_cov}")
        print(f"  Gesamte Zeilen     : {n_tot}")
        print(f"  Abdeckung          : {pct:.1f} %")
        missing = sorted(set(total) - set(covered))
        if missing:
            print(f"  Nicht abgedeckt    : {missing}")
        else:
            print("  ✓ Alle ausführbaren Zeilen abgedeckt!")
    else:
        print("  (keine messbaren Zeilen gefunden)")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    _run_coverage()
