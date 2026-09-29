"""pytest-Tests zu odd.py (Kapitel "Der ungerade Anteil der e-Reihe").

Ausführen:
    pytest -v test_odd.py
    pytest --cov=odd --cov-branch --cov-report=term-missing test_odd.py

Referenzwerte stammen aus math.exp/sinh/cosh/atan bzw. aus exakten Brüchen,
nicht aus den getesteten Funktionen selbst.
"""

import math
from fractions import Fraction

import pytest

import odd as ua

XS = [-6.0, -2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 6.0]
XS_NONZERO = [x for x in XS if x != 0.0]


# ---------------------------------------------------------------- Definition 1
class TestReihen:
    @pytest.mark.parametrize("x", XS)
    def test_e_series_matches_exp(self, x):
        assert ua.e_series(x) == pytest.approx(math.exp(x), rel=1e-12)

    @pytest.mark.parametrize("x", XS)
    def test_gerade_is_cosh(self, x):
        assert ua.e_gerade(x) == pytest.approx(math.cosh(x), rel=1e-12)

    @pytest.mark.parametrize("x", XS)
    def test_ungerade_is_sinh(self, x):
        assert ua.e_ungerade(x) == pytest.approx(math.sinh(x), rel=1e-12, abs=1e-15)

    @pytest.mark.parametrize("x", XS)
    def test_lemma1c_decomposition(self, x):
        assert ua.e_gerade(x) + ua.e_ungerade(x) == pytest.approx(ua.e_series(x), rel=1e-12)
        assert ua.e_gerade(x) - ua.e_ungerade(x) == pytest.approx(ua.e_series(-x), rel=1e-12)

    @pytest.mark.parametrize("x", XS)
    def test_parity(self, x):
        assert ua.e_gerade(-x) == pytest.approx(ua.e_gerade(x))
        assert ua.e_ungerade(-x) == pytest.approx(-ua.e_ungerade(x))

    @pytest.mark.parametrize("x", XS)
    def test_lemma1b_functional_equation(self, x):
        assert ua.e_series(x) * ua.e_series(-x) == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("x", XS)
    def test_lemma1d_identity(self, x):
        assert ua.e_gerade(x) ** 2 - ua.e_ungerade(x) ** 2 == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize("x", XS)
    def test_lemma1e_signs(self, x):
        assert ua.e_series(x) > 0
        assert ua.e_gerade(x) >= 1.0
        if x == 0:
            assert ua.e_ungerade(x) == 0.0
            assert ua.e_gerade(x) == 1.0
        else:
            assert math.copysign(1, ua.e_ungerade(x)) == math.copysign(1, x)

    def test_complex_argument_euler(self):
        theta = 0.7
        assert ua.e_series(1j * theta) == pytest.approx(complex(math.cos(theta), math.sin(theta)))
        g, u = ua.imaginary_axis_parts(theta)
        assert g == pytest.approx(math.cos(theta))
        assert u == pytest.approx(1j * math.sin(theta))

    @pytest.mark.parametrize("func", [ua.e_series, ua.e_gerade, ua.e_ungerade])
    def test_n_terms_validation(self, func):
        with pytest.raises(ValueError):
            func(1.0, 0)

    def test_truncation_error_decreases(self):
        errs = [abs(ua.e_ungerade(1.0, n) - math.sinh(1.0)) for n in range(1, 6)]
        assert all(a > b for a, b in zip(errs, errs[1:]))

    def test_single_term(self):
        assert ua.e_series(2.0, 1) == 1.0
        assert ua.e_gerade(2.0, 1) == 1.0
        assert ua.e_ungerade(2.0, 1) == 2.0


class TestPartialSums:
    def test_odd_partial_sum_exact(self):
        assert ua.odd_partial_sum(0) == Fraction(1)
        assert ua.odd_partial_sum(1) == Fraction(7, 6)
        assert ua.odd_partial_sum(2) == Fraction(141, 120)

    def test_even_partial_sum_exact(self):
        assert ua.even_partial_sum(0) == Fraction(1)
        assert ua.even_partial_sum(1) == Fraction(3, 2)
        assert ua.even_partial_sum(2) == Fraction(37, 24)

    def test_converge_to_sinh_cosh(self):
        assert float(ua.odd_partial_sum(12)) == pytest.approx(math.sinh(1), rel=1e-15)
        assert float(ua.even_partial_sum(12)) == pytest.approx(math.cosh(1), rel=1e-15)

    @pytest.mark.parametrize("func", [ua.odd_partial_sum, ua.even_partial_sum])
    def test_negative_rejected(self, func):
        with pytest.raises(ValueError):
            func(-1)


# ---------------------------------------------------- Lemma 2 / Bemerkung 1
class TestAbleitung:
    def test_series_coefficients(self):
        assert ua.series_coefficients("all", 3) == [Fraction(1), Fraction(1), Fraction(1, 2), Fraction(1, 6)]
        assert ua.series_coefficients("even", 4) == [Fraction(1), 0, Fraction(1, 2), 0, Fraction(1, 24)]
        assert ua.series_coefficients("odd", 3) == [0, Fraction(1), 0, Fraction(1, 6)]

    def test_series_coefficients_validation(self):
        with pytest.raises(ValueError):
            ua.series_coefficients("odd", -1)
        with pytest.raises(ValueError):
            ua.series_coefficients("mixed", 3)

    def test_derivative_of_odd_is_even(self):
        odd = ua.series_coefficients("odd", 9)
        assert ua.derivative_coefficients(odd) == ua.series_coefficients("even", 8)

    def test_derivative_of_even_is_odd_and_loses_constant(self):
        even = ua.series_coefficients("even", 8)
        d = ua.derivative_coefficients(even)
        assert d == ua.series_coefficients("odd", 7)[:len(d)]
        assert even[0] == 1 and d[0] == 0  # das konstante Glied 1 verschwindet
        assert d[1] == 1                   # x^2/2 wird zu x

    def test_second_derivative_reproduces_parts(self):
        for parity in ("odd", "even"):
            c = ua.series_coefficients(parity, 10)
            dd = ua.derivative_coefficients(ua.derivative_coefficients(c))
            assert dd == c[:len(dd)]

    def test_derivative_of_full_series(self):
        c = ua.series_coefficients("all", 8)
        assert ua.derivative_coefficients(c) == c[:-1]

    def test_derivative_of_constant(self):
        assert ua.derivative_coefficients([Fraction(5)]) == [Fraction(0)]


# ------------------------------------------------------------- Satz 1, Korollar 1
class TestRekonstruktion:
    @pytest.mark.parametrize("x", XS)
    def test_even_from_odd(self, x):
        assert ua.even_from_odd(math.sinh(x)) == pytest.approx(math.cosh(x), rel=1e-12)

    @pytest.mark.parametrize("x", XS)
    def test_exp_from_odd(self, x):
        assert ua.exp_from_odd(x) == pytest.approx(math.exp(x), rel=1e-11)

    @pytest.mark.parametrize("x", XS)
    def test_inv_exp_from_odd(self, x):
        assert ua.inv_exp_from_odd(x) == pytest.approx(math.exp(-x), rel=1e-9)

    def test_e_from_odd_converges(self):
        assert ua.e_from_odd(8) == pytest.approx(math.e, rel=1e-14)

    def test_e_from_odd_underestimates(self):
        for n in range(6):
            assert ua.e_from_odd(n) < math.e

    def test_even_coefficients_from_odd(self):
        got = ua.even_coefficients_from_odd(6)
        assert got == [Fraction(1, math.factorial(2 * k)) for k in range(7)]

    def test_even_coefficients_k_zero(self):
        assert ua.even_coefficients_from_odd(0) == [Fraction(1)]

    def test_chapter_example_coefficients(self):
        got = ua.even_coefficients_from_odd(3)
        assert got[1] == Fraction(1, 2)
        assert got[2] == Fraction(1, 24)
        assert got[3] == Fraction(1, 720)

    def test_even_coefficients_validation(self):
        with pytest.raises(ValueError):
            ua.even_coefficients_from_odd(-1)


# ---------------------------------------------------- Asymmetrie (Satz 2/3)
class TestAsymmetrie:
    def test_candidates_are_e_and_inverse(self):
        plus, minus = ua.exp_candidates_from_even(math.cosh(1))
        assert plus == pytest.approx(math.e, rel=1e-12)
        assert minus == pytest.approx(1 / math.e, rel=1e-12)
        assert plus * minus == pytest.approx(1.0)

    def test_even_part_identical_for_plus_and_minus_one(self):
        assert ua.e_gerade(1.0) == pytest.approx(ua.e_gerade(-1.0))

    def test_exp_from_even_needs_sign(self):
        c = ua.e_gerade(1.0)
        assert ua.exp_from_even(c, +1) == pytest.approx(math.e, rel=1e-12)
        assert ua.exp_from_even(c, -1) == pytest.approx(math.exp(-1), rel=1e-12)

    def test_exp_from_even_validation(self):
        with pytest.raises(ValueError):
            ua.exp_from_even(1.5, 0)
        with pytest.raises(ValueError):
            ua.exp_candidates_from_even(0.99)

    def test_exp_candidates_at_one(self):
        assert ua.exp_candidates_from_even(1.0) == (1.0, 1.0)

    @pytest.mark.parametrize("x", XS_NONZERO)
    def test_no_function_of_even_part_gives_odd_part(self, x):
        # Satz 2(b): gleicher gerader Anteil bei x und -x, verschiedener ungerader
        assert ua.e_gerade(x) == pytest.approx(ua.e_gerade(-x))
        assert ua.e_ungerade(x) != pytest.approx(ua.e_ungerade(-x))

    def test_split_parity_lemma3(self):
        def f(x):
            return x ** 3 + 2 * x ** 2 + 5 * x + 1
        f_g, f_u = ua.split_parity(f)
        for x in XS:
            assert f(x) - f(-x) == pytest.approx(2 * f_u(x))
            assert f_g(x) + f_u(x) == pytest.approx(f(x))
            assert f_g(x) == pytest.approx(f_g(-x))
            assert f_u(-x) == pytest.approx(-f_u(x))

    def test_split_parity_of_exp(self):
        f_g, f_u = ua.split_parity(math.exp)
        for x in XS:
            assert f_g(x) == pytest.approx(math.cosh(x))
            assert f_u(x) == pytest.approx(math.sinh(x), abs=1e-15)

    @pytest.mark.parametrize("y", [-50.0, -1.0, 0.0, 0.5, 3.0, 100.0])
    def test_odd_part_bijective_inverse(self, y):
        x = ua.arsinh_via_odd(y)
        assert x == pytest.approx(math.asinh(y), abs=1e-12)
        assert ua.e_ungerade(x) == pytest.approx(y, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize("y", [-50.0, -1.0, 0.0, 0.5, 3.0, 100.0])
    def test_inverse_odd_bisection(self, y):
        assert ua.inverse_odd(y) == pytest.approx(math.asinh(y), abs=1e-9)

    def test_odd_part_strictly_increasing(self):
        vals = [ua.e_ungerade(x) for x in sorted(XS)]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_sign_entropy_symmetric_sample(self):
        sample = [-3.0, -2.0, -0.5, 0.5, 2.0, 3.0]
        assert ua.conditional_sign_entropy(sample, ua.e_gerade) == pytest.approx(1.0)
        assert ua.conditional_sign_entropy(sample, ua.e_ungerade) == pytest.approx(0.0)

    def test_sign_entropy_asymmetric_sample_between(self):
        sample = [-1.0, 1.0, 2.0]
        h = ua.conditional_sign_entropy(sample, ua.e_gerade)
        assert 0.0 < h < 1.0

    def test_sign_entropy_rejects_zero(self):
        with pytest.raises(ValueError):
            ua.conditional_sign_entropy([-1.0, 0.0, 1.0], ua.e_gerade)


# ----------------------------------------------------- Satz 4 / Korollar 3
class TestZeitrichtung:
    @pytest.mark.parametrize("t", [0.0, 0.1, 1.0, 3.0])
    @pytest.mark.parametrize("tau", [0.2, 1.0, 5.0])
    def test_decay_parts(self, t, tau):
        g, u, decay = ua.rc_decay_parts(t, tau)
        assert g == pytest.approx(math.cosh(t / tau), rel=1e-12)
        assert u == pytest.approx(math.sinh(t / tau), rel=1e-12, abs=1e-15)
        # cosh - sinh loescht sich fuer grosses t/tau aus (Verlust ~ 1e-16 * cosh)
        assert decay == pytest.approx(math.exp(-t / tau), rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize("t", [0.0, 0.5, 2.0, 8.0])
    def test_step_response(self, t):
        assert ua.rc_step_response(t, 1.5) == pytest.approx(1 - math.exp(-t / 1.5), rel=1e-9, abs=1e-15)

    def test_step_response_monotone_and_bounded(self):
        vals = [ua.rc_step_response(t, 1.0) for t in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)]
        assert vals[0] == pytest.approx(0.0, abs=1e-15)
        assert all(a < b for a, b in zip(vals, vals[1:]))
        assert all(v < 1.0 for v in vals)

    def test_growth_and_decay_share_even_part(self):
        t, tau = 2.0, 1.0
        g_dec = ua.e_gerade(-t / tau)
        g_grow = ua.e_gerade(t / tau)
        assert g_dec == pytest.approx(g_grow)
        assert ua.e_ungerade(-t / tau) == pytest.approx(-ua.e_ungerade(t / tau))

    def test_tau_validation(self):
        with pytest.raises(ValueError):
            ua.rc_decay_parts(1.0, 0.0)
        with pytest.raises(ValueError):
            ua.rc_step_response(1.0, -1.0)


# ------------------------------------------------------------------ Satz 5
class TestFundamentalloesung:
    @pytest.mark.parametrize("x", XS)
    def test_impulse_response(self, x):
        assert ua.general_solution(0.0, 1.0, x) == pytest.approx(math.sinh(x), rel=1e-12, abs=1e-15)

    @pytest.mark.parametrize("y0,v0", [(0.0, 1.0), (1.0, 0.0), (2.0, -3.0), (-1.5, 0.7)])
    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
    def test_general_solution_matches_rk4(self, y0, v0, x):
        assert ua.general_solution(y0, v0, x) == pytest.approx(ua.rk4_second_order(y0, v0, x), rel=1e-9)

    def test_initial_conditions(self):
        assert ua.general_solution(2.0, -3.0, 0.0) == 2.0

    def test_rk4_validation(self):
        with pytest.raises(ValueError):
            ua.rk4_second_order(1.0, 0.0, 1.0, steps=0)


# ------------------------------------------------------------------ Satz 6
class TestGewicht:
    @pytest.mark.parametrize("x", XS)
    def test_even_dominates_by_exactly_exp_minus_abs(self, x):
        assert ua.magnitude_gap(x) == pytest.approx(math.exp(-abs(x)), rel=1e-9)
        assert abs(ua.e_ungerade(x)) < ua.e_gerade(x)

    @pytest.mark.parametrize("x", XS)
    def test_ratio_is_tanh(self, x):
        assert ua.odd_even_ratio(x) == pytest.approx(math.tanh(abs(x)), rel=1e-12, abs=1e-15)

    def test_ratio_increases_to_one(self):
        r = [ua.odd_even_ratio(x) for x in (0.5, 1.0, 2.0, 5.0, 10.0)]
        assert all(a < b for a, b in zip(r, r[1:]))
        assert r[-1] == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("a", [0.1, 0.5, 1.0, 3.0])
    def test_energy_norms_against_numeric_integration(self, a):
        n = 4000
        h = 2 * a / n
        xs = [-a + (i + 0.5) * h for i in range(n)]
        int_g = sum(math.cosh(x) ** 2 for x in xs) * h
        int_u = sum(math.sinh(x) ** 2 for x in xs) * h
        int_gu = sum(math.cosh(x) * math.sinh(x) for x in xs) * h
        int_e = sum(math.exp(2 * x) for x in xs) * h
        g, u, mixed, total = ua.energy_norms(a)
        assert g == pytest.approx(int_g, rel=1e-5)
        assert u == pytest.approx(int_u, rel=1e-5)
        assert mixed == 0.0 and int_gu == pytest.approx(0.0, abs=1e-9)
        assert total == pytest.approx(int_e, rel=1e-5)
        assert g + u == pytest.approx(total)

    def test_energy_share(self):
        for a in (0.2, 1.0, 4.0):
            share = ua.odd_energy_share(a)
            assert share == pytest.approx(0.5 - a / math.sinh(2 * a), rel=1e-9)
            assert 0.0 < share < 0.5
        assert ua.odd_energy_share(15.0) == pytest.approx(0.5, abs=1e-8)

    def test_energy_validation(self):
        with pytest.raises(ValueError):
            ua.energy_norms(0.0)
        with pytest.raises(ValueError):
            ua.energy_norms(-1.0)


# ------------------------------------------------------------------ Satz 7
class TestFehlerabschaetzung:
    @pytest.mark.parametrize("n", range(0, 9))
    def test_tail_bound_holds(self, n):
        # exakter Rest als Bruch (Gleitkomma-Differenz waere fuer grosse n reine Rundung)
        r_n = sum(Fraction(1, math.factorial(2 * k + 1)) for k in range(n + 1, n + 25))
        assert 0 < float(r_n) <= ua.tail_bound(n)

    @pytest.mark.parametrize("n", range(0, 6))
    def test_error_bound_holds_and_is_tight(self, n):
        err = ua.error_odd(n)
        bound = ua.error_bound(n)
        assert 0 < err < bound
        assert err > 0.9 * bound

    def test_bound_constant(self):
        assert ua.error_bound(0) == pytest.approx((1 + math.tanh(1)) * 16 / 15 / 6)
        assert ua.error_bound(3) < 1.88 / math.factorial(9)

    def test_odd_beats_even_for_equal_number_of_terms(self):
        for n in range(1, 6):
            assert ua.error_odd(n) < ua.error_even(n)

    def test_error_table_matches_chapter(self):
        tab = ua.error_table(5)
        assert len(tab) == 6
        assert tab[3]["N"] == 3
        assert tab[3]["g(S_N)"] == pytest.approx(2.71827692956, abs=1e-11)
        assert tab[3]["error_odd"] == pytest.approx(4.90e-6, rel=2e-3)
        assert tab[3]["bound"] == pytest.approx(5.18e-6, rel=2e-3)
        assert tab[3]["error_even"] == pytest.approx(5.80e-5, rel=2e-3)
        assert tab[0]["S_N"] == 1.0
        assert tab[0]["error_odd"] == pytest.approx(0.304, abs=5e-4)
        assert tab[0]["error_even"] == pytest.approx(1.72, abs=5e-3)
        for row in tab:
            assert row["error_odd"] < row["bound"]

    def test_validation(self):
        with pytest.raises(ValueError):
            ua.tail_bound(-1)
        with pytest.raises(ValueError):
            ua.error_table(-1)


# ------------------------------------------------------- Satz 8-11 (Leibniz)
class TestLeibniz:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_leibniz_criterion_bound(self, n):
        assert abs(ua.leibniz_partial(n) - math.pi / 4) <= 1 / (2 * n + 3)

    def test_leibniz_alternates_around_limit(self):
        assert ua.leibniz_partial(4) > math.pi / 4 > ua.leibniz_partial(5)

    def test_leibniz_first_terms(self):
        assert ua.leibniz_partial(0) == 1.0
        assert ua.leibniz_partial(1) == pytest.approx(2 / 3)

    def test_leibniz_validation(self):
        with pytest.raises(ValueError):
            ua.leibniz_partial(-1)

    def test_averaging_accelerates(self):
        n = 200
        plain = abs(ua.leibniz_partial(n) - math.pi / 4)
        avg = abs(ua.averaged(ua.leibniz_partial, n) - math.pi / 4)
        assert avg < plain / 50

    def test_log_series_at_i_converges_to_log_1_plus_i(self):
        n = 4001
        s = 0.5 * (ua.log_series_at_i(n) + ua.log_series_at_i(n + 1))
        # Realteil aendert sich nur bei geraden n -> Mittelung glaettet ihn (Fehler ~1/n^2);
        # Imaginaerteil ist die Leibniz-Partialsumme mit Fehler <= 1/(2N+3) (Leibniz-Kriterium)
        assert s.real == pytest.approx(0.5 * math.log(2), abs=1e-6)
        assert s.imag == pytest.approx(math.pi / 4, abs=3e-4)
        assert abs(s - complex(math.log(math.sqrt(2)), math.atan(1.0))) < 3e-4

    def test_even_odd_split_of_log_series(self):
        n = 5000
        even, odd = ua.log_series_even_odd(n)
        full = ua.log_series_at_i(n)
        assert even == pytest.approx(full.real, abs=1e-12)
        assert odd == pytest.approx(full.imag, abs=1e-12)

    def test_odd_index_part_is_leibniz_series(self):
        # n_terms = 2N+1 ungerade Indizes 1,3,...,2N+1 -> Leibniz-Partialsumme L_N
        for big_n in (0, 3, 25):
            _, odd = ua.log_series_even_odd(2 * big_n + 1)
            assert odd == pytest.approx(ua.leibniz_partial(big_n), abs=1e-12)

    def test_even_index_part_is_half_alternating_harmonic(self):
        even, _ = ua.log_series_even_odd(2 * 50)
        harmonic = sum((-1) ** (k + 1) / k for k in range(1, 51))
        assert even == pytest.approx(0.5 * harmonic, abs=1e-12)

    def test_conjugation_flips_only_odd_part(self):
        # Satz 10: l(-i) = conj(l(i)); gerader Anteil gleich, ungerader mit Vorzeichen
        n = 301
        val_plus = ua.log_series_at_i(n)
        val_minus = sum(((-1) ** (k + 1)) * ((-1j) ** k) / k for k in range(1, n + 1))
        assert val_minus == pytest.approx(val_plus.conjugate())
        assert val_minus.real == pytest.approx(val_plus.real)
        assert val_minus.imag == pytest.approx(-val_plus.imag)

    def test_log_series_validation(self):
        with pytest.raises(ValueError):
            ua.log_series_at_i(0)
        with pytest.raises(ValueError):
            ua.log_series_even_odd(0)

    def test_polar_form(self):
        modulus, phase = ua.polar_from_log_parts()
        assert modulus == pytest.approx(math.sqrt(2))
        assert phase == pytest.approx((1 + 1j) / math.sqrt(2))
        assert modulus * phase == pytest.approx(1 + 1j)

    def test_e_of_4iL_is_minus_one(self):
        L = math.pi / 4
        assert ua.e_series(4j * L) == pytest.approx(-1.0, abs=1e-12)

    def test_balance_angle_is_leibniz_value(self):
        assert ua.balance_angle() == pytest.approx(math.pi / 4, abs=1e-12)

    def test_balance_values(self):
        g, u = ua.imaginary_axis_parts(math.pi / 4)
        assert g == pytest.approx(1 / math.sqrt(2))
        assert u == pytest.approx(1j / math.sqrt(2))
        assert abs(g) == pytest.approx(abs(u))

    @pytest.mark.parametrize("theta", [0.1, 0.5, 0.7])
    def test_before_balance_even_dominates(self, theta):
        g, u = ua.imaginary_axis_parts(theta)
        assert abs(g) > abs(u)

    @pytest.mark.parametrize("theta", [0.9, 1.2, 1.5])
    def test_after_balance_odd_dominates(self, theta):
        g, u = ua.imaginary_axis_parts(theta)
        assert abs(u) > abs(g)
