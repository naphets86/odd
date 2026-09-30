"""Tests for ecos.ship_freight: every theorem of the chapter is checked numerically."""

import dataclasses
import math
import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq

from ecos import ship_freight as sf
from ecos.ship_freight import ShipMarketParams

# Chapter example
D0, DH, KL, T = 4000.0, 1000.0, 2500.0, 1.0
G_EX = 1500.0


def numeric_gap(d0, d_hat, K_L, T, n=400_001):
    t = np.linspace(0.0, T, n)
    u = np.maximum(d0 + d_hat * np.sin(2 * np.pi * t / T) - K_L, 0.0)
    return float(np.trapezoid(u, t))


class TestDemandAndGap(unittest.TestCase):
    def test_demand_rate_values_and_periodicity(self):
        self.assertAlmostEqual(sf.demand_rate(0.0, D0, DH, T), D0)
        self.assertAlmostEqual(sf.demand_rate(0.25, D0, DH, T), D0 + DH)
        self.assertAlmostEqual(sf.demand_rate(0.3, D0, DH, T), sf.demand_rate(1.3, D0, DH, T))

    def test_demand_rate_validation(self):
        for args in [(0, 4000.0, 0.0, 1.0), (0, 4000.0, -1.0, 1.0),
                     (0, 4000.0, 1000.0, 0.0), (0, 1000.0, 1000.0, 1.0),
                     (0, 500.0, 1000.0, 1.0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sf.demand_rate(*args)

    def test_gap_rate(self):
        self.assertAlmostEqual(sf.gap_rate(0.25, D0, DH, KL, T), 2500.0)
        self.assertEqual(sf.gap_rate(0.75, D0, DH, 6000.0, T), 0.0)
        with self.assertRaises(ValueError):
            sf.gap_rate(0.0, D0, DH, -1.0, T)

    def test_period_gap_closed_form_matches_integral(self):
        for K in [0.0, 1000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0,
                  5500.0, 6000.0, 7000.0]:
            with self.subTest(K_L=K):
                self.assertAlmostEqual(sf.period_gap(D0, DH, K, T),
                                       numeric_gap(D0, DH, K, T), delta=1e-3)

    def test_period_gap_cases_and_boundaries(self):
        self.assertEqual(sf.period_gap(D0, DH, 5000.0, T), 0.0)      # c = -1
        self.assertEqual(sf.period_gap(D0, DH, 6000.0, T), 0.0)      # c < -1
        self.assertAlmostEqual(sf.period_gap(D0, DH, 3000.0, T), 1000.0)   # c = 1
        self.assertAlmostEqual(sf.period_gap(D0, DH, KL, 2.0), 3000.0)
        eps = 1e-9
        self.assertAlmostEqual(sf.period_gap(D0, DH, 3000.0 - eps, T),
                               sf.period_gap(D0, DH, 3000.0 + eps, T), places=5)
        self.assertLess(sf.period_gap(D0, DH, 5000.0 - eps, T), 1e-6)

    def test_period_gap_validation(self):
        with self.assertRaises(ValueError):
            sf.period_gap(D0, DH, -1.0, T)
        with self.assertRaises(ValueError):
            sf.period_gap(500.0, DH, KL, T)

    def test_cumulative_gap_matches_numeric_for_all_regimes(self):
        for K in [2500.0, 3500.0, 4500.0, 5500.0, 6000.0]:   # c>=1, c>0, c<0, c<0, c<=-1
            t = np.linspace(0.0, 3.0, 300_001)
            u = np.maximum(D0 + DH * np.sin(2 * np.pi * t) - K, 0.0)
            ref = cumulative_trapezoid(u, t, initial=0.0)
            for tt in [0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.7, 2.9, 3.0]:
                idx = int(round(tt / 3.0 * 300_000))
                with self.subTest(K=K, t=tt):
                    self.assertAlmostEqual(sf.cumulative_gap(tt, D0, DH, K, T),
                                           ref[idx], delta=2e-3)

    def test_cumulative_gap_structure(self):
        G = sf.period_gap(D0, DH, 3500.0, T)
        self.assertAlmostEqual(sf.cumulative_gap(1.0, D0, DH, 3500.0, T), G)
        self.assertAlmostEqual(sf.cumulative_gap(2.3, D0, DH, 3500.0, T),
                               2 * G + sf.cumulative_gap(0.3, D0, DH, 3500.0, T))
        ts = np.linspace(0, 2, 200)
        vals = [sf.cumulative_gap(float(x), D0, DH, 3500.0, T) for x in ts]
        self.assertTrue(np.all(np.diff(vals) >= -1e-9))
        with self.assertRaises(ValueError):
            sf.cumulative_gap(-0.1, D0, DH, KL, T)


class TestBalance(unittest.TestCase):
    def test_load_and_departures(self):
        self.assertAlmostEqual(sf.load_for_departures(1500.0, 20), 75.0)
        self.assertAlmostEqual(sf.departures_for_load(1500.0, 75.0), 20.0)
        self.assertAlmostEqual(sf.period_imbalance(20, 75.0, 1500.0), 0.0)
        self.assertAlmostEqual(sf.period_imbalance(20, 80.0, 1500.0), 100.0)
        with self.assertRaises(ValueError):
            sf.load_for_departures(1500.0, 0)
        with self.assertRaises(ValueError):
            sf.load_for_departures(0.0, 5)
        with self.assertRaises(ValueError):
            sf.departures_for_load(0.0, 5.0)
        with self.assertRaises(ValueError):
            sf.departures_for_load(1500.0, 0.0)

    def test_time_estimates_values(self):
        self.assertAlmostEqual(sf.overflow_time_estimate(500, 250, 1600, 1500, 1.0), 2.5)
        self.assertAlmostEqual(sf.overflow_time_tolerance(1600, 1500, 1.0), 16.0)
        self.assertAlmostEqual(sf.shortage_time_estimate(500, 250, 1200, 1500, 1.0), 250 / 300)
        self.assertAlmostEqual(sf.shortage_time_tolerance(1200, 1500, 1.0), 5.0)

    def test_time_estimates_validation(self):
        with self.assertRaises(ValueError):
            sf.overflow_time_estimate(500, 250, 1500, 1500, 1.0)     # Q = G
        with self.assertRaises(ValueError):
            sf.overflow_time_estimate(500, 600, 1600, 1500, 1.0)     # I0 > B
        with self.assertRaises(ValueError):
            sf.overflow_time_estimate(500, -1, 1600, 1500, 1.0)      # I0 < 0
        with self.assertRaises(ValueError):
            sf.overflow_time_estimate(0, 0, 1600, 1500, 1.0)         # B = 0
        with self.assertRaises(ValueError):
            sf.overflow_time_estimate(500, 250, 1600, 1500, 0.0)     # T = 0
        with self.assertRaises(ValueError):
            sf.overflow_time_tolerance(1500, 1500, 1.0)
        with self.assertRaises(ValueError):
            sf.overflow_time_tolerance(1600, 1500, 0.0)
        with self.assertRaises(ValueError):
            sf.shortage_time_estimate(500, 250, 1500, 1500, 1.0)
        with self.assertRaises(ValueError):
            sf.shortage_time_estimate(500, 250, 1200, 1500, 0.0)
        with self.assertRaises(ValueError):
            sf.shortage_time_tolerance(1500, 1500, 1.0)
        with self.assertRaises(ValueError):
            sf.shortage_time_tolerance(1200, 1500, 0.0)

    def test_stock_at_calls_validation(self):
        base = dict(N=20, L=75.0, d0=D0, d_hat=DH, K_L=KL, T=T, I0=100.0)
        for change in [dict(N=0), dict(L=0.0), dict(T=0.0), dict(t0=-0.01),
                       dict(t0=0.05), dict(n_periods=0)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                sf.stock_at_calls(**{**base, **change})

    def test_period_shift_lemma(self):
        # X(t + nT) = X(t) + n (Q - G), and the bound I0 - G <= X <= I0 + Q on [0, T).
        for L in [70.0, 75.0, 80.0]:
            tr = sf.stock_at_calls(20, L, D0, DH, KL, T, I0=100.0, t0=0.013, n_periods=3)
            Q = 20 * L
            self.assertAlmostEqual(tr.G, G_EX)
            for k in range(40):
                self.assertAlmostEqual(tr.before[k + 20] - tr.before[k], Q - tr.G, places=6)
            first = slice(0, 20)
            self.assertTrue(np.all(tr.before[first] >= 100.0 - tr.G - 1e-9))
            self.assertTrue(np.all(tr.after[first] <= 100.0 + Q + 1e-9))

    def test_balance_theorem_overflow(self):
        B = 500.0
        for N, L, t0, I0 in [(20, 80.0, 0.0, 250.0), (20, 78.0, 0.02, 0.0),
                             (10, 160.0, 0.05, 400.0), (30, 52.0, 0.01, 500.0)]:
            with self.subTest(N=N, L=L):
                Q = N * L
                tr = sf.stock_at_calls(N, L, D0, DH, KL, T, I0, t0, n_periods=40)
                tB = sf.first_overflow_time(tr, B)
                self.assertIsNotNone(tB)
                t_star = sf.overflow_time_estimate(B, I0, Q, tr.G, T)
                self.assertLessEqual(abs(tB - t_star), sf.overflow_time_tolerance(Q, tr.G, T))

    def test_balance_theorem_shortage(self):
        for N, L, t0, I0 in [(20, 60.0, 0.0, 250.0), (20, 70.0, 0.02, 100.0),
                             (10, 100.0, 0.05, 500.0)]:
            with self.subTest(N=N, L=L):
                Q = N * L
                tr = sf.stock_at_calls(N, L, D0, DH, KL, T, I0, t0, n_periods=40)
                tF = sf.first_shortage_time(tr, I0, D0, DH, KL, T)
                self.assertIsNotNone(tF)
                t_star = sf.shortage_time_estimate(500.0, I0, Q, tr.G, T)
                self.assertLessEqual(abs(tF - t_star), sf.shortage_time_tolerance(Q, tr.G, T))

    def test_shortage_at_time_zero_branch(self):
        # X(t0^-) < 0 already before the first call: root is located in (0, t0].
        tr = sf.stock_at_calls(20, 75.0, D0, DH, KL, T, I0=0.0, t0=0.01)
        self.assertLess(tr.before[0], 0.0)
        self.assertEqual(sf.first_shortage_time(tr, 0.0, D0, DH, KL, T), 0.0)

    def test_shortage_root_between_calls(self):
        tr = sf.stock_at_calls(20, 60.0, D0, DH, KL, T, I0=100.0, t0=0.0, n_periods=10)
        tF = sf.first_shortage_time(tr, 100.0, D0, DH, KL, T)
        k = int(np.nonzero(tr.before < 0)[0][0])
        self.assertGreater(k, 0)
        self.assertGreater(tF, tr.times[k - 1])
        self.assertLessEqual(tF, tr.times[k])
        x = 100.0 + 60.0 * (k) - sf.cumulative_gap(tF, D0, DH, KL, T)
        self.assertAlmostEqual(x, 0.0, places=6)

    def test_no_violation_when_balanced_and_buffer_large(self):
        N = 20
        tr = sf.stock_at_calls(N, G_EX / N, D0, DH, KL, T, I0=1000.0, t0=0.0, n_periods=5)
        self.assertIsNone(sf.first_overflow_time(tr, 10_000.0))
        self.assertIsNone(sf.first_shortage_time(tr, 1000.0, D0, DH, KL, T))
        # Necessity (Satz 1.1): with Q = G the stock stays periodic, bounded.
        self.assertLess(tr.after.max() - tr.before.min(), 1_000.0)


class TestBuffer(unittest.TestCase):
    def test_season_buffer_and_limits(self):
        self.assertAlmostEqual(sf.season_buffer(DH, T), 318.3098861837907)
        self.assertAlmostEqual(sf.buffer_upper_bound(75.0, DH, T), 75.0 + 318.3098861837907)
        self.assertAlmostEqual(sf.buffer_limit_sufficient(500.0, DH, T), 181.6901138162093)
        with self.assertRaises(ValueError):
            sf.season_buffer(0.0, T)
        with self.assertRaises(ValueError):
            sf.season_buffer(DH, 0.0)
        with self.assertRaises(ValueError):
            sf.buffer_upper_bound(0.0, DH, T)
        with self.assertRaises(ValueError):
            sf.buffer_limit_sufficient(0.0, DH, T)

    def test_buffer_requirement_matches_simulation_and_bounds(self):
        for N in [1, 2, 3, 7, 12, 27, 50]:
            for t0_frac in [0.0, 0.3, 0.9]:
                with self.subTest(N=N, t0_frac=t0_frac):
                    t0 = t0_frac * T / N
                    Bmin = sf.buffer_requirement(N, D0, DH, KL, T, t0)
                    L = G_EX / N
                    tr = sf.stock_at_calls(N, L, D0, DH, KL, T, I0=0.0, t0=t0)
                    self.assertAlmostEqual(Bmin, tr.after.max() - tr.before.min(), places=6)
                    self.assertGreaterEqual(Bmin, L - 1e-9)                      # Satz 2.2
                    self.assertLessEqual(Bmin, sf.buffer_upper_bound(L, DH, T) + 1e-9)  # 2.3

    def test_buffer_bound_is_sharp_for_many_departures(self):
        N = 50
        extra = sf.buffer_requirement(N, D0, DH, KL, T) - G_EX / N
        self.assertGreater(extra, 0.99 * sf.season_buffer(DH, T))

    def test_buffer_requirement_when_gap_is_seasonal(self):
        # K_L = 3500: c = 0.5, the year-round-gap assumption does not hold.
        for N in [1, 4, 9]:
            Bmin = sf.buffer_requirement(N, D0, DH, 3500.0, T)
            self.assertGreaterEqual(Bmin, sf.period_gap(D0, DH, 3500.0, T) / N - 1e-9)

    def test_buffer_requirement_validation(self):
        with self.assertRaises(ValueError):
            sf.buffer_requirement(0, D0, DH, KL, T)
        with self.assertRaises(ValueError):
            sf.buffer_requirement(10, D0, DH, 6000.0, T)          # G = 0
        with self.assertRaises(ValueError):
            sf.buffer_requirement(10, D0, DH, KL, T, t0=-0.01)
        with self.assertRaises(ValueError):
            sf.buffer_requirement(10, D0, DH, KL, T, t0=0.1)

    def test_admissible_initial_stock(self):
        N = 12
        Bmin = sf.buffer_requirement(N, D0, DH, KL, T)
        interval = sf.admissible_initial_stock(N, Bmin + 10.0, D0, DH, KL, T)
        self.assertIsNotNone(interval)
        lo, hi = interval
        self.assertAlmostEqual(hi - lo, 10.0, places=6)
        for I0 in (lo, 0.5 * (lo + hi), hi):        # all admissible: 0 <= X <= B
            tr = sf.stock_at_calls(N, G_EX / N, D0, DH, KL, T, I0, n_periods=4)
            self.assertGreaterEqual(tr.before.min(), -1e-6)
            self.assertLessEqual(tr.after.max(), Bmin + 10.0 + 1e-6)
        self.assertIsNone(sf.admissible_initial_stock(N, Bmin - 10.0, D0, DH, KL, T))


class TestProfitability(unittest.TestCase):
    kappa = 0.04

    def test_storage_cost_and_margin(self):
        self.assertAlmostEqual(sf.storage_cost_coefficient(120.0, 1.0, 1500.0), 0.04)
        self.assertAlmostEqual(sf.unit_margin(100.0, 40.0, 1200.0, 0.04), 24.0)
        for args in [(0.0, 1.0, 1500.0), (120.0, 0.0, 1500.0), (120.0, 1.0, 0.0)]:
            with self.assertRaises(ValueError):
                sf.storage_cost_coefficient(*args)
        for args in [(0.0, 40.0, 1200.0, 0.04), (100.0, 40.0, 0.0, 0.04),
                     (100.0, 40.0, 1200.0, 0.0)]:
            with self.assertRaises(ValueError):
                sf.unit_margin(*args)

    def test_lemma_structure_of_margin(self):
        LE = sf.optimal_load(1200.0, self.kappa)
        self.assertAlmostEqual(LE, 173.20508075688772)
        mmax = sf.max_margin(40.0, 1200.0, self.kappa)
        self.assertAlmostEqual(mmax, 40 - 2 * math.sqrt(0.04 * 1200))
        self.assertAlmostEqual(sf.unit_margin(LE, 40.0, 1200.0, self.kappa), mmax)
        grid = np.linspace(1.0, 2000.0, 4000)
        vals = np.array([sf.unit_margin(x, 40.0, 1200.0, self.kappa) for x in grid])
        self.assertLessEqual(vals.max(), mmax + 1e-9)               # maximum
        self.assertTrue(np.all(np.diff(vals[grid < LE]) > 0))       # increasing
        self.assertTrue(np.all(np.diff(vals[grid > LE]) < 0))       # decreasing
        second = vals[2:-2] - 2 * vals[1:-3] + vals[:-4]
        self.assertTrue(np.all(second[:-1] < 0))                    # strictly concave
        with self.assertRaises(ValueError):
            sf.optimal_load(0.0, self.kappa)
        with self.assertRaises(ValueError):
            sf.optimal_load(1200.0, 0.0)
        with self.assertRaises(ValueError):
            sf.max_margin(40.0, 0.0, self.kappa)
        with self.assertRaises(ValueError):
            sf.max_margin(40.0, 1200.0, 0.0)

    def test_viable_interval(self):
        # Satz 3 (eps = 0): profit threshold 31.0
        lo0, hi0 = sf.viable_load_interval(40.0, 0.0, 1200.0, self.kappa)
        self.assertAlmostEqual(lo0, 31.0, places=1)
        self.assertAlmostEqual(lo0 * hi0, 1200.0 / self.kappa)
        self.assertAlmostEqual(lo0 + hi0, 40.0 / self.kappa)
        eps = 19.814775241282053
        lo, hi = sf.viable_load_interval(40.0, eps, 1200.0, self.kappa)
        self.assertAlmostEqual(lo, 68.84047360724233)
        self.assertAlmostEqual(hi, 435.7901453607063)
        for L in (lo, hi):
            self.assertAlmostEqual(sf.unit_margin(L, 40.0, 1200.0, self.kappa), eps)
        LE = sf.optimal_load(1200.0, self.kappa)
        self.assertLess(lo, LE)
        self.assertGreater(hi, LE)
        # boundary eps = m_max: single point L_E
        mmax = sf.max_margin(40.0, 1200.0, self.kappa)
        a, b = sf.viable_load_interval(40.0, mmax, 1200.0, self.kappa)
        self.assertAlmostEqual(a, LE, places=5)
        self.assertAlmostEqual(b, LE, places=5)
        self.assertIsNone(sf.viable_load_interval(40.0, mmax + 0.01, 1200.0, self.kappa))
        with self.assertRaises(ValueError):
            sf.viable_load_interval(40.0, -1.0, 1200.0, self.kappa)

    def test_viable_interval_monotone_in_eps(self):
        prev = None
        for eps in np.linspace(0.5, 25.5, 30):
            lo, hi = sf.viable_load_interval(40.0, float(eps), 1200.0, self.kappa)
            if prev is not None:
                self.assertGreater(lo, prev[0])
                self.assertLess(hi, prev[1])
            prev = (lo, hi)

    def test_degradation_and_safety_margin(self):
        self.assertAlmostEqual(sf.degradation_exponential(0.0, 2.0, 0.1), 1.0)
        self.assertAlmostEqual(sf.degradation_exponential(6.0, 2.0, 0.1),
                               2.0 - math.exp(-0.6))
        self.assertAlmostEqual(sf.degradation_exponential(1e6, 2.0, 0.1), 2.0)
        for args in [(-1.0, 2.0, 0.1), (1.0, 2.0, 0.0), (1.0, 0.5, 0.1)]:
            with self.assertRaises(ValueError):
                sf.degradation_exponential(*args)
        eps = sf.safety_margin(10.0, 1.4511883639059735, 0.05)
        self.assertAlmostEqual(eps, 19.814775241282053, places=9)
        self.assertAlmostEqual(sf.safety_margin(10.0, 1.0, 0.5), 0.0)
        self.assertGreater(sf.safety_margin(10.0, 2.0, 0.05), sf.safety_margin(10.0, 1.0, 0.05))
        for args in [(0.0, 1.0, 0.05), (10.0, 0.0, 0.05), (10.0, 1.0, 0.0), (10.0, 1.0, 0.6)]:
            with self.assertRaises(ValueError):
                sf.safety_margin(*args)

    def test_safety_margin_probability_guarantee(self):
        # P[m_tilde >= 0] >= 1 - p  iff  m >= eps   (Lemma 3)
        from scipy.stats import norm
        sigma_m, delta, p = 10.0, 1.4511883639059735, 0.05
        eps = sf.safety_margin(sigma_m, delta, p)
        sigma = sigma_m * math.sqrt(delta)
        self.assertAlmostEqual(norm.cdf(eps / sigma), 1 - p)
        self.assertGreaterEqual(norm.cdf((eps + 0.1) / sigma), 1 - p)
        self.assertLess(norm.cdf((eps - 0.1) / sigma), 1 - p)

    def test_critical_delta_and_horizon(self):
        mmax = sf.max_margin(40.0, 1200.0, 0.04)
        dcrit = sf.critical_delta(mmax, 10.0, 0.05)
        self.assertAlmostEqual(dcrit, 2.5262, places=3)
        for args in [(0.0, 10.0, 0.05), (mmax, 0.0, 0.05), (mmax, 10.0, 0.0),
                     (mmax, 10.0, 0.5)]:
            with self.assertRaises(ValueError):
                sf.critical_delta(*args)
        self.assertEqual(sf.critical_horizon(dcrit, 2.0, 0.1), math.inf)   # never critical
        self.assertEqual(sf.critical_horizon(2.0, 2.0, 0.1), math.inf)
        self.assertIsNone(sf.critical_horizon(0.9, 2.0, 0.1))
        tau = sf.critical_horizon(1.5, 2.0, 0.1)
        self.assertAlmostEqual(sf.degradation_exponential(tau, 2.0, 0.1), 1.5)
        self.assertEqual(sf.critical_horizon(1.0, 2.0, 0.1), 0.0)
        with self.assertRaises(ValueError):
            sf.critical_horizon(1.5, 2.0, 0.0)
        with self.assertRaises(ValueError):
            sf.critical_horizon(1.5, 1.0, 0.1)

    def test_critical_horizon_iff_viability(self):
        # Satz 4.4: V nonempty iff delta(tau) <= delta_crit.
        p, a0, Cf, kappa = 0.05, 40.0, 1200.0, 0.04
        mmax = sf.max_margin(a0, Cf, kappa)
        for sigma in (12.0, 14.0):
            dcrit = sf.critical_delta(mmax, sigma, p)
            self.assertTrue(1.0 <= dcrit < 3.0)
            tau_c = sf.critical_horizon(dcrit, 3.0, 0.1)
            for tau in (tau_c * 0.9, tau_c * 1.1):
                d = sf.degradation_exponential(tau, 3.0, 0.1)
                eps = sf.safety_margin(sigma, d, p)
                viable = sf.viable_load_interval(a0, eps, Cf, kappa) is not None
                self.assertEqual(viable, tau <= tau_c)

    def test_ship_size_theorem(self):
        cost = lambda K: sf.power_law_fixed_cost(K, 60.0, 0.8)
        L = 80.0
        margins = [sf.ship_unit_margin(L, K, 40.0, 0.04, cost) for K in (80.0, 120.0, 400.0)]
        self.assertTrue(margins[0] > margins[1] > margins[2])          # strictly decreasing
        self.assertAlmostEqual(margins[0], sf.unit_margin(L, 40.0, cost(80.0), 0.04))
        with self.assertRaises(ValueError):
            sf.ship_unit_margin(80.0, 50.0, 40.0, 0.04, cost)
        with self.assertRaises(ValueError):
            sf.ship_unit_margin(0.0, 50.0, 40.0, 0.04, cost)
        # economies of scale at full load
        self.assertGreater(cost(80.0) / 80.0, cost(400.0) / 400.0)
        for args in [(0.0, 60.0, 0.8), (80.0, 0.0, 0.8), (80.0, 60.0, 0.0)]:
            with self.assertRaises(ValueError):
                sf.power_law_fixed_cost(*args)


class TestMarket(unittest.TestCase):
    a0, Cf, kappa, gamma, eps = 40.0, 1200.0, 0.04, 0.02, 19.814775241282053

    def test_equilibrium_freight(self):
        self.assertEqual(sf.equilibrium_freight(10.0, self.eps, 0.02, 1500.0), 0.0)
        self.assertAlmostEqual(sf.equilibrium_freight(self.eps, self.eps, 0.02, 1500.0), 1500.0)
        self.assertAlmostEqual(sf.equilibrium_freight(24.0, 20.0, 0.02, 1500.0), 1700.0)
        with self.assertRaises(ValueError):
            sf.equilibrium_freight(24.0, 20.0, 0.0, 1500.0)
        with self.assertRaises(ValueError):
            sf.equilibrium_freight(24.0, 20.0, 0.02, 0.0)

    def test_equilibrium_oversupply(self):
        self.assertEqual(sf.equilibrium_oversupply(10.0, 20.0, 0.02), 0.0)
        self.assertAlmostEqual(sf.equilibrium_oversupply(24.0, 20.0, 0.02), 200.0)
        with self.assertRaises(ValueError):
            sf.equilibrium_oversupply(24.0, 20.0, 0.0)

    def test_hauptsatz(self):
        Lm, Lp = sf.viable_load_interval(self.a0, self.eps, self.Cf, self.kappa)
        LE = sf.optimal_load(self.Cf, self.kappa)
        over = lambda L: sf.equilibrium_oversupply(
            sf.unit_margin(L, self.a0, self.Cf, self.kappa), self.eps, self.gamma)
        # (1) zero exactly at the two ends, positive inside, zero (unserved) outside
        self.assertAlmostEqual(over(Lm), 0.0, places=9)
        self.assertAlmostEqual(over(Lp), 0.0, places=9)
        for L in np.linspace(Lm + 1, Lp - 1, 50):
            self.assertGreater(over(float(L)), 0.0)
        for L in (Lm * 0.9, Lp * 1.1):
            self.assertEqual(over(L), 0.0)
            m = sf.unit_margin(L, self.a0, self.Cf, self.kappa)
            self.assertEqual(sf.equilibrium_freight(m, self.eps, self.gamma, 1500.0), 0.0)
        # (2) monotone on both sides of L_E and maximal at L_E
        left = np.array([over(float(L)) for L in np.linspace(Lm, LE, 60)])
        right = np.array([over(float(L)) for L in np.linspace(LE, Lp, 60)])
        self.assertTrue(np.all(np.diff(left) > 0))
        self.assertTrue(np.all(np.diff(right) < 0))
        dmax = sf.max_oversupply(self.a0, self.Cf, self.kappa, self.eps, self.gamma)
        self.assertAlmostEqual(dmax, over(LE))
        self.assertAlmostEqual(dmax, (26.143593539448982 - self.eps) / self.gamma)
        # Q_eq = G exactly at the ends (knife edge)
        self.assertAlmostEqual(sf.equilibrium_freight(self.eps, self.eps, self.gamma, 1500.0), 1500.0)

    def test_hauptsatz_random_parameters(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            a0 = rng.uniform(20, 80)
            Cf = rng.uniform(100, 3000)
            kappa = rng.uniform(0.005, 0.1)
            eps = rng.uniform(0.0, 0.9) * sf.max_margin(a0, Cf, kappa)
            if sf.max_margin(a0, Cf, kappa) <= 0:
                continue
            Lm = sf.just_worth_load(a0, eps, Cf, kappa)
            Lp = sf.viable_load_interval(a0, eps, Cf, kappa)[1]
            grid = np.linspace(0.2 * Lm, 1.5 * Lp, 300)
            over = np.array([sf.equilibrium_oversupply(
                sf.unit_margin(float(L), a0, Cf, kappa), eps, 0.02) for L in grid])
            inside = (grid > Lm) & (grid < Lp)
            self.assertTrue(np.all(over[inside] > 0))
            self.assertTrue(np.all(over[~inside] < 1e-9))
            self.assertLessEqual(Lm, sf.optimal_load(Cf, kappa) + 1e-9)

    def test_just_worth_load(self):
        self.assertAlmostEqual(sf.just_worth_load(self.a0, self.eps, self.Cf, self.kappa),
                               68.84047360724233)
        self.assertIsNone(sf.just_worth_load(self.a0, 30.0, self.Cf, self.kappa))

    def test_tolerance_band(self):
        lower, upper = sf.tolerance_band(self.a0, self.eps, self.Cf, self.kappa, self.gamma, 0.0)
        self.assertAlmostEqual(lower[0], lower[1])          # degenerate: strictly no oversupply
        self.assertAlmostEqual(upper[0], upper[1])
        lower, upper = sf.tolerance_band(self.a0, self.eps, self.Cf, self.kappa, self.gamma, 100.0)
        over = lambda L: sf.equilibrium_oversupply(
            sf.unit_margin(L, self.a0, self.Cf, self.kappa), self.eps, self.gamma)
        self.assertAlmostEqual(over(lower[1]), 100.0, places=6)
        self.assertAlmostEqual(over(upper[0]), 100.0, places=6)
        for L in np.linspace(lower[0], lower[1], 10):
            self.assertLessEqual(over(float(L)), 100.0 + 1e-6)
        for L in np.linspace(upper[0], upper[1], 10):
            self.assertLessEqual(over(float(L)), 100.0 + 1e-6)
        mid = 0.5 * (lower[1] + upper[0])
        self.assertGreater(over(mid), 100.0)
        # tolerance so large that every viable load is compatible
        full, none = sf.tolerance_band(self.a0, self.eps, self.Cf, self.kappa, self.gamma, 1e6)
        self.assertIsNone(none)
        self.assertEqual(full, sf.viable_load_interval(self.a0, self.eps, self.Cf, self.kappa))
        # nothing viable
        self.assertIsNone(sf.tolerance_band(self.a0, 30.0, self.Cf, self.kappa, self.gamma, 1.0))
        with self.assertRaises(ValueError):
            sf.tolerance_band(self.a0, self.eps, self.Cf, self.kappa, self.gamma, -1.0)
        with self.assertRaises(ValueError):
            sf.tolerance_band(self.a0, self.eps, self.Cf, self.kappa, 0.0, 1.0)

    def test_integer_departures(self):
        res = sf.integer_departures(1500.0, self.a0, self.eps, self.Cf, self.kappa, self.gamma)
        self.assertEqual(res.N, 21)
        self.assertAlmostEqual(res.load, 1500.0 / 21)
        self.assertAlmostEqual(res.margin, 20.34285714285714)
        self.assertAlmostEqual(res.oversupply, 26.404095078754786, places=6)
        self.assertFalse(res.beyond_eoq)
        self.assertGreaterEqual(res.margin, self.eps)
        # N = 22 is not viable, N = 21 is the largest viable integer
        self.assertLess(sf.unit_margin(1500.0 / 22, self.a0, self.Cf, self.kappa), self.eps)
        # oversupply minimal among viable N with load <= L_E
        for N in range(8, 21):
            m = sf.unit_margin(1500.0 / N, self.a0, self.Cf, self.kappa)
            self.assertGreater(sf.equilibrium_oversupply(m, self.eps, self.gamma),
                               res.oversupply)
        # nothing viable / fewer than one departure / beyond EOQ
        self.assertIsNone(sf.integer_departures(1500.0, self.a0, 30.0, self.Cf, self.kappa, 0.02))
        self.assertIsNone(sf.integer_departures(50.0, self.a0, self.eps, self.Cf, self.kappa, 0.02))
        beyond = sf.integer_departures(200.0, 27.0, 12.5, 1200.0, 0.04, 0.02)
        self.assertEqual(beyond.N, 1)
        self.assertTrue(beyond.beyond_eoq)
        with self.assertRaises(ValueError):
            sf.integer_departures(0.0, self.a0, self.eps, self.Cf, self.kappa, 0.02)


class TestPriceDynamics(unittest.TestCase):
    alpha, eta, beta = 4.0, 6.0, 0.02

    def test_amplitude_and_phase(self):
        A = sf.price_amplitude(self.alpha, self.eta, 1.0)
        self.assertAlmostEqual(A, 6 / math.sqrt(16 + 4 * math.pi ** 2))
        self.assertAlmostEqual(A, 0.8055439082194726)
        self.assertAlmostEqual(sf.price_phase(self.alpha, 1.0), -math.atan(2 * math.pi / 4))
        for args in [(0.0, 6.0, 1.0), (4.0, 0.0, 1.0), (4.0, 6.0, 0.0)]:
            with self.assertRaises(ValueError):
                sf.price_amplitude(*args)
        with self.assertRaises(ValueError):
            sf.price_phase(0.0, 1.0)
        with self.assertRaises(ValueError):
            sf.price_phase(4.0, 0.0)

    def test_limit_cycle_from_ode(self):
        A = sf.price_amplitude(self.alpha, self.eta, 1.0)
        phi = sf.price_phase(self.alpha, 1.0)
        f = lambda t, x: [-self.alpha * x[0] + self.eta * math.sin(2 * math.pi * t)]
        sol = solve_ivp(f, (0, 6), [5.0], rtol=1e-11, atol=1e-12, dense_output=True)
        for t in (5.0, 5.3, 5.71, 6.0):
            self.assertAlmostEqual(sol.sol(t)[0], A * math.sin(2 * math.pi * t + phi), places=6)
        # difference decays as e^{-alpha t} (Lemma 4)
        d0 = sol.sol(0.0)[0] - A * math.sin(phi)
        d1 = sol.sol(1.0)[0] - A * math.sin(2 * math.pi + phi)
        self.assertAlmostEqual(d1 / d0, math.exp(-self.alpha), places=6)

    def test_disturbance_matches_impulse_simulation(self):
        for N, delta in [(4, 40.0), (10, 100.0), (25, 150.0)]:
            with self.subTest(N=N):
                o = delta / N
                Ts = 1.0 / N
                x = 0.0
                for _ in range(400):                       # jump then decay, until steady
                    x = x - self.beta * o
                    x_after = x
                    x = x * math.exp(-self.alpha * Ts)
                self.assertAlmostEqual(-x_after, sf.peak_disturbance(
                    self.beta, delta, self.alpha, 1.0, N), places=9)
                self.assertAlmostEqual(sf.disturbance_profile(
                    0.0, self.beta, o, self.alpha, Ts), x_after, places=9)
                s = 0.4 * Ts
                self.assertAlmostEqual(sf.disturbance_profile(s, self.beta, o, self.alpha, Ts),
                                       x_after * math.exp(-self.alpha * s), places=9)
                # time average over one inter-arrival interval
                ss = np.linspace(0, Ts, 20001, endpoint=False)
                vals = [sf.disturbance_profile(float(v), self.beta, o, self.alpha, Ts) for v in ss]
                self.assertAlmostEqual(float(np.mean(vals)),
                                       sf.mean_disturbance(self.beta, delta, self.alpha, 1.0),
                                       delta=abs(x_after) * 1e-3)

    def test_disturbance_validation(self):
        for args in [(0.0, 0.0, 1.0, 4.0, 0.1), (0.0, 0.02, 0.0, 4.0, 0.1),
                     (0.0, 0.02, 1.0, 0.0, 0.1), (0.0, 0.02, 1.0, 4.0, 0.0),
                     (-0.01, 0.02, 1.0, 4.0, 0.1), (0.1, 0.02, 1.0, 4.0, 0.1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sf.disturbance_profile(args[0], *args[1:])
        for args in [(0.0, 10.0, 4.0, 1.0, 5), (0.02, 0.0, 4.0, 1.0, 5),
                     (0.02, 10.0, 0.0, 1.0, 5), (0.02, 10.0, 4.0, 0.0, 5),
                     (0.02, 10.0, 4.0, 1.0, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sf.peak_disturbance(*args)
        for args in [(0.0, 10.0, 4.0, 1.0), (0.02, 0.0, 4.0, 1.0),
                     (0.02, 10.0, 0.0, 1.0), (0.02, 10.0, 4.0, 0.0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sf.mean_disturbance(*args)

    def test_tolerance_equivalence_and_monotonicity(self):
        A = sf.price_amplitude(self.alpha, self.eta, 1.0)
        # peak <= kappa_p A  <=>  Delta <= Delta_tol(N)
        for N in (3, 8, 21):
            tol = sf.oversupply_tolerance(N, 1.0, A, self.beta, self.alpha, 1.0)
            self.assertAlmostEqual(sf.peak_disturbance(self.beta, tol, self.alpha, 1.0, N), A)
            self.assertLess(sf.peak_disturbance(self.beta, 0.99 * tol, self.alpha, 1.0, N), A)
            self.assertGreater(sf.peak_disturbance(self.beta, 1.01 * tol, self.alpha, 1.0, N), A)
        Ns = np.arange(1, 400)
        tols = np.array([sf.oversupply_tolerance(float(n), 1.0, A, self.beta, self.alpha, 1.0)
                         for n in Ns])
        self.assertTrue(np.all(np.diff(tols) > 0))
        limit = A * self.alpha * 1.0 / self.beta
        self.assertTrue(np.all(tols < limit))
        self.assertAlmostEqual(tols[-1], limit, delta=0.02 * limit)
        self.assertAlmostEqual(sf.oversupply_tolerance(21.789507268035315, 1.0, A, 0.02, 4.0, 1.0),
                               147.2, delta=0.1)

    def test_tolerance_validation(self):
        for args in [(0, 1.0, 0.8, 0.02, 4.0, 1.0), (5, 0.0, 0.8, 0.02, 4.0, 1.0),
                     (5, 1.0, 0.0, 0.02, 4.0, 1.0), (5, 1.0, 0.8, 0.0, 4.0, 1.0),
                     (5, 1.0, 0.8, 0.02, 0.0, 1.0), (5, 1.0, 0.8, 0.02, 4.0, 0.0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sf.oversupply_tolerance(*args)


class TestCells(unittest.TestCase):
    def test_shares_and_limits(self):
        gaps, buffers = (900.0, 600.0), (300.0, 200.0)
        np.testing.assert_allclose(sf.cell_shares(gaps), [0.6, 0.4])
        self.assertAlmostEqual(sf.cell_load_limit(gaps, buffers), 500.0)
        self.assertAlmostEqual(sf.pool_load_limit(gaps, buffers), 500.0)   # equal rho: equality
        gaps, buffers = (900.0, 600.0), (60.0, 400.0)
        self.assertAlmostEqual(sf.cell_load_limit(gaps, buffers), 100.0)
        self.assertAlmostEqual(sf.pool_load_limit(gaps, buffers), 460.0)

    def test_pool_dominates_cells_randomized(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            n = int(rng.integers(1, 6))
            gaps = tuple(rng.uniform(10, 1000, n))
            buffers = tuple(rng.uniform(10, 500, n))
            self.assertLessEqual(sf.cell_load_limit(gaps, buffers),
                                 sf.pool_load_limit(gaps, buffers) + 1e-9)

    def test_cells_validation(self):
        with self.assertRaises(ValueError):
            sf.cell_shares(())
        with self.assertRaises(ValueError):
            sf.cell_shares((1.0, 0.0))
        for gaps, buffers in [((), ()), ((1.0, 2.0), (1.0,)), ((0.0,), (1.0,)),
                              ((1.0,), (0.0,))]:
            with self.subTest(gaps=gaps), self.assertRaises(ValueError):
                sf.cell_load_limit(gaps, buffers)
            with self.subTest(gaps=gaps), self.assertRaises(ValueError):
                sf.pool_load_limit(gaps, buffers)


class TestParamsAndAnalysis(unittest.TestCase):
    P = ShipMarketParams()

    def test_defaults_reproduce_chapter_example(self):
        P = self.P
        self.assertAlmostEqual(P.G, 1500.0)
        self.assertAlmostEqual(P.kappa, 0.04)
        self.assertAlmostEqual(P.L_E, 173.2, places=1)
        self.assertAlmostEqual(P.m_max, 26.14, places=2)
        self.assertAlmostEqual(P.eps, 19.81, places=2)
        self.assertAlmostEqual(P.A, 0.806, places=3)

    def test_param_validation(self):
        bad = [dict(d_hat=0.0), dict(T=0.0), dict(d0=900.0), dict(K_L=-1.0),
               dict(tau=-1.0), dict(delta_tol=-1.0), dict(delta_max=0.5),
               dict(p=0.0), dict(p=0.6), dict(K_L=6000.0),
               dict(cell_gaps=(1500.0,)), dict(cell_buffers=(500.0,)),
               dict(cell_gaps=(1000.0, 400.0), cell_buffers=(300.0, 200.0)),
               dict(cell_gaps=(1500.0, 0.0), cell_buffers=(300.0, 200.0))]
        bad += [{name: 0.0} for name in ("B", "a0", "Cf", "h", "gamma", "sigma_m",
                                         "lam", "alpha", "eta", "beta", "kappa_p")]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(self.P, **change)
        ok = dataclasses.replace(self.P, cell_gaps=(900.0, 600.0), cell_buffers=(300.0, 200.0))
        self.assertEqual(ok.cell_gaps, (900.0, 600.0))
        self.assertTrue(dataclasses.is_dataclass(ok))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ok.B = 1.0

    def test_price_margin_and_compatibility(self):
        Lm = sf.just_worth_load(self.P.a0, self.P.eps, self.P.Cf, self.P.kappa)
        self.assertAlmostEqual(sf.price_margin(Lm, self.P), 147.2, delta=0.1)   # Delta_eq = 0
        self.assertTrue(sf.is_price_compatible(Lm, self.P))
        self.assertAlmostEqual(sf.price_margin(100.0, self.P), 141.4 - 209.3, delta=0.2)
        self.assertFalse(sf.is_price_compatible(100.0, self.P))

    def test_max_price_compatible_load_matches_chapter(self):
        self.assertAlmostEqual(sf.max_price_compatible_load(self.P), 86.7034, places=3)
        Lmax = sf.max_price_compatible_load(self.P)
        self.assertAlmostEqual(sf.price_margin(Lmax, self.P), 0.0, places=6)
        # the upper crossing quoted in the chapter (373.2 kt)
        upper = brentq(lambda L: sf.price_margin(L, self.P), 300.0, 435.0)
        self.assertAlmostEqual(upper, 373.23, places=1)
        # uniqueness on the lower branch: price_margin strictly decreasing on [L-, L_E]
        Lm = sf.just_worth_load(self.P.a0, self.P.eps, self.P.Cf, self.P.kappa)
        vals = [sf.price_margin(float(L), self.P) for L in np.linspace(Lm, self.P.L_E, 200)]
        self.assertTrue(np.all(np.diff(vals) < 0))
        # very weak price reaction: whole lower branch compatible, returns L_E
        lax = dataclasses.replace(self.P, beta=1e-6)
        self.assertAlmostEqual(sf.max_price_compatible_load(lax), lax.L_E)
        # nothing viable
        hopeless = dataclasses.replace(self.P, sigma_m=100.0)
        self.assertIsNone(sf.max_price_compatible_load(hopeless))

    def test_large_ships_are_more_fragile(self):
        # Korollar 6: tolerance Delta_tol(G/L) strictly decreasing in L
        A = self.P.A
        tols = [sf.oversupply_tolerance(1500.0 / L, 1.0, A, 0.02, 4.0, 1.0)
                for L in np.linspace(20, 500, 100)]
        self.assertTrue(np.all(np.diff(tols) < 0))

    def test_tolerance_upper_bound(self):
        self.assertAlmostEqual(sf.tolerance_upper_bound(self.P), 68.84047360724233)
        wide = dataclasses.replace(self.P, delta_tol=100.0)
        self.assertGreater(sf.tolerance_upper_bound(wide), 68.84)
        huge = dataclasses.replace(self.P, delta_tol=1000.0)          # eps' > m_max
        self.assertAlmostEqual(sf.tolerance_upper_bound(huge), 435.7901453607063)
        self.assertIsNone(sf.tolerance_upper_bound(dataclasses.replace(self.P, sigma_m=100.0)))

    def test_admissible_interval_cases(self):
        lo, hi = sf.admissible_load_interval(self.P)                   # default: knife edge
        self.assertAlmostEqual(lo, 68.84047360724233)
        self.assertAlmostEqual(hi, lo)
        tol = dataclasses.replace(self.P, delta_tol=100.0)             # tolerance binds
        lo, hi = sf.admissible_load_interval(tol)
        self.assertAlmostEqual(lo, 68.84047360724233)
        self.assertAlmostEqual(hi, sf.tolerance_upper_bound(tol))
        self.assertLess(hi, 181.69)
        buf = dataclasses.replace(self.P, delta_tol=1000.0)            # buffer binds
        self.assertAlmostEqual(sf.admissible_load_interval(buf)[1], 181.69011381620930)
        cells = dataclasses.replace(self.P, delta_tol=1000.0,
                                    cell_gaps=(900.0, 600.0), cell_buffers=(60.0, 200.0))
        self.assertAlmostEqual(sf.admissible_load_interval(cells)[1], 100.0)   # cell binds
        weak = dataclasses.replace(self.P, delta_tol=1000.0,
                                   cell_gaps=(900.0, 600.0), cell_buffers=(30.0, 200.0))
        self.assertIsNone(sf.admissible_load_interval(weak))           # cell limit < L-
        self.assertIsNone(sf.admissible_load_interval(
            dataclasses.replace(self.P, sigma_m=100.0)))               # not viable
        small_B = dataclasses.replace(self.P, B=350.0)                 # L_B = 31.7 < L-
        self.assertIsNone(sf.admissible_load_interval(small_B))

    def test_analyze(self):
        a = sf.analyze(self.P)
        self.assertAlmostEqual(a.G, 1500.0)
        self.assertAlmostEqual(a.kappa, 0.04)
        self.assertAlmostEqual(a.L_E, 173.2, places=1)
        self.assertAlmostEqual(a.m_max, 26.14, places=2)
        self.assertAlmostEqual(a.eps, 19.81, places=2)
        self.assertAlmostEqual(a.viable[0], 68.84047360724233)
        self.assertAlmostEqual(a.viable[1], 435.7901453607063)
        self.assertAlmostEqual(a.just_worth_load, 68.84047360724233)
        self.assertAlmostEqual(a.departures_at_just_worth, 21.79, places=2)
        self.assertAlmostEqual(a.buffer_limit, 181.69, places=2)
        self.assertAlmostEqual(a.tolerance_upper, 68.84047360724233)
        self.assertAlmostEqual(a.price_compatible_max, 86.7034, places=3)
        self.assertAlmostEqual(a.admissible[0], 68.84047360724233)

    def test_analyze_when_nothing_is_viable(self):
        a = sf.analyze(dataclasses.replace(self.P, sigma_m=100.0))
        self.assertIsNone(a.viable)
        self.assertIsNone(a.just_worth_load)
        self.assertIsNone(a.departures_at_just_worth)
        self.assertIsNone(a.tolerance_upper)
        self.assertIsNone(a.price_compatible_max)
        self.assertIsNone(a.admissible)


if __name__ == "__main__":
    unittest.main()
