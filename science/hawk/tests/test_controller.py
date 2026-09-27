"""Tests for hawk.controller — full branch and statement coverage of Algorithm 1."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from hawk.controller import (
    BGSGController,
    BioFalconTracker,
    Phase,
    proportional_navigation,
)


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class TestPhase:
    def test_integer_values(self):
        assert Phase.P1 == 1
        assert Phase.P2 == 2
        assert Phase.P3 == 3
        assert Phase.P4 == 4
        assert Phase.P5 == 5

    def test_is_int_enum(self):
        assert isinstance(Phase.P3, int)


# ---------------------------------------------------------------------------
# proportional_navigation
# ---------------------------------------------------------------------------

class TestProportionalNavigation:
    def test_zero_los_returns_zero(self):
        """When p_target == p_self the command must be zero (no LOS)."""
        pos = np.array([5.0, 3.0, 1.0])
        result = proportional_navigation(pos, pos, np.array([1.0, 0.0, 0.0]))
        npt.assert_array_equal(result, np.zeros(3))

    def test_zero_closing_speed_returns_zero(self):
        """If v_rel is perpendicular to LOS there is no closing → zero command."""
        # LOS along x, v_rel along y (purely lateral — v_closing = 0)
        result = proportional_navigation(
            p_target=np.array([10.0, 0.0, 0.0]),
            p_self=np.zeros(3),
            v_rel=np.array([0.0, 5.0, 0.0]),
            nav_gain=4.0,
        )
        npt.assert_array_almost_equal(result, np.zeros(3), decimal=10)

    def test_known_geometry(self):
        """Verify explicit calculation for a known intercept geometry.

        Setup:
          p_target = [10, 0, 0],  p_self = [0, 0, 0]
          v_rel    = [-5, 5, 0]   (closing at 5 m/s + 5 m/s lateral)
          nav_gain = 4.0

        Expected:
          los = [10,0,0], los_hat = [1,0,0]
          v_closing = -dot([-5,5,0],[1,0,0]) = +5
          omega_los = cross([10,0,0],[-5,5,0])/100 = [0,0,50]/100 = [0,0,0.5]
          n_c = 4·5·cross([0,0,0.5],[1,0,0]) = 20·[0,0.5,0] = [0,10,0]
        """
        result = proportional_navigation(
            p_target=np.array([10.0, 0.0, 0.0]),
            p_self=np.zeros(3),
            v_rel=np.array([-5.0, 5.0, 0.0]),
            nav_gain=4.0,
        )
        npt.assert_array_almost_equal(result, np.array([0.0, 10.0, 0.0]))

    def test_nav_gain_scales_output(self):
        """Doubling nav_gain must double the command magnitude."""
        kwargs = dict(
            p_target=np.array([10.0, 0.0, 0.0]),
            p_self=np.zeros(3),
            v_rel=np.array([-5.0, 5.0, 0.0]),
        )
        r4 = proportional_navigation(**kwargs, nav_gain=4.0)
        r8 = proportional_navigation(**kwargs, nav_gain=8.0)
        npt.assert_array_almost_equal(r8, 2.0 * r4)

    def test_output_shape(self):
        result = proportional_navigation(
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            np.array([0.5, -0.5, 0.0]),
        )
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# BGSGController
# ---------------------------------------------------------------------------

class TestBGSGController:
    def _make(self, **kw):
        return BGSGController(**kw)

    def test_sliding_surface_formula(self):
        """s = ė + Λ·e."""
        ctrl = self._make(lambda_=2.0)
        e = np.array([1.0, 0.0, -1.0])
        e_dot = np.array([0.5, 0.5, 0.5])
        s = ctrl.sliding_surface(e, e_dot)
        expected = e_dot + 2.0 * e
        npt.assert_array_almost_equal(s, expected)

    def test_compute_output_shape(self):
        ctrl = self._make()
        u = ctrl.compute(np.zeros(3), np.zeros(3), dt=0.01)
        assert u.shape == (3,)

    def test_compute_zero_error_zero_output_except_switching(self):
        """At perfect tracking (e=0, ė=0) only the switching term is non-zero
        if integrals are also zero (first call)."""
        ctrl = self._make(k_sw=1.0, k_p=4.0, k_d=2.0, k_i=0.5, lambda_=2.0)
        u = ctrl.compute(np.zeros(3), np.zeros(3), dt=0.01)
        # sign(0) = 0, so u_sw = 0; integral is zero after first step → u = 0
        npt.assert_array_almost_equal(u, np.zeros(3))

    def test_compute_proportional_term(self):
        """With only error (ė=0, no integral from first call), check output."""
        ctrl = BGSGController(k_p=2.0, k_d=0.0, k_i=0.0, k_sw=0.0, lambda_=0.0)
        e = np.array([1.0, 0.0, 0.0])
        u = ctrl.compute(e, np.zeros(3), dt=0.01)
        # u = -k_p·e = -2·[1,0,0]
        npt.assert_array_almost_equal(u, np.array([-2.0, 0.0, 0.0]))

    def test_compute_derivative_term(self):
        """Pure derivative action: u = -K_d·ė."""
        ctrl = BGSGController(k_p=0.0, k_d=3.0, k_i=0.0, k_sw=0.0, lambda_=0.0)
        e_dot = np.array([0.0, 2.0, 0.0])
        u = ctrl.compute(np.zeros(3), e_dot, dt=0.01)
        npt.assert_array_almost_equal(u, np.array([0.0, -6.0, 0.0]))

    def test_compute_integral_accumulates(self):
        """Repeated calls accumulate the integral term."""
        ctrl = BGSGController(k_p=0.0, k_d=0.0, k_i=1.0, k_sw=0.0, lambda_=0.0)
        e = np.array([1.0, 0.0, 0.0])
        # After 10 steps of dt=0.1 → integral = 1.0 → u_i = -1.0
        for _ in range(10):
            u = ctrl.compute(e, np.zeros(3), dt=0.1)
        npt.assert_array_almost_equal(u, np.array([-1.0, 0.0, 0.0]))

    def test_compute_with_feedforward(self):
        """Feedforward is added to the control output."""
        ctrl = BGSGController(k_p=0.0, k_d=0.0, k_i=0.0, k_sw=0.0, lambda_=0.0)
        ff = np.array([3.0, 1.0, -1.0])
        u = ctrl.compute(np.zeros(3), np.zeros(3), dt=0.01, feedforward=ff)
        npt.assert_array_almost_equal(u, ff)

    def test_compute_without_feedforward_is_none_default(self):
        """No feedforward (None) must not raise and returns unmodified output."""
        ctrl = BGSGController(k_p=1.0, k_d=0.0, k_i=0.0, k_sw=0.0, lambda_=0.0)
        e = np.array([2.0, 0.0, 0.0])
        u = ctrl.compute(e, np.zeros(3), dt=0.01, feedforward=None)
        npt.assert_array_almost_equal(u, np.array([-2.0, 0.0, 0.0]))

    def test_reset_clears_integral(self):
        """After reset the integrator is zeroed."""
        ctrl = BGSGController(k_p=0.0, k_d=0.0, k_i=1.0, k_sw=0.0, lambda_=0.0)
        for _ in range(5):
            ctrl.compute(np.ones(3), np.zeros(3), dt=0.1)
        ctrl.reset()
        u = ctrl.compute(np.zeros(3), np.zeros(3), dt=0.01)
        npt.assert_array_almost_equal(u, np.zeros(3))

    def test_two_instances_independent(self):
        """Different instances must not share state."""
        c1 = BGSGController(k_i=1.0, k_p=0.0, k_d=0.0, k_sw=0.0, lambda_=0.0)
        c2 = BGSGController(k_i=1.0, k_p=0.0, k_d=0.0, k_sw=0.0, lambda_=0.0)
        for _ in range(5):
            c1.compute(np.ones(3), np.zeros(3), dt=0.1)
        # c2 has never been called — its integral should still be zero
        u2 = c2.compute(np.zeros(3), np.zeros(3), dt=0.01)
        npt.assert_array_almost_equal(u2, np.zeros(3))

    def test_switching_term_sign(self):
        """u_sw = -K_sw·sign(s); positive error produces negative switch contribution."""
        ctrl = BGSGController(k_p=0.0, k_d=0.0, k_i=0.0, k_sw=2.0, lambda_=0.0)
        # e=0, ė=[1,0,0] → s=[1,0,0] → u_sw = -2·[1,0,0]
        u = ctrl.compute(np.zeros(3), np.array([1.0, 0.0, 0.0]), dt=0.01)
        assert u[0] == pytest.approx(-2.0)

    def test_compute_reinitialises_integral_on_dim_change(self):
        """Integral is re-initialised when error dimension changes (lazy init branch)."""
        ctrl = BGSGController(k_p=0.0, k_d=0.0, k_i=1.0, k_sw=0.0, lambda_=0.0)
        # First call with 3-D error seeds the integrator shape to (3,)
        ctrl.compute(np.ones(3), np.zeros(3), dt=0.1)
        # Second call with 2-D error must trigger the re-initialisation branch
        u = ctrl.compute(np.array([1.0, 0.0]), np.zeros(2), dt=0.1)
        # After re-init the integral was zeroed then accumulated one step: ξ = 0.1·[1,0]
        # u = -k_i·ξ = -1·[0.1, 0.0]
        npt.assert_array_almost_equal(u, np.array([-0.1, 0.0]))


# ---------------------------------------------------------------------------
# BioFalconTracker — sub-step methods
# ---------------------------------------------------------------------------

class TestBioFalconTrackerSubSteps:
    def setup_method(self):
        self.tracker = BioFalconTracker()

    def test_compute_relative_vector(self):
        """Δp = p_target − p_self."""
        p_t = np.array([10.0, 5.0, -2.0])
        p_s = np.array([1.0, 1.0, 1.0])
        delta = self.tracker.compute_relative_vector(p_t, p_s)
        npt.assert_array_almost_equal(delta, np.array([9.0, 4.0, -3.0]))

    def test_compute_relative_vector_at_target(self):
        pos = np.array([3.0, 4.0, 5.0])
        delta = self.tracker.compute_relative_vector(pos, pos)
        npt.assert_array_equal(delta, np.zeros(3))

    def test_log_spiral_param_normal(self):
        """κ = arctan(|Δp| / (v·t_go))."""
        delta_p = np.array([3.0, 4.0, 0.0])   # |Δp| = 5
        v_mag = 10.0
        t_go = 0.5
        kappa = self.tracker.compute_log_spiral_param(delta_p, v_mag, t_go)
        expected = math.atan(5.0 / 5.0)       # arctan(1) = π/4
        assert kappa == pytest.approx(expected)

    def test_log_spiral_param_zero_denominator(self):
        """κ = π/2 when v·t_go → 0 (undefined → arctan(∞))."""
        delta_p = np.array([1.0, 0.0, 0.0])
        kappa = self.tracker.compute_log_spiral_param(delta_p, v_mag=0.0, t_go=0.0)
        assert kappa == pytest.approx(math.pi / 2.0)

    def test_log_spiral_param_zero_tgo(self):
        """t_go=0 (denominator zero) → π/2."""
        kappa = self.tracker.compute_log_spiral_param(
            np.array([5.0, 0.0, 0.0]), v_mag=10.0, t_go=0.0
        )
        assert kappa == pytest.approx(math.pi / 2.0)

    def test_feedforward_zero_velocity(self):
        """Zero velocity → zero feedforward vector."""
        ff = self.tracker._feedforward(0.032, np.zeros(3), h=0.0, beta_fold=0.5)
        npt.assert_array_equal(ff, np.zeros(3))

    def test_feedforward_positive_magnitude(self):
        """Non-zero velocity produces a non-zero feedforward along velocity."""
        v = np.array([0.0, 0.0, -30.0])
        ff = self.tracker._feedforward(0.032, v, h=0.0, beta_fold=0.0)
        # Must be along v̂ = [0,0,-1]
        assert ff[2] < 0.0
        assert ff[0] == pytest.approx(0.0)
        assert ff[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BioFalconTracker.step — Algorithm 1 branch coverage
# ---------------------------------------------------------------------------

class TestBioFalconTrackerStep:
    """Each branch of Algorithm 1 (P3, P4 partial, P4 clamped, P5, default) is
    exercised to ensure full statement and branch coverage."""

    _p_target = np.array([100.0, 0.0, 0.0])
    _p_self = np.zeros(3)
    _v = np.array([0.0, 0.0, -20.0])       # diving downward
    _v_target = np.zeros(3)
    _dt = 0.01

    def _step(self, phase, t=0.0, t_po=0.0, tau_po=1.2, **kw):
        tracker = BioFalconTracker(tau_pullout=tau_po)
        return tracker.step(
            self._p_target,
            self._p_self,
            self._v,
            self._v_target,
            phase=phase,
            t=t,
            t_pullout_start=t_po,
            dt=self._dt,
            **kw,
        )

    # --- Phase P3 -----------------------------------------------------------

    def test_p3_beta_fold_is_zero(self):
        """In Phase P3 wings must be fully folded (β_fold = 0)."""
        _, beta = self._step(Phase.P3)
        assert beta == 0.0

    def test_p3_returns_ndarray(self):
        u, _ = self._step(Phase.P3)
        assert isinstance(u, np.ndarray)
        assert u.shape == (3,)

    def test_p3_uses_bgsg_not_pn(self):
        """P3 uses BGSG, not PN — output changes when BGSG gains change."""
        t = BioFalconTracker(
            controller=BGSGController(k_p=10.0, k_d=0.0, k_i=0.0, k_sw=0.0, lambda_=0.0)
        )
        u, _ = t.step(
            self._p_target, self._p_self, self._v, self._v_target,
            phase=Phase.P3, t=0.0, t_pullout_start=0.0, dt=self._dt,
        )
        # Large k_p should produce large -k_p·e along the LOS direction
        assert np.linalg.norm(u) > 0.0

    # --- Phase P4 -----------------------------------------------------------

    def test_p4_beta_partial(self):
        """Part-way through pullout: β = (t − t_po) / τ_po < 1."""
        t_po, tau_po, t = 0.0, 2.0, 1.0      # 1 s into a 2 s pullout → β = 0.5
        _, beta = self._step(Phase.P4, t=t, t_po=t_po, tau_po=tau_po)
        assert beta == pytest.approx(0.5)

    def test_p4_beta_clamped_to_one(self):
        """After τ_po pullout is complete → β is clamped to 1."""
        t_po, tau_po, t = 0.0, 1.2, 10.0     # well past pullout duration
        _, beta = self._step(Phase.P4, t=t, t_po=t_po, tau_po=tau_po)
        assert beta == 1.0

    def test_p4_beta_exactly_at_boundary(self):
        """At t = t_po + τ_po the clamp is exactly 1."""
        t_po, tau_po = 5.0, 1.2
        t = t_po + tau_po
        _, beta = self._step(Phase.P4, t=t, t_po=t_po, tau_po=tau_po)
        assert beta == pytest.approx(1.0)

    def test_p4_returns_ndarray(self):
        u, _ = self._step(Phase.P4, t=0.5, t_po=0.0, tau_po=1.2)
        assert isinstance(u, np.ndarray)
        assert u.shape == (3,)

    # --- Phase P5 -----------------------------------------------------------

    def test_p5_beta_fold_is_one(self):
        """In Phase P5 wings are fully deployed."""
        _, beta = self._step(Phase.P5)
        assert beta == 1.0

    def test_p5_output_equals_pn(self):
        """P5 must return exactly the proportional_navigation command."""
        p_t = np.array([10.0, 0.0, 0.0])
        p_s = np.zeros(3)
        v = np.array([5.0, 0.0, 0.0])
        v_target = np.array([0.0, 5.0, 0.0])
        nav_gain = 4.0

        tracker = BioFalconTracker(nav_gain=nav_gain)
        u_star, _ = tracker.step(
            p_t, p_s, v, v_target,
            phase=Phase.P5, t=0.0, t_pullout_start=0.0, dt=self._dt,
        )
        expected = proportional_navigation(p_t, p_s, v_target - v, nav_gain)
        npt.assert_array_almost_equal(u_star, expected)

    def test_p5_on_target_zero_command(self):
        """Drone exactly at target in P5 → zero PN command."""
        pos = np.array([5.0, 0.0, 0.0])
        tracker = BioFalconTracker()
        u, _ = tracker.step(
            pos, pos, np.zeros(3), np.zeros(3),
            phase=Phase.P5, t=0.0, t_pullout_start=0.0, dt=self._dt,
        )
        npt.assert_array_equal(u, np.zeros(3))

    # --- Default (P1 / P2) --------------------------------------------------

    def test_p1_zero_command(self):
        """Phase P1 produces zero control and β=1."""
        u, beta = self._step(Phase.P1)
        npt.assert_array_equal(u, np.zeros(3))
        assert beta == 1.0

    def test_p2_zero_command(self):
        """Phase P2 produces zero control and β=1."""
        u, beta = self._step(Phase.P2)
        npt.assert_array_equal(u, np.zeros(3))
        assert beta == 1.0

    # --- Zero-velocity edge case for feedforward ----------------------------

    def test_p3_zero_velocity_feedforward_is_zero(self):
        """Zero own velocity in P3 → feedforward is zero; BGSG still active."""
        tracker = BioFalconTracker()
        u, beta = tracker.step(
            np.array([10.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),           # v = 0
            np.zeros(3),
            phase=Phase.P3, t=0.0, t_pullout_start=0.0, dt=self._dt,
        )
        assert beta == 0.0
        assert u.shape == (3,)     # no crash even with zero velocity

    # --- Return type sanity -------------------------------------------------

    def test_step_returns_tuple_of_array_and_float(self):
        u, beta = self._step(Phase.P3)
        assert isinstance(u, np.ndarray)
        assert isinstance(beta, float)


# ---------------------------------------------------------------------------
# Integration: public API import
# ---------------------------------------------------------------------------

def test_public_api_importable():
    """All public symbols documented in __init__ must be importable."""
    import hawk  # noqa: F401
    from hawk import (  # noqa: F401
        BGSGController,
        BioFalconTracker,
        Phase,
        air_density,
        drag_coefficient,
        drag_force,
        lift_coefficient,
        lift_force,
        proportional_navigation,
        reference_area,
        terminal_velocity,
    )
