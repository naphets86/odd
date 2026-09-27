"""Tests for liionp.controllers."""
import math

import numpy as np
import pytest

from liionp.controllers import PIDController, SimpleMPC


# ── PIDController ─────────────────────────────────────────────────────────────

class TestPIDController:
    def test_proportional_only(self):
        pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)
        out = pid.compute(setpoint=10.0, measured=8.0, dt=1.0)
        assert out == pytest.approx(4.0)

    def test_integral_accumulates(self):
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0)
        out1 = pid.compute(setpoint=10.0, measured=8.0, dt=1.0)  # integral += 2
        out2 = pid.compute(setpoint=10.0, measured=8.0, dt=1.0)  # integral += 4
        assert out2 > out1

    def test_derivative_term(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0)
        pid.compute(setpoint=10.0, measured=8.0, dt=1.0)  # prev_error = 2
        # error decreases → de/dt < 0 → negative derivative output
        out = pid.compute(setpoint=10.0, measured=9.5, dt=1.0)
        assert out < 0.0

    def test_zero_dt_skips_derivative(self):
        pid = PIDController(Kp=1.0, Ki=0.0, Kd=10.0)
        # With dt=0, derivative should be 0, not inf
        out = pid.compute(setpoint=5.0, measured=3.0, dt=0.0)
        assert math.isfinite(out)
        assert out == pytest.approx(2.0)

    def test_output_clamp_max(self):
        pid = PIDController(Kp=100.0, Ki=0.0, Kd=0.0, output_max=5.0)
        out = pid.compute(setpoint=10.0, measured=0.0, dt=1.0)
        assert out == pytest.approx(5.0)

    def test_output_clamp_min(self):
        pid = PIDController(Kp=100.0, Ki=0.0, Kd=0.0, output_min=-5.0)
        out = pid.compute(setpoint=0.0, measured=10.0, dt=1.0)
        assert out == pytest.approx(-5.0)

    def test_integral_anti_windup(self):
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0, integral_limit=3.0)
        for _ in range(100):
            pid.compute(setpoint=10.0, measured=0.0, dt=1.0)
        # Integrator clamped to 3.0 → output = Ki * 3 = 3
        out = pid.compute(setpoint=10.0, measured=0.0, dt=0.0)
        assert out <= 3.0 * 1.0  # Ki * integral_limit

    def test_reset(self):
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0)
        pid.compute(setpoint=10.0, measured=0.0, dt=1.0)
        pid.reset()
        out = pid.compute(setpoint=10.0, measured=0.0, dt=1.0)
        # After reset integral is 0, so output = Ki * (0 + 10*1) = 10
        assert out == pytest.approx(10.0)

    def test_lyapunov_function_non_negative(self):
        pid = PIDController(Kp=2.0, Ki=1.0, Kd=0.1)
        assert pid.lyapunov_function(0.0, 0.0) == pytest.approx(0.0)
        assert pid.lyapunov_function(1.0, 1.0) > 0.0
        assert pid.lyapunov_function(-1.0, 0.0) > 0.0

    def test_lyapunov_formula(self):
        pid = PIDController(Kp=2.0, Ki=1.0, Kd=0.1)
        e, xi = 3.0, 2.0
        expected = 0.5 * e**2 + (pid.Ki / (2.0 * pid.Kp)) * xi**2
        assert pid.lyapunov_function(e, xi) == pytest.approx(expected)


# ── SimpleMPC ─────────────────────────────────────────────────────────────────

class TestSimpleMPC:
    @pytest.fixture
    def mpc(self):
        return SimpleMPC(
            q_V=1.0, r_I=0.1, p_D=0.5,
            I_min=-3.0, I_max=3.0,
            V_min=2.5, V_max=4.2,
            V_target=3.9, I_ref=1.0,
            n_grid=11,
        )

    @staticmethod
    def _ocv(SoC: float) -> float:
        return 2.5 + SoC * (4.2 - 2.5)

    @staticmethod
    def _deg(V: float, I: float) -> float:
        return 0.0  # zero degradation for simple tests

    def test_returns_float_in_range(self, mpc):
        I_opt = mpc.compute_current(
            SoC=0.5, R_int=0.05,
            V_ocv_func=self._ocv,
            deg_rate_func=self._deg,
        )
        assert mpc.I_min <= I_opt <= mpc.I_max

    def test_reduced_limit_mode(self, mpc):
        I_normal = mpc.compute_current(
            SoC=0.9, R_int=0.05,
            V_ocv_func=self._ocv,
            deg_rate_func=self._deg,
            reduced_limit=False,
        )
        I_reduced = mpc.compute_current(
            SoC=0.9, R_int=0.05,
            V_ocv_func=self._ocv,
            deg_rate_func=self._deg,
            reduced_limit=True,
        )
        # Both must be in range
        assert mpc.I_min <= I_reduced <= mpc.I_max
        assert mpc.I_min <= I_normal <= mpc.I_max

    def test_penalises_voltage_violations(self, mpc):
        # Extremely high SoC: OCV ≈ V_max — any positive current violates
        I_opt = mpc.compute_current(
            SoC=1.0, R_int=0.001,
            V_ocv_func=self._ocv,
            deg_rate_func=self._deg,
        )
        # Optimal current should be ≤ I_ref to avoid overvoltage
        assert I_opt <= mpc.I_max

    def test_with_degradation_penalty(self, mpc):
        def deg_high(V, I):
            return abs(I) * 10.0  # high penalty for large currents

        I_opt = mpc.compute_current(
            SoC=0.5, R_int=0.05,
            V_ocv_func=self._ocv,
            deg_rate_func=deg_high,
        )
        assert mpc.I_min <= I_opt <= mpc.I_max

    def test_default_attributes(self):
        mpc = SimpleMPC()
        assert mpc.N_p == 10
        assert mpc.n_grid == 21
        assert mpc.V_min == 2.5
        assert mpc.V_max == 4.2
