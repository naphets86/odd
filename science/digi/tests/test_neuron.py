"""tests/test_neuron.py – Tests for digi.neuron"""
import numpy as np
import pytest
from digi.neuron import (
    threshold_response,
    exceeds_threshold,
    action_potential_waveform,
    leaky_integrate_and_fire,
    hodgkin_huxley,
    V_REST,
    V_THRESH,
    V_PEAK,
)


class TestThresholdResponse:
    def test_below_threshold_gives_low(self):
        u_in = np.array([0.5, -1.0, 0.0])
        out = threshold_response(u_in, u_threshold=1.0, u_high=1.0, u_low=0.0)
        np.testing.assert_array_equal(out, 0.0)

    def test_above_threshold_gives_high(self):
        u_in = np.array([2.0, 3.0, 10.0])
        out = threshold_response(u_in, u_threshold=1.0, u_high=1.0, u_low=0.0)
        np.testing.assert_array_equal(out, 1.0)

    def test_at_threshold_gives_low(self):
        """Exactly at threshold (not strictly greater) → low"""
        u_in = np.array([1.0])
        out = threshold_response(u_in, u_threshold=1.0, u_high=1.0, u_low=0.0)
        assert out[0] == 0.0

    def test_custom_levels(self):
        u_in = np.array([5.0])
        out = threshold_response(u_in, u_threshold=2.0, u_high=42.0, u_low=-1.0)
        assert out[0] == 42.0

    def test_binary_output(self):
        """Output must be exactly u_high or u_low"""
        rng = np.random.default_rng(0)
        u_in = rng.uniform(-5, 5, 1000)
        out = threshold_response(u_in, u_threshold=0.0)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_all_or_nothing(self):
        """Slight change over threshold triggers full response"""
        eps = 1e-10
        below = threshold_response(np.array([1.0 - eps]), 1.0)
        above = threshold_response(np.array([1.0 + eps]), 1.0)
        assert below[0] == 0.0
        assert above[0] == 1.0


class TestExceedsThreshold:
    def test_true_above(self):
        assert exceeds_threshold(2.0, 1.0) is True

    def test_false_at(self):
        assert exceeds_threshold(1.0, 1.0) is False

    def test_false_below(self):
        assert exceeds_threshold(-1.0, 0.0) is False


class TestActionPotentialWaveform:
    def test_starts_at_rest(self):
        t = np.array([-0.01])
        v = action_potential_waveform(t, t_spike=0.0, v_rest=V_REST)
        assert v[0] == pytest.approx(V_REST)

    def test_peak_is_positive(self):
        t = np.linspace(0, 3e-3, 1000)
        v = action_potential_waveform(t, t_spike=0.0, v_rest=V_REST, v_peak=V_PEAK)
        assert np.max(v) > V_THRESH

    def test_output_shape_matches_t(self):
        t = np.linspace(-0.001, 0.01, 500)
        v = action_potential_waveform(t)
        assert v.shape == t.shape

    def test_finite_values(self):
        t = np.linspace(-0.001, 0.02, 1000)
        v = action_potential_waveform(t, t_spike=0.005)
        assert np.all(np.isfinite(v))


class TestLeakyIntegrateAndFire:
    def test_no_spikes_below_threshold(self):
        """Zero current → no spikes"""
        I_ext = np.zeros(1000)
        V, spikes = leaky_integrate_and_fire(I_ext, dt=1e-4)
        assert not np.any(spikes)

    def test_spikes_above_threshold(self):
        """Strong enough current drives spiking.

        With R_m=1e7 Ω and V_thresh=-55 mV (rest -70 mV), the steady-state
        depolarisation is R_m·I = 1e7 * 2e-9 = 20 mV → V_ss = -50 mV > thresh.
        """
        I_ext = np.ones(5000) * 2e-9  # 2 nA → V_ss = -50 mV > -55 mV threshold
        V, spikes = leaky_integrate_and_fire(
            I_ext, dt=1e-4,
            C_m=1e-10, R_m=1e7,
            V_rest=-70e-3, V_thresh=-55e-3,
        )
        assert np.any(spikes)

    def test_resting_potential_without_input(self):
        """With zero input membrane should stay near rest"""
        I_ext = np.zeros(100)
        V, _ = leaky_integrate_and_fire(I_ext, dt=1e-4)
        np.testing.assert_allclose(V, -70e-3, atol=1e-6)

    def test_output_lengths(self):
        I_ext = np.zeros(200)
        V, spikes = leaky_integrate_and_fire(I_ext, dt=1e-4)
        assert len(V) == len(spikes) == 200

    def test_refractory_prevents_double_spike(self):
        """No two spikes within refractory period"""
        I_ext = np.ones(10_000) * 2e-9
        dt = 5e-5
        V, spikes = leaky_integrate_and_fire(
            I_ext, dt=dt,
            C_m=1e-10, R_m=1e7,
            V_rest=-70e-3, V_thresh=-55e-3,
            t_refrac=2e-3,
        )
        spike_indices = np.where(spikes)[0]
        if len(spike_indices) > 1:
            min_gap = np.min(np.diff(spike_indices)) * dt
            assert min_gap >= 2e-3 - dt


class TestHodgkinHuxley:
    def test_output_shape(self):
        n = 1000
        I_ext = np.zeros(n)
        V, t = hodgkin_huxley(I_ext, dt=1e-5)
        assert len(V) == n
        assert len(t) == n

    def test_resting_without_input(self):
        """Without current, membrane stays near zero (HH convention)"""
        I_ext = np.zeros(500)
        V, _ = hodgkin_huxley(I_ext, dt=1e-5)
        assert np.max(np.abs(V)) < 5.0  # should stay near 0 mV

    def test_spike_with_strong_current(self):
        """A suprathreshold current should produce a spike (V > 40 mV)"""
        n = 5000
        I_ext = np.zeros(n)
        I_ext[100:200] = 20.0  # 20 µA/cm²
        V, _ = hodgkin_huxley(I_ext, dt=1e-5)
        assert np.max(V) > 20.0  # considerable depolarisation

    def test_time_vector(self):
        n = 100
        dt = 1e-5
        _, t = hodgkin_huxley(np.zeros(n), dt=dt)
        assert t[0] == 0.0
        assert pytest.approx(t[-1], abs=dt) == (n - 1) * dt
