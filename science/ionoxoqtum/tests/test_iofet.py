"""
tests/test_iofet.py
===================
Tests für core/iofet.py
"""

import pytest
import numpy as np
from src.iofet import IoFET, IoFETGeometry
from src.ewod import standard_al2o3_system


class TestIoFETGeometry:

    def test_cross_section(self):
        geo = IoFETGeometry(1e-6, 100e-9, 100e-9)
        assert geo.cross_section_area == pytest.approx((100e-9)**2, rel=1e-10)

    def test_electrode_area(self):
        geo = IoFETGeometry(1e-6, 100e-9, 100e-9)
        assert geo.electrode_area == pytest.approx(1e-6 * 100e-9, rel=1e-10)


class TestIoFETBaseConduction:

    def test_base_conductance_formula(self):
        """G_0 = σ A / L."""
        sigma = 1.0
        geo = IoFETGeometry(1e-6, 100e-9, 100e-9)
        t = IoFET(conductivity=sigma, geometry=geo)
        expected = sigma * geo.cross_section_area / geo.channel_length
        assert t.base_conductance == pytest.approx(expected, rel=1e-10)

    def test_higher_conductivity_higher_G0(self):
        t1 = IoFET(conductivity=1.0)
        t2 = IoFET(conductivity=2.0)
        assert t2.base_conductance > t1.base_conductance


class TestIoFETDrainCurrent:

    def test_below_threshold_small_current(self):
        t = IoFET(threshold_voltage=0.1)
        I = t.drain_current(V_gs=0.0, V_ds=0.1)
        assert I >= 0

    def test_above_threshold_positive_current(self):
        t = IoFET(threshold_voltage=0.05)
        I = t.drain_current(V_gs=0.5, V_ds=0.1)
        assert I > 0

    def test_saturation_gt_linear(self):
        """Sättigungsstrom unabhängig von V_DS für V_DS >> overdrive."""
        t = IoFET(threshold_voltage=0.1)
        I_sat1 = t.drain_current(0.5, 1.0)
        I_sat2 = t.drain_current(0.5, 2.0)
        assert I_sat1 == pytest.approx(I_sat2, rel=0.01)

    def test_linear_region_increases_with_vds(self):
        t = IoFET(threshold_voltage=0.05)
        I1 = t.drain_current(0.5, 0.01)
        I2 = t.drain_current(0.5, 0.02)
        assert I2 > I1

    def test_current_increases_with_vgs(self):
        t = IoFET(threshold_voltage=0.05)
        I1 = t.drain_current(0.3, 0.1)
        I2 = t.drain_current(0.6, 0.1)
        assert I2 > I1

    def test_transfer_characteristics_shape(self):
        t = IoFET()
        V_gs, I_ds = t.transfer_characteristics(V_ds=0.1, n_points=100)
        assert len(V_gs) == 100
        assert len(I_ds) == 100

    def test_output_characteristics_shape(self):
        t = IoFET()
        out = t.output_characteristics([0.3, 0.5, 0.7], n_points=50)
        assert len(out) == 3
        for V_gs, (V_ds, I_ds) in out.items():
            assert len(V_ds) == 50
            assert len(I_ds) == 50

    def test_switching_energy_positive(self):
        t = IoFET()
        E = t.switching_energy()
        assert E > 0 or np.isnan(E)

    def test_on_off_ratio_positive(self):
        t = IoFET(threshold_voltage=0.05)
        ratio = t.on_off_ratio(V_gs_on=0.5, V_ds=0.1)
        assert ratio > 1.0

    def test_transistor_density(self):
        t = IoFET()
        density = t.transistor_density_3d(pitch=100e-9)
        assert density > 0

    def test_repr(self):
        t = IoFET()
        s = repr(t)
        assert "IoFET" in s
