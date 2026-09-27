"""Tests für irf.transistor"""

import math
import pytest

from src.transistor import IoFET, IoFETState, TransferCharacteristic
from src.physics import EWODModel, IonicConductivity
from src.utils import ValidationError


@pytest.fixture
def ewod():
    return EWODModel(
        dielectric_thickness_m=10e-9,
        dielectric_epsilon_r=9.0,
        contact_angle_0_deg=110.0,
        surface_tension_N_m=0.0728,
    )


@pytest.fixture
def conductivity():
    return IonicConductivity.from_kcl(0.1)


@pytest.fixture
def iofet(ewod, conductivity):
    return IoFET(
        ewod=ewod,
        conductivity_model=conductivity,
        channel_length_m=100e-9,
        channel_width_m=50e-9,
        channel_height_m=50e-9,
        threshold_angle_deg=90.0,
    )


class TestIoFET:
    def test_threshold_voltage_positive(self, iofet):
        assert iofet.threshold_voltage_V > 0

    def test_channel_cross_section(self, iofet):
        expected = 50e-9 * 50e-9
        assert iofet.channel_cross_section_m2 == pytest.approx(expected)

    def test_g0_positive(self, iofet):
        assert iofet.g0 > 0

    def test_eta_ewod_positive(self, iofet):
        assert iofet.eta_ewod > 0

    def test_state_off(self, iofet):
        # VGS sehr negativ → OFF
        vt = iofet.threshold_voltage_V
        assert iofet.state(0.0, 0.1) == IoFETState.OFF

    def test_state_subthreshold(self, iofet):
        vt = iofet.threshold_voltage_V
        vgs_sub = vt * 0.75  # zwischen 0.5*VT und VT
        s = iofet.state(vgs_sub, 0.01)
        assert s == IoFETState.SUBTHRESHOLD

    def test_state_linear(self, iofet):
        vt = iofet.threshold_voltage_V
        vgs = vt + 0.5
        vds = 0.01  # VDS ≪ VGS - VT
        assert iofet.state(vgs, vds) == IoFETState.LINEAR

    def test_state_saturation(self, iofet):
        vt = iofet.threshold_voltage_V
        vgs = vt + 0.5
        vds = vgs  # VDS >= VGS - VT
        assert iofet.state(vgs, vds) == IoFETState.SATURATION

    def test_drain_current_off(self, iofet):
        ids = iofet.drain_current_A(0.0, 0.1)
        assert ids == pytest.approx(0.0)

    def test_drain_current_subthreshold_positive(self, iofet):
        vt = iofet.threshold_voltage_V
        ids = iofet.drain_current_A(vt * 0.75, 0.01)
        assert ids >= 0.0

    def test_drain_current_linear_positive(self, iofet):
        vt = iofet.threshold_voltage_V
        ids = iofet.drain_current_A(vt + 0.5, 0.01)
        assert ids > 0

    def test_drain_current_saturation_positive(self, iofet):
        vt = iofet.threshold_voltage_V
        ids = iofet.drain_current_A(vt + 0.5, vt + 1.0)
        assert ids > 0

    def test_drain_current_increases_with_vgs_linear(self, iofet):
        vt = iofet.threshold_voltage_V
        i1 = iofet.drain_current_A(vt + 0.2, 0.01)
        i2 = iofet.drain_current_A(vt + 0.5, 0.01)
        assert i2 > i1

    def test_transconductance_off(self, iofet):
        assert iofet.transconductance_S(0.0, 0.1) == pytest.approx(0.0)

    def test_transconductance_subthreshold(self, iofet):
        vt = iofet.threshold_voltage_V
        gm = iofet.transconductance_S(vt * 0.75, 0.01)
        assert gm == pytest.approx(0.0)

    def test_transconductance_linear(self, iofet):
        vt = iofet.threshold_voltage_V
        gm = iofet.transconductance_S(vt + 0.5, 0.01)
        assert gm > 0

    def test_transconductance_saturation(self, iofet):
        vt = iofet.threshold_voltage_V
        gm = iofet.transconductance_S(vt + 0.5, 2.0)
        assert gm > 0

    def test_switching_energy_positive(self, iofet):
        e = iofet.switching_energy_J(0.12)
        assert e > 0

    def test_transfer_characteristic(self, iofet):
        tc = iofet.transfer_characteristic(vds=0.1, n_points=50)
        assert tc.num_points == 50
        assert len(tc.vgs_values) == 50
        assert tc.threshold_voltage_V == pytest.approx(iofet.threshold_voltage_V)

    def test_transfer_characteristic_invalid_points(self, iofet):
        with pytest.raises(ValidationError):
            iofet.transfer_characteristic(0.1, n_points=1)

    def test_output_characteristic(self, iofet):
        result = iofet.output_characteristic(vgs=0.5, n_points=20)
        assert len(result) == 20
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_output_characteristic_invalid_points(self, iofet):
        with pytest.raises(ValidationError):
            iofet.output_characteristic(0.5, n_points=1)

    def test_invalid_channel_length(self, ewod, conductivity):
        with pytest.raises(ValidationError):
            IoFET(ewod, conductivity, 0.0, 50e-9, 50e-9)

    def test_invalid_channel_width(self, ewod, conductivity):
        with pytest.raises(ValidationError):
            IoFET(ewod, conductivity, 100e-9, 0.0, 50e-9)

    def test_invalid_channel_height(self, ewod, conductivity):
        with pytest.raises(ValidationError):
            IoFET(ewod, conductivity, 100e-9, 50e-9, 0.0)

    def test_invalid_threshold_angle(self, ewod, conductivity):
        with pytest.raises(ValidationError):
            IoFET(ewod, conductivity, 100e-9, 50e-9, 50e-9, threshold_angle_deg=0.0)
        with pytest.raises(ValidationError):
            IoFET(ewod, conductivity, 100e-9, 50e-9, 50e-9, threshold_angle_deg=180.0)


class TestTransferCharacteristic:
    def test_add_point(self):
        tc = TransferCharacteristic()
        tc.add_point(0.1, 1e-9)
        tc.add_point(0.2, 2e-9)
        assert tc.num_points == 2

    def test_on_off_ratio(self):
        tc = TransferCharacteristic()
        tc.add_point(0.0, 0.0)   # OFF: wird ignoriert
        tc.add_point(0.1, 1e-12)
        tc.add_point(0.5, 1e-6)
        ratio = tc.on_off_ratio
        assert ratio == pytest.approx(1e-6 / 1e-12, rel=1e-3)

    def test_on_off_ratio_empty(self):
        tc = TransferCharacteristic()
        assert tc.on_off_ratio == 0.0

    def test_on_off_ratio_all_zero(self):
        tc = TransferCharacteristic()
        tc.add_point(0.0, 0.0)
        assert tc.on_off_ratio == float("inf")

    def test_num_points_empty(self):
        tc = TransferCharacteristic()
        assert tc.num_points == 0
