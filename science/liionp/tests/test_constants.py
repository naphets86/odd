"""Tests for liionp.constants."""
import liionp.constants as c


def test_physical_constants():
    assert c.R_GAS == 8.314
    assert c.F_FARADAY == 96485.0


def test_cell_defaults():
    assert c.Q0_DEFAULT == 3.0
    assert c.V_NOM_DEFAULT == 3.7
    assert c.V_MAX_DEFAULT == 4.2
    assert c.V_MIN_DEFAULT == 2.5
    assert c.R_INT_DEFAULT == 0.050
    assert c.E_A_SEI_DEFAULT == 50_000.0
    assert c.M_CELL_CP_DEFAULT == 45.0
    assert c.R_TH_DEFAULT == 3.5
    assert c.D_EOL == 0.20


def test_sei_defaults():
    assert c.K_SEI_DEFAULT == 1e-17
    assert c.ALPHA_SEI_DEFAULT == 1e-4


def test_voltage_defaults():
    assert c.K_OV_DEFAULT == 0.01
    assert c.BETA_OV_DEFAULT == 10.0
    assert c.K_UV_DEFAULT == 0.01
    assert c.BETA_UV_DEFAULT == 10.0


def test_gamma_defaults():
    assert c.GAMMA_T_DEFAULT == 0.65
    assert c.GAMMA_V_DEFAULT == 0.35
    assert c.GAMMA_C_DEFAULT == 0.10
    assert c.GAMMA_M_DEFAULT == 0.15


def test_mechanical_defaults():
    assert c.D_SEP_0 == 25e-6
    assert c.D_SEP_MIN == 20e-6
    assert c.D_SEP_MAX == 35e-6
    assert c.TAU_0 == 2.5
    assert c.GAMMA_TAU == 1.8
    assert c.EPS_SEP == 0.40
    assert c.SIGMA_EL == 1.0
    assert c.K_MECH_DEFAULT == 0.5
    assert c.M_MECH_DEFAULT == 2.5
    assert c.SIGMA_YIELD_DEFAULT == 1.0


def test_swelling_coeffs():
    a0, a1, a2, a3 = c.EPS_VOL_COEFFS
    assert a0 == 0.0
    assert a1 == 0.062
    assert a2 == 0.015
    assert a3 == 0.003


def test_ev_cold_defaults():
    assert c.KAPPA_T == 0.007
    assert c.T_REF_COLD == 293.0
