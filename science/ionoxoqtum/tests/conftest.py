"""
tests/conftest.py
=================
Gemeinsame Fixtures für alle Tests.
"""

import pytest
import numpy as np
import sys
import os


@pytest.fixture(scope="session")
def kcl_05():
    """KCl-Lösung 0.5 mol/l (Session-Fixture)."""
    from src.ionic import kcl_solution
    return kcl_solution(0.5)


@pytest.fixture(scope="session")
def standard_ewod():
    """Standard Al₂O₃-EWOD-System."""
    from src.ewod import standard_al2o3_system
    return standard_al2o3_system()


@pytest.fixture(scope="session")
def default_iofet():
    """Standard IoFET mit typischen Parametern."""
    from src.iofet import IoFET
    return IoFET(conductivity=1.0, threshold_voltage=0.05)


@pytest.fixture(scope="session")
def fp_250nm():
    """FP-Resonator: d₀=250nm, n₀=1.40, R=0.5."""
    from src.fabry_perot import FabryPerotResonator
    return FabryPerotResonator(d0=250e-9, n0=1.40, R1=0.5, R2=0.5)


@pytest.fixture(scope="session")
def pdms_polymer():
    """PDMS-ähnliches Polymer (chi=0.45, Vc=5e-4 m³/mol)."""
    from src.polymer import FloryRehnerModel, PolymerParameters
    return FloryRehnerModel(PolymerParameters(chi=0.45, Vm=1.8e-5, Vc=5.0e-4))


@pytest.fixture(scope="session")
def small_ebl():
    """Kleines EBL-Modell für schnelle Tests."""
    from src.ebl import ProximityEffectModel, EBLParameters
    return ProximityEffectModel(
        EBLParameters(beta_f=10e-9, beta_b=5e-6, eta=0.5),
        grid_size_nm=500.0,
        n_grid=32,
    )


@pytest.fixture(scope="session")
def green_target_spectrum():
    """Gauss-Zielspektrum bei 550 nm (Grün)."""
    lam = np.linspace(380, 780, 80)
    R = 0.5 * np.exp(-((lam - 550)**2) / (2 * 40**2))
    return lam, R
