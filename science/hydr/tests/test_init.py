"""Tests für hydr.__init__ – stellt sicher, dass alle Submodule importierbar sind."""

import pytest


def test_import_hydr():
    import hydr  # noqa: F401


def test_all_submodules_accessible():
    import hydr
    assert hasattr(hydr, "bulk_modulus")
    assert hasattr(hydr, "viscosity")
    assert hasattr(hydr, "viscoelastic")
    assert hasattr(hydr, "hydraulics")
    assert hasattr(hydr, "lifetime")
    assert hasattr(hydr, "surface")
    assert hasattr(hydr, "mixing")
    assert hasattr(hydr, "monitoring")


def test_bulk_modulus_accessible():
    from hydr import bulk_modulus
    assert callable(bulk_modulus.isentropic_modulus)


def test_viscosity_accessible():
    from hydr import viscosity
    assert callable(viscosity.relaxation_time)


def test_viscoelastic_accessible():
    from hydr import viscoelastic
    assert callable(viscoelastic.storage_modulus)


def test_hydraulics_accessible():
    from hydr import hydraulics
    assert callable(hydraulics.hydraulic_eigenfrequency)


def test_lifetime_accessible():
    from hydr import lifetime
    assert callable(lifetime.oil_lifetime)


def test_surface_accessible():
    from hydr import surface
    assert callable(surface.laplace_pressure)


def test_mixing_accessible():
    from hydr import mixing
    assert callable(mixing.maxwell_emulsion_modulus)


def test_monitoring_accessible():
    from hydr import monitoring
    assert callable(monitoring.ultrasonic_bulk_modulus)
