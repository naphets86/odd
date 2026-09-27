"""
hydr – Elastizität hydraulischer Öle und Flüssigkeiten im industriellen Maschinenbau

Submodule:
  bulk_modulus   Kompressionsmodul, Tait-Gleichung, Temperatur-/Druckabhängigkeit
  viscosity      Viskositätsmodelle, Barus-Gesetz, VII-Additive, Scherabbau
  viscoelastic   Zener-Modell, Speicher-/Verlustmodul, Verlustfaktor
  hydraulics     Hydraulische Eigenfrequenz, Kompressibilitätsfehler, SEA
  lifetime       Lebensdauerformel, Alterungsmodelle
  surface        Laplace-Druck, Kavitationsdruck, Henry-Gesetz
  mixing         Maxwell-Mischungsregel für Emulsionen
  monitoring     Ultraschall-Bulk-Modul-Sensor
"""

from hydr import (
    bulk_modulus,
    viscosity,
    viscoelastic,
    hydraulics,
    lifetime,
    surface,
    mixing,
    monitoring,
)

__all__ = [
    "bulk_modulus",
    "viscosity",
    "viscoelastic",
    "hydraulics",
    "lifetime",
    "surface",
    "mixing",
    "monitoring",
]
