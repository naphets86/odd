"""
nanooptik.materials
===================
Materialdatenbank mit Brechungsindizes für häufig verwendete Materialien
in der aktiven Nanooptik.

Alle Werte bei Raumtemperatur (25 °C) und λ = 550 nm, sofern nicht anders
angegeben. Komplexe Werte: ñ = n + i·κ.

Quellen:
  - PDMS:       Mata et al., J. Micromech. Microeng. 2005
  - Au:         Palik, Handbook of Optical Constants, 1985
  - SiO2:       Malitson, J. Opt. Soc. Am. 1965
  - H2O:        Hale & Querry, Appl. Opt. 1973
  - PMMA:       Zhang et al., Polym. Int. 2009
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

# ---------------------------------------------------------------------------
# Dataclass für Materialeigenschaften
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """Optische und physikalische Eigenschaften eines Materials."""
    name: str
    n_real: float                          # Realteil des Brechungsindex bei 550 nm
    n_imag: float = 0.0                    # Imaginärteil (Extinktionskoeffizient)
    dispersion: Optional[Callable] = None  # n(λ_nm) → complex, falls vorhanden
    density: float = 1.0                   # [g/cm³]
    description: str = ""

    @property
    def n(self) -> complex:
        """Komplexer Brechungsindex bei 550 nm."""
        return complex(self.n_real, self.n_imag)

    def n_at(self, wavelength_nm: float) -> complex:
        """Brechungsindex bei gegebener Wellenlänge [nm]."""
        if self.dispersion is not None:
            return self.dispersion(wavelength_nm)
        return self.n  # Näherung: kein Dispersion

    def __repr__(self) -> str:
        return f"Material({self.name!r}, n={self.n:.4f})"


# ---------------------------------------------------------------------------
# Sellmeier-Dispersionsformeln
# ---------------------------------------------------------------------------

def sellmeier_sio2(lam_nm: float) -> complex:
    """Sellmeier-Formel für Quarzglas (SiO2), λ in nm."""
    lam_um = lam_nm / 1000.0
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [0.0684043**2, 0.1162414**2, 9.896161**2]
    n2 = 1.0 + sum(b * lam_um**2 / (lam_um**2 - c) for b, c in zip(B, C))
    return complex(np.sqrt(max(n2, 1.0)), 0.0)


def drude_lorentz_gold(lam_nm: float) -> complex:
    """
    Vereinfachtes Drude-Lorentz-Modell für Gold, λ in nm.
    Kalibriert auf Palik-Daten bei 400–800 nm.
    """
    # Tabellierte Palik-Stützwerte (λ_nm, n, κ)
    data = np.array([
        [400, 1.658, 1.956],
        [450, 1.426, 1.767],
        [500, 0.873, 1.922],
        [550, 0.370, 2.398],
        [600, 0.196, 2.981],
        [650, 0.170, 3.392],
        [700, 0.166, 3.842],
        [750, 0.163, 4.284],
        [800, 0.161, 4.726],
    ])
    n_interp = float(np.interp(lam_nm, data[:, 0], data[:, 1]))
    k_interp = float(np.interp(lam_nm, data[:, 0], data[:, 2]))
    return complex(n_interp, k_interp)


def cauchy_pdms(lam_nm: float) -> complex:
    """
    Cauchy-Approximation für PDMS (Polydimethylsiloxan), λ in nm.
    n(λ) = A + B/λ² + C/λ⁴ (λ in µm)
    """
    lam_um = lam_nm / 1000.0
    # Mata et al. (2005), Sylgard 184 PDMS:
    # n(632nm)≈1.400, n(550nm)≈1.403, n(450nm)≈1.408
    A, B, C = 1.3980, 0.0028, 0.00004
    n = A + B / lam_um**2 + C / lam_um**4
    return complex(n, 0.0)


def water_dispersion(lam_nm: float) -> complex:
    """Brechungsindex von Wasser (Hale & Querry), λ in nm."""
    # Cauchy-Näherung für sichtbares Spektrum
    lam_um = lam_nm / 1000.0
    n = 1.3195 + 6.26e-3 / lam_um**2
    # Absorption im Sichtbaren vernachlässigbar
    return complex(n, 0.0)


# ---------------------------------------------------------------------------
# Materialdatenbank
# ---------------------------------------------------------------------------

MATERIALS: dict[str, Material] = {
    # --- Polymere / Hydrogele ---
    "PDMS": Material(
        name="PDMS",
        n_real=1.40,
        n_imag=0.0,
        dispersion=cauchy_pdms,
        density=1.03,
        description="Polydimethylsiloxan – Standardpolymer für Hydrogel-FP-Resonatoren",
    ),
    "PMMA": Material(
        name="PMMA",
        n_real=1.492,
        n_imag=0.0,
        density=1.18,
        description="Polymethylmethacrylat – E-Beam-Resist",
    ),
    "Hydrogel_PEG": Material(
        name="Hydrogel_PEG",
        n_real=1.345,
        n_imag=0.0,
        density=1.05,
        description="PEG-basiertes Hydrogel (stark gequollen)",
    ),

    # --- Metalle (Spiegel) ---
    "Au": Material(
        name="Au",
        n_real=0.37,
        n_imag=2.40,
        dispersion=drude_lorentz_gold,
        density=19.3,
        description="Gold – halbdurchlässiger Spiegel für FP-Resonatoren",
    ),
    "Ag": Material(
        name="Ag",
        n_real=0.054,
        n_imag=3.43,
        density=10.5,
        description="Silber – hochreflektiver Spiegel",
    ),

    # --- Dielektrika / Substrate ---
    "SiO2": Material(
        name="SiO2",
        n_real=1.458,
        n_imag=0.0,
        dispersion=sellmeier_sio2,
        density=2.20,
        description="Quarzglas – optisches Substrat",
    ),
    "Air": Material(
        name="Air",
        n_real=1.000,
        n_imag=0.0,
        density=0.0012,
        description="Luft – Eingangsmedium",
    ),

    # --- Lösungsmittel ---
    "Water": Material(
        name="Water",
        n_real=1.333,
        n_imag=0.0,
        dispersion=water_dispersion,
        density=1.00,
        description="Wasser – polares Quellungsmittel (χ ≈ 0.43 für PDMS)",
    ),
    "Ethanol": Material(
        name="Ethanol",
        n_real=1.361,
        n_imag=0.0,
        density=0.789,
        description="Ethanol – Quellungsmittel (χ ≈ 0.25 für PDMS)",
    ),
    "Isopropanol": Material(
        name="Isopropanol",
        n_real=1.377,
        n_imag=0.0,
        density=0.786,
        description="Isopropanol – Quellungsmittel (χ ≈ 0.22 für PDMS)",
    ),
    "nButanol": Material(
        name="nButanol",
        n_real=1.399,
        n_imag=0.0,
        density=0.810,
        description="n-Butanol – Quellungsmittel (χ ≈ 0.30 für PDMS)",
    ),
}

# Flory-Huggins-Parameter für PDMS mit verschiedenen Lösungsmitteln
FLORY_HUGGINS_CHI: dict[str, float] = {
    "Ethanol":      0.25,
    "Isopropanol":  0.22,
    "nButanol":     0.30,
    "Water":        0.43,
    "Methanol":     0.15,
    "Acetone":      0.20,
    "Toluene":      0.35,
    "Chloroform":   0.40,
}

# Molares Lösungsmittelvolumen V1 [cm³/mol]
MOLAR_VOLUME: dict[str, float] = {
    "Water":        18.0,
    "Ethanol":      58.4,
    "Isopropanol":  76.9,
    "nButanol":     91.5,
    "Methanol":     40.7,
    "Acetone":      73.4,
    "Toluene":      106.3,
    "Chloroform":   80.7,
}
