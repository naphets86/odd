"""
ebl.py
=================
Elektronenstrahl-Lithographie (EBL) – Proximity-Effekte und Korrektur.

Implementiert:
  - Chang-Modell (Doppel-Gauss PSF)
  - Energiedepositonsberechnung via Faltung
  - FFT-basierte Proximity-Effekt-Korrektur (SPECTRE-ähnlich)
  - Bethe-Reichweite für Elektronen

Gleichungen:
  PSF(r) = 1/(π(1+η)) · [1/β_f² · exp(−r²/β_f²) + η/β_b² · exp(−r²/β_b²)]
  E_dep(r) = ∫ I(r') · PSF(r−r') d²r'  (Faltung)
  Î(k) = Ê_Ziel(k) / PSF_hat(k)  (Dekonvolution im Fourier-Raum)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EBLParameters:
    """Parametrierung des Elektronenstrahl-Systems.

    Parameters
    ----------
    beta_f : float
        Vorwärts-Streuradius β_f [m]. Typisch: 10 nm bei 100 keV.
    beta_b : float
        Rückwärts-Streuradius β_b [m]. Typisch: 10 µm bei 100 keV.
    eta : float
        Verhältnis Rückwärts- zu Vorwärtsstreuung η (typ. 0.5–0.8).
    beam_energy_keV : float
        Strahlenergie [keV].
    """
    beta_f: float = 10e-9    # m
    beta_b: float = 10e-6    # m
    eta: float = 0.6
    beam_energy_keV: float = 100.0

    @property
    def bethe_range_m(self) -> float:
        """Bethe-Reichweite für Elektronen in organischen Materialien.

        Empirische Formel nach Kanaya-Okayama (vereinfacht für Polymere):
        R_B ≈ 0.0276 · A / (ρ Z^0.889) · E^1.67  [µm]
        Hier: Näherungsformel für Polymer (A≈12, ρ≈1.2 g/cm³, Z≈6).
        """
        E = self.beam_energy_keV
        return 0.0276 * 12.0 / (1.2 * 6.0**0.889) * E**1.67 * 1e-6  # m


class ProximityEffectModel:
    """Modell für EBL Proximity-Effekte.

    Parameters
    ----------
    params : EBLParameters
        EBL-Parameter.
    grid_size_nm : float
        Gittergröße für Simulation [nm].
    n_grid : int
        Anzahl Gitterpunkte pro Dimension.
    """

    def __init__(
        self,
        params: Optional[EBLParameters] = None,
        grid_size_nm: float = 2000.0,
        n_grid: int = 256,
    ) -> None:
        self.params = params or EBLParameters()
        self.grid_size_nm = grid_size_nm
        self.n_grid = n_grid

    @property
    def dx_nm(self) -> float:
        """Gitterschrittweite [nm]."""
        return self.grid_size_nm / self.n_grid

    def _make_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Erstellt 2D-Koordinatengitter [nm]."""
        coords = np.linspace(
            -self.grid_size_nm / 2,
             self.grid_size_nm / 2,
             self.n_grid,
        )
        return np.meshgrid(coords, coords)

    # ── Chang-PSF ──────────────────────────────────────────────────────────────

    def psf_1d(self, r_nm: np.ndarray) -> np.ndarray:
        """Chang-Doppel-Gauss PSF in 1D.

        PSF(r) = 1/(π(1+η)) · [1/β_f² · exp(−r²/β_f²) + η/β_b² · exp(−r²/β_b²)]

        Parameters
        ----------
        r_nm : ndarray
            Radiale Abstände [nm].

        Returns
        -------
        ndarray
            PSF-Werte [1/nm²].
        """
        bf = self.params.beta_f * 1e9   # m → nm
        bb = self.params.beta_b * 1e9
        eta = self.params.eta
        r2 = r_nm**2
        forward  = np.exp(-r2 / bf**2) / (bf**2)
        backward = eta * np.exp(-r2 / bb**2) / (bb**2)
        return (forward + backward) / (np.pi * (1.0 + eta))

    def psf_2d(self) -> np.ndarray:
        """Chang-PSF auf dem 2D-Gitter.

        Returns
        -------
        ndarray
            2D-PSF-Array, normiert auf Integral = 1.
        """
        X, Y = self._make_grid()
        R = np.sqrt(X**2 + Y**2)
        psf = self.psf_1d(R)
        # Normierung
        psf /= psf.sum() * self.dx_nm**2
        return psf

    # ── Energiedeposition ──────────────────────────────────────────────────────

    def energy_deposition(self, pattern: np.ndarray) -> np.ndarray:
        """Energiedeposition durch Faltung mit PSF.

        E_dep(r) = I(r) * PSF(r)  (Faltung via FFT)

        Parameters
        ----------
        pattern : ndarray (n_grid × n_grid)
            Belichtungsmuster I(r) [normiert].

        Returns
        -------
        ndarray
            Energiedepositon E_dep(r) [gleiche Einheit wie pattern].
        """
        psf = self.psf_2d()
        # FFT-basierte Faltung (zirkulär)
        I_hat = np.fft.fft2(pattern)
        PSF_hat = np.fft.fft2(psf)
        E_hat = I_hat * PSF_hat
        E_dep = np.real(np.fft.ifft2(E_hat))
        return E_dep * self.dx_nm**2

    # ── Proximity-Korrektur (Dekonvolution) ───────────────────────────────────

    def proximity_correction(
        self,
        target_energy: np.ndarray,
        regularization: float = 0.01,
    ) -> np.ndarray:
        """FFT-Dekonvolution zur Proximity-Korrektur.

        Î(k) = Ê_Ziel(k) / PSF_hat(k)  mit Wiener-Regularisierung.

        Komplexität: O(M² log M) für M×M-Gitter.

        Parameters
        ----------
        target_energy : ndarray (n_grid × n_grid)
            Gewünschte Energiedeposition.
        regularization : float
            Wiener-Regularisierungsparameter ε (verhindert Rauschen bei
            PSF_hat ≈ 0).

        Returns
        -------
        ndarray
            Korrigiertes Belichtungsmuster I_korr(r).
        """
        psf = self.psf_2d()
        E_hat = np.fft.fft2(target_energy)
        PSF_hat = np.fft.fft2(psf)
        # Wiener-Filter
        denom = np.abs(PSF_hat)**2 + regularization
        I_hat_corr = E_hat * np.conj(PSF_hat) / denom
        I_corr = np.real(np.fft.ifft2(I_hat_corr))
        # Nicht-negativ
        return np.maximum(I_corr, 0.0)

    # ── Quellung-Kodierung ─────────────────────────────────────────────────────

    def dose_to_swelling(
        self,
        dose_map: np.ndarray,
        Q_min: float = 1.0,
        Q_max: float = 4.0,
        dose_min: Optional[float] = None,
        dose_max: Optional[float] = None,
    ) -> np.ndarray:
        """Konvertiert Dosierungskarte in Quellung-Karte Q(r).

        Lineare Interpolation: Q ∝ 1/dose (höhere Dosis → stärkere Vernetzung
        → geringere Quellung).

        Parameters
        ----------
        dose_map : ndarray
            Belichtungsdosis [normiert].
        Q_min, Q_max : float
            Quellungsbereich.
        dose_min, dose_max : float, optional
            Normierungsbereich.

        Returns
        -------
        ndarray
            Quellung-Karte Q(r).
        """
        if dose_min is None:
            dose_min = dose_map.min()
        if dose_max is None:
            dose_max = dose_map.max()

        if dose_max == dose_min:
            return np.full_like(dose_map, (Q_min + Q_max) / 2.0)

        dose_norm = (dose_map - dose_min) / (dose_max - dose_min)
        # Invers: hohe Dosis → geringe Quellung
        Q_map = Q_max - dose_norm * (Q_max - Q_min)
        return Q_map

    def encode_texture(
        self,
        texture_target: np.ndarray,
        Q_range: Tuple[float, float] = (1.0, 4.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kodiert ein Zieltexturmuster in ein EBL-Belichtungsmuster.

        Parameters
        ----------
        texture_target : ndarray
            Normiertes Zieltexturmuster (0...1).
        Q_range : tuple
            (Q_min, Q_max) des Quellungsbereichs.

        Returns
        -------
        (corrected_pattern, swelling_map) : tuple[ndarray, ndarray]
        """
        Q_min, Q_max = Q_range
        # Umkehrung: texture → inverse Quellung → Dosis
        dose_target = 1.0 - texture_target
        corrected = self.proximity_correction(dose_target)
        Q_map = self.dose_to_swelling(corrected, Q_min, Q_max)
        return corrected, Q_map

    def __repr__(self) -> str:
        return (
            f"ProximityEffectModel("
            f"β_f={self.params.beta_f*1e9:.1f}nm, "
            f"β_b={self.params.beta_b*1e6:.1f}µm, "
            f"η={self.params.eta:.1f})"
        )
