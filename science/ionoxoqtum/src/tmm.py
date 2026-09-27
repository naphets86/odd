"""
tmm.py
=================
Transfermatrix-Methode (TMM) für beliebige optische Schichtsysteme.

Implementiert die charakteristische Matrix-Methode nach Born & Wolf (1999)
für s- und p-Polarisation sowie unpolarisiertes Licht.

Gleichungen:
  M_j = [[cos δ_j,  -i/η_j · sin δ_j],
          [-i η_j · sin δ_j,  cos δ_j]]

  δ_j = 2π ñ_j d_j cos θ_j / λ
  η_j^(s) = ñ_j cos θ_j / Z_0
  η_j^(p) = ñ_j / cos θ_j / Z_0  (für p-Pol.)

Wichtige Analogie:
  Die TMM ist formal identisch mit dem QM-Tunnelmatrix-Formalismus.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Impedanz des Vakuums
Z_0 = 376.730_313_668  # Ω


@dataclass
class OpticalLayer:
    """Einzelne optische Schicht.

    Parameters
    ----------
    n : complex
        Komplexer Brechungsindex ñ = n + iκ.
    d : float
        Schichtdicke [m]. 0 für halb-unendliches Medium.
    label : str
        Beschriftung.
    """
    n: complex
    d: float
    label: str = ""

    @property
    def is_semi_infinite(self) -> bool:
        return self.d == 0.0


class TransferMatrixMethod:
    """Transfermatrix-Methode für geschichtete optische Medien.

    Parameters
    ----------
    incident_medium : OpticalLayer
        Eingangsmedium (halb-unendlich, d=0).
    layers : list[OpticalLayer]
        Schichtstapel in Ausbreitungsrichtung.
    substrate : OpticalLayer
        Substrat (halb-unendlich, d=0).
    theta_i_deg : float
        Einfallswinkel [Grad].
    """

    def __init__(
        self,
        incident_medium: OpticalLayer,
        layers: List[OpticalLayer],
        substrate: OpticalLayer,
        theta_i_deg: float = 0.0,
    ) -> None:
        self.medium = incident_medium
        self.layers = layers
        self.substrate = substrate
        self.theta_i = np.radians(theta_i_deg)

    # ── Snellius ───────────────────────────────────────────────────────────────

    def _snell(self, n_in: complex, n_out: complex, theta_in: complex) -> complex:
        """Snellsches Brechungsgesetz: n_in sin θ_in = n_out sin θ_out."""
        sin_out = n_in * np.sin(theta_in) / n_out
        # Für absorbierende Medien: arcsin komplex
        return np.arcsin(sin_out)

    def _refraction_angles(self) -> List[complex]:
        """Brechungswinkel in allen Schichten."""
        all_n = [self.medium.n] + [l.n for l in self.layers] + [self.substrate.n]
        angles = [self.theta_i]
        for i in range(1, len(all_n)):
            theta_next = self._snell(all_n[i - 1], all_n[i], angles[-1])
            angles.append(theta_next)
        return angles

    # ── Admittanzen ────────────────────────────────────────────────────────────

    def _admittance_s(self, n: complex, theta: complex) -> complex:
        """Optische Admittanz η_s = n cos θ / Z_0."""
        return n * np.cos(theta) / Z_0

    def _admittance_p(self, n: complex, theta: complex) -> complex:
        """Optische Admittanz η_p = n / (cos θ · Z_0)."""
        cos_t = np.cos(theta)
        if np.abs(cos_t) < 1e-12:
            return np.inf
        return n / (cos_t * Z_0)

    # ── Charakteristische Matrix ───────────────────────────────────────────────

    def _layer_matrix_s(
        self,
        n: complex,
        d: float,
        theta: complex,
        wavelength: float,
    ) -> np.ndarray:
        """Charakteristische Matrix M_j für s-Polarisation.

        M_j = [[cos δ,   -i/η sin δ],
                [-iη sin δ,  cos δ  ]]
        """
        delta = 2.0 * np.pi * n * d * np.cos(theta) / wavelength
        eta = self._admittance_s(n, theta)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        return np.array([
            [cos_d,              -1j / eta * sin_d],
            [-1j * eta * sin_d,   cos_d           ],
        ])

    def _layer_matrix_p(
        self,
        n: complex,
        d: float,
        theta: complex,
        wavelength: float,
    ) -> np.ndarray:
        """Charakteristische Matrix M_j für p-Polarisation."""
        delta = 2.0 * np.pi * n * d * np.cos(theta) / wavelength
        eta = self._admittance_p(n, theta)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        return np.array([
            [cos_d,              -1j / eta * sin_d],
            [-1j * eta * sin_d,   cos_d           ],
        ])

    # ── Gesamtmatrix ───────────────────────────────────────────────────────────

    def _system_matrix(
        self,
        wavelength: float,
        polarization: str = "s",
    ) -> np.ndarray:
        """Gesamtmatrix M_ges = ∏ M_j."""
        angles = self._refraction_angles()
        mat_fn = self._layer_matrix_s if polarization == "s" else self._layer_matrix_p
        M = np.eye(2, dtype=complex)
        for i, layer in enumerate(self.layers):
            M_j = mat_fn(layer.n, layer.d, angles[i + 1], wavelength)
            M = M @ M_j
        return M

    # ── Reflexions- und Transmissionskoeffizienten ─────────────────────────────

    def _rt_coefficients(
        self,
        wavelength: float,
        polarization: str = "s",
    ) -> Tuple[complex, complex]:
        """Reflexions- und Transmissionsamplitude (r, t).

        Aus Gleichung (7.3.x) Born & Wolf.
        """
        angles = self._refraction_angles()
        adm_fn = self._admittance_s if polarization == "s" else self._admittance_p
        eta_0 = adm_fn(self.medium.n, angles[0])
        eta_s = adm_fn(self.substrate.n, angles[-1])
        M = self._system_matrix(wavelength, polarization)
        m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        numerator_r = m11 * eta_0 + m12 * eta_0 * eta_s - m21 - m22 * eta_s
        denominator  = m11 * eta_0 + m12 * eta_0 * eta_s + m21 + m22 * eta_s
        if np.abs(denominator) < 1e-30:
            return 1.0 + 0j, 0.0 + 0j
        r = numerator_r / denominator
        t = 2.0 * eta_0 / denominator
        return r, t

    # ── Öffentliche API ────────────────────────────────────────────────────────

    def reflectance(
        self,
        wavelength: float,
        polarization: str = "avg",
    ) -> float:
        """Reflexionsgrad R = |r|² bei gegebener Wellenlänge.

        Parameters
        ----------
        wavelength : float
            Wellenlänge [m].
        polarization : str
            's', 'p', oder 'avg' (Mittelwert).

        Returns
        -------
        float
            Reflexionsgrad R ∈ [0, 1].
        """
        if polarization == "avg":
            rs, _ = self._rt_coefficients(wavelength, "s")
            rp, _ = self._rt_coefficients(wavelength, "p")
            return 0.5 * (abs(rs)**2 + abs(rp)**2)
        else:
            r, _ = self._rt_coefficients(wavelength, polarization)
            return abs(r) ** 2

    def transmittance(
        self,
        wavelength: float,
        polarization: str = "avg",
    ) -> float:
        """Transmissionsgrad T = 1 − R (für verlustfreie Medien)."""
        return 1.0 - self.reflectance(wavelength, polarization)

    def spectrum(
        self,
        lam_min_nm: float = 380.0,
        lam_max_nm: float = 780.0,
        n_points: int = 400,
        polarization: str = "avg",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Berechnetes Reflexionsspektrum R(λ).

        Returns
        -------
        (wavelengths_nm, R) : tuple[ndarray, ndarray]
        """
        lam_nm = np.linspace(lam_min_nm, lam_max_nm, n_points)
        R = np.array([
            self.reflectance(l * 1e-9, polarization) for l in lam_nm
        ])
        return lam_nm, R

    def phase_matrix(self, wavelength: float) -> np.ndarray:
        """Rückgabe der Gesamtphasenmatrix (für QM-Analogie).

        Die formale Identität mit dem QM-Zeitentwicklungsoperator
        wird durch diese Struktur sichtbar.
        """
        return self._system_matrix(wavelength, "s")

    def ellipsometry_params(self, wavelength: float) -> Tuple[float, float]:
        """Ellipsometrie-Parameter Ψ und Δ.

        tan Ψ · e^(iΔ) = r_p / r_s

        Returns
        -------
        (Psi_deg, Delta_deg) : tuple[float, float]
        """
        rp, _ = self._rt_coefficients(wavelength, "p")
        rs, _ = self._rt_coefficients(wavelength, "s")
        if abs(rs) < 1e-30:
            return 0.0, 0.0
        rho = rp / rs
        psi = np.degrees(np.arctan(abs(rho)))
        delta = np.degrees(np.angle(rho))
        return psi, delta

    def __repr__(self) -> str:
        return (
            f"TMM({len(self.layers)} Schichten, "
            f"θ_i={np.degrees(self.theta_i):.1f}°)"
        )


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def fresnel_rs(n1: complex, n2: complex, theta_i_deg: float = 0.0) -> complex:
    """Fresnel-Amplitudenreflexionskoeffizient r_s an ebener Grenzfläche."""
    theta_i = np.radians(theta_i_deg)
    sin_t = np.sin(theta_i)
    cos_i = np.cos(theta_i)
    cos_t = np.sqrt(1.0 - (n1 / n2 * sin_t)**2 + 0j)
    return (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)


def fresnel_rp(n1: complex, n2: complex, theta_i_deg: float = 0.0) -> complex:
    """Fresnel-Amplitudenreflexionskoeffizient r_p an ebener Grenzfläche."""
    theta_i = np.radians(theta_i_deg)
    sin_t = np.sin(theta_i)
    cos_i = np.cos(theta_i)
    cos_t = np.sqrt(1.0 - (n1 / n2 * sin_t)**2 + 0j)
    return (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)


def build_hydrogel_stack(
    d0_nm: float = 250.0,
    n_polymer: float = 1.40,
    Q: float = 1.0,
    R_mirror: float = 0.5,
) -> TransferMatrixMethod:
    """Erstellt einen TMM-Stack für ein Hydrogel-FP-System.

    Aufbau: Luft | Au-Spiegel | Hydrogel | Au-Spiegel | Glas

    Parameters
    ----------
    d0_nm : float
        Trockene Hydrogeldicke [nm].
    n_polymer : float
        Brechungsindex des trockenen Polymers.
    Q : float
        Quellungsgrad.
    R_mirror : float
        Spiegel-Reflexivität.

    Returns
    -------
    TransferMatrixMethod
    """
    n_gel = 1.0 + (n_polymer - 1.0) / Q
    d_gel = Q * d0_nm * 1e-9
    d_mirror = 10e-9  # 10 nm Au-Spiegel
    n_au = 0.47 + 2.83j  # Gold bei 550 nm (Näherung)
    layers = [
        OpticalLayer(n_au,  d_mirror, "Au-Spiegel 1"),
        OpticalLayer(n_gel, d_gel,    f"Hydrogel (Q={Q:.1f})"),
        OpticalLayer(n_au,  d_mirror, "Au-Spiegel 2"),
    ]
    medium   = OpticalLayer(1.0,  0.0, "Luft")
    substrate = OpticalLayer(1.52, 0.0, "Borosilikatglas")
    return TransferMatrixMethod(medium, layers, substrate)
