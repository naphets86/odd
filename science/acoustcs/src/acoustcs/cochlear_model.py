"""
Cochlear Model: Basilarmembran-Dynamik und Frequenzselektion.

Implementiert:
- Eindimensionale Wellengleichung der Cochlea (Eq. 10.1)
- Helmholtz-Gleichung im Frequenzbereich (Eq. 10.2)
- Impedanzmodelle (Eq. 10.3)
- Resonanzfrequenzen und Q-Faktoren
- Nichtlineare Mechanik der äußeren Haarzellen
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal


class BasilarMembrane:
    """
    Modell der Basilarmembran mit ortsabhängigen mechanischen Parametern.
    
    Die Wellengleichung (Eq. 10.1):
        m(x)*w_tt + c(x)*w_t + k(x)*w = p_in - p_out
    
    wird gelöst durch räumliche Diskretisierung mit variablen Parametern.
    """
    
    def __init__(
        self,
        cochlea_length_mm: float = 35.0,
        f_min_hz: float = 20.0,
        f_max_hz: float = 20000.0,
        n_positions: int = 100,
    ):
        """
        Initialisiere die Basilarmembran.
        
        Args:
            cochlea_length_mm: Länge der Cochlea in mm
            f_min_hz: Untere Resonanzfrequenz (apikal)
            f_max_hz: Obere Resonanzfrequenz (basal)
            n_positions: Anzahl der diskretisierten Positionen
        """
        self.L = cochlea_length_mm
        self.f_min = f_min_hz
        self.f_max = f_max_hz
        self.n_positions = n_positions
        
        # Räumliche Diskretisierung
        self.positions = np.linspace(0, self.L, n_positions)
        
        # Greenwood-Frequenzkarte für alle Positionen
        self.resonance_frequencies = self._greenwood_map(self.positions)
    
    def _greenwood_map(self, x: np.ndarray) -> np.ndarray:
        """
        Greenwood's cochlear frequency map (Eq. 10.4).
        
        f(x) = f_base * (10^(1 - x/L) - 0.5)
        
        Args:
            x: Position(en) entlang der Cochlea in mm
        
        Returns:
            Resonanzfrequenz(en) in Hz
        """
        freqs = self.f_max * np.exp(
            -x / self.L * np.log(self.f_max / self.f_min)
        )
        return np.clip(freqs, self.f_min, self.f_max)
    
    def mass_per_length(self, x: float) -> float:
        """
        Massenbelegung m(x) der Basilarmembran.
        
        Nimmt logarithmisch von basal zu apikal zu (empirisches Modell).
        
        Args:
            x: Position in mm
        
        Returns:
            Masse pro Länge in kg/m
        """
        m_basal = 0.2  # kg/m bei x=0 (basal)
        m_apical = 1.0  # kg/m bei x=L (apikal)
        
        # Logarithmische Interpolation
        m = m_basal * (m_apical / m_basal) ** (x / self.L)
        return m
    
    def stiffness_per_length(self, x: float) -> float:
        """
        Steifigkeit k(x) der Basilarmembran.
        
        Nimmt exponentiell von basal zu apikal ab (Hartley et al., 1992).
        
        Args:
            x: Position in mm
        
        Returns:
            Steifigkeit pro Länge in N/m²
        """
        k_basal = 1e8  # N/m² bei x=0 (basal)
        k_apical = 1e6  # N/m² bei x=L (apikal)
        
        # Exponentielle Abnahme
        k = k_basal * (k_apical / k_basal) ** (x / self.L)
        return k
    
    def damping_per_length(self, x: float) -> float:
        """
        Dämpfungskoeffizient c(x).
        
        Wird durch die cochleäre Flüssigkeit und die Steifigkeit bestimmt.
        
        Args:
            x: Position in mm
        
        Returns:
            Dämpfungskoeffizient in N*s/m²
        """
        # Qualitätsfaktor ist basal niedrig, apikal hoch
        q_basal = 1.5
        q_apical = 8.0
        q = q_basal + (q_apical - q_basal) * (x / self.L)
        
        # c = k / (2 * Q) (aus Q = sqrt(k/m) / c)
        m = self.mass_per_length(x)
        k = self.stiffness_per_length(x)
        omega_r = np.sqrt(k / m)
        c = omega_r * m / q
        return c
    
    def quality_factor(self, x: float) -> float:
        """
        Berechne den Gütefaktor Q an Position x.
        
        Q(x) = ω_r(x) * m(x) / c(x)
        
        Args:
            x: Position in mm
        
        Returns:
            Q-Faktor (dimensionslos)
        """
        m = self.mass_per_length(x)
        c = self.damping_per_length(x)
        k = self.stiffness_per_length(x)
        
        omega_r = np.sqrt(k / m)
        q = omega_r * m / c
        return q
    
    def mechanical_impedance(
        self, x: float, freq_hz: float
    ) -> complex:
        """
        Berechne die mechanische Impedanz Z_BM (Eq. 10.3).
        
        Z = (k - ω²*m) / (-i*ω) + c
        
        Args:
            x: Position in mm
            freq_hz: Frequenz in Hz
        
        Returns:
            Komplexe mechanische Impedanz in N*s/m
        """
        omega = 2 * np.pi * freq_hz
        m = self.mass_per_length(x)
        k = self.stiffness_per_length(x)
        c = self.damping_per_length(x)
        
        z = (k - omega**2 * m) / (-1j * omega) + c
        return z
    
    def amplitude_response(
        self, x: float, freq_hz: float
    ) -> float:
        """
        Berechne die Auslenkungsamplitude bei gegebener Frequenz.
        
        Für ein Pressungseingang von 1 Pa.
        
        Args:
            x: Position in mm
            freq_hz: Frequenz in Hz
        
        Returns:
            Relative Auslenkungsamplitude (normalisiert)
        """
        z = self.mechanical_impedance(x, freq_hz)
        # Admittanz Y = 1/Z
        y = 1.0 / z
        # Normalisiere auf maximale Auslenkung bei dieser Position
        freq_res = self.resonance_frequencies[np.argmin(np.abs(self.positions - x))]
        z_res = self.mechanical_impedance(x, freq_res)
        y_res = 1.0 / z_res
        
        return abs(y / y_res)
    
    def frequency_response_at_position(
        self, x: float, frequencies_hz: np.ndarray
    ) -> np.ndarray:
        """
        Berechne die Frequenzantwort an einer Position.
        
        Args:
            x: Position in mm
            frequencies_hz: Array von Frequenzen in Hz
        
        Returns:
            Normalisierte Amplituden (0 bis 1)
        """
        responses = np.array([
            self.amplitude_response(x, f) for f in frequencies_hz
        ])
        return responses / np.max(responses)


class CochlearModel:
    """
    Vereinigtes Modell der Cochlea mit allen Komponenten.
    """
    
    def __init__(
        self,
        cochlea_length_mm: float = 35.0,
        f_min_hz: float = 20.0,
        f_max_hz: float = 20000.0,
        n_positions: int = 50,
    ):
        """
        Initialisiere das Cochlea-Modell.
        
        Args:
            cochlea_length_mm: Cochlea-Länge
            f_min_hz: Untere Grenzfrequenz
            f_max_hz: Obere Grenzfrequenz
            n_positions: Anzahl der Diskretisierungspositionen
        """
        self.basilar = BasilarMembrane(
            cochlea_length_mm, f_min_hz, f_max_hz, n_positions
        )
    
    def ohc_amplification(
        self, basilar_displacement_um: float
    ) -> float:
        """
        Modelliere die Verstärkung durch äußere Haarzellen.
        
        Nichtlineare sigmoidale Funktion (Eq. 10.13).
        
        Args:
            basilar_displacement_um: Auslenkung der Basilarmembran in µm
        
        Returns:
            Multiplikativer Verstärkungsfaktor
        """
        # Parameter der OHC-Transduktion
        d_thresh = 0.1  # Schwellenwert in µm
        sigma_d = 0.05  # Steigung
        
        # Verstärkung: 1 + A*tanh((x - x_th) / sigma)
        max_gain = 3.0  # Maximale Verstärkung 3x
        gain = 1.0 + max_gain * np.tanh(
            (basilar_displacement_um - d_thresh) / sigma_d
        )
        return np.clip(gain, 0.1, max_gain)
    
    def ihc_response(
        self, basilar_displacement_um: float
    ) -> float:
        """
        Modelliere die Antwort der inneren Haarzellen (IHCs).
        
        Sigmoide Eingabe-Ausgabe-Funktion (Eq. 10.14).
        
        Args:
            basilar_displacement_um: Auslenkung in µm
        
        Returns:
            Normalisierte IHC-Antwort (0 bis 1)
        """
        # Sigmoid mit Schwellenwert
        d_thresh = 0.05  # µm
        sigma = 0.02  # µm
        
        response = 1.0 / (
            1.0 + np.exp(-(basilar_displacement_um - d_thresh) / sigma)
        )
        return response
    
    def afferent_firing_rate(
        self, ihc_response: float, max_rate: float = 250.0
    ) -> float:
        """
        Konvertiere IHC-Antwort in Feuerrate afferenter Nervenfasern.
        
        Args:
            ihc_response: Normalisierte IHC-Antwort
            max_rate: Maximale Feuerrate in spikes/s
        
        Returns:
            Feuerrate in spikes/s
        """
        return ihc_response * max_rate
    
    def neurotransmitter_concentration(
        self,
        ihc_response: float,
        dt: float = 0.001,
        release_rate: float = 1000.0,
        reuptake_rate: float = 100.0,
    ) -> float:
        """
        Modelliere die Neurotransmitter-Konzentration an der Synapse.
        
        Differenzialgleichung (Eq. 10.15):
            d[NT]/dt = r_release * I_Ca - λ_reuptake * [NT]
        
        Args:
            ihc_response: IHC-Antwort (proportional I_Ca)
            dt: Zeitschritt
            release_rate: Freisetzungsrate
            reuptake_rate: Rückaufnahmerate
        
        Returns:
            Neurotransmitter-Konzentration (arbitrary units)
        """
        # Einfaches exponentielles Modell
        k_release = release_rate * ihc_response
        k_reuptake = reuptake_rate
        
        # Stationärer Wert
        nt_steady = k_release / k_reuptake
        return nt_steady
    
    def critical_band_erb(self, freq_hz: float) -> float:
        """
        Berechne die äquivalente rektangulare Bandbreite (ERB).
        
        Menschliche Formel (Glasberg & Moore 1990):
        ERB(f) = 24.7 + 0.108*f Hz
        
        Args:
            freq_hz: Frequenz in Hz
        
        Returns:
            ERB in Hz
        """
        erb = 24.7 + 0.108 * freq_hz
        return erb
    
    def get_summary(self) -> str:
        """Gib eine Zusammenfassung aus."""
        return (
            f"Cochlear Model:\n"
            f"  Length: {self.basilar.L} mm\n"
            f"  Frequency range: {self.basilar.f_min:.1f} Hz - "
            f"{self.basilar.f_max:.1f} Hz\n"
            f"  Discretization points: {self.basilar.n_positions}\n"
        )
