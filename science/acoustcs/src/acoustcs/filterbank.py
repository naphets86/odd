"""
Cochlear Filterbank: Gammatone-Filter und diskretisierte Filterbank.

Implementiert:
- Gammatone-Filter im Zeit- und Frequenzbereich (Eq. 10.7-10.8)
- Bandbreitenfunktion (Eq. 10.9)
- Filterbank-Ausgänge (Eq. 10.10)
- Digitale IIR-Filter (Eq. 10.11)
"""

from typing import Tuple, Optional, List
import numpy as np
from scipy import signal, fft
from dataclasses import dataclass


@dataclass
class FilterParameters:
    """Parameter für einen einzelnen Gammatone-Filter."""
    
    center_freq_hz: float
    bandwidth_hz: float
    order: int = 4
    q_factor: float = 1.0


class GammaToneFilter:
    """
    Gammatone-Filter mit konfigurierbaren Parametern.
    
    Zeit-Bereich-Antwort (Eq. 10.7):
        g(t) = A * t^(n-1) * exp(-2π*B*Δf*t) * cos(2π*f_c*t + φ)
    
    Frequenz-Bereich (Eq. 10.8):
        G(f) = (A * (2π*B*Δf)^n) / ((f - f_c + i*B*Δf)^n * (f + f_c + i*B*Δf)^n)
    """
    
    def __init__(
        self,
        center_freq_hz: float,
        bandwidth_hz: float,
        order: int = 4,
        sample_rate_hz: float = 96000.0,
    ):
        """
        Initialisiere einen Gammatone-Filter.
        
        Args:
            center_freq_hz: Mittenfrequenz
            bandwidth_hz: Bandbreite
            order: Filterordnung (typisch 4)
            sample_rate_hz: Abtastrate
        """
        self.f_c = center_freq_hz
        self.bandwidth = bandwidth_hz
        self.order = order
        self.fs = sample_rate_hz
        
        # Berechne Filter-Koeffizienten für digitale Implementierung
        self._compute_coefficients()
    
    def _compute_coefficients(self) -> None:
        """Berechne IIR-Filter-Koeffizienten via Bilinear-Transformation."""
        # Normalisierte Frequenz
        wn = 2 * self.f_c / self.fs
        
        # Design eines Butterworth-Bandpass-Filters
        # Der Gammatone wird hier durch einen äquivalenten Butterworth approximiert
        sos = signal.butter(
            self.order,
            [wn - self.bandwidth / self.fs, wn + self.bandwidth / self.fs],
            btype='band',
            output='sos'
        )
        self.sos = sos
    
    def filter_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Wende den Filter auf ein Signal an.
        
        Args:
            x: Eingangssignal (1D-Array)
        
        Returns:
            Gefiltertes Signal
        """
        return signal.sosfilt(self.sos, x)
    
    def frequency_response(
        self, frequencies_hz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechne die Frequenzantwort des Filters.
        
        Args:
            frequencies_hz: Frequenzwerte zum Evaluieren
        
        Returns:
            Tupel (Magnitude, Phase) in dB und Radian
        """
        w = 2 * np.pi * frequencies_hz / self.fs
        w, h = signal.sosfreqz(self.sos, w)
        
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        phase = np.angle(h)
        
        return frequencies_hz, magnitude_db, phase
    
    def impulse_response(
        self, duration_ms: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechne die Impulsantwort des Filters.
        
        Args:
            duration_ms: Dauer der Impulsantwort in ms
        
        Returns:
            Tupel (Zeit-Array, Impulsantwort)
        """
        n_samples = int(duration_ms * self.fs / 1000.0)
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0
        
        h = self.filter_signal(impulse)
        t = np.arange(n_samples) / self.fs
        
        return t, h
    
    def group_delay(self) -> float:
        """Berechne die Gruppenlaufzeit des Filters."""
        w, gd = signal.group_delay(self.sos, fs=self.fs)
        # Rückgabe des Mittelwerts in ms
        return np.mean(gd) * 1000.0 / self.fs
    
    def summary(self) -> str:
        """Gib eine Zusammenfassung aus."""
        return (
            f"GammaToneFilter:\n"
            f"  Center Frequency: {self.f_c:.1f} Hz\n"
            f"  Bandwidth: {self.bandwidth:.1f} Hz\n"
            f"  Order: {self.order}\n"
            f"  Sample Rate: {self.fs:.0f} Hz\n"
        )


class CochlearFilterBank:
    """
    Filterbank mit mehreren Gammatone-Filtern zur Modellierung der Cochlea.
    
    Implementiert die logarithmische Frequenzkarte nach Greenwood und
    die Diskretisierung (Eq. 10.10).
    """
    
    def __init__(
        self,
        f_min_hz: float = 20.0,
        f_max_hz: float = 20000.0,
        n_channels: int = 128,
        sample_rate_hz: float = 96000.0,
        cochlea_length_mm: float = 35.0,
    ):
        """
        Initialisiere die Cochlear Filterbank.
        
        Args:
            f_min_hz: Untere Frequenzgrenze
            f_max_hz: Obere Frequenzgrenze
            n_channels: Anzahl der Filter-Kanäle
            sample_rate_hz: Abtastrate
            cochlea_length_mm: Cochlea-Länge für Greenwood-Abbildung
        """
        self.f_min = f_min_hz
        self.f_max = f_max_hz
        self.n_channels = n_channels
        self.fs = sample_rate_hz
        self.L = cochlea_length_mm
        
        # Generiere logarithmisch verteilte Kanäle
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Erstelle die Filter-Kanäle mit logarithmischen Mittenfrequenzen."""
        # Logarithmische Frequenzverteilung
        self.center_freqs = np.logspace(
            np.log10(self.f_max),
            np.log10(self.f_min),
            self.n_channels
        )
        
        # Bandbreiten nach Glasberg & Moore
        self.bandwidths = 24.7 + 0.108 * self.center_freqs
        
        # Erstelle Filter
        self.filters: List[GammaToneFilter] = []
        for fc, bw in zip(self.center_freqs, self.bandwidths):
            f = GammaToneFilter(fc, bw, sample_rate_hz=self.fs)
            self.filters.append(f)
    
    def apply_filterbank(self, x: np.ndarray) -> np.ndarray:
        """
        Wende die komplette Filterbank auf ein Signal an.
        
        Args:
            x: Eingangssignal (1D-Array)
        
        Returns:
            2D-Array mit Shape (n_channels, len(x))
                Jede Zeile ist die Ausgabe eines Filters
        """
        outputs = np.zeros((self.n_channels, len(x)))
        
        for i, filt in enumerate(self.filters):
            outputs[i, :] = filt.filter_signal(x)
        
        return outputs
    
    def compute_spectrogram(
        self, x: np.ndarray, window_duration_ms: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Berechne ein Cochleagramm (Zeit-Frequenz-Darstellung).
        
        Args:
            x: Eingangssignal
            window_duration_ms: Fenster-Dauer für Zeitauflösung
        
        Returns:
            Tupel (Zeit-Array, Frequenz-Array, Cochleagramm)
        """
        window_samples = int(window_duration_ms * self.fs / 1000.0)
        hop_samples = window_samples // 2
        
        # Wende Filterbank an
        filtered = self.apply_filterbank(x)
        
        # Berechne Spektrogramm für jeden Kanal
        n_frames = (len(x) - window_samples) // hop_samples + 1
        cochleagram = np.zeros((self.n_channels, n_frames))
        
        for i, frame_idx in enumerate(range(0, len(x) - window_samples, hop_samples)):
            frame = x[frame_idx:frame_idx + window_samples]
            frame_filtered = self.apply_filterbank(frame)
            
            # RMS-Energie pro Kanal
            cochleagram[:, i] = np.sqrt(np.mean(frame_filtered**2, axis=1))
        
        time_axis = np.arange(n_frames) * hop_samples / self.fs
        freq_axis = self.center_freqs
        
        return time_axis, freq_axis, cochleagram
    
    def tonotopic_position(self, freq_hz: float) -> float:
        """
        Berechne die cochleäre Position für eine Frequenz.
        
        Args:
            freq_hz: Frequenz in Hz
        
        Returns:
            Position in mm
        """
        if freq_hz < self.f_min or freq_hz > self.f_max:
            raise ValueError(
                f"Frequency {freq_hz} Hz outside range [{self.f_min}, {self.f_max}] Hz"
            )
        
        # Logarithmic map from the basal high-frequency end to the apical end.
        position = self.L * np.log(self.f_max / freq_hz) / np.log(
            self.f_max / self.f_min
        )
        return position
    
    def frequency_for_position(self, position_mm: float) -> float:
        """
        Berechne die Frequenz für eine cochleäre Position.
        
        Args:
            position_mm: Position in mm
        
        Returns:
            Frequenz in Hz
        """
        if position_mm < 0 or position_mm > self.L:
            raise ValueError(f"Position must be in [0, {self.L}] mm")
        
        freq = self.f_max * np.exp(
            -position_mm / self.L * np.log(self.f_max / self.f_min)
        )
        return float(np.clip(freq, self.f_min, self.f_max))
    
    def get_channel_for_frequency(self, freq_hz: float) -> int:
        """
        Finde den Channel-Index für eine Frequenz.
        
        Args:
            freq_hz: Frequenz in Hz
        
        Returns:
            Channel-Index
        """
        idx = np.argmin(np.abs(self.center_freqs - freq_hz))
        return int(idx)
    
    def summary(self) -> str:
        """Gib eine Zusammenfassung aus."""
        return (
            f"CochlearFilterBank:\n"
            f"  Number of channels: {self.n_channels}\n"
            f"  Frequency range: {self.f_min:.1f} Hz - {self.f_max:.1f} Hz\n"
            f"  Center frequencies (first 5): {self.center_freqs[:5]}\n"
            f"  Center frequencies (last 5): {self.center_freqs[-5:]}\n"
            f"  Sample rate: {self.fs:.0f} Hz\n"
        )
