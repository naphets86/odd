"""
Analog-to-Digital Conversion with Cochlear Filterbank.

Implementiert:
- Sample-and-Hold (Eq. 10.16)
- Quantisierung (Eq. 10.17-10.19)
- Sigma-Delta-Modulation (Eq. 10.20)
- Digitale Nachverarbeitung und Normalisierung
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal as scipy_signal


class AudioConverter:
    """
    A/D-Wandler mit Cochlea-inspirierter Filterbank.
    """
    
    def __init__(
        self,
        sample_rate_hz: float = 96000.0,
        bit_depth: int = 24,
        v_max_v: float = 10.0,
        use_sigma_delta: bool = False,
        oversampling_factor: int = 64,
    ):
        """
        Initialisiere den A/D-Wandler.
        
        Args:
            sample_rate_hz: Abtastrate in Hz
            bit_depth: Bit-Tiefe (16, 24, 32)
            v_max_v: Maximale Eingangsspannung in Volt
            use_sigma_delta: Nutze Sigma-Delta-Modulation
            oversampling_factor: Überabtastungsfaktor für SDM
        """
        if bit_depth not in [16, 24, 32]:
            raise ValueError("bit_depth must be 16, 24, or 32")
        
        self.fs = sample_rate_hz
        self.bit_depth = bit_depth
        self.v_max = v_max_v
        self.use_sdm = use_sigma_delta
        self.osf = oversampling_factor
        
        # Quantisierungsparameter
        self.n_levels = 2 ** bit_depth
        self.delta = 2 * v_max_v / self.n_levels  # Quantisierungsstufe (Eq. 10.17)
        
        # Quantisierungsrauschen-Leistung (Eq. 10.19)
        self.snr_quantization_db = 6.02 * bit_depth + 1.76  # dB
    
    def sample_and_hold(
        self, analog_signal: np.ndarray, t_analog: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample-and-Hold Stufe (Eq. 10.16).
        
        Args:
            analog_signal: Analoges Eingangssignal
            t_analog: Zeit-Vektor des analogen Signals
        
        Returns:
            Tupel (digitale Samples, Zeit-Vektor)
        """
        # Berechne Abtast-Indizes
        t_digital = np.arange(
            0, t_analog[-1], 1.0 / self.fs
        )
        
        # Interpoliere auf digitale Zeitpunkte
        samples = np.interp(t_digital, t_analog, analog_signal)
        
        return samples, t_digital
    
    def quantize(self, samples: np.ndarray) -> np.ndarray:
        """
        Quantisiere die Samples.
        
        Formula (Eq. 10.17): y[k] = Δ * floor(x[k]/Δ + 0.5)
        
        Args:
            samples: Digitale Samples (floating-point)
        
        Returns:
            Quantisierte Samples (integer-äquivalent)
        """
        # Begrenzen auf [-V_max, +V_max]
        clipped = np.clip(samples, -self.v_max, self.v_max)
        
        # Quantisierung mit Runden
        quantized = self.delta * np.round(clipped / self.delta)
        
        return quantized
    
    def quantization_error(self, samples: np.ndarray, quantized: np.ndarray) -> np.ndarray:
        """
        Berechne den Quantisierungsfehler.
        
        Args:
            samples: Ursprüngliche Samples
            quantized: Quantisierte Samples
        
        Returns:
            Fehler-Signal
        """
        return samples - quantized
    
    def quantization_snr(self, signal_power: Optional[float] = None) -> float:
        """
        Berechne das Signal-zu-Quantisierungsrausch-Verhältnis.
        
        Formula (Eq. 10.19): SNR = P_signal / P_quantization
        
        Args:
            signal_power: Optionale Signalleistung (normalisiert auf 1.0 wenn None)
        
        Returns:
            SNR in dB
        """
        if signal_power is None:
            return self.snr_quantization_db
        
        # Quantisierungsrauschen-Leistung
        p_quantization = self.delta**2 / 12.0
        
        snr = 10 * np.log10(signal_power / p_quantization)
        return snr
    
    def sigma_delta_modulation(
        self, samples: np.ndarray, n_bits: int = 1
    ) -> np.ndarray:
        """
        Sigma-Delta-Modulation für höhere effektive Auflösung.
        
        Args:
            samples: Eingangssignale
            n_bits: Bit-Tiefe der Ausgabe (1 für ein Bit)
        
        Returns:
            Gepulste Ausgabe mit hoher Abtastrate
        """
        # Überabgetastete Version
        osf = self.osf
        samples_os = np.repeat(samples, osf)
        
        # Sigma-Delta-Loop mit Integrator
        output = np.zeros_like(samples_os)
        integrator_state = 0.0
        
        for i, sample in enumerate(samples_os):
            # Integrator
            integrator_state += sample
            
            # Komparator
            if integrator_state > 0:
                output[i] = 1.0
                integrator_state -= 1.0
            else:
                output[i] = -1.0
                integrator_state += 1.0
        
        return output
    
    def decimation_filter(
        self, os_signal: np.ndarray, decimation_factor: int
    ) -> np.ndarray:
        """
        Dezimations-Tiefpass-Filter für Sigma-Delta.
        
        Args:
            os_signal: Überabgetastetes Signal
            decimation_factor: Dezimationsfaktor
        
        Returns:
            Dezimiertes und gefiltertes Signal
        """
        # Design eines Tiefpass-Filters
        nyquist_freq = self.fs / 2.0 / decimation_factor
        normalized_cutoff = nyquist_freq / (self.fs * self.osf / 2.0)
        normalized_cutoff = min(0.99, normalized_cutoff)
        
        # FIR-Filter (Hamming-Fenster)
        h = scipy_signal.firwin(
            64, normalized_cutoff, window='hamming'
        )
        
        # Filterung und Dezimation
        filtered = scipy_signal.convolve(
            os_signal, h, mode='same'
        )
        decimated = filtered[::decimation_factor]
        
        return decimated
    
    def convert_analog_to_digital(
        self, analog_signal: np.ndarray, t_analog: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Führe eine vollständige A/D-Wandlung durch.
        
        Args:
            analog_signal: Analoges Eingangssignal
            t_analog: Zeit-Vektor
        
        Returns:
            Tupel (digitale Samples, Zeit-Vektor, Metadaten)
        """
        metadata = {}
        
        # Sample-and-Hold
        samples, t_digital = self.sample_and_hold(analog_signal, t_analog)
        metadata['after_sampling'] = {
            'n_samples': len(samples),
            'fs': self.fs,
            't_min': t_digital[0],
            't_max': t_digital[-1],
        }
        
        if self.use_sdm:
            # Sigma-Delta-Modulation
            os_output = self.sigma_delta_modulation(samples)
            
            # Dezimation
            digital = self.decimation_filter(os_output, self.osf)
            
            metadata['sdm'] = {
                'oversampling_factor': self.osf,
                'os_output_length': len(os_output),
            }
        else:
            # Standard-Quantisierung
            digital = self.quantize(samples)
            
            metadata['quantization'] = {
                'bit_depth': self.bit_depth,
                'snr_db': self.snr_quantization_db,
                'delta_v': self.delta,
            }
        
        metadata['final'] = {
            'n_samples': len(digital),
            'min': np.min(digital),
            'max': np.max(digital),
            'mean': np.mean(digital),
            'std': np.std(digital),
        }
        
        return digital, t_digital[:len(digital)], metadata
    
    def estimate_snr(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Schätze das Signal-zu-Rausch-Verhältnis.
        
        Args:
            original: Ursprüngliches Signal
            quantized: Quantisiertes Signal
        
        Returns:
            SNR in dB
        """
        error = original - quantized
        
        signal_power = np.mean(original**2)
        noise_power = np.mean(error**2)
        
        if noise_power == 0:
            return np.inf
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def memory_usage(self, n_samples: int) -> Tuple[float, str]:
        """
        Berechne den Speicherverbrauch für digitalisierte Daten.
        
        Args:
            n_samples: Anzahl der Samples
        
        Returns:
            Tupel (Größe in MB, formatierte Zeichenkette)
        """
        bytes_per_sample = self.bit_depth // 8
        total_bytes = n_samples * bytes_per_sample
        total_mb = total_bytes / (1024 * 1024)
        
        formatted = f"{total_mb:.2f} MB ({total_bytes:,} bytes)"
        return total_mb, formatted
    
    def data_rate_mbps(self) -> float:
        """
        Berechne die Datenrate in Mbit/s.
        
        Returns:
            Datenrate in Mbps
        """
        return self.fs * self.bit_depth / 1e6
    
    def summary(self) -> str:
        """Gib eine Zusammenfassung aus."""
        return (
            f"AudioConverter:\n"
            f"  Sample Rate: {self.fs:.0f} Hz\n"
            f"  Bit Depth: {self.bit_depth} bits\n"
            f"  V_max: {self.v_max:.1f} V\n"
            f"  Quantization Levels: {self.n_levels:,}\n"
            f"  Quantization Step: {self.delta*1e6:.3f} µV\n"
            f"  Theoretical SNR: {self.snr_quantization_db:.1f} dB\n"
            f"  Data Rate: {self.data_rate_mbps():.1f} Mbps\n"
            f"  Sigma-Delta: {'Yes' if self.use_sdm else 'No'}\n"
        )
