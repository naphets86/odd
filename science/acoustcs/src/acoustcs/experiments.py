"""Numerische Experimente fuer die Modelle des ``acoustcs``-Pakets.

Das Modul ist bewusst kein Testmodul. Es dient dazu, die Modelle mit kleinen,
reproduzierbaren Signalen auszufuehren und ihre wichtigsten Ergebnisse zu
demonstrieren.

Aus dem Projektverzeichnis ausfuehren:

    python -m acoustcs.experiments
    python -m acoustcs.experiments --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio_converter import AudioConverter
from .cochlear_model import BasilarMembrane, CochlearModel
from .feline_auditory import FelineAuditory, Species
from .filterbank import CochlearFilterBank, GammaToneFilter
from .hearing_comparison import HearingComparison
from .spatial_localization import SpatialLocalization


def experiment_hearing_comparison() -> dict[str, Any]:
    """Vergleiche Frequenzumfang, Selektivitaet und Lokalisierung der Arten."""
    comparison = HearingComparison()
    systems = {
        species.name.lower(): {
            "frequency_range_hz": system.frequency_bandwidth()[:2],
            "octaves": system.frequency_range_octaves(),
            "phase_locking_khz": system.phase_locking_frequency() / 1e3,
            "localization_deg": (
                system.localization_accuracy_fixed(),
                system.localization_accuracy_mobile(),
            ),
        }
        for species, system in comparison.systems.items()
    }
    systems["cat_vs_dog_superiority"] = comparison.geometric_mean_superiority()[
        Species.CAT.value
    ]

    print("\n=== Hoervergleich ===")
    print(comparison.summary_table())
    return systems


def experiment_spatial_localization() -> dict[str, Any]:
    """Untersuche ITD, ILD und den Einfluss eines Hoerverlusts."""
    localization = SpatialLocalization(Species.CAT)
    azimuths = np.linspace(-90.0, 90.0, 13)
    itds_us = np.array(
        [localization.itd_woodworth_schlosberg(float(angle)) * 1e6 for angle in azimuths]
    )
    frequencies_hz = np.array([500.0, 4_000.0, 8_000.0, 16_000.0])
    ild_db = np.array(
        [localization.ild_frequency_dependent(float(freq), 60.0) for freq in frequencies_hz]
    )
    jnd_normal, jnd_loss = localization.hearing_loss_effect(4_000.0, 40.0)

    result = {
        "azimuth_deg": azimuths,
        "itd_us": itds_us,
        "ild_db_at_60_deg": dict(zip(frequencies_hz, ild_db)),
        "itd_max_us": localization.itd_maximum() * 1e6,
        "localization_accuracy_deg": localization.localization_accuracy(10.0),
        "jnd_us": (jnd_normal * 1e6, jnd_loss * 1e6),
        "spatial_entropy_bits": localization.informational_entropy_spatial(
            n_sources=36, hearing_loss_db=40.0
        ),
    }

    print("\n=== Raumlokalisierung ===")
    print(f"Maximale ITD: {result['itd_max_us']:.2f} us")
    print(f"Lokalisierungsgenauigkeit bei 10 dB SNR: "
          f"{result['localization_accuracy_deg']:.2f} Grad")
    print(f"ILD bei 60 Grad: {result['ild_db_at_60_deg']}")
    print(f"ITD-JND normal / mit 40 dB Verlust: {result['jnd_us']}")
    return result


def experiment_cochlear_mechanics() -> dict[str, Any]:
    """Erzeuge eine tonotope Frequenzantwort der Basilarmembran."""
    membrane = BasilarMembrane(n_positions=80)
    position_mm = 17.5
    frequencies_hz = np.logspace(2, 4.2, 180)
    response = membrane.frequency_response_at_position(position_mm, frequencies_hz)
    model = CochlearModel(n_positions=40)
    displacements_um = np.linspace(0.0, 0.3, 7)
    ihc = np.array([model.ihc_response(value) for value in displacements_um])

    result = {
        "position_mm": position_mm,
        "resonance_hz": float(
            membrane.resonance_frequencies[np.argmin(np.abs(membrane.positions - position_mm))]
        ),
        "frequencies_hz": frequencies_hz,
        "response": response,
        "displacements_um": displacements_um,
        "ohc_gain": np.array([model.ohc_amplification(value) for value in displacements_um]),
        "ihc_response": ihc,
        "erb_at_4khz_hz": model.critical_band_erb(4_000.0),
    }

    print("\n=== Cochlea-Mechanik ===")
    print(f"Resonanz bei {position_mm:.1f} mm: {result['resonance_hz']:.1f} Hz")
    print(f"ERB bei 4 kHz: {result['erb_at_4khz_hz']:.1f} Hz")
    print(f"IHC-Antwort bei 0.3 um: {ihc[-1]:.3f}")
    return result


def experiment_filterbank() -> dict[str, Any]:
    """Analysiere einen Zwei-Ton-Impuls mit Filter und Filterbank."""
    sample_rate_hz = 48_000.0
    duration_s = 0.25
    time = np.arange(int(sample_rate_hz * duration_s)) / sample_rate_hz
    signal = 0.7 * np.sin(2.0 * np.pi * 800.0 * time)
    signal += 0.35 * np.sin(2.0 * np.pi * 4_000.0 * time)

    single_filter = GammaToneFilter(4_000.0, 500.0, sample_rate_hz=sample_rate_hz)
    filtered = single_filter.filter_signal(signal)
    filterbank = CochlearFilterBank(
        f_min_hz=100.0,
        f_max_hz=12_000.0,
        n_channels=16,
        sample_rate_hz=sample_rate_hz,
    )
    outputs = filterbank.apply_filterbank(signal)
    time_axis, frequency_axis, cochleagram = filterbank.compute_spectrogram(signal)

    result = {
        "time_s": time,
        "signal": signal,
        "filtered_4khz": filtered,
        "center_frequencies_hz": filterbank.center_freqs,
        "channel_rms": np.sqrt(np.mean(outputs**2, axis=1)),
        "spectrogram_time_s": time_axis,
        "spectrogram_frequencies_hz": frequency_axis,
        "cochleagram": cochleagram,
        "tonotopic_position_4khz_mm": filterbank.tonotopic_position(4_000.0),
    }

    print("\n=== Cochlea-Filterbank ===")
    print(f"Aktive Kanaele: {filterbank.n_channels}")
    print(f"Position von 4 kHz: {result['tonotopic_position_4khz_mm']:.2f} mm")
    print(f"Staerkster Kanal: {filterbank.center_freqs[np.argmax(result['channel_rms'])]:.1f} Hz")
    return result


def experiment_audio_conversion() -> dict[str, Any]:
    """Vergleiche Quantisierung und Sigma-Delta-Wandlung."""
    sample_rate_hz = 48_000.0
    time = np.arange(int(sample_rate_hz * 0.1)) / sample_rate_hz
    analog = 2.5 * np.sin(2.0 * np.pi * 1_000.0 * time)
    analog += 0.2 * np.sin(2.0 * np.pi * 7_000.0 * time)

    converter = AudioConverter(sample_rate_hz=sample_rate_hz, bit_depth=16, v_max_v=5.0)
    digital, digital_time, metadata = converter.convert_analog_to_digital(analog, time)
    sigma_delta = AudioConverter(
        sample_rate_hz=sample_rate_hz,
        bit_depth=16,
        v_max_v=5.0,
        use_sigma_delta=True,
        oversampling_factor=8,
    )
    sdm_digital, _, sdm_metadata = sigma_delta.convert_analog_to_digital(analog, time)

    result = {
        "time_s": digital_time,
        "analog": analog,
        "quantized": digital,
        "quantized_snr_db": converter.estimate_snr(analog[: len(digital)], digital),
        "sigma_delta": sdm_digital,
        "metadata": metadata,
        "sigma_delta_metadata": sdm_metadata,
        "data_rate_mbps": converter.data_rate_mbps(),
    }

    print("\n=== A/D-Wandlung ===")
    print(f"Geschaetztes Quantisierungs-SNR: {result['quantized_snr_db']:.2f} dB")
    print(f"Datenrate: {result['data_rate_mbps']:.2f} Mbit/s")
    print(f"Sigma-Delta-Ausgabe: {len(sdm_digital)} Samples")
    return result


def _save_plots(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Speichere die zwei aussagekraeftigsten Experimentgrafiken."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    mechanics = results["cochlear_mechanics"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogx(mechanics["frequencies_hz"], mechanics["response"])
    axis.set(xlabel="Frequenz (Hz)", ylabel="Normierte Antwort", title="Basilarmembran")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "basilar_membrane_response.png", dpi=150)
    plt.close(figure)

    bank = results["filterbank"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    image = axis.imshow(
        bank["cochleagram"],
        aspect="auto",
        origin="lower",
        extent=(
            bank["spectrogram_time_s"][0],
            bank["spectrogram_time_s"][-1],
            bank["spectrogram_frequencies_hz"][0],
            bank["spectrogram_frequencies_hz"][-1],
        ),
    )
    axis.set(xlabel="Zeit (s)", ylabel="Kanal-Frequenz (Hz)", title="Cochleagramm")
    figure.colorbar(image, ax=axis, label="RMS")
    figure.tight_layout()
    figure.savefig(output_dir / "cochleagram.png", dpi=150)
    plt.close(figure)


def _json_default(value: Any) -> Any:
    """Konvertiere NumPy-Werte fuer die JSON-Ausgabe."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Wert ist nicht JSON-serialisierbar: {type(value)!r}")


def save_results_json(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Speichere alle numerischen Experimentdaten als formatiertes JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, default=_json_default, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_results_pdf(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Erzeuge einen kompakten PDF-Bericht mit Grafiken und Befunden."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mechanics = results["cochlear_mechanics"]
    localization = results["spatial_localization"]
    bank = results["filterbank"]
    conversion = results["audio_conversion"]

    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.92, "Experiment-Ergebnisse acoustcs", fontsize=20, weight="bold")
        figure.text(0.08, 0.87, "Numerische Auswertung der biologischen Akustikmodelle", fontsize=12)
        figure.text(
            0.08,
            0.78,
            "Untersucht wurden Hoervergleich, binaurale Raumlokalisierung, "
            "Cochlea-Mechanik, Filterbank und A/D-Wandlung.\n"
            "Die Zahlen stammen direkt aus experiments.py; sie sind Modellresultate "
            "und keine Messdaten.",
            fontsize=11,
            va="top",
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        axes[0].plot(localization["azimuth_deg"], localization["itd_us"], marker="o")
        axes[0].set(xlabel="Azimut (Grad)", ylabel="ITD (us)", title="Binaurale Zeitdifferenz")
        axes[0].grid(True, alpha=0.25)
        axes[1].bar(
            [str(freq / 1000).rstrip("0").rstrip(".") + " kHz"
             for freq in localization["ild_db_at_60_deg"]],
            list(localization["ild_db_at_60_deg"].values()),
        )
        axes[1].set(xlabel="Frequenz", ylabel="ILD (dB)", title="Frequenzabhaengige Pegeldifferenz")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        axes[0].semilogx(mechanics["frequencies_hz"], mechanics["response"])
        axes[0].set(xlabel="Frequenz (Hz)", ylabel="Normierte Antwort", title="Basilarmembran")
        axes[0].grid(True, which="both", alpha=0.25)
        axes[1].plot(mechanics["displacements_um"], mechanics["ihc_response"], marker="o")
        axes[1].set(xlabel="Auslenkung (um)", ylabel="IHC-Antwort", title="Innere Haarzellen")
        axes[1].grid(True, alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.27, 6.2))
        image = axis.imshow(
            bank["cochleagram"],
            aspect="auto",
            origin="lower",
            extent=(
                bank["spectrogram_time_s"][0],
                bank["spectrogram_time_s"][-1],
                bank["spectrogram_frequencies_hz"][0],
                bank["spectrogram_frequencies_hz"][-1],
            ),
        )
        axis.set(xlabel="Zeit (s)", ylabel="Kanal-Frequenz (Hz)", title="Cochleagramm")
        figure.colorbar(image, ax=axis, label="RMS")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.92, "Zusammenfassung und Einordnung", fontsize=18, weight="bold")
        summary = (
            f"Hoervergleich: Die Katze erreicht im Modell {results['hearing_comparison']['cat']['frequency_range_hz'][1]:.0f} Hz obere Grenzfrequenz.\n"
            f"Raumlokalisierung: maximale ITD {localization['itd_max_us']:.2f} us; "
            f"40 dB Hoerverlust erhoeht die ITD-JND auf {localization['jnd_us'][1]:.2f} us.\n"
            f"Cochlea: Resonanz bei 17.5 mm etwa {mechanics['resonance_hz']:.1f} Hz; "
            f"ERB bei 4 kHz {mechanics['erb_at_4khz_hz']:.1f} Hz.\n"
            f"Filterbank: 16 Kanaele zwischen 100 Hz und 12 kHz; die 4-kHz-Komponente "
            f"liegt bei {bank['tonotopic_position_4khz_mm']:.2f} mm.\n"
            f"A/D-Wandlung: geschaetztes Quantisierungs-SNR "
            f"{conversion['quantized_snr_db']:.2f} dB bei 16 Bit und 48 kHz.\n\n"
            "Die Ergebnisse illustrieren die interne Konsistenz der Implementierung. "
            "Sie ersetzen keine kalibrierten Messungen, da Parameter, Frequenzkarten und "
            "biologische Vergleichswerte modellhaft vorgegeben sind."
        )
        figure.text(0.08, 0.84, summary, fontsize=11, va="top", linespacing=1.6)
        pdf.savefig(figure)
        plt.close(figure)


def run_all_experiments(save_plots: bool = False, output_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Fuehre alle Experimente aus und gebe ihre Rohdaten zurueck."""
    results = {
        "hearing_comparison": experiment_hearing_comparison(),
        "spatial_localization": experiment_spatial_localization(),
        "cochlear_mechanics": experiment_cochlear_mechanics(),
        "filterbank": experiment_filterbank(),
        "audio_conversion": experiment_audio_conversion(),
    }
    if save_plots:
        _save_plots(results, output_dir or Path("experiment_output"))
    return results


def main() -> None:
    """Kommandozeileneinstieg fuer die Experimente."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_results_dir = Path(__file__).resolve().parent / "results"
    parser.add_argument("--plot", action="store_true", help="Zusaetzliche PNG-Grafiken speichern")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help="Zielordner fuer JSON und PDF",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_output"),
        help="Zielordner fuer Grafiken",
    )
    args = parser.parse_args()
    results = run_all_experiments(save_plots=args.plot, output_dir=args.output_dir)
    save_results_json(results, args.results_dir / "experiment_results.json")
    save_results_pdf(results, args.results_dir / "experiment_results.pdf")
    print(f"\nErgebnisse gespeichert in: {args.results_dir}")


if __name__ == "__main__":
    main()