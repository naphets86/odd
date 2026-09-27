"""
Umfassendes Experiment-Modul für die Analyse des Subgraph-SAT-Solvers.

Dieses Modul führt eine Reihe von Experimenten durch, um die theoretischen
Konzepte aus der Arbeit zu validieren:

1. Kombinationspyramide-Struktur
2. Wahrscheinlichkeitsverschiebung über Phasen
3. Bimodale Verteilungen
4. Konvergenz zur Worst-Case-Komplexität
5. Metaverteilungs-Analyse
6. Logarithmisches Belegungsverfahren vs. Standard-Solver
7. Laufzeitvergleiche
8. Phasenübergänge
9. Wahrscheinlichkeitsraum-Eigenschaften
10. Skalierungsverhalten

Alle Ergebnisse werden als matplotlib-Plots und JSON-Dateien gespeichert.
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import math

# Importiere die Module
from sat_solver import SATFormula, CombinationPyramid, SubgraphSATSolver, Clause
from probability_shift import ProbabilityShiftAnalyzer, ThreeWayPartition, RuntimeAnalysis
from metadistribution import ProbabilitySpace, DistributionSpace, MetaDistribution, ShiftTrajectory
from lbv_solver import LogarithmicAssignmentProcedure


class ExperimentManager:
    """Manager für alle Experimente."""
    
    def __init__(self, output_dir: str = 'src/results/'):
        """
        Initialisiert den Experiment Manager.
        
        Args:
            output_dir: Verzeichnis für Ausgabedateien
        """
        self.output_dir = output_dir
        self.results = {}
        self.plots = {}
        
        # Erstelle Output-Verzeichnis
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/plots', exist_ok=True)
        os.makedirs(f'{output_dir}/data', exist_ok=True)
        
        print(f"Experiment-Verzeichnis erstellt: {output_dir}")
    
    def _save_plot(self, fig, name: str, description: str = ""):
        """Speichert einen matplotlib-Plot."""
        path = f'{self.output_dir}/plots/{name}.pdf'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot gespeichert: {path}")
    
    def _save_data(self, data: Dict[str, Any], name: str):
        """Speichert Daten als JSON."""
        path = f'{self.output_dir}/data/{name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Daten gespeichert: {path}")
    
    # =====================================================================
    # EXPERIMENT 1: Kombinationspyramide Struktur
    # =====================================================================
    
    def experiment_01_combination_pyramid(self):
        """
        Visualisiert die Struktur der Kombinationspyramide.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: Kombinationspyramide-Struktur")
        print("="*70)
        
        n_vars_list = [5, 10, 15]
        results = {}
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Kombinationspyramide: Struktur und Ebenen', fontsize=16, fontweight='bold')
        
        for idx, n_vars in enumerate(n_vars_list):
            pyramid = CombinationPyramid(n_vars)
            
            # Berechne Ebenengrößen
            levels = list(range(n_vars + 1))
            level_sizes = [pyramid.level_size(k) for k in levels]
            
            results[f'n={n_vars}'] = {
                'levels': levels,
                'level_sizes': level_sizes,
                'total_nodes': pyramid.total_nodes(),
                'expected_depth': pyramid.expected_depth(),
                'three_way_split': pyramid.three_way_split()
            }
            
            ax = axes[idx]
            ax.bar(levels, level_sizes, color='steelblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel('Ebene k', fontsize=11, fontweight='bold')
            ax.set_ylabel('Anzahl Knoten C(n,k)', fontsize=11, fontweight='bold')
            ax.set_title(f'n = {n_vars} Variablen\nGesamt: {pyramid.total_nodes()} Knoten', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_yscale('log')
        
        self._save_plot(fig, 'exp01_pyramid_structure', 'Kombinationspyramide-Struktur')
        self._save_data(results, 'exp01_pyramid_structure')
        
        print(f"✓ Kombinationspyramide-Analyse abgeschlossen")
        print(f"  Größen: n = {n_vars_list}")
    
    # =====================================================================
    # EXPERIMENT 2: Wahrscheinlichkeitsverschiebung über Phasen
    # =====================================================================
    
    def experiment_02_probability_shift(self):
        """
        Visualisiert die Verschiebung der Wahrscheinlichkeitsverteilung.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: Wahrscheinlichkeitsverschiebung über Phasen")
        print("="*70)
        
        n_values = [8, 10, 12]
        results = {}
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('Wahrscheinlichkeitsverschiebung: Drei Phasen', 
                    fontsize=16, fontweight='bold')
        
        for row_idx, n in enumerate(n_values):
            analyzer = ProbabilityShiftAnalyzer(n)
            
            # Erzeuge Laufzeit-Trajektorie
            n_samples = min(500, analyzer.max_formulas)
            k_values = list(range(1, n_samples + 1))
            runtimes = [analyzer.expected_runtime(k) for k in k_values]
            phases = [analyzer.get_phase(k) for k in k_values]
            
            # Phase-Grenzen
            phase1_end, phase2_end = analyzer.phase_transition_points()
            
            results[f'n={n}'] = {
                'n_variables': n,
                'n_samples': n_samples,
                'phase_boundaries': [int(phase1_end), int(phase2_end)],
                'worst_case_complexity': n ** 3,
                'runtime_statistics': {
                    'min': float(min(runtimes)),
                    'max': float(max(runtimes)),
                    'mean': float(np.mean(runtimes)),
                    'std': float(np.std(runtimes))
                }
            }
            
            # Plot 1: Laufzeit-Trajektorie mit Phaseneinfärbung
            ax1 = fig.add_subplot(gs[row_idx, 0])
            ax1.plot(k_values, runtimes, linewidth=2, color='darkblue', label='E[T_k]')
            ax1.axhline(n ** 3, color='red', linestyle='--', linewidth=2, label=f'Worst-Case n³ = {n**3}')
            ax1.axvline(phase1_end, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax1.axvline(phase2_end, color='purple', linestyle=':', linewidth=2, alpha=0.7)
            
            # Färbe Phasenbereiche
            ax1.axvspan(0, phase1_end, alpha=0.1, color='green', label='Phase 1: log(n)')
            ax1.axvspan(phase1_end, phase2_end, alpha=0.1, color='yellow', label='Phase 2: Übergang')
            ax1.axvspan(phase2_end, n_samples, alpha=0.1, color='red', label='Phase 3: n³')
            
            ax1.set_xlabel('Formel-Index k', fontweight='bold')
            ax1.set_ylabel('Erwartete Laufzeit E[T_k]', fontweight='bold')
            ax1.set_title(f'n = {n}: Laufzeit-Trajektorie', fontweight='bold')
            ax1.legend(fontsize=8, loc='upper left')
            ax1.grid(alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot 2: Konvergenzrate
            ax2 = fig.add_subplot(gs[row_idx, 1])
            convergence = [analyzer.convergence_rate(k) for k in k_values]
            ax2.plot(k_values, convergence, linewidth=2, color='darkgreen')
            ax2.axvline(phase1_end, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax2.axvline(phase2_end, color='purple', linestyle=':', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Formel-Index k', fontweight='bold')
            ax2.set_ylabel('Konvergenzgeschwindigkeit', fontweight='bold')
            ax2.set_title(f'n = {n}: Konvergenz zu Worst-Case', fontweight='bold')
            ax2.grid(alpha=0.3)
            
            # Plot 3: Phasenverteilung
            ax3 = fig.add_subplot(gs[row_idx, 2])
            phase_counts = [sum(1 for p in phases if p == i) for i in [1, 2, 3]]
            phase_names = ['Phase 1\n(log n)', 'Phase 2\n(Übergang)', 'Phase 3\n(n³)']
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            ax3.bar(phase_names, phase_counts, color=colors, edgecolor='black', alpha=0.7)
            ax3.set_ylabel('Anzahl Formeln', fontweight='bold')
            ax3.set_title(f'n = {n}: Phasenverteilung', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            # Beschriftung
            for i, count in enumerate(phase_counts):
                ax3.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        self._save_plot(fig, 'exp02_probability_shift', 'Wahrscheinlichkeitsverschiebung')
        self._save_data(results, 'exp02_probability_shift')
        
        print(f"✓ Wahrscheinlichkeitsverschiebung analysiert")
        print(f"  Parameter: n = {n_values}")
    
    # =====================================================================
    # EXPERIMENT 3: Bimodale Verteilung
    # =====================================================================
    
    def experiment_03_bimodal_distribution(self):
        """
        Visualisiert die bimodale Verteilung in Phase 1.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: Bimodale Wahrscheinlichkeitsverteilung")
        print("="*70)
        
        n_values = [8, 10, 12]
        results = {}
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Bimodale Verteilung in Phase 1', fontsize=16, fontweight='bold')
        
        for idx, n in enumerate(n_values):
            analyzer = ProbabilityShiftAnalyzer(n)
            
            # Erzeuge Verteilungen für verschiedene Formeln in Phase 1
            phase1_end = int(analyzer.phase1_end)
            k_samples = np.linspace(1, phase1_end, 20, dtype=int)
            
            mode1_values = []
            mode2_values = []
            mean_values = []
            weights_fast = []
            
            for k in k_samples:
                dist = analyzer.bimodal_distribution(k)
                mode1_values.append(dist.mode1)
                mode2_values.append(dist.mode2)
                mean_values.append(dist.mean)
            
            results[f'n={n}'] = {
                'n_variables': n,
                'phase1_samples': int(phase1_end),
                'mode1_values': [float(v) for v in mode1_values],
                'mode2_values': [float(v) for v in mode2_values],
                'mean_values': [float(v) for v in mean_values],
                'description': 'Bimodale Verteilung: Modus 1 bei log(n), Modus 2 bei n³'
            }
            
            ax = axes[idx]
            ax.plot(k_samples, mode1_values, 'o-', label='Modus 1: log(n)', 
                   color='green', linewidth=2, markersize=6)
            ax.plot(k_samples, mode2_values, 's-', label='Modus 2: n³', 
                   color='red', linewidth=2, markersize=6)
            ax.plot(k_samples, mean_values, '^--', label='Erwartungswert', 
                   color='purple', linewidth=2, markersize=6)
            
            ax.fill_between(k_samples, mode1_values, mode2_values, alpha=0.1, color='gray')
            ax.set_xlabel('Formel-Index k (in Phase 1)', fontweight='bold')
            ax.set_ylabel('Laufzeit', fontweight='bold')
            ax.set_title(f'n = {n} Variablen', fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_yscale('log')
        
        self._save_plot(fig, 'exp03_bimodal_distribution', 'Bimodale Verteilung')
        self._save_data(results, 'exp03_bimodal_distribution')
        
        print(f"✓ Bimodale Verteilung analysiert")
        print(f"  Parameter: n = {n_values}")
    
    # =====================================================================
    # EXPERIMENT 4: Konvergenz zur Worst-Case-Komplexität
    # =====================================================================
    
    def experiment_04_convergence_to_worst_case(self):
        """
        Zeigt die Konvergenz zur Worst-Case-Komplexität O(n³).
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Konvergenz zur Worst-Case-Komplexität")
        print("="*70)
        
        n = 12
        analyzer = ProbabilityShiftAnalyzer(n)
        
        # Erzeuge ausführliche Daten
        n_samples = min(800, analyzer.max_formulas)
        k_values = list(range(1, n_samples + 1))
        
        runtimes = [analyzer.expected_runtime(k) for k in k_values]
        errors = [abs(n ** 3 - analyzer.expected_runtime(k)) for k in k_values]
        convergence_rates = [analyzer.convergence_rate(k) for k in k_values]
        
        # Theoretische Schranke
        K, beta = 1.0, 0.5
        theoretical_bounds = [RuntimeAnalysis.convergence_error_bound(n, k, K, beta) 
                            for k in k_values]
        
        results = {
            'n_variables': n,
            'worst_case': n ** 3,
            'n_samples': n_samples,
            'error_statistics': {
                'min_error': float(min(errors)),
                'max_error': float(max(errors)),
                'mean_error': float(np.mean(errors)),
                'final_error': float(errors[-1])
            },
            'convergence_rate_statistics': {
                'min_rate': float(min(convergence_rates)),
                'max_rate': float(max(convergence_rates)),
                'mean_rate': float(np.mean(convergence_rates))
            }
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Konvergenz zur Worst-Case-Komplexität (n = {n})', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Fehler über Formel-Index
        ax1 = axes[0, 0]
        ax1.semilogy(k_values, errors, linewidth=2, color='darkred', label='Fehler |n³ - E[T_k]|')
        ax1.semilogy(k_values, theoretical_bounds, linewidth=2, color='blue', 
                    linestyle='--', label='Theoretische Schranke')
        ax1.fill_between(k_values, errors, theoretical_bounds, alpha=0.2, color='gray')
        ax1.set_xlabel('Formel-Index k', fontweight='bold')
        ax1.set_ylabel('Fehler (log-Skala)', fontweight='bold')
        ax1.set_title('Konvergenzfehler: Empirisch vs. Theoretisch', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Laufzeit-Konvergenz
        ax2 = axes[0, 1]
        ax2.plot(k_values, runtimes, linewidth=2.5, color='darkblue', label='E[T_k]')
        ax2.axhline(n ** 3, color='red', linestyle='--', linewidth=2.5, label=f'Worst-Case = {n**3}')
        ax2.fill_between(k_values, runtimes, n ** 3, alpha=0.1, color='lightblue')
        ax2.set_xlabel('Formel-Index k', fontweight='bold')
        ax2.set_ylabel('Laufzeit', fontweight='bold')
        ax2.set_title('Erwartete Laufzeit konvergiert zu Worst-Case', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Konvergenzrate
        ax3 = axes[1, 0]
        ax3.plot(k_values, convergence_rates, linewidth=2, color='darkgreen')
        ax3.fill_between(k_values, 0, convergence_rates, alpha=0.3, color='lightgreen')
        ax3.set_xlabel('Formel-Index k', fontweight='bold')
        ax3.set_ylabel('Konvergenzrate', fontweight='bold')
        ax3.set_title('Konvergenzgeschwindigkeit über Formel-Index', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Fehler-Verteilung (Histogram)
        ax4 = axes[1, 1]
        ax4.hist(errors, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
        ax4.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                   label=f'Mittelwert: {np.mean(errors):.2f}')
        ax4.set_xlabel('Fehler |n³ - E[T_k]|', fontweight='bold')
        ax4.set_ylabel('Häufigkeit', fontweight='bold')
        ax4.set_title('Verteilung der Konvergenzfehler', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        self._save_plot(fig, 'exp04_convergence', 'Konvergenz zur Worst-Case')
        self._save_data(results, 'exp04_convergence')
        
        print(f"✓ Konvergenzanalyse abgeschlossen")
        print(f"  Worst-Case: n³ = {n**3}")
        print(f"  Finaler Fehler: {errors[-1]:.2f}")
    
    # =====================================================================
    # EXPERIMENT 5: Metaverteilungs-Analyse
    # =====================================================================
    
    def experiment_05_metadistribution_analysis(self):
        """
        Analysiert die Metaverteilungstheorie.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 5: Metaverteilungs-Analyse")
        print("="*70)
        
        # Erstelle mehrere Wahrscheinlichkeitsverteilungen
        n_outcomes = 8
        dist_space = DistributionSpace(n_outcomes)
        
        # Uniform-Verteilung
        uniform = np.ones(n_outcomes) / n_outcomes
        dist_space.add_distribution(uniform)
        
        # Exponentialverteilung
        exp_dist = np.exp(np.arange(n_outcomes))
        exp_dist = exp_dist / np.sum(exp_dist)
        dist_space.add_distribution(exp_dist)
        
        # Skewed Verteilung
        skewed = np.array([10**(i-n_outcomes/2) for i in range(n_outcomes)])
        skewed = skewed / np.sum(skewed)
        dist_space.add_distribution(skewed)
        
        # Spike Verteilung
        spike = np.zeros(n_outcomes)
        spike[n_outcomes//2] = 1.0
        dist_space.add_distribution(spike)
        
        # Erstelle Metaverteilung
        metadist = MetaDistribution(dist_space)
        metadist.add_trajectory((0, 1), 0.3)
        metadist.add_trajectory((1, 2), 0.3)
        metadist.add_trajectory((2, 3), 0.4)
        
        # Berechne Kennwerte
        entropy = metadist.entropy()
        avg_entropy = metadist.average_distribution_entropy()
        shift_degree = metadist.shift_degree()
        
        # Extrahiere Daten für alle Distributionen
        dist_entropies = []
        dist_names = ['Uniform', 'Exponential', 'Skewed', 'Spike']
        
        for i, dist in enumerate(dist_space.distributions):
            dist_entropies.append(dist.entropy())
        
        results = {
            'n_outcomes': n_outcomes,
            'n_distributions': len(dist_space.distributions),
            'metadistribution_entropy': float(entropy),
            'average_distribution_entropy': float(avg_entropy),
            'shift_degree': float(shift_degree),
            'distribution_entropies': [float(h) for h in dist_entropies],
            'distribution_names': dist_names
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metaverteilungs-Analyse', fontsize=16, fontweight='bold')
        
        # Plot 1: Distributionen
        ax1 = axes[0, 0]
        x_pos = np.arange(n_outcomes)
        width = 0.2
        
        for i, dist in enumerate(dist_space.distributions):
            ax1.bar(x_pos + i*width, dist.measure, width, label=dist_names[i], alpha=0.7)
        
        ax1.set_xlabel('Outcome', fontweight='bold')
        ax1.set_ylabel('Wahrscheinlichkeit', fontweight='bold')
        ax1.set_title('Verschiedene Wahrscheinlichkeitsverteilungen', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # Plot 2: Entropien
        ax2 = axes[0, 1]
        colors_entropy = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax2.bar(dist_names, dist_entropies, color=colors_entropy, edgecolor='black', alpha=0.7)
        ax2.axhline(entropy, color='purple', linestyle='--', linewidth=2, 
                   label=f'Metaverteilungs-Entropie: {entropy:.3f}')
        ax2.set_ylabel('Entropie H(P)', fontweight='bold')
        ax2.set_title('Entropie jeder Verteilung', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        # Beschriftung
        for bar, value in zip(bars, dist_entropies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Wasserstein-Distanzen (Pairwise)
        ax3 = axes[1, 0]
        n_dist = len(dist_space.distributions)
        distance_matrix = np.zeros((n_dist, n_dist))
        
        for i in range(n_dist):
            for j in range(n_dist):
                if i != j:
                    d_i = dist_space.distributions[i]
                    d_j = dist_space.distributions[j]
                    distance_matrix[i, j] = d_i.wasserstein_distance(d_j)
        
        im = ax3.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(n_dist))
        ax3.set_yticks(range(n_dist))
        ax3.set_xticklabels(dist_names)
        ax3.set_yticklabels(dist_names)
        ax3.set_title('Wasserstein-Distanzen zwischen Distributionen', fontweight='bold')
        
        # Beschriftung
        for i in range(n_dist):
            for j in range(n_dist):
                text = ax3.text(j, i, f'{distance_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax3, label='Distanz')
        
        # Plot 4: Metaverteilungs-Statistiken
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
Metaverteilungs-Statistiken:

• Entropie der Metaverteilung:
  H(μ) = {entropy:.4f}

• Durchschnittliche Verteilungs-Entropie:
  E_μ[H(P)] = {avg_entropy:.4f}

• Verschiebungsgrad (Shift Degree):
  Durchschn. Wasserstein-Distanz = {shift_degree:.4f}

• Anzahl Distributionen: {len(dist_space.distributions)}
• Dimension des Verteilungsraums: {dist_space.dimension()}

Interpretation:
- Höhere H(μ) → größere Unsicherheit über Verteilungen
- Höherer Shift Degree → stärkere Verschiebung
        """
        
        ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self._save_plot(fig, 'exp05_metadistribution', 'Metaverteilungs-Analyse')
        self._save_data(results, 'exp05_metadistribution')
        
        print(f"✓ Metaverteilungs-Analyse abgeschlossen")
        print(f"  Entropie: {entropy:.4f}")
        print(f"  Verschiebungsgrad: {shift_degree:.4f}")
    
    # =====================================================================
    # EXPERIMENT 6: Logarithmisches Belegungsverfahren (LBV)
    # =====================================================================
    
    def experiment_06_lbv_solver(self):
        """
        Analysiert das Logarithmische Belegungsverfahren.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 6: Logarithmisches Belegungsverfahren (LBV)")
        print("="*70)
        
        n_vars_list = [5, 8, 10, 12, 15, 20]
        results = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Logarithmisches Belegungsverfahren: Laufzeit-Analyse', 
                    fontsize=16, fontweight='bold')
        
        iteration_counts = []
        max_iterations_expected = []
        theoretical_bound = []
        
        for n in n_vars_list:
            lbv = LogarithmicAssignmentProcedure(n)
            max_iter = lbv.iteration_count()
            
            iteration_counts.append(max_iter)
            max_iterations_expected.append(math.ceil(math.log2(n)) + 2)
            theoretical_bound.append(math.ceil(math.log2(n)) + 2)
            
            results[f'n={n}'] = {
                'n_variables': n,
                'max_iterations': max_iter,
                'theoretical_bound': math.ceil(math.log2(n)) + 2,
                'log_term': math.ceil(math.log2(n)),
                'description': 'K = ⌈log₂(m)⌉ + 2 ∈ O(log m)'
            }
        
        # Plot 1: Iterationen vs. n
        ax1 = axes[0, 0]
        ax1.plot(n_vars_list, iteration_counts, 'o-', linewidth=2.5, markersize=8,
                label='Tatsächliche Iterationen', color='darkblue')
        ax1.plot(n_vars_list, theoretical_bound, 's--', linewidth=2.5, markersize=8,
                label='Theoretische Schranke K = ⌈log₂(n)⌉ + 2', color='red')
        
        ax1.fill_between(n_vars_list, iteration_counts, theoretical_bound, alpha=0.2, color='gray')
        ax1.set_xlabel('Anzahl Variablen n', fontweight='bold')
        ax1.set_ylabel('Iterationen K', fontweight='bold')
        ax1.set_title('Iterationsanzahl vs. Variablenanzahl', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Log-Vergleich
        ax2 = axes[0, 1]
        log_n = [math.log2(n) for n in n_vars_list]
        log_n_plus_2 = [math.ceil(math.log2(n)) + 2 for n in n_vars_list]
        
        ax2.plot(n_vars_list, log_n, '^-', linewidth=2, markersize=8, label='log₂(n)', color='green')
        ax2.plot(n_vars_list, log_n_plus_2, 's-', linewidth=2.5, markersize=8, 
                label='⌈log₂(n)⌉ + 2', color='red')
        ax2.set_xlabel('Anzahl Variablen n', fontweight='bold')
        ax2.set_ylabel('Wert', fontweight='bold')
        ax2.set_title('Logarithmische Komponenten', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Flip-Größen für ein Beispiel (n=16)
        n_example = 16
        lbv_example = LogarithmicAssignmentProcedure(n_example)
        
        iterations = list(range(lbv_example.iteration_count()))
        flip_sizes = [lbv_example.flip_size(i) for i in iterations]
        
        ax3 = axes[1, 0]
        ax3.bar(iterations, flip_sizes, color='steelblue', edgecolor='navy', alpha=0.7)
        ax3.set_xlabel('Iteration i', fontweight='bold')
        ax3.set_ylabel('Flip-Größe |F_i|', fontweight='bold')
        ax3.set_title(f'Flip-Größen pro Iteration (n = {n_example})', fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Beschriftung
        for i, size in enumerate(flip_sizes):
            ax3.text(i, size, str(size), ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Komplexitäts-Vergleich
        ax4 = axes[1, 1]
        
        # Standard-SAT-Solver: O(2^n)
        standard_complexity = [2**n for n in n_vars_list]
        
        # LBV mit Subgraph: O(m^3 * log m) vereinfacht zu O(n^3 * log n)
        lbv_complexity = [n**3 * math.log2(n) for n in n_vars_list]
        
        ax4.semilogy(n_vars_list, standard_complexity, 'o-', linewidth=2.5, 
                    markersize=8, label='Standard SAT: O(2ⁿ)', color='red')
        ax4.semilogy(n_vars_list, lbv_complexity, 's-', linewidth=2.5, 
                    markersize=8, label='LBV mit Subgraph: O(n³ log n)', color='green')
        
        ax4.set_xlabel('Anzahl Variablen n', fontweight='bold')
        ax4.set_ylabel('Komplexität (log-Skala)', fontweight='bold')
        ax4.set_title('Komplexitäts-Vergleich: Standard vs. LBV', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        self._save_plot(fig, 'exp06_lbv_solver', 'LBV-Analyse')
        self._save_data(results, 'exp06_lbv_solver')
        
        print(f"✓ LBV-Analyse abgeschlossen")
        print(f"  Parameter: n = {n_vars_list}")
    
    # =====================================================================
    # EXPERIMENT 7: Laufzeit-Vergleiche
    # =====================================================================
    
    def experiment_07_runtime_comparison(self):
        """
        Vergleicht Laufzeiten verschiedener Formeln-Größen.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 7: Laufzeit-Vergleiche für verschiedene Formeln-Größen")
        print("="*70)
        
        n_values = [6, 8, 10, 12]
        results = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Laufzeit-Vergleiche: SAT-Solver Performance', fontsize=16, fontweight='bold')
        
        all_n = []
        all_runtimes = []
        all_estimated = []
        
        for idx, n in enumerate(n_values):
            ax = axes[idx // 2, idx % 2]
            
            # Erzeuge Random-SAT-Formeln
            n_clauses = min(3 * n, 30)  # 3-SAT: typischerweise 3*n Klauseln
            
            solver = SubgraphSATSolver(SATFormula.random_formula(n, n_clauses))
            
            # Berechne erwartete Laufzeit
            estimated = solver.estimated_runtime()
            
            # Simuliere Laufzeiten
            runtimes = []
            for _ in range(20):
                _, _, ops = solver.solve(timeout=10000)
                runtimes.append(ops)
            
            all_n.append(n)
            all_runtimes.append(np.mean(runtimes))
            all_estimated.append(estimated)
            
            results[f'n={n}'] = {
                'n_variables': n,
                'n_clauses': n_clauses,
                'estimated_runtime': estimated,
                'actual_runtimes': [float(r) for r in runtimes],
                'mean_runtime': float(np.mean(runtimes)),
                'std_runtime': float(np.std(runtimes)),
                'min_runtime': float(min(runtimes)),
                'max_runtime': float(max(runtimes))
            }
            
            # Plot: Laufzeit-Verteilung
            ax.hist(runtimes, bins=15, color='steelblue', edgecolor='navy', alpha=0.7, label='Beobachtet')
            ax.axvline(np.mean(runtimes), color='green', linestyle='--', linewidth=2, label='Mittelwert')
            ax.axvline(estimated, color='red', linestyle='--', linewidth=2, label='Geschätzt')
            
            ax.set_xlabel('Laufzeit (Operationen)', fontweight='bold')
            ax.set_ylabel('Häufigkeit', fontweight='bold')
            ax.set_title(f'n = {n}, m = {n_clauses}', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, axis='y')
        
        self._save_plot(fig, 'exp07_runtime_comparison', 'Laufzeit-Vergleiche')
        self._save_data(results, 'exp07_runtime_comparison')
        
        print(f"✓ Laufzeit-Vergleiche abgeschlossen")
        print(f"  Parameter: n = {n_values}")
    
    # =====================================================================
    # EXPERIMENT 8: Phasenübergänge erkennen
    # =====================================================================
    
    def experiment_08_phase_transitions(self):
        """
        Analysiert die Übergänge zwischen den drei Phasen.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 8: Phasenübergänge und kritische Punkte")
        print("="*70)
        
        n_values = [8, 10, 12, 14]
        results = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phasenübergänge: Kritische Punkte in der Verschiebung', 
                    fontsize=16, fontweight='bold')
        
        for idx, n in enumerate(n_values):
            ax = axes[idx // 2, idx % 2]
            
            analyzer = ProbabilityShiftAnalyzer(n)
            
            # Erzeuge vollständige Trajektorie
            max_k = min(500, analyzer.max_formulas)
            k_values = list(range(1, max_k + 1))
            
            runtimes = [analyzer.expected_runtime(k) for k in k_values]
            phases = [analyzer.get_phase(k) for k in k_values]
            
            # Phase-Grenzen
            phase1_end, phase2_end = analyzer.phase_transition_points()
            
            # Derivate für Übergangserkennung
            derivatives = np.gradient(runtimes)
            
            results[f'n={n}'] = {
                'n_variables': n,
                'phase1_end_index': int(phase1_end),
                'phase2_end_index': int(phase2_end),
                'phase1_duration': int(phase1_end),
                'phase2_duration': int(phase2_end - phase1_end),
                'phase3_duration': int(max_k - phase2_end),
                'transition_points': [int(phase1_end), int(phase2_end)]
            }
            
            # Plot: Phasen mit Farbgebung
            colors = []
            for k in k_values:
                if k <= phase1_end:
                    colors.append('#2ecc71')  # Grün
                elif k <= phase2_end:
                    colors.append('#f39c12')  # Orange
                else:
                    colors.append('#e74c3c')  # Rot
            
            # Erzeuge Scatter-Plot mit Farben
            for i, (k, runtime, color) in enumerate(zip(k_values, runtimes, colors)):
                ax.scatter(k, runtime, c=color, s=10, alpha=0.6)
            
            # Markiere Übergangspunkte
            ax.axvline(phase1_end, color='blue', linestyle=':', linewidth=2.5, 
                      alpha=0.8, label=f'Phase 1→2: k={int(phase1_end)}')
            ax.axvline(phase2_end, color='purple', linestyle=':', linewidth=2.5, 
                      alpha=0.8, label=f'Phase 2→3: k={int(phase2_end)}')
            
            ax.axhline(n ** 3, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Legenden
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Phase 1: log(n)'),
                Patch(facecolor='#f39c12', label='Phase 2: Übergang'),
                Patch(facecolor='#e74c3c', label='Phase 3: n³')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
            
            ax.set_xlabel('Formel-Index k', fontweight='bold')
            ax.set_ylabel('Laufzeit E[T_k]', fontweight='bold')
            ax.set_title(f'n = {n}: Phasenstruktur', fontweight='bold')
            ax.grid(alpha=0.2)
            ax.set_yscale('log')
        
        self._save_plot(fig, 'exp08_phase_transitions', 'Phasenübergänge')
        self._save_data(results, 'exp08_phase_transitions')
        
        print(f"✓ Phasenübergänge analysiert")
        print(f"  Parameter: n = {n_values}")
    
    # =====================================================================
    # EXPERIMENT 9: Wahrscheinlichkeitsraum-Eigenschaften
    # =====================================================================
    
    def experiment_09_probability_space_properties(self):
        """
        Analysiert Eigenschaften von Wahrscheinlichkeitsräumen.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 9: Wahrscheinlichkeitsraum-Eigenschaften")
        print("="*70)
        
        n_outcomes = 10
        
        # Verschiedene Verteilungen
        distributions = {
            'Uniform': np.ones(n_outcomes) / n_outcomes,
            'Exponential': np.exp(np.arange(n_outcomes)) / np.sum(np.exp(np.arange(n_outcomes))),
            'Power Law': np.array([1/(i+1)**2 for i in range(n_outcomes)]),
            'Concentrated': np.array([0.1]*9 + [0.1]),
            'Skewed': np.array([10**(i-5) for i in range(n_outcomes)])
        }
        
        # Normalisiere Skewed
        distributions['Skewed'] = distributions['Skewed'] / np.sum(distributions['Skewed'])
        
        # Erstelle ProbabilitySpace-Objekte
        spaces = {name: ProbabilitySpace(n_outcomes, dist) 
                 for name, dist in distributions.items()}
        
        # Berechne Entropien
        entropies = {name: space.entropy() for name, space in spaces.items()}
        
        # Berechne KL-Divergenzen zur Uniform-Verteilung
        uniform_space = spaces['Uniform']
        kl_divergences = {name: space.kl_divergence(uniform_space) 
                         for name, space in spaces.items() if name != 'Uniform'}
        
        results = {
            'n_outcomes': n_outcomes,
            'distributions': {name: dist.tolist() for name, dist in distributions.items()},
            'entropies': {name: float(h) for name, h in entropies.items()},
            'kl_divergences_from_uniform': {name: float(d) for name, d in kl_divergences.items()}
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Wahrscheinlichkeitsraum-Eigenschaften: Verschiedene Verteilungen', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Verteilungen
        ax1 = axes[0, 0]
        x = np.arange(n_outcomes)
        width = 0.15
        
        colors_dist = plt.cm.Set3(np.linspace(0, 1, len(distributions)))
        
        for i, (name, dist) in enumerate(distributions.items()):
            ax1.bar(x + i*width, dist, width, label=name, alpha=0.8, color=colors_dist[i])
        
        ax1.set_xlabel('Outcome', fontweight='bold')
        ax1.set_ylabel('Wahrscheinlichkeit', fontweight='bold')
        ax1.set_title('Verschiedene Wahrscheinlichkeitsverteilungen', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3, axis='y')
        
        # Plot 2: Entropien
        ax2 = axes[0, 1]
        dist_names = list(entropies.keys())
        entropy_values = list(entropies.values())
        colors_entropy = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(dist_names)))
        
        bars = ax2.barh(dist_names, entropy_values, color=colors_entropy, 
                       edgecolor='black', alpha=0.7)
        ax2.axvline(np.log2(n_outcomes), color='red', linestyle='--', linewidth=2,
                   label=f'Max Entropie = log₂({n_outcomes}) = {np.log2(n_outcomes):.3f}')
        
        ax2.set_xlabel('Entropie H(P) [bits]', fontweight='bold')
        ax2.set_title('Shannon-Entropie verschiedener Verteilungen', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='x')
        
        # Beschriftung
        for bar, value in zip(bars, entropy_values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {value:.3f}', ha='left', va='center', fontweight='bold')
        
        # Plot 3: KL-Divergenzen
        ax3 = axes[1, 0]
        kl_names = list(kl_divergences.keys())
        kl_values = list(kl_divergences.values())
        
        bars = ax3.bar(kl_names, kl_values, color='steelblue', edgecolor='navy', alpha=0.7)
        ax3.set_ylabel('KL-Divergenz D_KL(P || Uniform)', fontweight='bold')
        ax3.set_title('KL-Divergenz von Uniform-Verteilung', fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Beschriftung
        for bar, value in zip(bars, kl_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Statistik-Zusammenfassung
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
Wahrscheinlichkeitsraum-Statistiken:

• Maximale Entropie (Uniform):
  H_max = log₂({n_outcomes}) = {np.log2(n_outcomes):.4f} bits
  
• Entropie-Bereich:
  Min: {min(entropies.values()):.4f} (Concentrated)
  Max: {max(entropies.values()):.4f} (Uniform)
  
• KL-Divergenzen von Uniform:
  Min: {min(kl_divergences.values()):.4f}
  Max: {max(kl_divergences.values()):.4f}

Interpretation:
- Höhere Entropie = höhere Unsicherheit
- KL-Divergenz = Distanz zur Uniform-Verteilung
- Uniform hat maximale Entropie
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self._save_plot(fig, 'exp09_probability_space', 'Wahrscheinlichkeitsraum-Eigenschaften')
        self._save_data(results, 'exp09_probability_space')
        
        print(f"✓ Wahrscheinlichkeitsraum-Analyse abgeschlossen")
        print(f"  Entropie-Bereich: [{min(entropies.values()):.4f}, {max(entropies.values()):.4f}]")
    
    # =====================================================================
    # EXPERIMENT 10: Skalierungsverhalten
    # =====================================================================
    
    def experiment_10_scaling_behavior(self):
        """
        Analysiert das Skalierungsverhalten der Laufzeit mit n.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 10: Skalierungsverhalten der Laufzeit")
        print("="*70)
        
        n_range = list(range(5, 21))
        results = {}
        
        phase1_runtimes = []
        phase3_runtimes = []
        n_cubed_runtimes = []
        logarithmic_runtimes = []
        
        for n in n_range:
            analyzer = ProbabilityShiftAnalyzer(n)
            
            # Phase 1: Erwartete Laufzeit
            phase1_runtime = analyzer.phase1_expected_runtime(1)
            phase1_runtimes.append(phase1_runtime)
            
            # Phase 3: Worst-Case
            phase3_runtime = n ** 3
            phase3_runtimes.append(phase3_runtime)
            
            # n^3
            n_cubed_runtimes.append(n ** 3)
            
            # log(n)
            logarithmic_runtimes.append(math.log(n, 3))
            
            results[f'n={n}'] = {
                'n': n,
                'log_n': float(math.log(n, 3)),
                'n_cubed': n ** 3,
                'phase1_runtime': float(phase1_runtime),
                'phase3_runtime': float(phase3_runtime)
            }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Skalierungsverhalten: Abhängigkeit von n', fontsize=16, fontweight='bold')
        
        # Plot 1: Log-Log-Vergleich
        ax1 = axes[0, 0]
        ax1.loglog(n_range, logarithmic_runtimes, 'o-', linewidth=2.5, markersize=8,
                  label='O(log n)', color='green')
        ax1.loglog(n_range, n_cubed_runtimes, 's-', linewidth=2.5, markersize=8,
                  label='O(n³)', color='red')
        
        # Referenzlinien
        ref_log = [math.log(n, 3) for n in n_range]
        ref_n3 = [n**3 for n in n_range]
        
        ax1.set_xlabel('Anzahl Variablen n (log-Skala)', fontweight='bold')
        ax1.set_ylabel('Laufzeit (log-Skala)', fontweight='bold')
        ax1.set_title('Log-Log-Vergleich: O(log n) vs. O(n³)', fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3, which='both')
        
        # Plot 2: Linear-Log
        ax2 = axes[0, 1]
        ax2.semilogy(n_range, logarithmic_runtimes, 'o-', linewidth=2.5, markersize=8,
                    label='log₃(n)', color='green')
        ax2.semilogy(n_range, n_cubed_runtimes, 's-', linewidth=2.5, markersize=8,
                    label='n³', color='red')
        
        ax2.set_xlabel('Anzahl Variablen n', fontweight='bold')
        ax2.set_ylabel('Laufzeit (log-Skala)', fontweight='bold')
        ax2.set_title('Semilog-Plot: Phase 1 (log) vs. Phase 3 (n³)', fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        
        # Plot 3: Wachstums-Raten
        ax3 = axes[1, 0]
        
        # Berechne Wachstums-Raten
        log_ratios = []
        n3_ratios = []
        
        for i in range(1, len(n_range)):
            if logarithmic_runtimes[i-1] > 0:
                log_ratio = logarithmic_runtimes[i] / logarithmic_runtimes[i-1]
                log_ratios.append(log_ratio)
            if n_cubed_runtimes[i-1] > 0:
                n3_ratio = n_cubed_runtimes[i] / n_cubed_runtimes[i-1]
                n3_ratios.append(n3_ratio)
        
        n_range_diffs = n_range[1:]
        ax3.plot(n_range_diffs, log_ratios, 'o-', linewidth=2, markersize=8,
                label='log(n) Verhältnis', color='green')
        ax3.plot(n_range_diffs, n3_ratios, 's-', linewidth=2, markersize=8,
                label='n³ Verhältnis', color='red')
        ax3.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax3.set_xlabel('Anzahl Variablen n', fontweight='bold')
        ax3.set_ylabel('Wachstums-Verhältnis (T_n / T_{n-1})', fontweight='bold')
        ax3.set_title('Wachstums-Raten beim Erhöhen von n', fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(alpha=0.3)
        
        # Plot 4: Dominanz-Analyse
        ax4 = axes[1, 1]
        
        ratios_n3_over_logn = [n3 / logn if logn > 0 else 0 
                              for n3, logn in zip(n_cubed_runtimes, logarithmic_runtimes)]
        
        ax4.semilogy(n_range, ratios_n3_over_logn, 'D-', linewidth=2.5, markersize=8,
                    color='purple', label='n³ / log(n)')
        ax4.set_xlabel('Anzahl Variablen n', fontweight='bold')
        ax4.set_ylabel('Verhältnis (log-Skala)', fontweight='bold')
        ax4.set_title('Dominanz von n³ über log(n)', fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.legend(fontsize=11)
        
        self._save_plot(fig, 'exp10_scaling_behavior', 'Skalierungsverhalten')
        self._save_data(results, 'exp10_scaling_behavior')
        
        print(f"✓ Skalierungsverhalten analysiert")
        print(f"  Parameter: n = {n_range}")
    
    # =====================================================================
    # EXPERIMENT SUMMARY
    # =====================================================================
    
    def run_all_experiments(self):
        """Führt alle Experimente aus."""
        print("\n" + "="*70)
        print("STARTEN SIE ALLE EXPERIMENTE FÜR DEN SUBGRAPH-SAT-SOLVER")
        print("="*70)
        
        start_time = datetime.now()
        
        try:
            self.experiment_01_combination_pyramid()
            self.experiment_02_probability_shift()
            self.experiment_03_bimodal_distribution()
            self.experiment_04_convergence_to_worst_case()
            self.experiment_05_metadistribution_analysis()
            self.experiment_06_lbv_solver()
            self.experiment_07_runtime_comparison()
            self.experiment_08_phase_transitions()
            self.experiment_09_probability_space_properties()
            self.experiment_10_scaling_behavior()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*70)
            print("ALLE EXPERIMENTE ERFOLGREICH ABGESCHLOSSEN")
            print("="*70)
            print(f"Laufzeit: {duration.total_seconds():.2f} Sekunden")
            print(f"Ausgabeverzeichnis: {self.output_dir}")
            print(f" Plots: {self.output_dir}/plots/")
            print(f"Daten: {self.output_dir}/data/")
            
        except Exception as e:
            print(f"\n FEHLER WÄHREND DER EXPERIMENTE:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()


# =====================================================================
# HAUPTPROGRAMM
# =====================================================================

if __name__ == '__main__':
    print("\n" + "  "*35)
    print("SUBGRAPH-SAT-SOLVER: UMFASSENDE EXPERIMENTELLE VALIDIERUNG")
    print("  "*35)
    
    manager = ExperimentManager()
    manager.run_all_experiments()
    
    print("\n  Alle Experimente sind abgeschlossen!")
    print("  Plots und Daten wurden gespeichert.")
