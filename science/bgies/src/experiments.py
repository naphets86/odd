"""
Experiments Modul: Umfangreiche Experimente zur Bacteria-DNA Wechselwirkung

Experimente durchführen:
1. Genomische Modifikationen über Zeit
2. Populationsdynamik und Equilibrium
3. Mikrobiom-Netzwerk-Komplexität
4. Einflussmetrik-Komponenten-Analyse
5. Informationskaskaden
6. Co-Evolution von Wirt und Mikrobiom
7. Stress-Response-Dynamik
8. Funktionale Komplementarität

Für jedes Experiment:
- Daten sammeln
- Plots mit matplotlib erstellen
- Ergebnisse in JSON speichern
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path
import networkx as nx
from datetime import datetime

from bacteria_dna_model import (
    Genome, Bacterium, Host, ModificationMechanisms, GenomicInfluenceMetric,
    PopulationDynamics, MicrobiomeNetwork
)


class ExperimentResults:
    """Container für Experimentergebnisse"""
    
    def __init__(self, name: str):
        self.name = name
        self.timestamp = datetime.now().isoformat()
        self.data = {}
        self.plots = []
        self.summary = {}
    
    def add_data(self, key: str, value):
        """Füge Daten hinzu"""
        if isinstance(value, np.ndarray):
            self.data[key] = value.tolist()
        else:
            self.data[key] = value
    
    def add_plot(self, filename: str):
        """Registriere Plot-Datei"""
        self.plots.append(filename)
    
    def add_summary(self, key: str, value):
        """Füge Zusammenfassung hinzu"""
        self.summary[key] = value
    
    def save_to_json(self, output_dir: Path = None):
        """Speichere zu JSON"""
        if output_dir is None:
            output_dir = Path("experiments_results")
        
        output_dir.mkdir(exist_ok=True)
        
        result_dict = {
            'name': self.name,
            'timestamp': self.timestamp,
            'summary': self.summary,
            'data': self.data,
            'plots': self.plots
        }
        
        filename = output_dir / f"{self.name}_{self.timestamp.split('T')[0]}.json"
        
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        return filename


class Experiment1_ModificationMechanisms:
    """
    Experiment 1: Vergleich der sechs Modifikationsmechanismen
    
    Zeigt, wie verschiedene Mechanismen das Wirtsgenom verändern
    """
    
    @staticmethod
    def run() -> ExperimentResults:
        """Führe Experiment durch"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: Genomische Modifikationsmechanismen")
        print("="*80)
        
        results = ExperimentResults("exp1_modification_mechanisms")
        
        # Erstelle initiales Host-Genom
        base_genome = Genome("ATGC" * 1000)
        host = Host("human_gut", base_genome)
        
        # Appliziere jeden Mechanismus
        modifications = {}
        
        # 1. HGT
        print("Appliziere HGT...")
        donor = Genome("TTTT" * 1000)
        modified_hgt = ModificationMechanisms.horizontal_gene_transfer(
            donor, base_genome, transfer_size=500, efficiency=1.0
        )
        modifications['hgt'] = modified_hgt
        
        # 2. Plasmid
        print("Appliziere Plasmid-Insertion...")
        modified_plasmid = ModificationMechanisms.plasmid_insertion(
            base_genome, plasmid_size=2000, plasmid_genes=15
        )
        modifications['plasmid'] = modified_plasmid
        
        # 3. Mutagenese
        print("Appliziere Mutagenese...")
        modified_mut = ModificationMechanisms.mutagenesis(
            base_genome, mutation_rate=0.001, metabolite_concentration=0.8
        )
        modifications['mutagenese'] = modified_mut
        
        # 4. Epigenetik
        print("Appliziere epigenetische Modulation...")
        modified_epi = ModificationMechanisms.epigenetic_modification(
            base_genome, affected_genes_fraction=0.1, metabolite_effect=0.6
        )
        modifications['epigen'] = modified_epi
        
        # 5. Rekombination
        print("Appliziere Rekombination...")
        modified_rec = ModificationMechanisms.recombination(
            base_genome, recombination_sites=5
        )
        modifications['rekomb'] = modified_rec
        
        # 6. Stress-Response
        print("Appliziere Stress-Response...")
        modified_stress = ModificationMechanisms.stress_response_mutagenesis(
            base_genome, stress_level=0.9, adaptation_rate=0.02
        )
        modifications['stress_response'] = modified_stress
        
        # Speichere Daten
        results.add_data("base_genome_size", base_genome.size)
        
        for mech_name, mod_genome in modifications.items():
            results.add_data(f"{mech_name}_size_change", 
                           len(mod_genome.sequence) - base_genome.size)
            results.add_data(f"{mech_name}_gc_content", mod_genome.gc_content)
            results.add_data(f"{mech_name}_epigenetic_marks", 
                           len(mod_genome.epigenetic_marks))
        
        # Erstelle Plot
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Plot 1: Genomgröße-Änderungen
        ax1 = fig.add_subplot(gs[0, 0])
        mechanisms = list(modifications.keys())
        size_changes = [
            len(modifications[m].sequence) - base_genome.size
            for m in mechanisms
        ]
        colors = plt.cm.Set3(np.linspace(0, 1, len(mechanisms)))
        ax1.bar(range(len(mechanisms)), size_changes, color=colors)
        ax1.set_xticks(range(len(mechanisms)))
        ax1.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax1.set_ylabel('Größen-Änderung (bp)')
        ax1.set_title('Exp 1.1: Genomgröße-Änderungen nach Modifikation')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: GC-Content Veränderungen
        ax2 = fig.add_subplot(gs[0, 1])
        gc_values = [modifications[m].gc_content for m in mechanisms]
        base_gc = base_genome.gc_content
        ax2.barh(mechanisms, gc_values, color=colors, alpha=0.7)
        ax2.axvline(base_gc, color='red', linestyle='--', label='Basis GC-Content')
        ax2.set_xlabel('GC-Content')
        ax2.set_title('Exp 1.2: GC-Content nach Modifikation')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # Plot 3: HGT-Regionen
        ax3 = fig.add_subplot(gs[1, 0])
        hgt_region_counts = [
            len(modifications[m].hgt_regions) for m in mechanisms
        ]
        ax3.bar(range(len(mechanisms)), hgt_region_counts, color=colors)
        ax3.set_xticks(range(len(mechanisms)))
        ax3.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax3.set_ylabel('Anzahl HGT-Regionen')
        ax3.set_title('Exp 1.3: HGT-Regionen Markierung')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Epigenetische Marker
        ax4 = fig.add_subplot(gs[1, 1])
        epi_marker_counts = [
            len(modifications[m].epigenetic_marks) for m in mechanisms
        ]
        ax4.bar(range(len(mechanisms)), epi_marker_counts, color=colors)
        ax4.set_xticks(range(len(mechanisms)))
        ax4.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax4.set_ylabel('Anzahl epigenetischer Marker')
        ax4.set_title('Exp 1.4: Epigenetische Marker pro Mechanismus')
        ax4.grid(axis='y', alpha=0.3)
        
        # Plot 5: Zusammenfassung
        ax5 = fig.add_subplot(gs[2, :])
        summary_data = np.array([
            [len(modifications[m].hgt_regions) / 5 for m in mechanisms],
            [modifications[m].gc_content for m in mechanisms],
            [len(modifications[m].epigenetic_marks) / 100 for m in mechanisms],
            [(len(modifications[m].sequence) - base_genome.size) / 1000 for m in mechanisms]
        ])
        
        x = np.arange(len(mechanisms))
        width = 0.2
        
        ax5.bar(x - 1.5*width, summary_data[0], width, label='HGT-Regionen/5', color='C0')
        ax5.bar(x - 0.5*width, summary_data[1], width, label='GC-Content', color='C1')
        ax5.bar(x + 0.5*width, summary_data[2], width, label='Epi-Marker/100', color='C2')
        ax5.bar(x + 1.5*width, summary_data[3], width, label='Größe-Änderung/1000bp', color='C3')
        
        ax5.set_xlabel('Modifikationsmechanismus')
        ax5.set_ylabel('Normalisierte Metriken')
        ax5.set_title('Exp 1.5: Vergleich aller Modifikationsmetriken')
        ax5.set_xticks(x)
        ax5.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = "src/results/exp1_modification_mechanisms.pdf"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_file}")
        results.add_plot(plot_file)
        plt.close()
        
        # Zusammenfassung
        results.add_summary("num_mechanisms", len(mechanisms))
        results.add_summary("base_genome_size", int(base_genome.size))
        results.add_summary("mechanisms", mechanisms)
        
        print(f"✓ Experiment 1 abgeschlossen")
        return results


class Experiment2_PopulationDynamics:
    """
    Experiment 2: Populationsdynamik und Equilibrium
    
    Simuliert die zeitliche Dynamik von Bakterienpopulation und Wirtsfitnessreaktionen
    """
    
    @staticmethod
    def run() -> ExperimentResults:
        """Führe Experiment durch"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: Populationsdynamik und Equilibrium")
        print("="*80)
        
        results = ExperimentResults("exp2_population_dynamics")
        
        # Simulationsparameter
        time_points = np.linspace(0, 200, 100)  # 200 Tage
        
        # Verschiedene Immunantwort-Szenarien
        scenarios = {
            'low_immunity': np.linspace(0, 0.2, len(time_points)),
            'medium_immunity': np.linspace(0, 0.5, len(time_points)),
            'high_immunity': np.linspace(0, 0.9, len(time_points)),
            'fluctuating_immunity': 0.5 + 0.4 * np.sin(2 * np.pi * time_points / 200)
        }
        
        all_results = {}
        
        for scenario_name, immune_traj in scenarios.items():
            print(f"Simuliere Szenario: {scenario_name}...")
            
            time, bacteria, fitness = PopulationDynamics.simulate_dynamics(
                initial_bacteria=1e6,
                initial_host_fitness=1.0,
                resources=1.0,
                metabolite_level=0.5,
                immune_response_trajectory=immune_traj,
                time_points=time_points
            )
            
            all_results[scenario_name] = {
                'time': time,
                'bacteria': bacteria,
                'fitness': fitness,
                'immune': immune_traj
            }
        
        # Speichere Daten
        for scenario_name, data in all_results.items():
            results.add_data(f"{scenario_name}_final_bacteria", float(data['bacteria'][-1]))
            results.add_data(f"{scenario_name}_final_fitness", float(data['fitness'][-1]))
            results.add_data(f"{scenario_name}_mean_bacteria", float(np.mean(data['bacteria'])))
        
        # Erstelle Plots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Plot 1: Bakterienpopulation im Zeitverlauf
        ax1 = fig.add_subplot(gs[0, :])
        for scenario_name, data in all_results.items():
            ax1.plot(data['time'], data['bacteria'], label=scenario_name, linewidth=2)
        ax1.set_xlabel('Zeit (Tage)')
        ax1.set_ylabel('Bakterienpopulation')
        ax1.set_title('Exp 2.1: Bakterienpopulation-Dynamik unter verschiedenen Immunantworten')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Wirtsfitnessreaktion
        ax2 = fig.add_subplot(gs[1, :])
        for scenario_name, data in all_results.items():
            ax2.plot(data['time'], data['fitness'], label=scenario_name, linewidth=2)
        ax2.set_xlabel('Zeit (Tage)')
        ax2.set_ylabel('Wirtsfitnessreaktionen')
        ax2.set_title('Exp 2.2: Wirtsfitnessreaktion im Zeitverlauf')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Immunantwort-Szenarien
        ax3 = fig.add_subplot(gs[2, 0])
        for scenario_name, data in all_results.items():
            ax3.plot(data['time'], data['immune'], label=scenario_name, linewidth=2)
        ax3.set_xlabel('Zeit (Tage)')
        ax3.set_ylabel('Immunantwort-Level')
        ax3.set_title('Exp 2.3: Immunantwort-Trajektorien')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Endzustände
        ax4 = fig.add_subplot(gs[2, 1])
        scenarios_names = list(all_results.keys())
        final_bacteria = [all_results[s]['bacteria'][-1] for s in scenarios_names]
        final_fitness = [all_results[s]['fitness'][-1] for s in scenarios_names]
        
        x = np.arange(len(scenarios_names))
        width = 0.35
        
        # Normalisiere für Vergleich
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - width/2, final_bacteria, width, label='Final Bacteria', alpha=0.7, color='C0')
        bars2 = ax4_twin.bar(x + width/2, final_fitness, width, label='Final Fitness', alpha=0.7, color='C1')
        
        ax4.set_ylabel('Finale Bakterienpopulation', color='C0')
        ax4_twin.set_ylabel('Finale Wirtsfitnessreaktion', color='C1')
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenarios_names, rotation=45, ha='right')
        ax4.set_title('Exp 2.4: Endzustände nach 200 Tagen')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = "src/results/exp2_population_dynamics.pdf"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_file}")
        results.add_plot(plot_file)
        plt.close()
        
        # Zusammenfassung
        results.add_summary("num_scenarios", len(scenarios))
        results.add_summary("simulation_duration_days", 200)
        results.add_summary("scenarios", scenarios_names)
        
        print(f"✓ Experiment 2 abgeschlossen")
        return results


class Experiment3_MicrobiomeNetwork:
    """
    Experiment 3: Mikrobiom-Netzwerk-Komplexität
    
    Analysiert die Struktur und Dynamik von Mikrobiom-Netzwerken
    """
    
    @staticmethod
    def run() -> ExperimentResults:
        """Führe Experiment durch"""
        print("\n" + "="*80)
        print("EXPERIMENT 3: Mikrobiom-Netzwerk-Analyse")
        print("="*80)
        
        results = ExperimentResults("exp3_microbiome_network")
        
        # Erstelle Mikrobiom mit verschiedenen Komplexitätsstufen
        complexity_levels = [5, 10, 20, 40]
        network_results = {}
        
        for num_species in complexity_levels:
            print(f"Erstelle Netzwerk mit {num_species} Arten...")
            
            network = MicrobiomeNetwork()
            
            # Generiere Arten mit komplementären Funktionen
            functions_pool = [
                "fiber_degradation", "glucose_fermentation", "butyrate_production",
                "acetate_utilization", "lactate_metabolism", "protein_fermentation",
                "mucin_degradation", "phenolic_metabolism", "starch_digestion",
                "lipid_metabolism", "vitamin_synthesis", "h2_utilization"
            ]
            
            for i in range(num_species):
                genome = Genome("ATGC" * 500)
                bacterium = Bacterium(f"Species_{i}", genome)
                
                # Zufällig 2-4 Funktionen pro Art
                num_functions = np.random.randint(2, 5)
                functions = np.random.choice(functions_pool, num_functions, replace=False)
                bacterium.metabolic_functions = set(functions)
                
                network.add_species(bacterium)
            
            # Füge zufällige Interaktionen hinzu
            species_list = list(network.bacteria_species.keys())
            
            for i in range(num_species):
                for j in range(i+1, num_species):
                    if np.random.random() < 0.3:  # 30% Verbindungswahrscheinlichkeit
                        interaction_type = np.random.choice(
                            ["substrate", "synergy", "inhibition"]
                        )
                        strength = np.random.uniform(0.3, 1.0)
                        network.add_interaction(
                            species_list[i], species_list[j],
                            interaction_type, strength
                        )
            
            # Analysiere Netzwerk
            clusters = network.find_functional_clusters(min_cluster_size=2)
            
            # Berechne Komplementarität zwischen zufälligen Paaren
            complementarities = []
            for _ in range(min(10, num_species)):
                s1, s2 = np.random.choice(species_list, 2, replace=False)
                comp = network.calculate_complementarity(s1, s2)
                complementarities.append(comp)
            
            network_results[num_species] = {
                'network': network,
                'num_clusters': len(clusters),
                'cluster_sizes': [len(c) for c in clusters],
                'avg_complementarity': np.mean(complementarities) if complementarities else 0,
                'num_edges': network.graph.number_of_edges()
            }
            
            results.add_data(f"network_{num_species}_num_clusters", len(clusters))
            results.add_data(f"network_{num_species}_avg_complementarity", 
                           float(np.mean(complementarities)) if complementarities else 0)
            results.add_data(f"network_{num_species}_num_edges", network.graph.number_of_edges())
        
        # Erstelle Plots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Plot 1: Netzwerk-Visualisierung (ein Beispiel)
        ax1 = fig.add_subplot(gs[0, 0])
        example_network = network_results[10]['network']
        pos = nx.spring_layout(example_network.graph, k=2, iterations=50)
        
        nx.draw_networkx_nodes(example_network.graph, pos, ax=ax1, 
                              node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(example_network.graph, pos, ax=ax1, font_size=8)
        nx.draw_networkx_edges(example_network.graph, pos, ax=ax1, 
                              alpha=0.5, edge_color='gray')
        
        ax1.set_title('Exp 3.1: Beispiel Mikrobiom-Netzwerk (10 Arten)')
        ax1.axis('off')
        
        # Plot 2: Netzwerk-Metriken
        ax2 = fig.add_subplot(gs[0, 1])
        complexity_levels_array = np.array(complexity_levels)
        num_clusters = [network_results[c]['num_clusters'] for c in complexity_levels]
        num_edges = [network_results[c]['num_edges'] for c in complexity_levels]
        
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(complexity_levels_array, num_clusters, 'o-', 
                        color='C0', linewidth=2, markersize=8, label='Anzahl Cluster')
        line2 = ax2_twin.plot(complexity_levels_array, num_edges, 's-', 
                             color='C1', linewidth=2, markersize=8, label='Anzahl Kanten')
        
        ax2.set_xlabel('Netzwerk-Komplexität (Anzahl Arten)')
        ax2.set_ylabel('Anzahl funktionaler Cluster', color='C0')
        ax2_twin.set_ylabel('Anzahl Interaktions-Kanten', color='C1')
        ax2.set_title('Exp 3.2: Netzwerk-Struktur mit Komplexität')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Komplementarität
        ax3 = fig.add_subplot(gs[1, 0])
        avg_complementarities = [
            network_results[c]['avg_complementarity'] for c in complexity_levels
        ]
        ax3.plot(complexity_levels_array, avg_complementarities, 'D-', 
                color='C2', linewidth=2, markersize=8)
        ax3.fill_between(complexity_levels_array, 0, avg_complementarities, 
                        alpha=0.3, color='C2')
        ax3.set_xlabel('Netzwerk-Komplexität (Anzahl Arten)')
        ax3.set_ylabel('Durchschnittliche metabolische Komplementarität')
        ax3.set_title('Exp 3.3: Funktionale Komplementarität')
        ax3.set_ylim([0, 1])
        ax3.grid(alpha=0.3)
        
        # Plot 4: Cluster-Größen-Verteilung
        ax4 = fig.add_subplot(gs[1, 1])
        all_cluster_sizes = []
        complexity_labels = []
        
        for complexity in complexity_levels:
            sizes = network_results[complexity]['cluster_sizes']
            all_cluster_sizes.extend(sizes)
            complexity_labels.extend([str(complexity)] * len(sizes))
        
        data_for_box = [
            network_results[c]['cluster_sizes'] for c in complexity_levels
        ]
        
        bp = ax4.boxplot(data_for_box, labels=[str(c) for c in complexity_levels],
                        patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax4.set_xlabel('Netzwerk-Komplexität (Anzahl Arten)')
        ax4.set_ylabel('Cluster-Größe')
        ax4.set_title('Exp 3.4: Verteilung der funktionalen Cluster-Größen')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = "src/results/exp3_microbiome_network.pdf"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_file}")
        results.add_plot(plot_file)
        plt.close()
        
        # Zusammenfassung
        results.add_summary("complexity_levels", [int(c) for c in complexity_levels])
        results.add_summary("max_avg_complementarity", 
                          float(max(avg_complementarities)))
        
        print(f"✓ Experiment 3 abgeschlossen")
        return results


class Experiment4_InfluenceMetric:
    """
    Experiment 4: Genomische Einflussmetrik - Komponenten-Analyse
    
    Analysiert die Beiträge verschiedener Modifikationsmechanismen zur Gesamtmetrik
    """
    
    @staticmethod
    def run() -> ExperimentResults:
        """Führe Experiment durch"""
        print("\n" + "="*80)
        print("EXPERIMENT 4: Genomische Einflussmetrik Komponenten-Analyse")
        print("="*80)
        
        results = ExperimentResults("exp4_influence_metric")
        
        # Erstelle Host mit progressiven Modifikationen
        host_genome = Genome("ATGC" * 1000)
        original_genome = host_genome
        
        bacterium = Bacterium("Test_Bacterium", Genome("TTTT" * 1000))
        bacterium.metabolic_functions = {"func1", "func2"}
        
        # Appliziere Modifikationen schrittweise
        num_steps = 5
        modification_steps = []
        
        for step in range(num_steps):
            # Appliziere verschiedene Mechanismen mit zunehmender Intensität
            intensity = (step + 1) / num_steps
            
            # Kombiniere mehrere Mechanismen
            modified = ModificationMechanisms.horizontal_gene_transfer(
                bacterium.genome, host_genome,
                transfer_size=int(500 * intensity), efficiency=intensity
            )
            
            modified = ModificationMechanisms.epigenetic_modification(
                modified,
                affected_genes_fraction=0.05 * intensity,
                metabolite_effect=intensity
            )
            
            modified = ModificationMechanisms.mutagenesis(
                modified,
                mutation_rate=0.0001 * intensity,
                metabolite_concentration=intensity
            )
            
            host_genome = modified
            
            # Berechne Einflussmetrik
            host = Host("human", modified)
            influence_result = GenomicInfluenceMetric.calculate_total_influence(
                host, original_genome, [bacterium]
            )
            
            modification_steps.append({
                'step': step + 1,
                'intensity': intensity,
                'total_influence': influence_result['total'],
                'components': influence_result['components'].copy()
            })
            
            results.add_data(f"step_{step+1}_total_influence", 
                           influence_result['total'])
        
        # Erstelle Plots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Plot 1: Gesamte Einflussmetrik über Schritte
        ax1 = fig.add_subplot(gs[0, :])
        steps = [s['step'] for s in modification_steps]
        total_influences = [s['total_influence'] for s in modification_steps]
        
        ax1.plot(steps, total_influences, 'o-', color='C0', linewidth=3, 
                markersize=10, label='Gesamt Einflussmetrik')
        ax1.fill_between(steps, 0, total_influences, alpha=0.3, color='C0')
        ax1.set_xlabel('Modifikationsschritt')
        ax1.set_ylabel('Genomische Einflussmetrik (0-1)')
        ax1.set_title('Exp 4.1: Kumulative Einflussmetrik')
        ax1.set_xticks(steps)
        ax1.set_ylim([0, 1])
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Plot 2: Komponenten-Beiträge
        ax2 = fig.add_subplot(gs[1, 0])
        component_names = list(modification_steps[0]['components'].keys())
        
        for component in component_names:
            component_values = [s['components'][component] for s in modification_steps]
            ax2.plot(steps, component_values, 'o-', linewidth=2, 
                    label=component, markersize=6)
        
        ax2.set_xlabel('Modifikationsschritt')
        ax2.set_ylabel('Komponenten-Beitrag')
        ax2.set_title('Exp 4.2: Beiträge einzelner Mechanismen')
        ax2.set_xticks(steps)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        
        # Plot 3: Stacked Area Chart der Komponenten
        ax3 = fig.add_subplot(gs[1, 1])
        component_data = {
            comp: [s['components'][comp] for s in modification_steps]
            for comp in component_names
        }
        
        ax3.stackplot(steps, *component_data.values(), labels=component_names, alpha=0.7)
        ax3.set_xlabel('Modifikationsschritt')
        ax3.set_ylabel('Normalisierter Beitrag')
        ax3.set_title('Exp 4.3: Stacked Komponenten-Anteile')
        ax3.set_xticks(steps)
        ax3.legend(fontsize=8, loc='upper left')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = "src/results/exp4_influence_metric.pdf"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_file}")
        results.add_plot(plot_file)
        plt.close()
        
        # Zusammenfassung
        results.add_summary("num_modification_steps", num_steps)
        results.add_summary("final_influence_metric", float(total_influences[-1]))
        results.add_summary("component_names", component_names)
        
        print(f"✓ Experiment 4 abgeschlossen")
        return results


class Experiment5_InformationCascade:
    """
    Experiment 5: Informationskaskaden durch Mikrobiom-Netzwerke
    
    Simuliert die Ausbreitung von Informationen/Metaboliten durch ein Netzwerk
    """
    
    @staticmethod
    def run() -> ExperimentResults:
        """Führe Experiment durch"""
        print("\n" + "="*80)
        print("EXPERIMENT 5: Informationskaskaden in Mikrobiom-Netzwerken")
        print("="*80)
        
        results = ExperimentResults("exp5_information_cascade")
        
        # Erstelle lineares Netzwerk
        network = MicrobiomeNetwork()
        num_species = 10
        
        for i in range(num_species):
            genome = Genome("ATGC" * 300)
            bacterium = Bacterium(f"Species_{i}", genome)
            bacterium.metabolic_functions = {f"pathway_{i % 3}"}
            network.add_species(bacterium)
        
        # Erstelle lineares Netzwerk (Kette)
        for i in range(num_species - 1):
            network.add_interaction(
                f"Species_{i}", f"Species_{i+1}",
                "substrate", 0.8
            )
        
        # Simuliere Kaskaden von verschiedenen Startpunkten
        cascade_sources = [0, 5, 9]
        cascade_results = {}
        
        for source_idx in cascade_sources:
            print(f"Simuliere Kaskade von Species_{source_idx}...")
            
            cascade_data = {}
            
            for step in range(6):
                cascade = network.propagate_information_cascade(
                    f"Species_{source_idx}", "metabolite", steps=step
                )
                cascade_data[f"step_{step}"] = cascade
            
            cascade_results[source_idx] = cascade_data
        
        # Speichere Daten
        for source_idx, cascade_data in cascade_results.items():
            results.add_data(f"cascade_source_{source_idx}", cascade_data)
        
        # Erstelle Plots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Plot 1: Kaskade-Ausbreitung von unterschiedlichen Quellen
        ax1 = fig.add_subplot(gs[0, :])
        
        species_range = range(num_species)
        colors = plt.cm.viridis(np.linspace(0, 1, len(cascade_sources)))
        
        for color_idx, source_idx in enumerate(cascade_sources):
            cascade_final = cascade_results[source_idx]["step_5"]
            concentrations = [cascade_final[f"Species_{i}"] for i in species_range]
            ax1.plot(species_range, concentrations, 'o-', 
                    color=colors[color_idx], linewidth=2, markersize=8,
                    label=f'Quelle: Species_{source_idx}')
        
        ax1.set_xlabel('Spezies-Index')
        ax1.set_ylabel('Metaboliten-Konzentration')
        ax1.set_title('Exp 5.1: Informationskaskade-Ausbreitung von verschiedenen Quellen')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Zeitliche Ausbreitung vom zentralen Knoten
        ax2 = fig.add_subplot(gs[1, 0])
        
        source_idx = 5
        cascade_data = cascade_results[source_idx]
        
        species_labels = [f"S_{i}" for i in species_range]
        
        for step in [0, 2, 4]:
            cascade = cascade_data[f"step_{step}"]
            concentrations = [cascade[f"Species_{i}"] for i in species_range]
            ax2.plot(species_range, concentrations, 'o-', 
                    label=f'Schritt {step}', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Spezies-Index')
        ax2.set_ylabel('Metaboliten-Konzentration')
        ax2.set_title(f'Exp 5.2: Zeitliche Ausbreitung von Species_5')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Heatmap der Kaskaden-Ausbreitung
        ax3 = fig.add_subplot(gs[1, 1])
        
        cascade_matrix = np.zeros((num_species, num_species))
        
        for source_idx in range(num_species):
            cascade = network.propagate_information_cascade(
                f"Species_{source_idx}", "metabolite", steps=5
            )
            for target_idx in range(num_species):
                cascade_matrix[source_idx, target_idx] = cascade[f"Species_{target_idx}"]
        
        im = ax3.imshow(cascade_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xlabel('Ziel-Spezies')
        ax3.set_ylabel('Quell-Spezies')
        ax3.set_title('Exp 5.3: Heatmap der Informations-Erreichbarkeit')
        plt.colorbar(im, ax=ax3, label='Kaskaden-Stärke')
        
        plt.tight_layout()
        plot_file = "src/results/exp5_information_cascade.pdf"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot gespeichert: {plot_file}")
        results.add_plot(plot_file)
        plt.close()
        
        # Zusammenfassung
        results.add_summary("network_size", num_species)
        results.add_summary("cascade_sources", [int(s) for s in cascade_sources])
        
        print(f"✓ Experiment 5 abgeschlossen")
        return results


class ExperimentRunner:
    """Runner für alle Experimente"""
    
    @staticmethod
    def run_all(output_dir: Path = None) -> List[ExperimentResults]:
        """Führe alle Experimente durch"""
        
        if output_dir is None:
            output_dir = Path("src/results/")
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("STARTEN ALLER EXPERIMENTE ZUR BACTERIA-DNA WECHSELWIRKUNG")
        print("="*80)
        
        all_results = []
        
        # Experiment 1
        try:
            exp1_results = Experiment1_ModificationMechanisms.run()
            exp1_results.save_to_json(output_dir)
            all_results.append(exp1_results)
        except Exception as e:
            print(f"✗ Experiment 1 fehlgeschlagen: {e}")
        
        # Experiment 2
        try:
            exp2_results = Experiment2_PopulationDynamics.run()
            exp2_results.save_to_json(output_dir)
            all_results.append(exp2_results)
        except Exception as e:
            print(f"✗ Experiment 2 fehlgeschlagen: {e}")
        
        # Experiment 3
        try:
            exp3_results = Experiment3_MicrobiomeNetwork.run()
            exp3_results.save_to_json(output_dir)
            all_results.append(exp3_results)
        except Exception as e:
            print(f"✗ Experiment 3 fehlgeschlagen: {e}")
        
        # Experiment 4
        try:
            exp4_results = Experiment4_InfluenceMetric.run()
            exp4_results.save_to_json(output_dir)
            all_results.append(exp4_results)
        except Exception as e:
            print(f"✗ Experiment 4 fehlgeschlagen: {e}")
        
        # Experiment 5
        try:
            exp5_results = Experiment5_InformationCascade.run()
            exp5_results.save_to_json(output_dir)
            all_results.append(exp5_results)
        except Exception as e:
            print(f"✗ Experiment 5 fehlgeschlagen: {e}")
        
        # Erstelle Zusammenfassungs-Report
        ExperimentRunner._create_summary_report(all_results, output_dir)
        
        print("\n" + "="*80)
        print(f"✓ ALLE EXPERIMENTE ABGESCHLOSSEN")
        print(f"✓ Ergebnisse gespeichert in: {output_dir}")
        print(f"✓ Anzahl erfolgreicher Experimente: {len(all_results)}")
        print("="*80 + "\n")
        
        return all_results
    
    @staticmethod
    def _create_summary_report(all_results: List[ExperimentResults], output_dir: Path):
        """Erstelle Zusammenfassungs-Report"""
        
        summary = {
            'total_experiments': len(all_results),
            'experiments': [
                {
                    'name': result.name,
                    'timestamp': result.timestamp,
                    'plots': result.plots,
                    'summary': result.summary
                }
                for result in all_results
            ]
        }
        
        report_file = output_dir / "experiment_summary.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nZusammenfassungs-Report erstellt: {report_file}")


if __name__ == "__main__":
    # Führe alle Experimente durch
    results = ExperimentRunner.run_all()
