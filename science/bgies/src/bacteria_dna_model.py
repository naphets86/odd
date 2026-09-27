"""
Bacteria-DNA Wechselwirkung: Formale Modellierung und Implementierung

Dieses Modul implementiert alle Konzepte aus der wissenschaftlichen Arbeit:
- Bacterial-Genomic Information Exchange System (BGIES)
- Modifikationsmechanismen (HGT, Plasmide, Mutagenese, Epigenetik, Rekombination, StressResp)
- Genomische Einflussmetrik
- Funktionaler Raum
- Populationsdynamik
- Mikrobiom-Netzwerke
"""

import numpy as np
from typing import Tuple, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from scipy.integrate import odeint
from scipy.spatial.distance import cdist
import json


class ModificationMechanism(Enum):
    """Sechs Hauptmechanismen der genomischen Modifikation"""
    HGT = "horizontaler_gentransfer"
    PLASMID = "plasmid_insertion"
    MUTAGENESE = "mutagenese"
    EPIGEN = "epigenetik"
    REKOMB = "rekombination"
    STRESS_RESP = "stress_response"


@dataclass
class Genome:
    """Darstellung eines Genoms als Sequenz"""
    sequence: str  # DNA-Sequenz (A, T, G, C)
    gc_content: float = 0.0
    size: int = 0
    hgt_regions: List[Tuple[int, int]] = field(default_factory=list)  # HGT-Regionen
    epigenetic_marks: Dict[int, float] = field(default_factory=dict)  # Position -> Methylierungsgrad
    
    def __post_init__(self):
        """Berechne Genomstatistiken"""
        self.size = len(self.sequence)
        if self.size > 0:
            gc_count = self.sequence.count('G') + self.sequence.count('C')
            self.gc_content = gc_count / self.size
    
    def to_dict(self):
        """Konvertiere zu Dictionary für JSON-Export"""
        return {
            'size': self.size,
            'gc_content': float(self.gc_content),
            'sequence_preview': self.sequence[:100],
            'hgt_regions_count': len(self.hgt_regions),
            'epigenetic_marks_count': len(self.epigenetic_marks)
        }


@dataclass
class Bacterium:
    """Darstellung eines Bakteriums als lokales Informationssystem"""
    name: str
    genome: Genome
    metabolic_functions: Set[str] = field(default_factory=set)
    local_information: Dict[str, float] = field(default_factory=dict)
    
    def functional_space(self) -> Set[str]:
        """Funktionaler Raum: Was kann dieses Bakterium tun?"""
        return self.metabolic_functions.copy()
    
    def has_function(self, function: str) -> bool:
        """Kann dieses Bakterium eine bestimmte Funktion ausführen?"""
        return function in self.metabolic_functions
    
    def to_dict(self):
        """Konvertiere zu Dictionary"""
        return {
            'name': self.name,
            'genome': self.genome.to_dict(),
            'functions': list(self.metabolic_functions),
            'local_info_keys': list(self.local_information.keys())
        }


@dataclass
class Host:
    """Darstellung eines Wirt-Organismus"""
    name: str
    genome: Genome
    fitness: float = 1.0
    immune_response: float = 0.0
    stress_level: float = 0.0
    
    def to_dict(self):
        """Konvertiere zu Dictionary"""
        return {
            'name': self.name,
            'genome': self.genome.to_dict(),
            'fitness': float(self.fitness),
            'immune_response': float(self.immune_response),
            'stress_level': float(self.stress_level)
        }


class ModificationMechanisms:
    """Implementierung der sechs Modifikationsmechanismen"""
    
    @staticmethod
    def horizontal_gene_transfer(
        donor_genome: Genome,
        recipient_genome: Genome,
        transfer_size: int = 1000,
        efficiency: float = 0.01
    ) -> Genome:
        """
        Mechanismus 1: Horizontaler Gentransfer
        
        Args:
            donor_genome: Spender-Genom
            recipient_genome: Empfänger-Genom
            transfer_size: Größe des übertragenen DNA-Fragments
            efficiency: Übertragungseffizienz (0-1)
        
        Returns:
            Modifiziertes Empfänger-Genom
        """
        if np.random.random() > efficiency:
            return recipient_genome
        
        # Wähle zufälliges Fragment aus Donor
        if len(donor_genome.sequence) > transfer_size:
            start = np.random.randint(0, len(donor_genome.sequence) - transfer_size)
            fragment = donor_genome.sequence[start:start + transfer_size]
        else:
            fragment = donor_genome.sequence
        
        # Integriere in Recipient-Genom
        insert_pos = np.random.randint(0, len(recipient_genome.sequence))
        new_sequence = (
            recipient_genome.sequence[:insert_pos] +
            fragment +
            recipient_genome.sequence[insert_pos:]
        )
        
        # Markiere als HGT-Region
        new_genome = Genome(new_sequence)
        new_genome.hgt_regions = recipient_genome.hgt_regions.copy()
        new_genome.hgt_regions.append((insert_pos, insert_pos + len(fragment)))
        
        return new_genome
    
    @staticmethod
    def plasmid_insertion(
        recipient_genome: Genome,
        plasmid_size: int = 5000,
        plasmid_genes: int = 10
    ) -> Genome:
        """
        Mechanismus 2: Plasmid-Insertion
        
        Args:
            recipient_genome: Empfänger-Genom
            plasmid_size: Größe des Plasmids
            plasmid_genes: Anzahl der Gene im Plasmid
        
        Returns:
            Modifiziertes Genom mit integriertem Plasmid
        """
        # Generiere künstliches Plasmid
        bases = ['A', 'T', 'G', 'C']
        plasmid_seq = ''.join(np.random.choice(bases, plasmid_size))
        
        # Integriere zufällig
        insert_pos = np.random.randint(0, len(recipient_genome.sequence))
        new_sequence = (
            recipient_genome.sequence[:insert_pos] +
            plasmid_seq +
            recipient_genome.sequence[insert_pos:]
        )
        
        new_genome = Genome(new_sequence)
        new_genome.hgt_regions = recipient_genome.hgt_regions.copy()
        
        return new_genome
    
    @staticmethod
    def mutagenesis(
        genome: Genome,
        mutation_rate: float = 1e-6,
        metabolite_concentration: float = 0.5
    ) -> Genome:
        """
        Mechanismus 3: Mutagenese durch Metaboliten
        
        Args:
            genome: Originalgenome
            mutation_rate: Basis-Mutationsrate
            metabolite_concentration: Konzentration mutagener Metaboliten (0-1)
        
        Returns:
            Genom mit Mutationen
        """
        # Effektive Mutationsrate erhöht sich mit Metaboliten
        effective_rate = mutation_rate * (1 + 100 * metabolite_concentration)
        
        sequence = list(genome.sequence)
        bases = ['A', 'T', 'G', 'C']
        
        for i in range(len(sequence)):
            if np.random.random() < effective_rate:
                current_base = sequence[i]
                new_base = np.random.choice(
                    [b for b in bases if b != current_base]
                )
                sequence[i] = new_base
        
        new_genome = Genome(''.join(sequence))
        new_genome.hgt_regions = genome.hgt_regions.copy()
        new_genome.epigenetic_marks = genome.epigenetic_marks.copy()
        
        return new_genome
    
    @staticmethod
    def epigenetic_modification(
        genome: Genome,
        affected_genes_fraction: float = 0.1,
        metabolite_effect: float = 0.5
    ) -> Genome:
        """
        Mechanismus 4: Epigenetische Modulation
        
        Args:
            genome: Originalgenome
            affected_genes_fraction: Fraktion der betroffenen Gene
            metabolite_effect: Stärke der epigenetischen Modifikation
        
        Returns:
            Genom mit epigenetischen Markern
        """
        new_genome = Genome(genome.sequence)
        new_genome.hgt_regions = genome.hgt_regions.copy()
        new_genome.epigenetic_marks = genome.epigenetic_marks.copy()
        
        # Modifiziere zufällig ausgewählte Positionen
        num_marks = int(len(genome.sequence) * affected_genes_fraction)
        positions = np.random.choice(len(genome.sequence), num_marks, replace=False)
        
        for pos in positions:
            # Methylierungsgrad (0-1)
            methylation = np.random.uniform(0, metabolite_effect)
            new_genome.epigenetic_marks[int(pos)] = methylation
        
        return new_genome
    
    @staticmethod
    def recombination(
        genome: Genome,
        recombination_sites: int = 3
    ) -> Genome:
        """
        Mechanismus 5: Rekombination und Umstrukturierung
        
        Args:
            genome: Originalgenome
            recombination_sites: Anzahl der Rekombinationsstellen
        
        Returns:
            Umstrukturiertes Genom
        """
        sequence = list(genome.sequence)
        
        for _ in range(recombination_sites):
            if len(sequence) < 100:
                continue
            
            # Wähle zwei Positionen
            pos1 = np.random.randint(0, len(sequence) - 50)
            pos2 = np.random.randint(pos1 + 50, len(sequence))
            
            # Invertiere Segment (Inversion)
            if np.random.random() > 0.5:
                sequence[pos1:pos2] = reversed(sequence[pos1:pos2])
            # Oder lösche Segment (Deletion)
            else:
                del_size = min(50, pos2 - pos1)
                sequence[pos1:pos1 + del_size] = []
        
        new_genome = Genome(''.join(sequence))
        new_genome.hgt_regions = genome.hgt_regions.copy()
        new_genome.epigenetic_marks = genome.epigenetic_marks.copy()
        
        return new_genome
    
    @staticmethod
    def stress_response_mutagenesis(
        genome: Genome,
        stress_level: float,
        adaptation_rate: float = 0.01
    ) -> Genome:
        """
        Mechanismus 6: Stress-induzierte adaptive Mutagenese
        
        Args:
            genome: Originalgenome
            stress_level: Stresslevel (0-1)
            adaptation_rate: Adaptationsrate
        
        Returns:
            Genom mit adaptiven Mutationen
        """
        # Unter Stress erhöht sich die Mutationsrate
        stress_induced_rate = adaptation_rate * stress_level * 100
        
        return ModificationMechanisms.mutagenesis(
            genome,
            mutation_rate=stress_induced_rate,
            metabolite_concentration=0.0
        )


class GenomicInfluenceMetric:
    """Berechnung der Genomischen Einflussmetrik"""
    
    @staticmethod
    def calculate_hgt_component(host_genome: Genome, bacteria: List[Bacterium]) -> float:
        """
        Komponente 1: HGT-Metrik
        
        Misst die Anzahl und Größe der HGT-Ereignisse
        """
        if not bacteria or not host_genome.hgt_regions:
            return 0.0
        
        total_hgt_size = sum(end - start for start, end in host_genome.hgt_regions)
        hgt_fraction = total_hgt_size / host_genome.size if host_genome.size > 0 else 0.0
        
        # Normalisiere auf [0, 1]
        return min(1.0, hgt_fraction / 0.1)  # 10% ist die Referenz
    
    @staticmethod
    def calculate_plasmid_component(host_genome: Genome) -> float:
        """
        Komponente 2: Plasmid-Metrik
        
        Misst die relative Größe integrierter Plasmide
        """
        # Schätze basierend auf GC-Content-Variation
        base_gc = 0.5
        gc_deviation = abs(host_genome.gc_content - base_gc)
        
        return min(1.0, gc_deviation * 2)
    
    @staticmethod
    def calculate_mutagenesis_component(
        host_genome: Genome,
        original_genome: Genome
    ) -> float:
        """
        Komponente 3: Mutagenese-Metrik
        
        Misst die Mutationsrate
        """
        if len(host_genome.sequence) != len(original_genome.sequence):
            # Genome unterschiedliche Länge
            return 0.5
        
        # Berechne Hamming-Distanz
        mutations = sum(
            1 for a, b in zip(host_genome.sequence, original_genome.sequence)
            if a != b
        )
        
        mutation_rate = mutations / len(host_genome.sequence) if host_genome.size > 0 else 0.0
        
        # Normalisiere: 1% Mutation = maximale Komponente
        return min(1.0, mutation_rate / 0.01)
    
    @staticmethod
    def calculate_epigenetic_component(host_genome: Genome) -> float:
        """
        Komponente 4: Epigenetische Modulation
        
        Misst den Anteil epigenetisch modifizierter Regionen
        """
        if host_genome.size == 0:
            return 0.0
        
        marked_positions = len(host_genome.epigenetic_marks)
        fraction = marked_positions / host_genome.size
        
        # Normalisiere
        return min(1.0, fraction / 0.1)  # 10% ist Referenz
    
    @staticmethod
    def calculate_recombination_component(host_genome: Genome) -> float:
        """
        Komponente 5: Rekombinations-Metrik
        
        Misst strukturelle Variationen
        """
        # Schätze aus HGT-Regionen (Vereinfachung)
        total_variation = sum(end - start for start, end in host_genome.hgt_regions)
        
        return min(1.0, total_variation / (host_genome.size * 0.1))
    
    @staticmethod
    def calculate_stress_response_component(host: Host) -> float:
        """
        Komponente 6: Stress-Response-Metrik
        
        Misst den Stresslevel des Wirtes
        """
        return host.stress_level
    
    @staticmethod
    def calculate_total_influence(
        host: Host,
        host_original_genome: Genome,
        bacteria: List[Bacterium],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Berechne die Gesamte Genomische Einflussmetrik
        
        Args:
            host: Wirt-Organismus
            host_original_genome: Original-Genom des Wirtes
            bacteria: Liste der Bakterien
            weights: Gewichte für die Komponenten
        
        Returns:
            Normalisierte Metrik (0-1)
        """
        if weights is None:
            # Standard-Gewichte
            weights = {
                'hgt': 0.2,
                'plasmid': 0.15,
                'mutagenese': 0.25,
                'epigen': 0.15,
                'rekomb': 0.15,
                'stress': 0.1
            }
        
        components = {
            'hgt': GenomicInfluenceMetric.calculate_hgt_component(
                host.genome, bacteria
            ),
            'plasmid': GenomicInfluenceMetric.calculate_plasmid_component(
                host.genome
            ),
            'mutagenese': GenomicInfluenceMetric.calculate_mutagenesis_component(
                host.genome, host_original_genome
            ),
            'epigen': GenomicInfluenceMetric.calculate_epigenetic_component(
                host.genome
            ),
            'rekomb': GenomicInfluenceMetric.calculate_recombination_component(
                host.genome
            ),
            'stress': GenomicInfluenceMetric.calculate_stress_response_component(
                host
            )
        }
        
        # Gewichtete Summe
        total = sum(
            components[key] * weights[key]
            for key in weights.keys()
        )
        
        return {
            'total': float(min(1.0, total)),
            'components': {k: float(v) for k, v in components.items()},
            'weights': weights
        }


class PopulationDynamics:
    """
    Modellierung der Populationsdynamik
    
    System von Differentialgleichungen:
    dN_B/dt = r_B(G_H, G_B) * N_B - d_B(W) * N_B
    dW/dt = alpha * F_new - beta * C_costs
    """
    
    @staticmethod
    def growth_rate(
        bacteria_population: int,
        host_genome_size: int,
        metabolite_level: float,
        resources: float = 1.0
    ) -> float:
        """
        Wachstumsrate der Bakterien
        
        Abhängig von:
        - Verfügbaren Ressourcen
        - Genomkomplementarität
        - Metaboliten
        """
        # Logistisches Wachstum
        carrying_capacity = 1e9  # Typische Trägerkapazität
        
        growth = (
            resources * metabolite_level *
            (1 - bacteria_population / carrying_capacity)
        )
        
        return max(0.0, growth)
    
    @staticmethod
    def death_rate(
        immune_response: float,
        host_fitness: float,
        bacteria_population: int
    ) -> float:
        """
        Sterberate der Bakterien
        
        Abhängig von Wirtsimmunität und Fitness
        """
        base_death = 0.1
        immune_death = immune_response * 2.0
        
        return base_death + immune_death
    
    @staticmethod
    def fitness_change(
        new_functions: Set[str],
        metabolite_costs: float,
        stress_level: float
    ) -> float:
        """
        Änderung der Wirtsfitnessrate
        
        Alpha * F_new - Beta * C_costs
        """
        alpha = 0.1  # Fitnessvorteil pro neue Funktion
        beta = 0.05  # Kosten pro Stressreaktionen
        
        benefit = alpha * len(new_functions)
        cost = beta * (metabolite_costs + stress_level)
        
        return benefit - cost
    
    @staticmethod
    def simulate_dynamics(
        initial_bacteria: int,
        initial_host_fitness: float,
        resources: float,
        metabolite_level: float,
        immune_response_trajectory: np.ndarray,
        time_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simuliere die gekoppelten Differentialgleichungen
        
        Returns:
            (time, bacteria_population, host_fitness)
        """
        def derivatives(state, t):
            N_B, W = state
            
            # Interpoliere Immunantwort
            immune_t = np.interp(t, time_points, immune_response_trajectory)
            
            # dN_B/dt
            growth = PopulationDynamics.growth_rate(
                N_B, 4600000, metabolite_level, resources
            )
            death = PopulationDynamics.death_rate(immune_t, W, N_B)
            dN_B_dt = (growth - death) * N_B
            
            # dW/dt
            dW_dt = PopulationDynamics.fitness_change(
                set(), metabolite_level, immune_t
            )
            
            return [dN_B_dt, dW_dt]
        
        state0 = [initial_bacteria, initial_host_fitness]
        
        trajectory = odeint(
            derivatives,
            state0,
            time_points
        )
        
        return time_points, trajectory[:, 0], trajectory[:, 1]


class MicrobiomeNetwork:
    """
    Modellierung des Mikrobioms als Graph-Netzwerk
    
    Knoten: Bakterienarten
    Kanten: Interaktionen (Nährstoff-Austausch, DNA-Transfer, etc.)
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.bacteria_species: Dict[str, Bacterium] = {}
    
    def add_species(self, bacterium: Bacterium):
        """Füge Bakterienart zum Netzwerk hinzu"""
        self.bacteria_species[bacterium.name] = bacterium
        self.graph.add_node(bacterium.name)
    
    def add_interaction(
        self,
        source: str,
        target: str,
        interaction_type: str,
        strength: float
    ):
        """
        Füge Interaktion zwischen zwei Arten hinzu
        
        Args:
            source: Quellbakterium
            target: Zielbakterium
            interaction_type: "substrate", "hgt", "inhibition", "synergy"
            strength: Stärke der Interaktion (0-1)
        """
        self.graph.add_edge(
            source, target,
            type=interaction_type,
            strength=strength
        )
    
    def find_functional_clusters(self, min_cluster_size: int = 2) -> List[List[str]]:
        """
        Identifiziere funktionale Bacterial Clusters (FBCs)
        
        Ein FBC ist ein dichter Subgraph mit komplementären Funktionen
        
        Returns:
            Liste von Clustern (jeweils Liste von Bakteriennamen)
        """
        # Berechne Modularität mittels Community Detection
        from networkx.algorithms import community
        
        try:
            communities = community.greedy_modularity_communities(
                self.graph.to_undirected()
            )
            
            # Filtere nach Mindestgröße
            clusters = [
                list(c) for c in communities
                if len(c) >= min_cluster_size
            ]
            
            return clusters
        except:
            # Fallback: Return all species as single cluster
            return [list(self.bacteria_species.keys())]
    
    def calculate_complementarity(self, species1: str, species2: str) -> float:
        """
        Berechne metabolische Komplementarität zwischen zwei Arten
        
        Returns:
            Komplementaritätsgrad (0-1)
        """
        if species1 not in self.bacteria_species or species2 not in self.bacteria_species:
            return 0.0
        
        b1 = self.bacteria_species[species1]
        b2 = self.bacteria_species[species2]
        
        f1 = b1.functional_space()
        f2 = b2.functional_space()
        
        if len(f1) == 0 or len(f2) == 0:
            return 0.0
        
        # Komplementarität = 1 - (Intersection / Union)
        intersection = len(f1 & f2)
        union = len(f1 | f2)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        complementarity = 1.0 - jaccard_similarity
        
        return float(complementarity)
    
    def propagate_information_cascade(
        self,
        initial_species: str,
        metabolite: str,
        steps: int = 5
    ) -> Dict[str, float]:
        """
        Simuliere eine Informationskaskade durch das Netzwerk
        
        Args:
            initial_species: Startbakterium
            metabolite: Metabolit, der verbreitet wird
            steps: Anzahl der Propagationsschritte
        
        Returns:
            Konzentration des Metaboliten bei jeder Art
        """
        concentrations = {species: 0.0 for species in self.bacteria_species.keys()}
        concentrations[initial_species] = 1.0
        
        for _ in range(steps):
            new_concentrations = concentrations.copy()
            
            for source in self.bacteria_species.keys():
                if concentrations[source] > 0.0:
                    # Propagiere zu Nachbarn
                    for target in self.graph.neighbors(source):
                        edge_strength = self.graph[source][target].get('strength', 0.5)
                        propagated = concentrations[source] * edge_strength * 0.5
                        new_concentrations[target] += propagated
            
            concentrations = {
                k: min(1.0, v) for k, v in new_concentrations.items()
            }
        
        return concentrations
    
    def to_dict(self):
        """Konvertiere zu Dictionary für JSON-Export"""
        nodes = [
            {
                'id': node,
                'data': self.bacteria_species[node].to_dict()
            }
            for node in self.graph.nodes()
        ]
        
        edges = [
            {
                'source': source,
                'target': target,
                'type': data.get('type', 'unknown'),
                'strength': float(data.get('strength', 0.0))
            }
            for source, target, data in self.graph.edges(data=True)
        ]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
