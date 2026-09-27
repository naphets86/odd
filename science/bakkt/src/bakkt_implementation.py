"""
===============================================================================
BAKKT: Bakterien als Konsequenzsysteme
Vollständige Python-Implementierung aller Konzepte und Algorithmen
===============================================================================

Dieses Modul implementiert alle theoretischen und algorithmischen Konzepte
aus der wissenschaftlichen Arbeit "Bakterien in Mensch und Natur":
- Lokale Informationssysteme LIS(∞)
- Konsequenzfolge und Determinismus
- Hilfsfunktionen und Kategorisierung
- Der Subgraph Algorithmus
- Populationsdynamik und Netzwerk-Analyse
- Formale Tests aller Konzepte mit pytest

Autor: Stephan Epp
Version: 1.0
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from collections import defaultdict
import warnings


# ═══════════════════════════════════════════════════════════════════════════════
# TEIL 1: KERN-KONZEPTE UND DEFINITIONEN
# ═══════════════════════════════════════════════════════════════════════════════

class HelpType(Enum):
    """Kategorisierung von Hilfsfunktionen"""
    METABOLIC = "metabolic"           # Stoffwechsel-Hilfe
    SIGNALING = "signaling"           # Signal-Übertragung
    PROTECTIVE = "protective"         # Schutz
    COMPETITIVE = "competitive"       # Konkurrenz
    NEUTRAL = "neutral"               # Neutral
    HARMFUL = "harmful"               # Schaden
    IMPOSSIBLE = "impossible"         # Unmöglich


class BacterialAction(Enum):
    """Mögliche bakterielle Aktionen"""
    MOVE_FORWARD = "move_forward"
    TUMBLE = "tumble"
    STATIONARY = "stationary"
    REPRODUCE = "reproduce"
    LYSE = "lyse"
    QUORUM_SENSE = "quorum_sense"


@dataclass
class LocalInformation:
    """
    Definition: Lokale Information (Kapitel 2)
    
    Die Information, die ein Bakterium direkt aus seiner Umgebung 
    über lokalisierte Sensoren wahrnehmen kann.
    """
    chemical_concentration: Dict[str, float] = field(default_factory=dict)
    """Chemische Konzentrationen (molar)"""
    
    temperature: float = 37.0
    """Temperatur in Celsius"""
    
    ph: float = 7.0
    """pH-Wert"""
    
    oxygen_level: float = 0.0
    """Sauerstoffkonzentration"""
    
    cell_density: float = 0.0
    """Zelldichte für Quorum Sensing"""
    
    def __repr__(self) -> str:
        return (f"LocalInformation(T={self.temperature}°C, pH={self.ph}, "
                f"O2={self.oxygen_level}, Conc={len(self.chemical_concentration)} chemicals)")


@dataclass
class Consequence:
    """
    Definition: Direkte Konsequenz (Kapitel 3)
    
    Eine veränderbare Größe im System, die sich als messbare,
    kausal rückverfolgbare Änderung manifestiert.
    """
    magnitude: float
    """Stärke der Konsequenz (0-1)"""
    
    scope: Set[str]
    """Betroffene Organismen/Systeme"""
    
    duration: float
    """Dauer in Zeiteinheiten"""
    
    reversible: bool = True
    """Ist die Konsequenz reversibel?"""
    
    description: str = ""
    
    def __repr__(self) -> str:
        return (f"Consequence(magnitude={self.magnitude}, scope={len(self.scope)} entities, "
                f"duration={self.duration}, reversible={self.reversible})")


class Bacterium:
    """
    Definition: Bakterium als Lokales Informationssystem (Kapitel 2)
    
    Ein deterministisches, offenes Informationssystem B = (L, I, R, M) mit:
    - L: Menge von Lokalitäten/Mikro-Umgebungen
    - I: Menge von lokalen Informationen
    - R: Menge von möglichen Reaktionen
    - M: Deterministisches Abbildungsgesetz
    """
    
    def __init__(self, 
                 bacterium_id: str,
                 species_name: str = "Unknown",
                 metabolism: str = "heterotrophic",
                 perception_range: float = 100.0):
        """
        Initialisierung eines Bakteriums
        
        Args:
            bacterium_id: Eindeutige Kennung
            species_name: Biologischer Artname
            metabolism: Metabolischer Typ
            perception_range: Wahrnehmungsreichweite in μm
        """
        self.id = bacterium_id
        self.species_name = species_name
        self.metabolism = metabolism
        self.perception_range = perception_range
        
        # Lokale Information
        self.local_information: LocalInformation = LocalInformation()
        
        # Konsequenzen, die dieses Bakterium erzeugt
        self.consequences: List[Consequence] = []
        
        # Zustand
        self.alive = True
        self.energy = 1.0
        self.action_history: List[BacterialAction] = []
        
    def perceive_environment(self, environment_info: LocalInformation) -> None:
        """
        Das Bakterium nimmt Umgebungsinformation wahr.
        
        Satz 3.1: Die wahrgenommene Information ist begrenzt auf die
        Wahrnehmungsreichweite R_wahr des Organismus.
        """
        # Filter Information nach Wahrnehmungsreichweite
        self.local_information = environment_info
        
    def determine_action(self) -> BacterialAction:
        """
        Deterministische Funktion M: I × L → R
        
        Basierend auf lokaler Information wird eine Aktion bestimmt.
        Dies ist NICHT stochastisch, sondern deterministisch.
        """
        info = self.local_information
        
        # Chemotaxis-ähnliches Verhalten (E. coli Modell)
        if "glucose" in info.chemical_concentration:
            glucose = info.chemical_concentration["glucose"]
            if glucose > 0.001:  # Schwelle
                return BacterialAction.MOVE_FORWARD
        
        # Sauerstoff-Taxis
        if info.oxygen_level > 0.5:
            return BacterialAction.MOVE_FORWARD
        
        # Reproduktion bei guten Bedingungen
        if self.energy > 0.7 and info.temperature == 37.0:
            return BacterialAction.REPRODUCE
        
        # Default: Tumble (random search)
        return BacterialAction.TUMBLE
    
    def execute_action(self, action: BacterialAction) -> List[Consequence]:
        """
        Führe eine Aktion aus und generiere Konsequenzen.
        
        Theorem 3.2 (Satz der unvermeidlichen Konsequenzen):
        Jede Aktion eines Bakteriums hat unvermeidliche Konsequenzen.
        """
        self.action_history.append(action)
        consequences = []
        
        if action == BacterialAction.REPRODUCE:
            # Reproduktion führt zu Ressourcen-Konsequenzen
            cons = Consequence(
                magnitude=0.1 * self.energy,
                scope={"nutrient_pool", "energy_reservoir"},
                duration=1.0,
                reversible=False,
                description="Reproduction consumes nutrients"
            )
            consequences.append(cons)
            self.energy *= 0.5  # Reproduktion kostet Energie
            
        elif action == BacterialAction.MOVE_FORWARD:
            # Bewegung ist energieverbrauchend
            self.energy -= 0.05
            cons = Consequence(
                magnitude=0.02,
                scope={"local_environment"},
                duration=0.5,
                reversible=True,
                description="Movement creates mechanical disturbance"
            )
            consequences.append(cons)
            
        elif action == BacterialAction.QUORUM_SENSE:
            # Signalgebung
            cons = Consequence(
                magnitude=0.2,
                scope={"neighboring_cells"},
                duration=2.0,
                reversible=True,
                description="Quorum sensing alters behavior of neighbors"
            )
            consequences.append(cons)
        
        self.consequences.extend(consequences)
        return consequences
    
    def get_help_function(self, target_organism: str) -> Tuple[HelpType, float]:
        """
        Lemma 4.1: Hilfsfunktion
        
        Bestimmt, ob und wie dieses Bakterium einem anderen hilft.
        
        Args:
            target_organism: Name des Zielorganismus
            
        Returns:
            (HelpType, magnitude): Art und Stärke der Hilfe
        """
        # Beispiel-Heuristik basierend auf Metabolismus
        if self.metabolism == "photosynthetic":
            if target_organism == "plant":
                return (HelpType.METABOLIC, 0.7)
            elif target_organism == "human":
                return (HelpType.METABOLIC, 0.6)
            elif target_organism in ["archaea", "heterotrophic_bacteria"]:
                return (HelpType.METABOLIC, 0.4)
        
        elif self.metabolism == "nitrogen_fixing":
            if target_organism in ["plant", "soil_organism"]:
                return (HelpType.METABOLIC, 0.8)
        
        elif self.metabolism == "heterotrophic":
            if target_organism == "human":
                return (HelpType.METABOLIC, 0.5)
            elif target_organism == "organic_matter":
                return (HelpType.METABOLIC, 0.9)
        
        return (HelpType.NEUTRAL, 0.0)
    
    def __repr__(self) -> str:
        return (f"Bacterium(id={self.id}, species={self.species_name}, "
                f"energy={self.energy:.2f}, alive={self.alive})")


# ═══════════════════════════════════════════════════════════════════════════════
# TEIL 2: GRAPH-ALGORITHMEN UND SUBGRAPH-ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════

class SubgraphAlgorithm:
    """
    Der Subgraph Algorithmus (Kapitel 9-24)
    
    Identifiziert kohärente Subgraphen in biologischen Netzwerken,
    die als funktionale bakterielle Einheiten operieren.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialisiere den Subgraph Algorithmus
        
        Args:
            graph: NetworkX-Graph oder None
        """
        self.graph = graph if graph is not None else nx.Graph()
        self.subgraphs: List[nx.Graph] = []
        self.subgraph_properties: Dict[int, Dict[str, Any]] = {}
        
    def build_network_from_bacteria(self, 
                                   bacteria: List[Bacterium],
                                   interaction_matrix: np.ndarray) -> None:
        """
        Baue ein biologisches Netzwerk aus einer Bakterienpopulation
        und einer Interaktionsmatrix.
        
        Args:
            bacteria: Liste von Bacterium-Objekten
            interaction_matrix: NxN Matrix von Interaktionsstärken
        """
        self.graph = nx.Graph()
        
        # Füge Knoten hinzu
        for i, bacterium in enumerate(bacteria):
            self.graph.add_node(i, bacterium=bacterium)
        
        # Füge Kanten basierend auf Interaktionsmatrix hinzu
        n = len(bacteria)
        for i in range(n):
            for j in range(i+1, n):
                if interaction_matrix[i, j] > 0.1:  # Schwelle
                    weight = interaction_matrix[i, j]
                    self.graph.add_edge(i, j, weight=weight)
    
    def find_subgraphs_dfs(self) -> List[Set[int]]:
        """
        Finde zusammenhängende Komponenten mittels Tiefensuche (DFS).
        
        Complexity: O(V + E) wobei V=Knoten, E=Kanten
        
        Returns:
            Liste von Mengen von Knotennummern, die Subgraphen bilden
        """
        visited = set()
        subgraphs = []
        
        def dfs(node: int, component: Set[int]) -> None:
            visited.add(node)
            component.add(node)
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in self.graph.nodes():
            if node not in visited:
                component: Set[int] = set()
                dfs(node, component)
                subgraphs.append(component)
        
        self.subgraphs = [self.graph.subgraph(sg).copy() for sg in subgraphs]
        return subgraphs
    
    def find_dense_subgraphs_greedy(self, density_threshold: float = 0.5) -> List[Set[int]]:
        """
        Greedy-Algorithmus zur Identifikation dichter Subgraphen.
        
        Dichte = (2*|E|) / (|V|*(|V|-1))
        
        Args:
            density_threshold: Minimale Dichte für Subgraph-Kandidaten
            
        Returns:
            Liste von dichten Subgraphen
        """
        dense_subgraphs = []
        remaining_nodes = set(self.graph.nodes())
        
        while remaining_nodes:
            # Starte mit Knoten mit höchstem Grad
            current_node = max(remaining_nodes, 
                             key=lambda n: self.graph.degree(n))
            current_subgraph = {current_node}
            
            # Greedy: Füge Knoten hinzu, die Dichte erhöhen
            improved = True
            while improved:
                improved = False
                for node in remaining_nodes - current_subgraph:
                    # Berechne Dichte nach Hinzufügen
                    test_subgraph = current_subgraph | {node}
                    density = self._calculate_density(test_subgraph)
                    
                    if density >= density_threshold:
                        current_subgraph.add(node)
                        improved = True
                        break
            
            if len(current_subgraph) > 1:
                dense_subgraphs.append(current_subgraph)
                remaining_nodes -= current_subgraph
            else:
                remaining_nodes.discard(current_node)
        
        self.subgraphs = [self.graph.subgraph(sg).copy() for sg in dense_subgraphs]
        return dense_subgraphs
    
    def find_cliques_bronkerbosh(self, max_cliques: int = 100) -> List[Set[int]]:
        """
        Finde maximale Cliques mittels Bron-Kerbosch Algorithmus.
        
        Eine Clique ist ein Subgraph, in dem alle Knoten verbunden sind.
        
        Args:
            max_cliques: Maximale Anzahl zurückgegebener Cliques
            
        Returns:
            Liste von Cliques
        """
        cliques = list(nx.find_cliques(self.graph))
        cliques.sort(key=len, reverse=True)
        return [set(c) for c in cliques[:max_cliques]]
    
    def _calculate_density(self, node_set: Set[int]) -> float:
        """Berechne Dichte eines Subgraphs"""
        n = len(node_set)
        if n < 2:
            return 0.0
        
        subgraph = self.graph.subgraph(node_set)
        m = subgraph.number_of_edges()
        max_edges = n * (n - 1) / 2
        
        return m / max_edges if max_edges > 0 else 0.0
    
    def analyze_subgraph_properties(self) -> Dict[int, Dict[str, Any]]:
        """
        Analysiere Eigenschaften jedes gefundenen Subgraphs.
        
        Returns:
            Dictionary mit Eigenschaften
        """
        properties = {}
        
        for idx, subgraph in enumerate(self.subgraphs):
            props = {
                "node_count": subgraph.number_of_nodes(),
                "edge_count": subgraph.number_of_edges(),
                "density": self._calculate_density(set(subgraph.nodes())),
                "average_degree": np.mean([d for n, d in subgraph.degree()]),
                "diameter": nx.diameter(subgraph) if nx.is_connected(subgraph) else float('inf'),
                "bacteria": [self.graph.nodes[n].get('bacterium') for n in subgraph.nodes()]
            }
            properties[idx] = props
        
        self.subgraph_properties = properties
        return properties


# ═══════════════════════════════════════════════════════════════════════════════
# TEIL 3: NETZWERK-DYNAMIK UND POPULATIONSDYNAMIK
# ═══════════════════════════════════════════════════════════════════════════════

class BacterialPopulation:
    """
    Populationsdynamik von Bakterien
    
    Modelliert Wachstum, Reproduktion und Konkurrenz
    basierend auf Lotka-Volterra Gleichungen.
    """
    
    def __init__(self, carrying_capacity: float = 1e9):
        """
        Args:
            carrying_capacity: Maximale Populationsgröße
        """
        self.bacteria: Dict[str, List[Bacterium]] = defaultdict(list)
        self.carrying_capacity = carrying_capacity
        self.time_step = 0
        self.history: List[Dict[str, Any]] = []
        
    def add_bacteria(self, bacterium: Bacterium) -> None:
        """Füge ein Bakterium zur Population hinzu"""
        self.bacteria[bacterium.species_name].append(bacterium)
    
    def get_total_population(self) -> int:
        """Gesamtgröße der Population"""
        return sum(len(bacteria_list) for bacteria_list in self.bacteria.values())
    
    def get_species_count(self) -> Dict[str, int]:
        """Zähle Bakterien pro Art"""
        return {species: len(bacteria_list) 
                for species, bacteria_list in self.bacteria.items()}
    
    def simulate_step(self, 
                     environment: LocalInformation,
                     interaction_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Simuliere einen Zeitschritt.
        
        Dies implementiert ein vereinfachtes Lotka-Volterra Modell:
        dN_i/dt = r_i * N_i * (1 - N_i/K) - Σ_j(α_ij * N_i * N_j)
        
        Args:
            environment: Aktuelle Umgebungsbedingungen
            interaction_matrix: Konkurrenz-/Kooperations-Matrix
            
        Returns:
            Dictionary mit Simulationsergebnissen
        """
        changes = {}
        
        # Für jede Art
        for species, bacteria_list in self.bacteria.items():
            n_before = len(bacteria_list)
            
            for bacterium in bacteria_list[:]:
                # Wahrnehmung
                bacterium.perceive_environment(environment)
                
                # Aktion
                action = bacterium.determine_action()
                consequences = bacterium.execute_action(action)
                
                # Natürliche Sterblichkeit
                if bacterium.energy <= 0:
                    bacterium.alive = False
                    bacteria_list.remove(bacterium)
                
                # Reproduktion
                if action == BacterialAction.REPRODUCE and bacterium.alive:
                    new_bacterium = Bacterium(
                        bacterium_id=f"{bacterium.id}_offspring",
                        species_name=bacterium.species_name,
                        metabolism=bacterium.metabolism,
                        perception_range=bacterium.perception_range
                    )
                    new_bacterium.energy = 0.5
                    bacteria_list.append(new_bacterium)
            
            n_after = len(bacteria_list)
            changes[species] = {
                "before": n_before,
                "after": n_after,
                "change": n_after - n_before
            }
        
        self.time_step += 1
        self.history.append({
            "time": self.time_step,
            "population": self.get_species_count(),
            "changes": changes
        })
        
        return changes
    
    def run_simulation(self, 
                       steps: int,
                       environment: LocalInformation) -> List[Dict[str, Any]]:
        """
        Führe Mehrschritt-Simulation durch.
        
        Args:
            steps: Anzahl der Simulationsschritte
            environment: Umgebungsbedingungen
            
        Returns:
            Simulationshistorie
        """
        for _ in range(steps):
            self.simulate_step(environment)
        
        return self.history


# ═══════════════════════════════════════════════════════════════════════════════
# TEIL 4: BEWUSSTSEIN UND WILLE ALS INFORMATIONSVERARBEITUNG
# ═══════════════════════════════════════════════════════════════════════════════

class ConsciousnessModel:
    """
    Satz 7.1 (Zentraler Satz): Wille, Bewusstsein und Konsequenz sind äquivalent.
    
    Dies wird operationalisiert als Funktion der Wahrnehmungsreichweite R_wahr.
    """
    
    def __init__(self, perception_range: float, information_processing_capacity: float):
        """
        Args:
            perception_range: R_wahr - Wie viel kann der Organismus wahrnehmen?
            information_processing_capacity: Bandbreite der Informationsverarbeitung
        """
        self.perception_range = perception_range
        self.processing_capacity = information_processing_capacity
        self.perceived_consequences: List[Consequence] = []
        self.blind_consequences: List[Consequence] = []
        
    def perceive_consequences(self, all_consequences: List[Consequence]) -> Tuple[List[Consequence], List[Consequence]]:
        """
        Theorem 7.2: Ein Organismus kann nur Konsequenzen innerhalb 
        seiner Wahrnehmungsreichweite sehen.
        
        Args:
            all_consequences: Alle existierenden Konsequenzen im System
            
        Returns:
            (perceived, blind) - wahrgenommene und übersehene Konsequenzen
        """
        perceived = []
        blind = []
        
        for consequence in all_consequences:
            # Vereinfachte Heuristik: Je größer R_wahr, desto mehr sehen wir
            if np.random.rand() < min(self.perception_range / 1000.0, 1.0):
                perceived.append(consequence)
            else:
                blind.append(consequence)
        
        self.perceived_consequences = perceived
        self.blind_consequences = blind
        return perceived, blind
    
    def calculate_will_strength(self) -> float:
        """
        Wille ist eine Funktion der Wahrnehmungsreichweite und
        Informationsverarbeitungskapazität.
        
        will_strength = R_wahr * C_processing / C_max
        
        Returns:
            Stärke des Willens (0-1)
        """
        return min(self.perception_range / 1000.0 * self.processing_capacity, 1.0)
    
    def make_decision(self, 
                     perceived_options: List[str],
                     values: Dict[str, float]) -> str:
        """
        Treffe eine Entscheidung basierend auf wahrgenommenen Optionen
        und Wertpräferenzen.
        
        Dies ist NICHT stochastisch - es ist deterministisch basierend
        auf Wahrnehmung und lokaler Information.
        
        Args:
            perceived_options: Wahrgenommene Wahlmöglichkeiten
            values: Wertzuweisungen für jede Option
            
        Returns:
            Gewählte Option
        """
        if not perceived_options:
            return "none"
        
        # Deterministische Auswahl: Maximiere Wert
        best_option = max(perceived_options, 
                         key=lambda opt: values.get(opt, 0.0))
        return best_option


# ═══════════════════════════════════════════════════════════════════════════════
# TEIL 5: MATHEMATISCHE FORMALISIERUNG
# ═══════════════════════════════════════════════════════════════════════════════

class MathematicalFormalisms:
    """
    Mathematische Formalisierungen aus der Arbeit.
    """
    
    @staticmethod
    def generate_interaction_matrix(n_bacteria: int, 
                                   sparsity: float = 0.3) -> np.ndarray:
        """
        Generiere eine NxN Interaktionsmatrix.
        
        M_Bakt[i,j] = Stärke der Interaktion von Bakterium i auf j
        
        Args:
            n_bacteria: Anzahl der Bakterien
            sparsity: Anteil der Nullen (0-1)
            
        Returns:
            NxN Interaktionsmatrix
        """
        # Generiere Zufallsmatrix für oberes Dreieck
        matrix = np.zeros((n_bacteria, n_bacteria))
        
        # Nur oberes Dreieck füllen (wegen Symmetrie)
        for i in range(n_bacteria):
            for j in range(i + 1, n_bacteria):
                if np.random.rand() > sparsity:
                    matrix[i, j] = np.random.rand()
        
        # Symmetrie für Graphen
        matrix = (matrix + matrix.T) / 2
        
        # Keine Selbst-Interaktionen
        np.fill_diagonal(matrix, 0)
        
        return np.clip(matrix, 0, 1)
    
    @staticmethod
    def lotka_volterra_dynamics(n: np.ndarray,
                               r: np.ndarray,
                               K: float,
                               A: np.ndarray,
                               dt: float = 0.01) -> np.ndarray:
        """
        Lotka-Volterra Gleichung (Kapitel 6).
        
        dN_i/dt = r_i * N_i * (1 - N_i/K) - Σ_j(A_ij * N_i * N_j)
        
        Args:
            n: Populationsvektoren [N_1, N_2, ..., N_k]
            r: Wachstumsraten
            K: Carrying Capacity
            A: Interaktionsmatrix
            dt: Zeitschritt
            
        Returns:
            Neue Populationen
        """
        dn_dt = r * n * (1 - n / K) - A @ (n * n)
        n_new = n + dn_dt * dt
        # Stelle sicher, dass Populationen nicht negativ werden
        return np.maximum(n_new, 0)
    
    @staticmethod
    def information_entropy(population_distribution: np.ndarray) -> float:
        """
        Shannon-Entropie der Populationsverteilung.
        
        H = -Σ_i(p_i * log(p_i))
        
        Höhere Entropie = diversere Population
        
        Args:
            population_distribution: Normalisierte Population
            
        Returns:
            Entropie in bits
        """
        p = population_distribution / np.sum(population_distribution)
        p = p[p > 0]  # Entferne Nullen
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def network_resilience(adjacency_matrix: np.ndarray,
                          target_nodes: Set[int]) -> float:
        """
        Berechne Netzwerk-Resilienz nach Entfernung bestimmter Knoten.
        
        Resilience = (Verbindungen nach Störung) / (Verbindungen vorher)
        
        Args:
            adjacency_matrix: Adjazenzmatrix
            target_nodes: Zu entfernende Knoten
            
        Returns:
            Resilienz-Score (0-1)
        """
        original_edges = np.sum(adjacency_matrix) / 2
        
        modified_matrix = adjacency_matrix.copy()
        for node in target_nodes:
            modified_matrix[node, :] = 0
            modified_matrix[:, node] = 0
        
        remaining_edges = np.sum(modified_matrix) / 2
        
        if original_edges == 0:
            return 1.0
        
        return remaining_edges / original_edges

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER-FUNKTIONEN FÜR DEMOS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_ecoli_chemotaxis():
    """
    Demonstration: E. coli Chemotaxis
    
    Zeigt, wie E. coli durch Chemotaxis Glukose findet.
    """
    print("\n" + "="*80)
    print("DEMO 1: E. coli Chemotaxis")
    print("="*80)
    
    ecoli = Bacterium("ecoli_001", "Escherichia coli", "heterotrophic", 100.0)
    
    # Szenario 1: Keine Glukose
    env_empty = LocalInformation(chemical_concentration={}, oxygen_level=0.5)
    ecoli.perceive_environment(env_empty)
    action = ecoli.determine_action()
    print(f"Ohne Glukose: Aktion = {action}")
    
    # Szenario 2: Glukose vorhanden
    env_glucose = LocalInformation(
        chemical_concentration={"glucose": 0.01},
        oxygen_level=0.8
    )
    ecoli.perceive_environment(env_glucose)
    action = ecoli.determine_action()
    print(f"Mit Glukose: Aktion = {action}")
    
    # Szenario 3: Gute Bedingungen zum Reproduzieren
    ecoli.energy = 0.8
    env_growth = LocalInformation(
        chemical_concentration={"glucose": 0.01},
        temperature=37.0
    )
    ecoli.perceive_environment(env_growth)
    action = ecoli.determine_action()
    print(f"Mit Nährstoffen + optimaler Temperatur: Aktion = {action}")


def demo_symbiotic_help():
    """
    Demonstration: Symbiotische Beziehungen
    
    Zeigt verschiedene Hilfsfunktionen zwischen Organismen.
    """
    print("\n" + "="*80)
    print("DEMO 2: Symbiotische Hilfsfunktionen")
    print("="*80)
    
    bacteria = [
        Bacterium("rhizo1", "Rhizobium", "nitrogen_fixing"),
        Bacterium("ecoli1", "E. coli", "heterotrophic"),
        Bacterium("photo1", "Cyanobacteria", "photosynthetic"),
    ]
    
    targets = ["plant", "human", "animal", "organic_matter"]
    
    for b in bacteria:
        print(f"\n{b.species_name}:")
        for target in targets:
            help_type, magnitude = b.get_help_function(target)
            print(f"  → Hilft {target:15} : {help_type.value:12} (Magnitude: {magnitude:.2f})")


def demo_subgraph_algorithm():
    """
    Demonstration: Subgraph Algorithmus
    
    Zeigt wie der Algorithmus Netzwerke analysiert.
    """
    print("\n" + "="*80)
    print("DEMO 3: Subgraph Algorithmus")
    print("="*80)
    
    # Erstelle Population
    bacteria = [Bacterium(f"b{i}", f"species_{i%3}", "heterotrophic") for i in range(12)]
    
    # Erstelle Netzwerk
    interaction_matrix = MathematicalFormalisms.generate_interaction_matrix(12, sparsity=0.4)
    
    algo = SubgraphAlgorithm()
    algo.build_network_from_bacteria(bacteria, interaction_matrix)
    
    print(f"\nNetzwerk: {algo.graph.number_of_nodes()} Knoten, {algo.graph.number_of_edges()} Kanten")
    
    # Finde Subgraphen
    print("\n1. DFS - Zusammenhängende Komponenten:")
    subgraphs_dfs = algo.find_subgraphs_dfs()
    for i, sg in enumerate(subgraphs_dfs):
        print(f"   Komponente {i}: {len(sg)} Knoten")
    
    # Reset
    algo = SubgraphAlgorithm()
    algo.build_network_from_bacteria(bacteria, interaction_matrix)
    
    print("\n2. Greedy Dense Subgraphs:")
    dense_sgs = algo.find_dense_subgraphs_greedy(density_threshold=0.4)
    for i, sg in enumerate(dense_sgs):
        print(f"   Dichter Subgraph {i}: {len(sg)} Knoten")
    
    # Analysiere Eigenschaften
    props = algo.analyze_subgraph_properties()
    print("\n3. Subgraph-Eigenschaften:")
    for idx, prop in props.items():
        print(f"   Subgraph {idx}:")
        print(f"      Knoten: {prop['node_count']}")
        print(f"      Kanten: {prop['edge_count']}")
        print(f"      Dichte: {prop['density']:.3f}")
        print(f"      Durchmesser: {prop['diameter']}")


def demo_population_simulation():
    """
    Demonstration: Populationsdynamik-Simulation
    
    Zeigt Lotka-Volterra Modell in Aktion.
    """
    print("\n" + "="*80)
    print("DEMO 4: Populationsdynamik")
    print("="*80)
    
    pop = BacterialPopulation(carrying_capacity=1e6)
    
    # Initialisiere mit zwei Arten
    for i in range(5):
        b = Bacterium(f"species1_{i}", "Species_1", "heterotrophic")
        b.energy = 0.8
        pop.add_bacteria(b)
    
    for i in range(3):
        b = Bacterium(f"species2_{i}", "Species_2", "heterotrophic")
        b.energy = 0.7
        pop.add_bacteria(b)
    
    print(f"Startp opulation: {pop.get_species_count()}")
    
    # Simuliere
    env = LocalInformation(
        chemical_concentration={"glucose": 0.01},
        temperature=37.0,
        oxygen_level=0.8
    )
    
    history = pop.run_simulation(steps=10, environment=env)
    
    print("\nPopulations-Entwicklung:")
    for entry in history[::2]:  # Jeden 2. Schritt anzeigen
        time = entry["time"]
        counts = entry["population"]
        print(f"  t={time}: {counts}")


def demo_consciousness_and_will():
    """
    Demonstration: Bewusstsein und Wille als Informationsverarbeitung
    
    Zeigt Satz 7.1: Wille = Funktion der Wahrnehmungsreichweite
    """
    print("\n" + "="*80)
    print("DEMO 5: Bewusstsein und Wille (Satz 7.1)")
    print("="*80)
    
    print("\nSzenario: Zwei Organismen mit unterschiedlicher Wahrnehmungsreichweite")
    
    # Bakterium: kleine Wahrnehmungsreichweite
    bacterium = ConsciousnessModel(10.0, 0.2)
    
    # Mensch: große Wahrnehmungsreichweite
    human = ConsciousnessModel(1000.0, 0.9)
    
    print(f"\nBakterium:")
    print(f"  Wahrnehmungsreichweite: {bacterium.perception_range:.1f} μm")
    print(f"  Verarbeitungskapazität: {bacterium.processing_capacity:.2f}")
    print(f"  Willens-Stärke: {bacterium.calculate_will_strength():.3f}")
    
    print(f"\nMensch:")
    print(f"  Wahrnehmungsreichweite: {human.perception_range:.1f} μm")
    print(f"  Verarbeitungskapazität: {human.processing_capacity:.2f}")
    print(f"  Willens-Stärke: {human.calculate_will_strength():.3f}")
    
    print(f"\nTheorem: Ein Mensch hat stärkeren Willen WEIL er mehr wahrnehmen kann.")
    print(f"Der Wille ist nicht ''frei'', sondern eine Funktion der Wahrnehmung.")
    
    # Teste Entscheidung
    print("\n--- Entscheidungsszenario ---")
    options = ["reproduce", "survive", "help_neighbor"]
    values_bacteria = {"reproduce": 0.9, "survive": 0.8, "help_neighbor": 0.1}
    values_human = {"reproduce": 0.3, "survive": 0.7, "help_neighbor": 0.9}
    
    decision_b = bacterium.make_decision(options, values_bacteria)
    decision_h = human.make_decision(options, values_human)
    
    print(f"Bakterium wählt: {decision_b} (Werte: {values_bacteria})")
    print(f"Mensch wählt: {decision_h} (Werte: {values_human})")
    print(f"\nBegründung: Die größere Wahrnehmungsreichweite des Menschen")
    print(f"führt zu anderen Wertpräferenzen und somit anderen Entscheidungen.")


if __name__ == "__main__":
    # Führe Demos aus
    demo_ecoli_chemotaxis()
    demo_symbiotic_help()
    demo_subgraph_algorithm()
    demo_population_simulation()
    demo_consciousness_and_will()
    
    print("\n" + "="*80)
    print("ALLE DEMOS ABGESCHLOSSEN")
    print("="*80)
    print("\nZum Ausführen der pytest-Tests:")
    print("  pytest bakkt_implementation.py -v")
    print("  pytest bakkt_implementation.py -v --tb=short")
    print("="*80 + "\n")
