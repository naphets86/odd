"""
Biologische Netzwerkanalyse mit dem Subgraph Algorithmus

Analysiert metabolische Pfade und Protein Netzwerke mit dem Subgraph Algorithmus

Autor: Stephan Epp
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json

from subgraph import Subgraph

class BiologicalNetwork:
    """Repräsentiert ein biologisches Netzwerk (metabolischer Pfad, Protein-Interaktion, etc.)"""
    
    def __init__(self, name: str, nodes: List[str], adjacency_matrix: np.ndarray):
        self.name = name
        self.nodes = nodes  # Knotenbeschriftungen (z.B. Metabolitnamen, Protein-IDs)
        self.matrix = adjacency_matrix
        self.n = len(nodes)
        
        if self.matrix.shape != (self.n, self.n):
            raise ValueError(f"Adjazenzmatrix muss {self.n}x{self.n} sein")
    
    def __repr__(self):
        return f"BiologicalNetwork('{self.name}', n={self.n})"
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Gibt Liste von Kanten als (Quelle, Ziel) Tupel zurück"""
        edges = []
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i, j] == 1:
                    edges.append((self.nodes[i], self.nodes[j]))
        return edges
    
    def to_dict(self) -> Dict:
        """Exportiert Netzwerk in Wörterbuch-Format"""
        return {
            "name": self.name,
            "nodes": self.nodes,
            "adjacency_matrix": self.matrix.tolist(),
            "edges": self.get_edges()
        }


class PathwayAnalyzer:
    """Analysiert biologische Pfade mit dem Subgraph Algorithmus"""
    
    def __init__(self):
        self.comparator = Subgraph()
        self.networks: Dict[str, BiologicalNetwork] = {}
    
    def add_network(self, network: BiologicalNetwork):
        """Fügt ein biologisches Netzwerk zum Analyzer hinzu"""
        self.networks[network.name] = network
        print(f"✓ Netzwerk hinzugefügt: {network.name} ({network.n} Knoten)")
    
    def compare_networks(self, name_a: str, name_b: str) -> Dict:
        """Vergleicht zwei biologische Netzwerke"""
        if name_a not in self.networks or name_b not in self.networks:
            raise ValueError("Beide Netzwerke müssen zuerst hinzugefügt werden")
        
        net_a = self.networks[name_a]
        net_b = self.networks[name_b]
        
        print(f"\n🔬 Vergleich: {name_a} vs {name_b}")
        print(f"   {name_a}: {net_a.n} Knoten, {len(net_a.get_edges())} Kanten")
        print(f"   {name_b}: {net_b.n} Knoten, {len(net_b.get_edges())} Kanten")
        
        # Verwende Subgraph Algorithmus
        result = self.comparator.compare_graphs(net_a.matrix, net_b.matrix)
        
        return {
            "network_a": name_a,
            "network_b": name_b,
            "decision": result,
            "interpretation": self._interpret_result(result, name_a, name_b)
        }
    
    def _interpret_result(self, result: str, name_a: str, name_b: str) -> str:
        """Interpretiert das Vergleichsergebnis"""
        if result == "keep_A":
            return f"'{name_a}' enthält mehr Informationen (Obermenge)"
        elif result == "keep_B":
            return f"'{name_b}' enthält mehr Informationen (Obermenge)"
        elif result == "keep_both":
            return "Beide Netzwerke enthalten einzigartige Informationen (unvergleichbar)"
        elif result == "keep_either":
            return "Netzwerke sind identisch"
        else:
            return f"Unbekanntes Ergebnis: {result}"
    
    def find_common_pathways(self) -> List[Tuple[str, str, str]]:
        """Findet alle Netzwerke mit Subgraph-Beziehungen"""
        results = []
        network_names = list(self.networks.keys())
        
        for i, name_a in enumerate(network_names):
            for name_b in network_names[i+1:]:
                comparison = self.compare_networks(name_a, name_b)
                if comparison["decision"] in ["keep_A", "keep_B", "keep_either"]:
                    results.append((name_a, name_b, comparison["decision"]))
        
        return results


# Beispiel biologische Netzwerke
def create_glycolysis_pathway() -> BiologicalNetwork:
    """Vereinfachter Glykolyse-Pfad (Glukoseabbau)"""
    nodes = ["Glucose", "G6P", "F6P", "FBP", "DHAP", "G3P", "Pyruvate"]
    # Linearer Pfad: Glucose -> G6P -> F6P -> FBP -> DHAP/G3P -> Pyruvate
    matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0],  # Glucose -> G6P
        [0, 0, 1, 0, 0, 0, 0],  # G6P -> F6P
        [0, 0, 0, 1, 0, 0, 0],  # F6P -> FBP
        [0, 0, 0, 0, 1, 1, 0],  # FBP -> DHAP/G3P
        [0, 0, 0, 0, 0, 1, 0],  # DHAP -> G3P
        [0, 0, 0, 0, 0, 0, 1],  # G3P -> Pyruvate
        [0, 0, 0, 0, 0, 0, 0]   # Pyruvate (Ende)
    ])
    return BiologicalNetwork("Glycolysis", nodes, matrix)


def create_partial_glycolysis() -> BiologicalNetwork:
    """Teilweise Glykolyse (Teilmenge des vollständigen Pfads)"""
    nodes = ["Glucose", "G6P", "F6P", "FBP"]
    matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    return BiologicalNetwork("Partial_Glycolysis", nodes, matrix)


def create_krebs_cycle_fragment() -> BiologicalNetwork:
    """Fragment des Krebs-Zyklus (TCA-Zyklus)"""
    nodes = ["Citrate", "Isocitrate", "α-Ketoglutarate", "Succinyl-CoA"]
    matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    return BiologicalNetwork("Krebs_Cycle_Fragment", nodes, matrix)


def create_protein_interaction_network() -> BiologicalNetwork:
    """Einfaches Protein-Protein-Interaktionsnetzwerk"""
    nodes = ["p53", "MDM2", "ATM", "DNA-PK"]
    # p53 <-> MDM2, ATM -> p53, DNA-PK -> ATM
    matrix = np.array([
        [0, 1, 0, 0],  # p53 -> MDM2
        [1, 0, 0, 0],  # MDM2 -> p53 (bidirektional)
        [1, 0, 0, 0],  # ATM -> p53
        [0, 0, 1, 0]   # DNA-PK -> ATM
    ])
    return BiologicalNetwork("DNA_Damage_Response", nodes, matrix)


def main():
    """Hauptdemonstration des Gen-Projekts"""
    print("=" * 60)
    print("GEN - Biologische Netzwerkanalyse mit Subgraph Algorithmus")
    print("=" * 60)
    
    # Analyzer initialisieren
    analyzer = PathwayAnalyzer()
    
    # Biologische Netzwerke erstellen und hinzufügen
    print("\nErstelle biologische Netzwerke...")
    glycolysis = create_glycolysis_pathway()
    partial = create_partial_glycolysis()
    krebs = create_krebs_cycle_fragment()
    proteins = create_protein_interaction_network()
    
    analyzer.add_network(glycolysis)
    analyzer.add_network(partial)
    analyzer.add_network(krebs)
    analyzer.add_network(proteins)
    
    # Netzwerke vergleichen
    print("\n" + "=" * 60)
    print("PFAD-VERGLEICHE")
    print("=" * 60)
    
    # Vergleiche vollständige vs. teilweise Glykolyse
    result1 = analyzer.compare_networks("Glycolysis", "Partial_Glycolysis")
    print(f"Ergebnis: {result1['decision']}")
    print(f"   {result1['interpretation']}\n")
    
    # Vergleiche verschiedene Pfade
    result2 = analyzer.compare_networks("Partial_Glycolysis", "Krebs_Cycle_Fragment")
    print(f"Ergebnis: {result2['decision']}")
    print(f"   {result2['interpretation']}\n")
    
    # Vergleiche metabolisches vs. Protein-Netzwerk
    result3 = analyzer.compare_networks("Glycolysis", "DNA_Damage_Response")
    print(f"Ergebnis: {result3['decision']}")
    print(f"   {result3['interpretation']}\n")
    
    # Finde alle gemeinsamen Pfade
    print("\n" + "=" * 60)
    print("SUBGRAPH-BEZIEHUNGEN")
    print("=" * 60)
    common = analyzer.find_common_pathways()
    
    if common:
        print("\nSubgraph-Beziehungen gefunden:")
        for net_a, net_b, decision in common:
            print(f"  • {net_a} ↔ {net_b}: {decision}")
    else:
        print("\nKeine Subgraph-Beziehungen gefunden")
    
    # Beispiel-Netzwerk exportieren
    print("\n" + "=" * 60)
    print("NETZWERK-EXPORT")
    print("=" * 60)
    export = glycolysis.to_dict()
    print(f"\nGlykolyse-Netzwerk-Struktur:")
    print(json.dumps(export, indent=2))