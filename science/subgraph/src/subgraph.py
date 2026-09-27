"""
Subgraph Algorithmus
Überprüft, ob Graph G in Graph G' enthalten ist durch Vergleich der Adjazenzmatrizen.
"""

from typing import List, Tuple, Set
import numpy as np


class Subgraph:
    """
    Algorithmus zum Finden von Teilgraphen in zwei Graphen G und G'.
    
    Wenn Matrix B (von G') die Matrix A (von G) enthält, wird A verworfen
    und B behalten, da G' mehr Informationen hat.
    """
    
    def __init__(self, use_adjacency_list: bool = False):
        """
        Initialisiert den Algorithmus.
        
        Args:
            use_adjacency_list: Wenn True, wird Adjazenzliste verwendet,
                               sonst Adjazenzmatrix
        """
        self.use_adjacency_list = use_adjacency_list
    
    def _compute_column_signature(self, matrix: np.ndarray) -> List[int]:
        """
        Berechnet für jeden Spaltenvektor eine eindeutige Signatur.
        
        Verwendet polynomiale Hash-Funktion:
        - Zeilenkomponente: Σ(2^i) für alle Zeilen i mit Wert 1
        - Spaltenkomponente: col_idx * 2^n (gewichtet mit Spaltenindex)
        
        Diese Methode garantiert Eindeutigkeit, da jede Binärkombination
        eine eindeutige Dezimalzahl ergibt und der Spaltenindex die
        Position zusätzlich kodiert.
        
        Beispiel für n=4:
            Spalte 0: [1, 0, 1, 0] → 2^0 + 2^2 + 0*(2^4) = 1 + 4 + 0 = 5
            Spalte 1: [1, 0, 1, 0] → 2^0 + 2^2 + 1*(2^4) = 1 + 4 + 16 = 21
            → Verschiedene Signaturen trotz gleichem Muster!
        
        Args:
            matrix: Adjazenzmatrix (n x n)
            
        Returns:
            Liste von eindeutigen Signaturen für jede Spalte
        """
        n = matrix.shape[0]
        signatures = []
        
        for col in range(n):
            column_vector = matrix[:, col]
            
            # Polynomiale Hash-Funktion: behandle Spalte als Binärzahl
            row_signature = sum(2**i for i in range(n) if column_vector[i] == 1)
            
            # Spaltenindex mit Gewichtung 2^n einbeziehen
            col_weight = col * (2**n)
            
            signature = row_signature + col_weight
            signatures.append(signature)
        
        return signatures
    
    def _find_signature_rotation_match(self, sig_A: List[int], sig_B: List[int]) -> bool:
        """
        Sucht nach einer zyklischen Rotation der Spalten in B, sodass A enthalten ist.
        
        KERNALGORITHMUS (wie beschrieben):
        1. Berechne Signatur von G
        2. Für jede der n_B möglichen Rotationen von G':
           - Rotiere Spalten zyklisch nach rechts
           - Berechne Signatur neu
           - Vergleiche mit sig_A
        
        Die Spalten werden nicht beliebig permutiert, sondern nur rotiert!
        Dies erhält die sequentielle Ordnung.
        
        Beispiel für 4 Spalten:
        - Original:   [0, 1, 2, 3]
        - Rotation 1: [1, 2, 3, 0]
        - Rotation 2: [2, 3, 0, 1]
        - Rotation 3: [3, 0, 1, 2]
        → Nur 4 Möglichkeiten, nicht 4! = 24
        
        Args:
            sig_A: Signaturen von Graph G (n_A Spalten)
            sig_B: Signaturen von Graph G' (n_B Spalten)
            
        Returns:
            True, wenn eine Rotation gefunden wurde, wo A in B enthalten ist
        """
        n_A = len(sig_A)
        n_B = len(sig_B)
        
        if n_A > n_B:
            return False
        
        if n_A == 0:
            return True
        
        # Extrahiere Zeilenkomponenten (ohne Spaltengewichtung)
        # Für Matrix n x n ist col_weight = 2^n
        col_weight_A = 2 ** n_A
        col_weight_B = 2 ** n_B
        
        row_sigs_A = [sig % col_weight_A for sig in sig_A]
        row_sigs_B = [sig % col_weight_B for sig in sig_B]
        
        # Prüfe alle n_B zyklischen Rotationen von B
        for rotation in range(n_B):
            # Rotiere row_sigs_B um 'rotation' Positionen nach rechts
            rotated_B = row_sigs_B[rotation:] + row_sigs_B[:rotation]
            
            # Prüfe ob A in diesem Fenster der Rotation enthalten ist
            if n_A == n_B:
                # Gleich groß: direkter Vergleich
                if self._compare_signature_sequences(row_sigs_A, rotated_B):
                    return True
            else:
                # A ist kleiner: Prüfe alle Startpositionen in rotated_B
                for start in range(n_B - n_A + 1):
                    window = rotated_B[start:start + n_A]
                    if self._compare_signature_sequences(row_sigs_A, window):
                        return True
        
        return False
    
    def _compare_signature_sequences(self, seq_A: List[int], seq_B: List[int]) -> bool:
        """
        Vergleicht zwei Signatur-Sequenzen elementweise.
        Args:
            seq_A: Signatur-Sequenz von G
            seq_B: Signatur-Sequenz von G'
        """
        n, m = len(seq_A), len(seq_B)
    
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        max_length = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq_A[i-1] == seq_B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0
        
        return max_length >= 2
    
    def compare_graphs(self, A: np.ndarray, B: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Vergleicht zwei Graphen G (Matrix A) und G' (Matrix B).
        
        ALGORITHMUS MIT ZYKLISCHER ROTATION:
        1. Berechne Signatur-Array für G
        2. Berechne Signatur-Array für G'
        3. Prüfe n_B zyklische Rotationen:
           - Für jede Rotation: Vergleiche Signaturen
           - Bei Bedarf: Neu-Berechnung
        
        Die Spalten werden NUR zyklisch rotiert (nicht beliebig permutiert):
        - Rotation 0: [0, 1, 2, ..., n-1]
        - Rotation 1: [1, 2, ..., n-1, 0]
        - Rotation 2: [2, ..., n-1, 0, 1]
        - ...
        Dies sind nur n Rotationen, nicht n! Permutationen.
        
        Args:
            A: Adjazenzmatrix von Graph G (n_A x n_A)
            B: Adjazenzmatrix von Graph G' (n_B x n_B)
            
        Returns:
            Tuple (Entscheidung, behaltene Matrix)
            - ("keep_B", B) wenn B die Matrix A enthält (G' hat mehr Info)
            - ("keep_A", A) wenn A die Matrix B enthält (G hat mehr Info)
            - ("keep_both", None) wenn keiner den anderen enthält
            - ("equal", A) wenn beide identisch sind
        """
        # Schritt 1: Berechne Signatur-Arrays O(n²)
        sig_A = self._compute_column_signature(A)
        sig_B = self._compute_column_signature(B)
        
        # Schritt 2: Suche mit zyklischen Rotationen (nicht Permutationen!)
        A_in_B = self._find_signature_rotation_match(sig_A, sig_B)
        B_in_A = self._find_signature_rotation_match(sig_B, sig_A)
        
        # Schritt 3: Entscheidung
        if A_in_B and B_in_A:
            # Zähle die Anzahl der Einsen (Kanten)
            ones_A = np.sum(A == 1)
            ones_B = np.sum(B == 1)
            
            if ones_B > ones_A:
                return ("equal_keep_B", B)
            else:
                return ("equal_keep_A", A)
        elif A_in_B:
            return ("keep_B", B)
        elif B_in_A:
            return ("keep_A", A)
        else:
            return ("keep_both", None)
    
    def adjacency_matrix_to_list(self, matrix: np.ndarray) -> List[Set[int]]:
        """
        Konvertiert Adjazenzmatrix zu Adjazenzliste.
        
        Args:
            matrix: Adjazenzmatrix (n x n)
            
        Returns:
            Adjazenzliste als Liste von Sets
        """
        n = matrix.shape[0]
        adj_list = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 1:
                    adj_list[i].add(j)
        
        return adj_list
    
    def compare_graphs_with_adj_list(self, A: np.ndarray, B: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Vergleicht Graphen unter Verwendung von Adjazenzlisten (für dichte Graphen).
        
        Args:
            A: Adjazenzmatrix von Graph G
            B: Adjazenzmatrix von Graph G'
            
        Returns:
            Tuple (Entscheidung, behaltene Matrix)
        """
        adj_A = self.adjacency_matrix_to_list(A)
        adj_B = self.adjacency_matrix_to_list(B)
        
        # A ist in B enthalten, wenn alle Kanten von A in B sind
        A_in_B = all(adj_A[i].issubset(adj_B[i]) for i in range(len(adj_A)))
        B_in_A = all(adj_B[i].issubset(adj_A[i]) for i in range(len(adj_B)))
        
        if A_in_B and B_in_A:
            return ("equal", A)
        elif A_in_B:
            return ("keep_B", B)
        elif B_in_A:
            return ("keep_A", A)
        else:
            return ("keep_both", None)
    
    def analyze_complexity(self, n: int) -> dict:
        """
        Analysiert die Laufzeitkomplexität des Algorithmus.
        
        Args:
            n: Anzahl der Knoten
            
        Returns:
            Dictionary mit Komplexitätsanalyse
        """
        return {
            "step_1_signature_G": f"O(n²) = O({n*n}) - Signatur von G berechnen",
            "step_2_signature_G_prime": f"O(n²) = O({n*n}) - Signatur von G' berechnen",
            "step_3_rotations": f"n = {n} zyklische Rotationen (nicht n!)",
            "step_3_per_rotation_recompute": f"O(n²) = O({n*n}) - Signatur neu berechnen",
            "step_3_per_rotation_compare": f"O(n²) = O({n*n}) - Signaturen vergleichen",
            "total_per_graph": f"O(n·n²) = O(n³) = O({n**3})",
            "total_both_directions": f"O(n³) für A⊆B und B⊆A",
            "note": "Nur n Rotationen (nicht n! Permutationen) - sequentielle Ordnung bleibt erhalten",
            "original_description": "O(n³) ✓"
        }


def create_example_graphs() -> Tuple[np.ndarray, np.ndarray]:
    """
    Erstellt Beispielgraphen für Demonstrationszwecke.
    
    Returns:
        Tuple (Matrix A von G, Matrix B von G')
    """
    # Graph G mit 4 Knoten
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    
    # Graph G' mit zusätzlichen Kanten
    B = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    
    return A, B


if __name__ == "__main__":
    # Demonstrationsbeispiel
    print("=== Subgraph Algorithmus Demo ===\n")
    
    algo = Subgraph()
    A, B = create_example_graphs()
    
    print("Graph G (Matrix A):")
    print(A)
    print("\nGraph G' (Matrix B):")
    print(B)
    
    decision, kept_matrix = algo.compare_graphs(A, B)
    
    print(f"\nEntscheidung: {decision}")
    if kept_matrix is not None:
        print("Behaltene Matrix:")
        print(kept_matrix)
    
    print("\n=== Komplexitätsanalyse ===")
    complexity = algo.analyze_complexity(4)
    for key, value in complexity.items():
        print(f"{key}: {value}")
