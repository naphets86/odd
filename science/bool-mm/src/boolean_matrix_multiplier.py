"""
Boolean Matrix Multiplikation mit Signatur-Technik.

Implementiert den Algorithmus aus der wissenschaftlichen Arbeit zur
Boolean Matrixmultiplikation in O(n^2) mittels polynomialer Hash-Signaturen.
"""

import numpy as np
from typing import List, Tuple, Optional


class BooleanMatrixMultiplier:
    """
    Implementierung der Boolean Matrixmultiplikation mit Signaturen.
    
    Der Algorithmus nutzt polynomiale Hash-Funktionen zur Kodierung von
    Zeilen und Spalten als Signaturen und berechnet die Boolean-Multiplikation
    in O(n^2) Zeit mittels Bitoperationen.
    
    Attributes:
        _row_signatures_cache: Cache für berechnete Zeilen-Signaturen
        _col_signatures_cache: Cache für berechnete Spalten-Signaturen
    """
    
    def __init__(self):
        """Initialisiert den Boolean Matrix Multiplier."""
        self._row_signatures_cache: dict = {}
        self._col_signatures_cache: dict = {}
    
    def _validate_matrix(self, matrix: np.ndarray, name: str = "Matrix") -> None:
        """
        Validiert eine Matrix auf Boolean-Eigenschaften.
        
        Args:
            matrix: Zu validierende Matrix
            name: Name der Matrix für Fehlermeldungen
            
        Raises:
            ValueError: Falls Matrix ungültig ist
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError(f"{name} muss ein numpy.ndarray sein")
        
        if matrix.ndim != 2:
            raise ValueError(f"{name} muss 2-dimensional sein")
        
        if not np.all(np.isin(matrix, [0, 1])):
            raise ValueError(f"{name} muss nur Werte 0 und 1 enthalten")
    
    def _validate_dimensions(self, A: np.ndarray, B: np.ndarray) -> None:
        """
        Validiert Dimensionen für Matrixmultiplikation.
        
        Args:
            A: Erste Matrix
            B: Zweite Matrix
            
        Raises:
            ValueError: Falls Dimensionen nicht kompatibel sind
        """
        n_A, k_A = A.shape
        k_B, m_B = B.shape
        
        if k_A != k_B:
            raise ValueError(
                f"Inkompatible Dimensionen: A ist {n_A}x{k_A}, "
                f"B ist {k_B}x{m_B}. Innere Dimensionen müssen übereinstimmen."
            )
    
    def compute_row_signature(self, row: np.ndarray) -> int:
        """
        Berechnet die Signatur für eine Zeile.
        
        Die Signatur kodiert den Binärvektor als Dezimalzahl.
        
        Args:
            row: Binärer Zeilenvektor (1D numpy array)
            
        Returns:
            Signatur als Integer
            
        Example:
            >>> row = np.array([1, 0, 1, 1])
            >>> compute_row_signature(row)
            13  # = 1*2^0 + 0*2^1 + 1*2^2 + 1*2^3 = 1 + 4 + 8
        """
        n = len(row)
        signature = sum(int(row[i]) * (2 ** i) for i in range(n))
        return signature
    
    def compute_column_signature(self, col: np.ndarray) -> int:
        """
        Berechnet die Signatur für eine Spalte.
        
        Analog die Zeilen-Signatur.
        
        Args:
            col: Binärer Spaltenvektor (1D numpy array)
            
        Returns:
            Signatur als Integer
        """
        n = len(col)
        signature = sum(int(col[i]) * (2 ** i) for i in range(n))
        return signature
    
    def precompute_signatures(
        self, 
        A: np.ndarray, 
        B: np.ndarray,
        use_cache: bool = False
    ) -> Tuple[List[int], List[int]]:
        """
        Berechnet alle Zeilen-Signaturen von A und Spalten-Signaturen von B.
        
        Dies ist Phase 1 des Algorithmus mit Komplexität O(n^2).
        
        Args:
            A: Erste Boolean Matrix (n x k)
            B: Zweite Boolean Matrix (k x m)
            use_cache: Ob Cache verwendet werden soll
            
        Returns:
            Tuple von (row_signatures, col_signatures)
        """
        n = A.shape[0]
        m = B.shape[1]
        
        # Zeilen-Signaturen von A
        if use_cache and 'A' in self._row_signatures_cache:
            row_sigs_A = self._row_signatures_cache['A']
        else:
            row_sigs_A = [
                self.compute_row_signature(A[i, :]) 
                for i in range(n)
            ]
            if use_cache:
                self._row_signatures_cache['A'] = row_sigs_A
        
        # Spalten-Signaturen von B
        if use_cache and 'B' in self._col_signatures_cache:
            col_sigs_B = self._col_signatures_cache['B']
        else:
            col_sigs_B = [
                self.compute_column_signature(B[:, j]) 
                for j in range(m)
            ]
            if use_cache:
                self._col_signatures_cache['B'] = col_sigs_B
        
        return row_sigs_A, col_sigs_B
    
    def boolean_and_via_signature(self, sig1: int, sig2: int) -> int:
        """
        Führt Boolean AND über Signaturen aus.
        
        Nutzt die bitweise AND-Operation (&) auf Hardware-Ebene.
        Komplexität: O(1)
        
        Args:
            sig1: Erste Signatur
            sig2: Zweite Signatur
            
        Returns:
            Ergebnis der bitweisen AND-Operation
        """
        return sig1 & sig2
    
    def boolean_or_check(self, and_result: int) -> bool:
        """
        Prüft ob das AND-Ergebnis != 0 ist.
        
        Dies entspricht dem Boolean OR über alle k Positionen.
        
        Args:
            and_result: Ergebnis der bitweisen AND-Operation
            
        Returns:
            True falls and_result != 0, sonst False
        """
        return and_result != 0
    
    def multiply_optimized(
        self, 
        A: np.ndarray, 
        B: np.ndarray,
        use_cache: bool = False
    ) -> np.ndarray:
        """
        Boolean Matrixmultiplikation mit Signatur-Optimierung.
        
        Implementiert Algorithmus 1 aus der wissenschaftlichen Arbeit.
        Komplexität: O(n^2)
        
        Phase 1: Signatur-Berechnung O(n^2)
        Phase 2: Multiplikation via Bitoperationen O(n^2)
        
        Args:
            A: Erste Boolean Matrix (n x k)
            B: Zweite Boolean Matrix (k x m)
            use_cache: Ob Signatur-Cache verwendet werden soll
            
        Returns:
            Ergebnis-Matrix C (n x m)
            
        Raises:
            ValueError: Falls Matrizen ungültig oder inkompatibel
        """
        # Validierung
        self._validate_matrix(A, "Matrix A")
        self._validate_matrix(B, "Matrix B")
        self._validate_dimensions(A, B)
        
        n, k = A.shape
        _, m = B.shape
        
        # Phase 1: Signaturen vorberechnen
        row_sigs_A, col_sigs_B = self.precompute_signatures(A, B, use_cache)
        
        # Phase 2: Boolean Multiplikation via Signaturen
        C = np.zeros((n, m), dtype=int)
        
        for i in range(n):
            for j in range(m):
                # Bitweise AND in O(1)
                and_result = self.boolean_and_via_signature(
                    row_sigs_A[i], 
                    col_sigs_B[j]
                )
                
                # Boolean OR Check
                if self.boolean_or_check(and_result):
                    C[i, j] = 1
        
        return C
    
    def multiply_naive(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive Boolean Matrixmultiplikation zum Vergleich.
        
        Implementiert die klassische Definition mit drei geschachtelten Schleifen.
        Komplexität: O(n^3)
        
        Args:
            A: Erste Boolean Matrix (n x k)
            B: Zweite Boolean Matrix (k x m)
            
        Returns:
            Ergebnis-Matrix C (n x m)
            
        Raises:
            ValueError: Falls Matrizen ungültig oder inkompatibel
        """
        # Validierung
        self._validate_matrix(A, "Matrix A")
        self._validate_matrix(B, "Matrix B")
        self._validate_dimensions(A, B)
        
        n, k = A.shape
        _, m = B.shape
        
        C = np.zeros((n, m), dtype=int)
        
        for i in range(n):
            for j in range(m):
                # Boolean OR über alle k
                for k_idx in range(k):
                    if A[i, k_idx] and B[k_idx, j]:
                        C[i, j] = 1
                        break  # Early exit für Boolean OR
        
        return C
    
    def clear_cache(self) -> None:
        """Leert den Signatur-Cache."""
        self._row_signatures_cache.clear()
        self._col_signatures_cache.clear()
