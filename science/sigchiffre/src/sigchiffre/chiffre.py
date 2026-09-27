"""
sigchiffre.chiffre
==================
Hauptklasse SignatureChiffre: Verschlüsselung und Entschlüsselung.

Algorithmus:
  Enc(A, a, b, p) = ((a·σ(A,j)+b) mod p)_{j=0,…,n-1}
  Dec(C, a, b, p): s_j = a^{-1}·(c_j-b) mod p → Spaltenrekonstruktion

Korrektheit garantiert, da σ(A,j) < 2^(2n) < p (Satz 4.1).

Referenz: S. Epp, "Die Signatur-Chiffre", Def. 3.2/3.3, Satz 4.1.
"""
from __future__ import annotations

from math import gcd

import numpy as np

from .keygen import validate_key
from .signature import (
    sigma_vector,
    reconstruct_column,
    bytes_to_blocks,
    blocks_to_bytes,
)


class SignatureChiffre:
    """Symmetrische Blockchiffre auf Basis der injektiven Signaturfunktion.

    Schlüssel: Tripel (a, b, p) mit p prim, p > 2^(2n), a ∈ Z_p*, b ∈ Z_p.
    Blockgröße: n×n Bits.

    Parameters
    ----------
    n : int
        Blockgröße (Matrix ist n×n).
    a : int
        Affiner Faktor ∈ {1, …, p-1}, teilerfremd zu p.
    b : int
        Affiner Term ∈ {0, …, p-1}.
    p : int
        Geheime Primzahl, p > 2^(2n).
    """

    def __init__(self, n: int, a: int, b: int, p: int) -> None:
        validate_key(n, a, b, p)
        self.n = n
        self.a = a
        self.b = b
        self.p = p
        self.a_inv = pow(a, -1, p)  # erw. euklidischer Algorithmus (Python-Builtin)

    # ------------------------------------------------------------------
    # Einzelblock-API
    # ------------------------------------------------------------------

    def encrypt(self, A: np.ndarray) -> list[int]:
        """Verschlüsselt eine n×n Binärmatrix.

        Enc(A) = ((a·σ(A,j)+b) mod p)_{j=0,…,n-1}

        Parameters
        ----------
        A : np.ndarray, shape (n, n), Einträge ∈ {0, 1}

        Returns
        -------
        list[int] der Länge n, Werte ∈ Z_p
        """
        if A.shape != (self.n, self.n):
            raise ValueError(
                f"Erwartet ({self.n},{self.n})-Matrix, erhalten: {A.shape}"
            )
        sigs = sigma_vector(A)
        return [(self.a * s + self.b) % self.p for s in sigs]

    def decrypt(self, C: list[int]) -> np.ndarray:
        """Entschlüsselt ein Chiffrat.

        Dec(C):
          1. s_j = a^{-1}·(c_j - b) mod p
          2. A_{ij} = ⌊(s_j - j·2^n)/2^i⌋ mod 2

        Parameters
        ----------
        C : list[int] der Länge n, Werte ∈ Z_p

        Returns
        -------
        np.ndarray, shape (n, n), dtype int, Einträge ∈ {0, 1}
        """
        if len(C) != self.n:
            raise ValueError(
                f"Chiffrat muss Länge {self.n} haben, erhalten: {len(C)}"
            )
        n = self.n
        A = np.zeros((n, n), dtype=int)
        for j in range(n):
            s_j = (self.a_inv * (C[j] - self.b)) % self.p
            col = reconstruct_column(s_j, j, n)
            for i in range(n):
                A[i, j] = col[i]
        return A

    # ------------------------------------------------------------------
    # Bytes-API (ECB-Modus, Zero-Padding)
    # ------------------------------------------------------------------

    def encrypt_bytes(self, data: bytes) -> tuple[list[list[int]], int]:
        """Verschlüsselt beliebige Bytes blockweise.

        Parameters
        ----------
        data : Eingabedaten

        Returns
        -------
        (ciphertext_blocks, original_len)
          ciphertext_blocks : list[list[int]], je Block n Werte ∈ Z_p
          original_len      : ursprüngliche Länge von data (für Entschlüsselung)
        """
        blocks = bytes_to_blocks(data, self.n)
        encrypted = [self.encrypt(block) for block in blocks]
        return encrypted, len(data)

    def decrypt_bytes(
        self, ciphertext_blocks: list[list[int]], original_len: int
    ) -> bytes:
        """Entschlüsselt blockweise verschlüsselte Bytes.

        Parameters
        ----------
        ciphertext_blocks : list[list[int]]
        original_len      : ursprüngliche Länge (zum Abschneiden des Paddings)

        Returns
        -------
        bytes
        """
        plaintext_blocks = [self.decrypt(C) for C in ciphertext_blocks]
        return blocks_to_bytes(plaintext_blocks, self.n, original_len)

    # ------------------------------------------------------------------
    # Darstellung
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SignatureChiffre(n={self.n}, "
            f"a={self.a}, b={self.b}, p={self.p})"
        )
