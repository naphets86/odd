"""
sigchiffre.signature
====================
Injektive Signaturfunktion σ aus dem Subgraph Algorithmus.

Definition:
    σ(A, j) = ρ(A, j) + λ(j)
    ρ(A, j) = Σ_{i=0}^{n-1} A_{ij} · 2^i   (Zeilenkomponente)
    λ(j)    = j · 2^n                          (Spaltengewicht)

Eigenschaft: σ(A,j) ∈ [j·2^n, (j+1)·2^n) und σ(A,j) < 2^(2n).

Referenz: S. Epp, "Die Signatur-Chiffre", März 2026.
"""
from __future__ import annotations

import numpy as np


def rho(A: np.ndarray, j: int) -> int:
    """Zeilenkomponente ρ(A, j) = Σ A_{ij}·2^i (LSB = Zeile 0)."""
    result = 0
    for i in range(A.shape[0]):
        if A[i, j]:
            result += (1 << i)
    return int(result)


def lambda_(j: int, n: int) -> int:
    """Spaltengewicht λ(j) = j · 2^n."""
    return j * (1 << n)


def sigma(A: np.ndarray, j: int) -> int:
    """Signaturfunktion σ(A, j) = ρ(A, j) + j · 2^n."""
    return rho(A, j) + lambda_(j, A.shape[0])


def sigma_vector(A: np.ndarray) -> list[int]:
    """Signaturvektor Σ(A) = (σ(A,0), …, σ(A,n-1))."""
    return [sigma(A, j) for j in range(A.shape[0])]


def bits_to_column(value: int, n: int) -> list[int]:
    """Extrahiert n Bits aus value (LSB zuerst): A_{ij} = ⌊value/2^i⌋ mod 2."""
    return [(value >> i) & 1 for i in range(n)]


def reconstruct_column(s_j: int, j: int, n: int) -> list[int]:
    """Spalte aus Signaturwert rekonstruieren: ρ_j = s_j - j·2^n → Bits."""
    return bits_to_column(s_j - lambda_(j, n), n)


def matrix_to_bits(A: np.ndarray) -> list[int]:
    """n×n Matrix spaltenweise serialisieren: Bit (j·n+i) = A_{ij}."""
    n = A.shape[0]
    bits = []
    for j in range(n):
        for i in range(n):
            bits.append(int(A[i, j]))
    return bits


def bits_to_matrix(bits: list[int], n: int) -> np.ndarray:
    """Bitvektor der Länge n² in n×n Matrix deserialisieren (spaltenweise)."""
    A = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            A[i, j] = bits[j * n + i]
    return A


def bytes_to_blocks(data: bytes, n: int) -> list[np.ndarray]:
    """Bytes in n×n-Blöcke aufteilen (ECB, Zero-Padding im letzten Block)."""
    block_bits = n * n
    block_bytes = (block_bits + 7) // 8
    blocks = []
    for offset in range(0, len(data), block_bytes):
        chunk = data[offset: offset + block_bytes]
        bits: list[int] = []
        for byte in chunk:
            for bit_idx in range(7, -1, -1):
                bits.append((byte >> bit_idx) & 1)
        bits = bits[:block_bits]
        while len(bits) < block_bits:
            bits.append(0)
        blocks.append(bits_to_matrix(bits, n))
    return blocks


def blocks_to_bytes(blocks: list[np.ndarray], n: int, original_len: int) -> bytes:
    """n×n-Blöcke zurück in Bytes serialisieren; Padding auf original_len abschneiden."""
    all_bits: list[int] = []
    for A in blocks:
        all_bits.extend(matrix_to_bits(A))
    result = bytearray()
    for byte_idx in range(0, len(all_bits), 8):
        byte_bits = all_bits[byte_idx: byte_idx + 8]
        if len(byte_bits) < 8:
            byte_bits += [0] * (8 - len(byte_bits))
        result.append(sum(b << (7 - i) for i, b in enumerate(byte_bits)))
    return bytes(result[:original_len])

