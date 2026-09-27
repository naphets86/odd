"""
sigchiffre.hashes
=================
Kryptographische Hash-Konstruktionen auf Basis der Signaturfunktion.

- SignatureHash: Hashfunktion über Signaturvektor + affiner Reduktion
- MerkleTree:    Binärer Merkle-Baum mit Proof- und Verify-Methoden

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 14 (Hash-Funktionen).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from .chiffre import SignatureChiffre
from .signature import sigma_vector


class SignatureHash:
    """Hash-Funktion auf Basis der Signatur-Chiffre.

    H(data) = SHA-256(σ(B_1) || σ(B_2) || … || σ(B_k) || enc(B_i))

    Die Signaturwerte aller Blöcke werden mit der Enc-Funktion transformiert
    und über SHA-256 zu einem 256-Bit-Digest komprimiert.

    Parameters
    ----------
    chiffre : SignatureChiffre-Instanz (liefert n, a, b, p)
    """

    def __init__(self, chiffre: SignatureChiffre) -> None:
        self.chiffre = chiffre

    def digest(self, data: bytes) -> bytes:
        """Berechnet den 32-Byte-Hash von data.

        Parameters
        ----------
        data : Eingabedaten

        Returns
        -------
        bytes, Länge 32
        """
        from .signature import bytes_to_blocks
        blocks = bytes_to_blocks(data, self.chiffre.n)
        h = hashlib.sha256()
        for block in blocks:
            encrypted = self.chiffre.encrypt(block)
            for val in encrypted:
                h.update(val.to_bytes((val.bit_length() + 7) // 8 + 1, "big"))
        return h.digest()

    def hexdigest(self, data: bytes) -> str:
        """Gibt den Hash als Hex-String zurück."""
        return self.digest(data).hex()


@dataclass
class MerkleProof:
    """Merkle-Beweis für ein einzelnes Blatt."""
    leaf_index: int
    leaf_hash: bytes
    path: list[bytes]       # Geschwister-Hashes von Blatt bis Wurzel
    directions: list[bool]  # True = Geschwister ist rechts


class MerkleTree:
    """Binärer Merkle-Hash-Baum.

    Blätter werden mittels SHA-256 gehasht. Innere Knoten:
        H(left || right)

    Parameters
    ----------
    leaves : list[bytes]  – Rohdaten der Blätter
    """

    def __init__(self, leaves: list[bytes]) -> None:
        if not leaves:
            raise ValueError("MerkleTree benötigt mindestens ein Blatt")
        self._leaves = leaves
        self._leaf_hashes = [self._hash_leaf(d) for d in leaves]
        self._tree = self._build(self._leaf_hashes)

    @staticmethod
    def _hash_leaf(data: bytes) -> bytes:
        return hashlib.sha256(b"\x00" + data).digest()

    @staticmethod
    def _hash_pair(left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(b"\x01" + left + right).digest()

    def _build(self, layer: list[bytes]) -> list[list[bytes]]:
        """Baut den Baum von den Blättern bis zur Wurzel auf."""
        tree = [layer]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer = layer + [layer[-1]]  # dupliziere letztes Blatt
            layer = [
                self._hash_pair(layer[i], layer[i + 1])
                for i in range(0, len(layer), 2)
            ]
            tree.append(layer)
        return tree

    @property
    def root(self) -> bytes:
        """Merkle-Wurzel (32 Bytes)."""
        return self._tree[-1][0]

    def get_proof(self, index: int) -> MerkleProof:
        """Erzeugt einen Inklusionsbeweis für Blatt `index`.

        Parameters
        ----------
        index : Blattindex (0-basiert)

        Returns
        -------
        MerkleProof
        """
        if index < 0 or index >= len(self._leaf_hashes):
            raise IndexError(f"Index {index} außerhalb [0, {len(self._leaf_hashes)-1}]")
        path: list[bytes] = []
        directions: list[bool] = []
        idx = index
        for layer in self._tree[:-1]:
            padded = layer if len(layer) % 2 == 0 else layer + [layer[-1]]
            sibling_idx = idx ^ 1  # XOR mit 1 gibt Geschwister
            if sibling_idx < len(padded):
                path.append(padded[sibling_idx])
            else:
                path.append(padded[idx])
            directions.append(idx % 2 == 0)  # True = wir sind links
            idx //= 2
        return MerkleProof(
            leaf_index=index,
            leaf_hash=self._leaf_hashes[index],
            path=path,
            directions=directions,
        )

    @staticmethod
    def verify_proof(proof: MerkleProof, root: bytes) -> bool:
        """Verifiziert einen Merkle-Beweis gegen die gegebene Wurzel.

        Parameters
        ----------
        proof : MerkleProof
        root  : erwartete Merkle-Wurzel

        Returns
        -------
        bool
        """
        current = proof.leaf_hash
        for sibling, is_left in zip(proof.path, proof.directions):
            if is_left:
                current = MerkleTree._hash_pair(current, sibling)
            else:
                current = MerkleTree._hash_pair(sibling, current)
        return current == root
