"""
Fenster-Chiffre – Entschluesselung (Algorithmus 2, Definition 3.1).

Dec(sk, P) -> M

Ablauf:
  1. Berechne Dummy-Indizes I_D = { i | (i-1+s) mod k == 0 }
  2. Extrahiere echte Pakete {C_j} = {P_i | i nicht in I_D}
  3. Fuer jedes j: M_j = C_j^d mod n
  4. Gib M = M_1 || M_2 || ... || M_r zurueck
"""

from __future__ import annotations
from typing import List, Sequence

from src.keygen import PrivateKey
from src.encrypt import blocks_to_message


def decrypt(sk: PrivateKey, packets: Sequence[int]) -> bytes:
    """Entschluesselt eine Paketfolge mit der Fenster-Chiffre (Algorithmus 2).

    Parameters
    ----------
    sk      : PrivateKey     Privater Schluessel (d, s, n, k).
    packets : Sequence[int]  Empfangene Paketfolge P = (P_1, ..., P_l).

    Returns
    -------
    bytes  Rekonstruierter Klartext M.
    """
    d, s, n, k = sk.d, sk.s, sk.n, sk.k

    # Schritt 1+2: Echte Pakete extrahieren (Dummy-Positionen ueberspringen)
    real_ciphers: List[int] = []
    for i, packet in enumerate(packets, start=1):
        if (i - 1 + s) % k != 0:  # keine Dummy-Position
            real_ciphers.append(packet)

    # Schritt 3: RSA-Entschluesselung
    real_blocks: List[int] = [pow(c, d, n) for c in real_ciphers]

    # Schritt 4: Bloecke zu Nachricht zusammensetzen
    return blocks_to_message(real_blocks, n)
