"""
Fenster-Chiffre – Verschluesselung (Algorithmus 1, Definition 3.1).

Enc(pk, s, M) -> P

Ablauf:
  1. Zerlege M in Bloecke M_1,...,M_r  (jeder Block < n, bitweises Framing)
  2. Berechne Gesamtlaenge l = minimale Paketanzahl fuer genau r echte Slots
  3. Fuer i in 1..l:
       falls (i-1+s) mod k == 0  ->  Dummy-Paket  (Zufallszahl in [0,n))
       sonst                      ->  C_j = M_j^e mod n

Blockzerlegung (message_to_blocks):
  Jeder Block traegt B = floor(log2(n)) Nutzbits.
  Framing: [16-Bit-Laengenfeld | Nachrichtenbits | 0-Fuellbits]
  Das Laengenfeld kodiert len(message) in Bytes (max 65535).
  Damit ist die Rekonstruktion eindeutig fuer beliebig kleine n (auch n=55).
"""

from __future__ import annotations
import os
import math
from typing import List

from src.keygen import PublicKey


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _dummy_int(n: int) -> int:
    """Erzeugt eine gleichverteilte Zufallszahl in [0, n)."""
    byte_len = math.ceil(n.bit_length() / 8)
    raw = os.urandom(byte_len)
    return int.from_bytes(raw, byteorder="big") % n


def _block_bits(n: int) -> int:
    """Nutzbits pro Block: B = floor(log2(n)).

    Garantiert Block-Werte in [0, 2^B - 1] <= [0, n-1].
    """
    return n.bit_length() - 1


def _block_byte_length(n: int) -> int:
    """Bytes die noetig sind um Zahlen < n darzustellen."""
    return math.ceil(n.bit_length() / 8)


def _compute_packet_count(r: int, k: int, s: int) -> int:
    """Minimale Paketanzahl l so dass genau r echte Slots entstehen.

    Echte Slots = Positionen i in [1..l] mit (i-1+s) % k != 0.
    Die Approximation l = ceil(r*k/(k-1)) ist nur fuer s=0 exakt;
    diese Funktion korrigiert fuer alle Offsets.

    Parameters
    ----------
    r : int  Anzahl der echten Bloecke (>= 0).
    k : int  Fenster-Abstand (>= 2).
    s : int  Offset (0 <= s < k).

    Returns
    -------
    int  Minimales l >= 0 mit exakt r echten Slots.
    """
    if r == 0:
        return 0
    l = math.ceil(r * k / (k - 1))
    while True:
        real = sum(1 for i in range(1, l + 1) if (i - 1 + s) % k != 0)
        if real == r:
            return l
        if real < r:
            l += 1
        else:
            real_prev = sum(1 for i in range(1, l) if (i - 1 + s) % k != 0)
            return l - 1 if real_prev == r else l


# ---------------------------------------------------------------------------
# Oeffentliche API
# ---------------------------------------------------------------------------

def message_to_blocks(message: bytes, n: int) -> List[int]:
    """Zerlegt eine Bytefolge in Klartextbloecke < n.

    Jeder Block traegt B = floor(log2(n)) Nutzbits.
    Framing: [16-Bit-Laengenfeld | Nachrichtenbits | 0-Fuellbits auf Vielfaches von B].

    Parameters
    ----------
    message : bytes  Klartextnachricht (max 65535 Bytes).
    n       : int    RSA-Modulus (>= 3).

    Returns
    -------
    List[int]  Blockliste, jeder Wert in [0, 2^B - 1] < n.
    """
    if n < 3:
        raise ValueError(f"RSA-Modulus n muss >= 3 sein, erhalten: {n}")

    B = _block_bits(n)
    if B < 1:
        raise ValueError(f"n={n} zu klein fuer Blockzerlegung.")

    msg_len = len(message)
    if msg_len > 65535:
        raise ValueError(f"Nachricht zu lang: {msg_len} Bytes (max 65535).")

    # Bitstream: 16-Bit-Laengenfeld || Nachrichtenbits
    msg_int = int.from_bytes(message, byteorder="big") if message else 0
    payload = (msg_len << (msg_len * 8)) | msg_int
    payload_bit_len = 16 + msg_len * 8

    # Rechts mit 0-Bits auf Vielfaches von B auffuellen
    remainder = payload_bit_len % B
    if remainder != 0:
        pad = B - remainder
        payload <<= pad
        payload_bit_len += pad

    num_blocks = payload_bit_len // B
    mask = (1 << B) - 1

    blocks: List[int] = []
    for i in range(num_blocks - 1, -1, -1):
        val = (payload >> (i * B)) & mask
        assert val < n, f"Interner Fehler: Blockwert {val} >= n={n}"
        blocks.append(val)

    return blocks


def blocks_to_message(blocks: List[int], n: int) -> bytes:
    """Rekonstruiert eine Bytefolge aus Klartextbloecken (Inverse von message_to_blocks).

    Parameters
    ----------
    blocks : List[int]  Entschluesselten Bloecke.
    n      : int        RSA-Modulus.

    Returns
    -------
    bytes  Urspruengliche Nachricht.

    Raises
    ------
    ValueError  Bei ungueltiger Blockstruktur.
    """
    if not blocks:
        raise ValueError("Ungueltiges Padding: 0x80-Byte nicht gefunden.")

    B = _block_bits(n)
    total_bits = len(blocks) * B

    if total_bits < 16:
        raise ValueError("Ungueltiges Padding: 0x80-Byte nicht gefunden.")

    # Alle Bloecke zu einem Integer zusammensetzen
    combined = 0
    for b in blocks:
        combined = (combined << B) | b

    # Laengenfeld aus den obersten 16 Bits extrahieren.
    # Struktur: [16 Bit Laenge | msg_len*8 Datenbits | pad Fuellbits]
    # total_bits = 16 + msg_len*8 + pad,  0 <= pad < B
    header_shift = total_bits - 16
    candidate_len = (combined >> header_shift) & 0xFFFF

    data_bits = candidate_len * 8
    used_bits = 16 + data_bits
    pad = total_bits - used_bits

    if pad < 0 or pad >= B:
        raise ValueError("Ungueltiges Padding: 0x80-Byte nicht gefunden.")

    if candidate_len == 0:
        return b""

    msg_int = (combined >> pad) & ((1 << data_bits) - 1)
    return msg_int.to_bytes(candidate_len, byteorder="big")


def encrypt(
    pk: PublicKey,
    s: int,
    message: bytes,
) -> List[int]:
    """Verschluesselt eine Nachricht mit der Fenster-Chiffre (Algorithmus 1).

    Parameters
    ----------
    pk      : PublicKey  Oeffentlicher Schluessel (n, e, k).
    s       : int        Geheimer Offset (0 <= s < k).
    message : bytes      Klartextnachricht (max 65535 Bytes).

    Returns
    -------
    List[int]  Paketfolge P = (P_1, ..., P_l).
               Dummy-Pakete: gleichverteilte Zufallszahlen in [0, n).
               Echte Pakete: C_j = M_j^e mod n.
    """
    if not (0 <= s < pk.k):
        raise ValueError(f"Offset s={s} liegt nicht in [0, k-1={pk.k - 1}].")

    n, e, k = pk.n, pk.e, pk.k

    # Schritt 1: Blockzerlegung
    real_blocks = message_to_blocks(message, n)
    r = len(real_blocks)

    # Schritt 2: Gesamtlaenge (offset-korrekt: genau r echte Slots)
    total = _compute_packet_count(r, k, s)

    # Schritt 3: Paketfolge aufbauen
    packets: List[int] = []
    j = 0
    for i in range(1, total + 1):
        if (i - 1 + s) % k == 0:
            packets.append(_dummy_int(n))
        else:
            packets.append(pow(real_blocks[j], e, n))
            j += 1

    assert j == r, f"Interner Fehler: {j} Bloecke platziert, erwartet {r}"
    return packets


def count_dummy_positions(total: int, k: int, s: int) -> List[int]:
    """Gibt die Dummy-Indizes (1-basiert) einer Paketfolge zurueck.

    Parameters
    ----------
    total : int  Gesamtanzahl Pakete.
    k     : int  Fenster-Abstand.
    s     : int  Offset.

    Returns
    -------
    List[int]  1-basierte Dummy-Positionen.
    """
    return [i for i in range(1, total + 1) if (i - 1 + s) % k == 0]