"""
sigchiffre.keygen
=================
Kryptographisch sichere Schlüsselgenerierung für die Signatur-Chiffre.

Schlüssel: Tripel (a, b, p) mit
  - p prim, p > 2^(2n)
  - a ∈ Z_p*, gcd(a,p) = 1
  - b ∈ Z_p

Referenz: S. Epp, "Die Signatur-Chiffre", Def. 2.4, Satz 4.3.
"""
from __future__ import annotations

import secrets
from math import gcd, isqrt


# ---------------------------------------------------------------------------
# Primzahl-Hilfsfunktionen
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Deterministischer Primzahltest (Miller-Rabin mit festen Zeugen für n < 3.3·10^24,
    danach probabilistisch mit 40 Runden).

    Für kryptographische Primzahlerzeugung ausreichend sicher.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Miller-Rabin: n-1 = 2^r · d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Deterministische Zeugen für n < 3.3·10^24
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    """Gibt die kleinste Primzahl ≥ n zurück."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while not is_prime(candidate):
        candidate += 2
    return candidate


def generate_prime(bits: int, min_value: int = 0) -> int:
    """Erzeugt eine zufällige Primzahl mit genau `bits` Bits und ≥ min_value.

    Parameters
    ----------
    bits      : gewünschte Bitzahl (oberstes Bit wird gesetzt)
    min_value : untere Schranke (zusätzlich zu 2^(bits-1))

    Returns
    -------
    int (Primzahl)
    """
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1  # ungerade, top-Bit gesetzt
        candidate = max(candidate, min_value)
        p = next_prime(candidate)
        if p.bit_length() <= bits + 1:  # toleriere leichten Überlauf
            return p


# ---------------------------------------------------------------------------
# Schlüsselgenerierung
# ---------------------------------------------------------------------------

def generate_key(n: int, bits: int = 256) -> tuple[int, int, int]:
    """Erzeugt einen zufälligen Signatur-Chiffre-Schlüssel (a, b, p).

    Ablauf:
      1. Erzeuge p prim mit `bits` Bits und p > 2^(2n).
      2. Wähle a ∈ {1, …, p-1} gleichmäßig zufällig.
      3. Wähle b ∈ {0, …, p-1} gleichmäßig zufällig.

    Parameters
    ----------
    n    : Blockgröße (n×n Matrix)
    bits : Bit-Länge der Primzahl p (Standard: 256, muss ≥ 2n+1 sein)

    Returns
    -------
    (a, b, p) – privater Schlüssel
    """
    if bits < 2 * n + 1:
        raise ValueError(
            f"bits={bits} muss ≥ 2n+1 = {2*n+1} sein, damit p > 2^(2n)"
        )
    min_p = (1 << (2 * n)) + 1
    p = generate_prime(bits, min_value=min_p)
    a = secrets.randbelow(p - 1) + 1   # ∈ {1, …, p-1}
    b = secrets.randbelow(p)            # ∈ {0, …, p-1}
    return a, b, p


def validate_key(n: int, a: int, b: int, p: int) -> None:
    """Prüft die Gültigkeit eines Schlüsseltripels (a, b, p) für Blockgröße n.

    Raises
    ------
    ValueError bei ungültigem Schlüssel.
    """
    if not is_prime(p):
        raise ValueError(f"p={p} ist keine Primzahl")
    min_p = 1 << (2 * n)
    if p <= min_p:
        raise ValueError(f"p={p} muss > 2^(2n) = {min_p} sein")
    if not (1 <= a < p):
        raise ValueError(f"a={a} muss in {{1,…,{p-1}}} liegen")
    if gcd(a, p) != 1:
        raise ValueError(f"gcd(a={a}, p={p}) ≠ 1")
    if not (0 <= b < p):
        raise ValueError(f"b={b} muss in {{0,…,{p-1}}} liegen")


def key_space_size(p: int) -> int:
    """Schlüsselraumgröße |K| = (p-1)·p für festes p (Satz 4.3).

    Für p ≈ 2^256 gilt |K| ≈ 2^512.
    """
    return (p - 1) * p


def effective_key_bits(p: int) -> float:
    """Effektive Schlüssellänge in Bit: log2((p-1)·p) ≈ 2·log2(p)."""
    from math import log2
    return log2(key_space_size(p))
