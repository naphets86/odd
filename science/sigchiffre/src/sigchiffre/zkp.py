"""
sigchiffre.zkp
==============
Zero-Knowledge-Beweise: Schnorr-Protokoll und Pedersen-Commitment.

Schnorr-Protokoll (Sigma-Protokoll):
  Öffentlich: p, q, g, h = g^x mod p.
  Prüft Kenntnis von x ohne x preiszugeben.

Pedersen-Commitment:
  Commit(m, r) = g^m · h^r mod p  (hiding + binding)

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 17 (ZKP).
          C. P. Schnorr, "Efficient Signature Generation", 1989.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from .keygen import generate_prime, is_prime


@dataclass
class SchnorrParams:
    """Öffentliche Parameter für das Schnorr-Protokoll.

    Parameters
    ----------
    p : große Primzahl
    q : Primteiler von p-1 (Ordnung der Untergruppe)
    g : Erzeuger der Ordnung q in Z_p*
    h : h = g^x mod p (öffentlicher Schlüssel)
    """
    p: int
    q: int
    g: int
    h: int


class SchnorrProver:
    """Beweiser im Schnorr-Protokoll.

    Kennt den privaten Schlüssel x und beweist seine Kenntnis.

    Parameters
    ----------
    params : SchnorrParams
    x      : privater Schlüssel (h = g^x mod p)
    """

    def __init__(self, params: SchnorrParams, x: int) -> None:
        self.params = params
        self.x = x

    def commit(self) -> tuple[int, int]:
        """Phase 1: Commitment R = g^r mod p.

        Returns
        -------
        (R, r)  –  R öffentlich (an Verifier senden), r geheim
        """
        r = secrets.randbelow(self.params.q - 1) + 1
        R = pow(self.params.g, r, self.params.p)
        return R, r

    def respond(self, c: int, r: int) -> int:
        """Phase 3: Antwort s = (r - c·x) mod q.

        Parameters
        ----------
        c : Challenge vom Verifier
        r : private Nonce aus commit()

        Returns
        -------
        int  –  s
        """
        return (r - c * self.x) % self.params.q


class SchnorrVerifier:
    """Verifikator im Schnorr-Protokoll.

    Parameters
    ----------
    params : SchnorrParams
    """

    def __init__(self, params: SchnorrParams) -> None:
        self.params = params

    def challenge(self) -> int:
        """Phase 2: Zufällige Challenge c ∈ {0, …, q-1}."""
        return secrets.randbelow(self.params.q)

    def verify(self, R: int, c: int, s: int) -> bool:
        """Verifikation: g^s · h^c ≡ R mod p?

        Parameters
        ----------
        R : Commitment
        c : Challenge
        s : Antwort des Beweisers

        Returns
        -------
        bool
        """
        lhs = (pow(self.params.g, s, self.params.p) *
               pow(self.params.h, c, self.params.p)) % self.params.p
        return lhs == R


def schnorr_keygen(bits: int = 256) -> tuple[SchnorrParams, int]:
    """Erzeugt Schnorr-Parameter und privaten Schlüssel.

    Parameters
    ----------
    bits : Bit-Länge von p

    Returns
    -------
    (SchnorrParams, x)
    """
    # Einfache Konstruktion: q = (p-1)//2 (sicherer Prime)
    while True:
        q = generate_prime(bits - 1)
        p = 2 * q + 1
        if is_prime(p):
            break
    # g: Erzeuger der Ordnung q (g = 2^2 mod p für sichere Primzahlen oft geeignet)
    g = pow(2, 2, p)
    if pow(g, q, p) != 1:
        g = pow(3, 2, p)
    x = secrets.randbelow(q - 1) + 1
    h = pow(g, x, p)
    return SchnorrParams(p=p, q=q, g=g, h=h), x


def schnorr_prove_interactive(
    params: SchnorrParams, x: int
) -> tuple[int, int, int]:
    """Vollständiger interaktiver Schnorr-Beweis (simuliert).

    Returns
    -------
    (R, c, s)  –  Commitment, Challenge, Antwort
    """
    prover = SchnorrProver(params, x)
    verifier = SchnorrVerifier(params)
    R, r = prover.commit()
    c = verifier.challenge()
    s = prover.respond(c, r)
    return R, c, s


@dataclass
class PedersenParams:
    """Parameter für Pedersen-Commitments.

    Parameters
    ----------
    p : Primzahl
    g : Erzeuger
    h : zweiter Erzeuger (h = g^a mod p, a geheim)
    """
    p: int
    g: int
    h: int


class PedersenCommitment:
    """Pedersen-Commitment-Schema.

    Commit(m, r) = g^m · h^r mod p
    Hiding: Ohne r ist m nicht bestimmbar.
    Binding: Kein anderes (m', r') liefert dasselbe Commit.

    Parameters
    ----------
    params : PedersenParams
    """

    def __init__(self, params: PedersenParams) -> None:
        self.params = params

    def commit(self, m: int, r: int | None = None) -> tuple[int, int]:
        """Erzeugt Commitment C = g^m · h^r mod p.

        Parameters
        ----------
        m : Nachricht
        r : Zufallswert (wird zufällig gewählt, wenn None)

        Returns
        -------
        (C, r)
        """
        if r is None:
            r = secrets.randbelow(self.params.p - 1) + 1
        C = (pow(self.params.g, m, self.params.p) *
             pow(self.params.h, r, self.params.p)) % self.params.p
        return C, r

    def verify(self, C: int, m: int, r: int) -> bool:
        """Verifiziert C = g^m · h^r mod p."""
        expected = (pow(self.params.g, m, self.params.p) *
                    pow(self.params.h, r, self.params.p)) % self.params.p
        return C == expected

    def add(self, C1: int, C2: int) -> int:
        """Homomorphe Addition: Commit(m1+m2) = C1·C2 mod p."""
        return (C1 * C2) % self.params.p


def pedersen_setup(bits: int = 256) -> PedersenParams:
    """Erzeugt Pedersen-Parameter (p, g, h)."""
    p = generate_prime(bits)
    g = 2
    a = secrets.randbelow(p - 2) + 1  # geheimer Diskreter Log
    h = pow(g, a, p)
    return PedersenParams(p=p, g=g, h=h)
