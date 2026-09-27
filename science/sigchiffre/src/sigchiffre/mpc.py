"""
sigchiffre.mpc
==============
Schwellenwert-Secret-Sharing (Shamir, 1979) und ThresholdSignatureChiffre.

Shamir Secret Sharing (t, n):
  - Split secret s in n Shares: wähle zufälliges Polynom f(x) vom Grad t-1
    mit f(0) = s über Z_p; Share i = (i, f(i)).
  - Rekonstruktion mit Lagrange-Interpolation aus beliebigen t Shares.

ThresholdSignatureChiffre:
  - Schlüssel (a, b, p) wird als Secret über n Shares verteilt.
  - t von n Shares reichen zur Entschlüsselung.

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 16 (MPC).
          A. Shamir, "How to Share a Secret", CACM 1979.
"""
from __future__ import annotations

import secrets
from math import gcd

from .keygen import is_prime


# ---------------------------------------------------------------------------
# Lagrange-Interpolation über Z_p
# ---------------------------------------------------------------------------

def _lagrange_interpolate(x: int, shares: list[tuple[int, int]], p: int) -> int:
    """Lagrange-Interpolation: Wertet das durch `shares` definierte Polynom in x aus.

    Parameters
    ----------
    x      : Auswertungspunkt (0 für das Secret)
    shares : list[(x_i, y_i)]
    p      : Primzahl (Körper Z_p)

    Returns
    -------
    int  –  f(x) mod p
    """
    result = 0
    n = len(shares)
    xs = [s[0] for s in shares]
    ys = [s[1] for s in shares]
    for i in range(n):
        num, den = 1, 1
        for j in range(n):
            if i == j:
                continue
            num = (num * (x - xs[j])) % p
            den = (den * (xs[i] - xs[j])) % p
        result = (result + ys[i] * num % p * pow(den, -1, p)) % p
    return result


# ---------------------------------------------------------------------------
# Shamir Secret Sharing
# ---------------------------------------------------------------------------

class ShamirSecretSharing:
    """(t, n)-Schwellenwert-Secret-Sharing über Z_p.

    Parameters
    ----------
    t : Schwellenwert (Mindestanzahl Shares zur Rekonstruktion)
    n : Gesamtzahl Shares
    p : Primzahl > Secret
    """

    def __init__(self, t: int, n: int, p: int) -> None:
        if t > n:
            raise ValueError(f"Schwellenwert t={t} darf nicht > n={n} sein")
        if not is_prime(p):
            raise ValueError(f"p={p} ist keine Primzahl")
        self.t = t
        self.n = n
        self.p = p

    def split(self, secret: int) -> list[tuple[int, int]]:
        """Teilt `secret` in n Shares auf.

        Parameters
        ----------
        secret : int ∈ {0, …, p-1}

        Returns
        -------
        list[(i, f(i))] für i = 1, …, n
        """
        if not (0 <= secret < self.p):
            raise ValueError(f"Secret {secret} muss in Z_{self.p} liegen")
        # Zufälliges Polynom f(x) vom Grad t-1 mit f(0) = secret
        coeffs = [secret] + [secrets.randbelow(self.p) for _ in range(self.t - 1)]

        def poly(x: int) -> int:
            val = 0
            for i, c in enumerate(coeffs):
                val = (val + c * pow(x, i, self.p)) % self.p
            return val

        return [(i, poly(i)) for i in range(1, self.n + 1)]

    def reconstruct(self, shares: list[tuple[int, int]]) -> int:
        """Rekonstruiert das Secret aus mindestens t Shares.

        Parameters
        ----------
        shares : list[(x_i, y_i)], mindestens t Einträge

        Returns
        -------
        int  –  Secret
        """
        if len(shares) < self.t:
            raise ValueError(
                f"Mindestens {self.t} Shares nötig, erhalten: {len(shares)}"
            )
        return _lagrange_interpolate(0, shares[:self.t], self.p)


# ---------------------------------------------------------------------------
# ThresholdSignatureChiffre
# ---------------------------------------------------------------------------

class ThresholdSignatureChiffre:
    """(t, n)-Schwellenwert-Entschlüsselung der Signatur-Chiffre.

    Verteilung von a und b als getrennte Secrets über je n Shares.
    Entschlüsselung erst nach Zusammenführung von t Shares möglich.

    Parameters
    ----------
    n_block : Blockgröße der Signatur-Chiffre
    a, b, p : Signatur-Chiffre-Schlüssel
    t       : Schwellenwert
    n_shares: Anzahl Shares
    """

    def __init__(
        self, n_block: int, a: int, b: int, p: int, t: int, n_shares: int
    ) -> None:
        self.n_block = n_block
        self.p = p
        self.t = t
        self.n_shares = n_shares
        sss = ShamirSecretSharing(t, n_shares, p)
        self.shares_a = sss.split(a)
        self.shares_b = sss.split(b)
        self._sss = sss

    def get_share(self, i: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Share i (1-basiert) als (share_a, share_b)-Paar."""
        return self.shares_a[i - 1], self.shares_b[i - 1]

    def decrypt(
        self,
        C: list[int],
        selected_shares_a: list[tuple[int, int]],
        selected_shares_b: list[tuple[int, int]],
    ) -> "np.ndarray":
        """Entschlüsselt C aus t Shares von a und b.

        Parameters
        ----------
        C                : Chiffrat (n_block Werte)
        selected_shares_a: mindestens t (a-)Shares
        selected_shares_b: mindestens t (b-)Shares

        Returns
        -------
        np.ndarray, shape (n_block, n_block)
        """
        import numpy as np
        from .signature import reconstruct_column
        a = self._sss.reconstruct(selected_shares_a)
        b = self._sss.reconstruct(selected_shares_b)
        a_inv = pow(a, -1, self.p)
        n = self.n_block
        A = np.zeros((n, n), dtype=int)
        for j in range(n):
            s_j = (a_inv * (C[j] - b)) % self.p
            col = reconstruct_column(s_j, j, n)
            for i in range(n):
                A[i, j] = col[i]
        return A
