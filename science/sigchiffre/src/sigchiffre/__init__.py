"""
sigchiffre
==========
Wissenschaftliches Python-Paket zur Signatur-Chiffre.

Basierend auf: S. Epp, "Die Signatur-Chiffre – Eine strukturbasierte
Verschlüsselungsmethode auf Basis der injektiven Subgraph-Signaturfunktion",
März 2026.

Kernmodule:
  - signature   : Injektive Signaturfunktion σ und Matrix-Kodierung
  - keygen      : Schlüsselgenerierung und Primzahlfunktionen
  - chiffre     : SignatureChiffre (Enc/Dec)

Erweiterungsmodule:
  - elgamal     : ElGamal-Kryptosystem + PublicKeySignatureChiffre
  - homomorphic : HomomorphicSignatureChiffre + Paillier
  - lwe         : Learning With Errors + LWESignatureChiffre
  - hashes      : SignatureHash + MerkleTree
  - steganography: SignatureWatermark + StegaImage
  - mpc         : Shamir Secret Sharing + ThresholdSignatureChiffre
  - zkp         : Schnorr-ZKP + Pedersen-Commitment
  - mceliece    : McEliece-Kryptosystem

Schnellstart::

    from sigchiffre import SignatureChiffre
    import numpy as np

    sc = SignatureChiffre(n=4, a=7, b=3, p=1031)
    A = np.array([[1,0,1,0],[0,1,1,0],[1,0,0,1],[1,0,1,0]])
    C = sc.encrypt(A)   # [94, 129, 304, 367]
    A2 = sc.decrypt(C)  # == A
"""

from .chiffre import SignatureChiffre
from .keygen import generate_key, validate_key, key_space_size, effective_key_bits, is_prime, next_prime
from .signature import sigma, sigma_vector, rho, lambda_, reconstruct_column

__all__ = [
    "SignatureChiffre",
    "generate_key",
    "validate_key",
    "key_space_size",
    "effective_key_bits",
    "is_prime",
    "next_prime",
    "sigma",
    "sigma_vector",
    "rho",
    "lambda_",
    "reconstruct_column",
]

__version__ = "1.0.0"
__author__ = "Stephan Epp"
