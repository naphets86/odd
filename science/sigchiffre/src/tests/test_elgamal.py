"""Tests für sigchiffre.elgamal – ElGamal und PublicKeySignatureChiffre."""

import numpy as np
import pytest

from sigchiffre.elgamal import (
    ElGamalPrivateKey,
    ElGamalPublicKey,
    PublicKeySignatureChiffre,
    elgamal_keygen,
)


class TestElGamalKeygen:
    def test_returns_private_key(self):
        priv = elgamal_keygen(bits=64)
        assert isinstance(priv, ElGamalPrivateKey)

    def test_public_key_available(self):
        priv = elgamal_keygen(bits=64)
        assert priv.public_key is not None

    def test_p_g_set(self):
        priv = elgamal_keygen(bits=64)
        assert priv.p > 0
        assert priv.g > 0


class TestElGamalEncryptDecrypt:
    @pytest.fixture
    def keypair(self):
        return elgamal_keygen(bits=64)

    def test_encrypt_decrypt_roundtrip(self, keypair):
        pub = keypair.public_key
        m = 42
        c1, c2 = pub.encrypt(m)
        m_dec = keypair.decrypt(c1, c2)
        assert m_dec == m

    def test_encrypt_decrypt_m1(self, keypair):
        pub = keypair.public_key
        c1, c2 = pub.encrypt(1)
        assert keypair.decrypt(c1, c2) == 1

    def test_encrypt_randomized(self, keypair):
        """Wiederholte Verschlüsselung derselben Nachricht → verschiedene Chiffrate."""
        pub = keypair.public_key
        m = 100
        C_list = [pub.encrypt(m) for _ in range(5)]
        # Mit sehr hoher Wahrscheinlichkeit nicht alle gleich
        assert len(set(C_list)) > 1

    def test_invalid_message_raises(self, keypair):
        with pytest.raises(ValueError):
            keypair.public_key.encrypt(0)  # m=0 nicht in Z_p*

    def test_m_equals_p_minus_1(self, keypair):
        pub = keypair.public_key
        m = pub.p - 1
        c1, c2 = pub.encrypt(m)
        assert keypair.decrypt(c1, c2) == m

    def test_multiple_messages(self, keypair):
        pub = keypair.public_key
        for m in range(1, 10):
            c1, c2 = pub.encrypt(m)
            assert keypair.decrypt(c1, c2) == m


class TestElGamalHomomorphic:
    @pytest.fixture
    def keypair(self):
        return elgamal_keygen(bits=64)

    def test_multiplicative_homomorphism(self, keypair):
        """Enc(m1)·Enc(m2) = Enc(m1·m2) mod p."""
        pub = keypair.public_key
        m1, m2 = 3, 5
        C1 = pub.encrypt(m1)
        C2 = pub.encrypt(m2)
        C_prod = pub.homomorphic_multiply(C1, C2)
        m_dec = keypair.decrypt(*C_prod)
        assert m_dec == (m1 * m2) % pub.p


class TestPublicKeySignatureChiffre:
    @pytest.fixture
    def pksc(self):
        priv = elgamal_keygen(bits=64)
        return PublicKeySignatureChiffre(n=4, elgamal_private=priv)

    @pytest.fixture
    def A4(self):
        return np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ])

    def test_encrypt_returns_list(self, pksc, A4):
        C = pksc.encrypt(A4)
        assert len(C) == 4

    def test_decrypt_returns_matrix(self, pksc, A4):
        C = pksc.encrypt(A4)
        A_dec = pksc.decrypt(C)
        assert A_dec.shape == (4, 4)

    def test_roundtrip(self, pksc, A4):
        C = pksc.encrypt(A4)
        A_dec = pksc.decrypt(C)
        np.testing.assert_array_equal(A_dec, A4)

    def test_zero_matrix_columns_with_nonzero_sigma(self, pksc):
        """Nullmatrix: σ(A,0)=0 wird als 1 kodiert (0 ∉ Z_p*); nur Spalten j>0 korrekt."""
        A = np.zeros((4, 4), dtype=int)
        C = pksc.encrypt(A)
        A_dec = pksc.decrypt(C)
        # Spalten j=1,2,3 haben σ=j*2^n>0 → korrekt rekonstruiert
        for j in range(1, 4):
            np.testing.assert_array_equal(A_dec[:, j], A[:, j])
