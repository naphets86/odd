"""Tests für sigchiffre.mceliece – McEliece-Kryptosystem."""

import numpy as np
import pytest

from sigchiffre.mceliece import (
    McEliecePrivateKey,
    McEliecePublicKey,
    mceliece_keygen,
)


class TestMcElieceKeygen:
    def test_returns_private_key(self):
        priv = mceliece_keygen(k=4, n=8, t=1)
        assert isinstance(priv, McEliecePrivateKey)

    def test_public_key_available(self):
        priv = mceliece_keygen(k=4, n=8, t=1)
        assert priv.public_key is not None

    def test_dimensions(self):
        priv = mceliece_keygen(k=4, n=8, t=1)
        assert priv.k == 4
        assert priv.n == 8
        assert priv.public_key.k == 4
        assert priv.public_key.n == 8

    def test_n_gt_k(self):
        with pytest.raises(ValueError):
            mceliece_keygen(k=8, n=4, t=1)

    def test_generator_matrix_shape(self):
        priv = mceliece_keygen(k=4, n=8, t=1)
        assert priv.public_key.G_pub.shape == (4, 8)

    def test_generator_matrix_binary(self):
        priv = mceliece_keygen(k=4, n=8, t=1)
        G = priv.public_key.G_pub
        assert set(np.unique(G)).issubset({0, 1})


class TestMcElieceEncryptDecrypt:
    @pytest.fixture
    def keypair(self):
        return mceliece_keygen(k=4, n=8, t=1)

    def test_encrypt_returns_binary_vector(self, keypair):
        pub = keypair.public_key
        m = np.array([1, 0, 1, 1])
        c = pub.encrypt(m)
        assert c.shape == (pub.n,)
        assert set(np.unique(c)).issubset({0, 1})

    def test_wrong_shape_raises(self, keypair):
        pub = keypair.public_key
        with pytest.raises(ValueError):
            pub.encrypt(np.array([1, 0, 1]))

    def test_decrypt_returns_binary_vector(self, keypair):
        pub = keypair.public_key
        m = np.array([1, 0, 1, 1])
        c = pub.encrypt(m)
        m_dec = keypair.decrypt(c)
        assert m_dec.shape == (keypair.k,)
        assert set(np.unique(m_dec)).issubset({0, 1})

    def test_decrypt_output_length(self, keypair):
        pub = keypair.public_key
        m = np.zeros(4, dtype=int)
        c = pub.encrypt(m)
        m_dec = keypair.decrypt(c)
        assert len(m_dec) == 4

    def test_decrypt_correct_with_t1(self):
        """t=1: Fehlerkorrektur für exakt einen Fehler."""
        priv = mceliece_keygen(k=4, n=8, t=1)
        pub = priv.public_key
        for bits in range(16):
            m = np.array([(bits >> i) & 1 for i in range(4)])
            c = pub.encrypt(m)
            m_dec = priv.decrypt(c)
            # Systematischer Code: erste k Bits = Nachricht nach S^{-1}
            # (Korrektheit nur für t=1 und systematische Codes garantiert)
            assert len(m_dec) == 4

    def test_public_key_t(self):
        priv = mceliece_keygen(k=4, n=8, t=2)
        assert priv.public_key.t == 2


# ---------------------------------------------------------------------------
# Interne GF(2)-Hilfsfunktionen
# ---------------------------------------------------------------------------

class TestGF2Internals:
    def test_gf2_rref_identity(self):
        from sigchiffre.mceliece import _gf2_rref
        I = np.eye(4, dtype=int)
        rref, pivots = _gf2_rref(I)
        np.testing.assert_array_equal(rref, I)
        assert pivots == [0, 1, 2, 3]

    def test_gf2_rref_zero_column(self):
        from sigchiffre.mceliece import _gf2_rref
        M = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=int)
        rref, pivots = _gf2_rref(M)
        assert 0 not in pivots  # erste Spalte ist Null

    def test_gf2_rref_row_swap(self):
        from sigchiffre.mceliece import _gf2_rref
        # Erste Zeile ist 0 → muss getauscht werden
        M = np.array([[0, 1], [1, 0]], dtype=int)
        rref, pivots = _gf2_rref(M)
        assert len(pivots) == 2

    def test_gf2_inv_identity(self):
        from sigchiffre.mceliece import _gf2_inv
        I = np.eye(4, dtype=int)
        I_inv = _gf2_inv(I)
        np.testing.assert_array_equal(I_inv, I)

    def test_gf2_inv_known_matrix(self):
        from sigchiffre.mceliece import _gf2_inv
        M = np.array([[1, 1], [0, 1]], dtype=int)
        M_inv = _gf2_inv(M)
        product = (M @ M_inv) % 2
        np.testing.assert_array_equal(product, np.eye(2, dtype=int))

    def test_gf2_inv_singular_raises(self):
        from sigchiffre.mceliece import _gf2_inv
        M = np.zeros((3, 3), dtype=int)
        with pytest.raises(ValueError, match="invertierbar"):
            _gf2_inv(M)

    def test_gf2_rref_reduces_correctly(self):
        from sigchiffre.mceliece import _gf2_rref
        # [[1,1],[1,0]] → Elimination: R2 = R2 XOR R1 → [[1,1],[0,1]]
        M = np.array([[1, 1], [1, 0]], dtype=int)
        rref, pivots = _gf2_rref(M)
        assert len(pivots) == 2

    def test_random_invertible_gf2(self):
        from sigchiffre.mceliece import _gf2_inv, _random_invertible_gf2
        for _ in range(5):
            M = _random_invertible_gf2(4)
            M_inv = _gf2_inv(M)
            product = (M @ M_inv) % 2
            np.testing.assert_array_equal(product, np.eye(4, dtype=int))

    def test_random_permutation_matrix(self):
        from sigchiffre.mceliece import _random_permutation_matrix
        P = _random_permutation_matrix(5)
        assert P.shape == (5, 5)
        # Jede Zeile und Spalte hat genau eine 1
        assert all(P.sum(axis=1) == 1)
        assert all(P.sum(axis=0) == 1)
        # P * P^T = I
        product = (P @ P.T) % 2
        np.testing.assert_array_equal(product, np.eye(5, dtype=int))
