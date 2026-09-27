"""Tests für sigchiffre.chiffre – SignatureChiffre (Korrektheit, Fehlerbehandlung)."""

import numpy as np
import pytest

from sigchiffre.chiffre import SignatureChiffre


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sc_ch6():
    """SignatureChiffre mit Beispielparametern aus Kapitel 6."""
    return SignatureChiffre(n=4, a=7, b=3, p=1031)


@pytest.fixture
def sc_ch7():
    """SignatureChiffre mit Beispielparametern aus Kapitel 7 (Fermat-Primzahl)."""
    return SignatureChiffre(n=4, a=17, b=16, p=257)


@pytest.fixture
def A4():
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])


@pytest.fixture
def A_zero():
    return np.zeros((4, 4), dtype=int)


@pytest.fixture
def A_ones():
    return np.ones((4, 4), dtype=int)


# ---------------------------------------------------------------------------
# Konstruktor-Validierung
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_valid_ch6(self):
        sc = SignatureChiffre(4, 7, 3, 1031)
        assert sc.n == 4
        assert sc.a == 7
        assert sc.b == 3
        assert sc.p == 1031

    def test_valid_ch7(self):
        sc = SignatureChiffre(4, 17, 16, 257)
        assert sc.n == 4

    def test_p_not_prime_raises(self):
        with pytest.raises(ValueError):
            SignatureChiffre(4, 7, 3, 100)

    def test_p_too_small_raises(self):
        with pytest.raises(ValueError):
            SignatureChiffre(4, 7, 3, 100)  # 100 < 256

    def test_a_zero_raises(self):
        with pytest.raises(ValueError):
            SignatureChiffre(4, 0, 3, 1031)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError):
            SignatureChiffre(4, 7, -1, 1031)

    def test_a_inv_precomputed(self):
        sc = SignatureChiffre(4, 7, 3, 1031)
        # 7 * 442 = 3094 = 3*1031 + 1 ≡ 1 (mod 1031)
        assert sc.a_inv == 442

    def test_a_inv_ch7(self):
        sc = SignatureChiffre(4, 17, 16, 257)
        # 17 * 121 = 2057 = 8*257 + 1 ≡ 1 (mod 257)
        assert sc.a_inv == 121

    def test_repr(self):
        sc = SignatureChiffre(4, 7, 3, 1031)
        assert "SignatureChiffre" in repr(sc)
        assert "n=4" in repr(sc)


# ---------------------------------------------------------------------------
# Verschlüsselung (encrypt)
# ---------------------------------------------------------------------------

class TestEncrypt:
    def test_example_ch6(self, sc_ch6, A4):
        C = sc_ch6.encrypt(A4)
        assert C == [94, 129, 304, 367]

    def test_example_ch7(self, sc_ch7, A4):
        C = sc_ch7.encrypt(A4)
        assert C == [237, 65, 233, 129]

    def test_output_length(self, sc_ch6, A4):
        C = sc_ch6.encrypt(A4)
        assert len(C) == 4

    def test_output_in_Zp(self, sc_ch6, A4):
        C = sc_ch6.encrypt(A4)
        for c in C:
            assert 0 <= c < sc_ch6.p

    def test_zero_matrix(self, sc_ch6, A_zero):
        C = sc_ch6.encrypt(A_zero)
        # σ(0,j) = j·2^4 → c_j = (7·j·16 + 3) mod 1031
        assert C[0] == (7 * 0 + 3) % 1031
        assert C[1] == (7 * 16 + 3) % 1031

    def test_wrong_shape_raises(self, sc_ch6):
        with pytest.raises(ValueError):
            sc_ch6.encrypt(np.zeros((3, 4), dtype=int))

    def test_different_keys_different_ciphertexts(self, sc_ch6, sc_ch7, A4):
        """Gleicher Klartext, verschiedene Schlüssel → verschiedene Chiffrate (Satz 7.7)."""
        C1 = sc_ch6.encrypt(A4)
        C2 = sc_ch7.encrypt(A4)
        assert C1 != C2
        # Keine Komponente darf gleich sein (Satz 7.7)
        for c1, c2 in zip(C1, C2):
            assert c1 != c2

    def test_deterministic(self, sc_ch6, A4):
        """Enc ist deterministisch (keine Zufälligkeit in Basis-Enc)."""
        assert sc_ch6.encrypt(A4) == sc_ch6.encrypt(A4)


# ---------------------------------------------------------------------------
# Entschlüsselung (decrypt)
# ---------------------------------------------------------------------------

class TestDecrypt:
    def test_example_ch6(self, sc_ch6, A4):
        C = [94, 129, 304, 367]
        A_dec = sc_ch6.decrypt(C)
        np.testing.assert_array_equal(A_dec, A4)

    def test_example_ch7(self, sc_ch7, A4):
        C = [237, 65, 233, 129]
        A_dec = sc_ch7.decrypt(C)
        np.testing.assert_array_equal(A_dec, A4)

    def test_wrong_length_raises(self, sc_ch6):
        with pytest.raises(ValueError):
            sc_ch6.decrypt([1, 2, 3])  # Länge 3 statt 4

    def test_output_shape(self, sc_ch6):
        C = [94, 129, 304, 367]
        A = sc_ch6.decrypt(C)
        assert A.shape == (4, 4)

    def test_output_binary(self, sc_ch6):
        C = [94, 129, 304, 367]
        A = sc_ch6.decrypt(C)
        assert set(np.unique(A)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Korrektheit: Enc ∘ Dec = Id (Satz 4.1)
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_ch6_roundtrip(self, sc_ch6, A4):
        assert np.array_equal(sc_ch6.decrypt(sc_ch6.encrypt(A4)), A4)

    def test_ch7_roundtrip(self, sc_ch7, A4):
        assert np.array_equal(sc_ch7.decrypt(sc_ch7.encrypt(A4)), A4)

    def test_zero_matrix_roundtrip(self, sc_ch6, A_zero):
        assert np.array_equal(sc_ch6.decrypt(sc_ch6.encrypt(A_zero)), A_zero)

    def test_ones_matrix_roundtrip(self, sc_ch6, A_ones):
        assert np.array_equal(sc_ch6.decrypt(sc_ch6.encrypt(A_ones)), A_ones)

    def test_all_bit_patterns_n2(self):
        """Korrektheit für alle 16 möglichen 2×2-Matrizen."""
        sc = SignatureChiffre(n=2, a=3, b=1, p=19)  # 19 > 16 = 2^(2·2)
        for bits in range(16):
            A = np.array([[(bits >> k) & 1 for k in range(4)]]).reshape(2, 2, order='F')
            C = sc.encrypt(A)
            A_dec = sc.decrypt(C)
            np.testing.assert_array_equal(A_dec, A, err_msg=f"Fehler bei bits={bits:04b}")

    def test_random_keys_roundtrip(self):
        """Korrektheit für zufällig generierte Schlüssel."""
        from sigchiffre.keygen import generate_key
        for _ in range(5):
            a, b, p = generate_key(4, bits=32)
            sc = SignatureChiffre(4, a, b, p)
            A = np.random.randint(0, 2, (4, 4))
            assert np.array_equal(sc.decrypt(sc.encrypt(A)), A)


# ---------------------------------------------------------------------------
# Bytes-API
# ---------------------------------------------------------------------------

class TestBytesAPI:
    def test_encrypt_decrypt_bytes_simple(self, sc_ch6):
        data = b"Hello!"
        encrypted, orig_len = sc_ch6.encrypt_bytes(data)
        decrypted = sc_ch6.decrypt_bytes(encrypted, orig_len)
        assert decrypted == data

    def test_encrypt_decrypt_empty(self, sc_ch6):
        data = b""
        encrypted, orig_len = sc_ch6.encrypt_bytes(data)
        decrypted = sc_ch6.decrypt_bytes(encrypted, orig_len)
        assert decrypted == data

    def test_encrypt_decrypt_long(self, sc_ch6):
        data = bytes(range(256))
        encrypted, orig_len = sc_ch6.encrypt_bytes(data)
        assert orig_len == 256
        decrypted = sc_ch6.decrypt_bytes(encrypted, orig_len)
        assert decrypted == data

    def test_ciphertext_blocks_structure(self, sc_ch6):
        data = bytes([0xFF, 0x00])
        encrypted, _ = sc_ch6.encrypt_bytes(data)
        # Jeder Block: Liste von n Ints
        for block in encrypted:
            assert len(block) == sc_ch6.n
            for c in block:
                assert isinstance(c, int)
                assert 0 <= c < sc_ch6.p

    def test_encrypt_decrypt_binary_data(self, sc_ch7):
        data = bytes([0xDE, 0xAD, 0xBE, 0xEF])
        enc, orig = sc_ch7.encrypt_bytes(data)
        dec = sc_ch7.decrypt_bytes(enc, orig)
        assert dec == data
