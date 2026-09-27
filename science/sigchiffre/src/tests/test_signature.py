"""Tests für sigchiffre.signature – vollständige Abdeckung der Signaturfunktion."""

import numpy as np
import pytest

from sigchiffre.signature import (
    bits_to_column,
    bits_to_matrix,
    blocks_to_bytes,
    bytes_to_blocks,
    lambda_,
    matrix_to_bits,
    reconstruct_column,
    rho,
    sigma,
    sigma_vector,
)


# ---------------------------------------------------------------------------
# Fixtures: Beispiel aus Kapitel 6 (a=7, b=3, p=1031, n=4)
# ---------------------------------------------------------------------------

@pytest.fixture
def A4():
    """Klartextmatrix aus Kapitel 6 der Arbeit."""
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
# rho (Zeilenkomponente)
# ---------------------------------------------------------------------------

class TestRho:
    def test_example_ch6_col0(self, A4):
        # Spalte 0: (1,0,1,1) → 1+0+4+8 = 13
        assert rho(A4, 0) == 13

    def test_example_ch6_col1(self, A4):
        # Spalte 1: (0,1,0,0) → 0+2+0+0 = 2
        assert rho(A4, 1) == 2

    def test_example_ch6_col2(self, A4):
        # Spalte 2: (1,1,0,1) → 1+2+0+8 = 11
        assert rho(A4, 2) == 11

    def test_example_ch6_col3(self, A4):
        # Spalte 3: (0,0,1,0) → 0+0+4+0 = 4
        assert rho(A4, 3) == 4

    def test_zero_matrix(self, A_zero):
        for j in range(4):
            assert rho(A_zero, j) == 0

    def test_all_ones_col(self, A_ones):
        # Alle 1 → 2^4-1 = 15
        for j in range(4):
            assert rho(A_ones, j) == 15

    def test_bounds(self, A4):
        n = A4.shape[0]
        for j in range(n):
            val = rho(A4, j)
            assert 0 <= val <= (1 << n) - 1

    def test_single_row_set(self):
        A = np.zeros((4, 4), dtype=int)
        A[2, 0] = 1  # 2^2 = 4
        assert rho(A, 0) == 4

    def test_n2(self):
        A = np.array([[1, 0], [0, 1]])
        assert rho(A, 0) == 1   # Spalte 0: (1,0) → 1
        assert rho(A, 1) == 2   # Spalte 1: (0,1) → 2

    def test_return_type(self, A4):
        assert isinstance(rho(A4, 0), int)


# ---------------------------------------------------------------------------
# lambda_
# ---------------------------------------------------------------------------

class TestLambda:
    def test_j0(self):
        assert lambda_(0, 4) == 0

    def test_j1(self):
        assert lambda_(1, 4) == 16   # 1 * 2^4

    def test_j2(self):
        assert lambda_(2, 4) == 32

    def test_j3(self):
        assert lambda_(3, 4) == 48

    def test_general(self):
        for j in range(8):
            assert lambda_(j, 4) == j * 16

    def test_n2(self):
        assert lambda_(1, 2) == 4  # 1 * 2^2


# ---------------------------------------------------------------------------
# sigma (Signaturfunktion)
# ---------------------------------------------------------------------------

class TestSigma:
    def test_example_ch6_col0(self, A4):
        # σ(A,0) = 13 + 0 = 13
        assert sigma(A4, 0) == 13

    def test_example_ch6_col1(self, A4):
        # σ(A,1) = 2 + 16 = 18
        assert sigma(A4, 1) == 18

    def test_example_ch6_col2(self, A4):
        # σ(A,2) = 11 + 32 = 43
        assert sigma(A4, 2) == 43

    def test_example_ch6_col3(self, A4):
        # σ(A,3) = 4 + 48 = 52
        assert sigma(A4, 3) == 52

    def test_range_constraint(self, A4):
        """σ(A,j) ∈ [j·2^n, (j+1)·2^n)"""
        n = A4.shape[0]
        for j in range(n):
            val = sigma(A4, j)
            assert j * (1 << n) <= val < (j + 1) * (1 << n)

    def test_sigma_lt_2_2n(self, A4):
        """σ(A,j) < 2^(2n) für alle j (Korrektheitsbedingung)."""
        n = A4.shape[0]
        for j in range(n):
            assert sigma(A4, j) < (1 << (2 * n))

    def test_zero_matrix(self, A_zero):
        n = A_zero.shape[0]
        for j in range(n):
            assert sigma(A_zero, j) == lambda_(j, n)

    def test_injectivity(self):
        """Injektivität: verschiedene (A,j) → verschiedene σ-Werte."""
        n = 3
        seen = {}
        for bits in range(1 << (n * n)):
            A = np.zeros((n, n), dtype=int)
            for pos in range(n * n):
                A[pos // n, pos % n] = (bits >> pos) & 1
            for j in range(n):
                val = sigma(A, j)
                key = (j, val)
                col_tuple = tuple(A[:, j])
                if key in seen:
                    assert seen[key] == col_tuple, "Injektivitätsverletzung!"
                seen[key] = col_tuple

    def test_injectivity_distinct_columns(self):
        """Gleiche Spalte, verschiedener Index → verschiedene σ-Werte."""
        A = np.ones((4, 4), dtype=int)
        s0 = sigma(A, 0)
        s1 = sigma(A, 1)
        assert s0 != s1


# ---------------------------------------------------------------------------
# sigma_vector
# ---------------------------------------------------------------------------

class TestSigmaVector:
    def test_example_ch6(self, A4):
        assert sigma_vector(A4) == [13, 18, 43, 52]

    def test_length(self, A4):
        sv = sigma_vector(A4)
        assert len(sv) == 4

    def test_zero_matrix(self, A_zero):
        n = A_zero.shape[0]
        sv = sigma_vector(A_zero)
        for j in range(n):
            assert sv[j] == lambda_(j, n)

    def test_return_type(self, A4):
        sv = sigma_vector(A4)
        assert isinstance(sv, list)
        assert all(isinstance(v, int) for v in sv)


# ---------------------------------------------------------------------------
# bits_to_column / reconstruct_column
# ---------------------------------------------------------------------------

class TestBitsToColumn:
    def test_value_13(self):
        # 13 = 1101_2 → Bits (LSB first): 1, 0, 1, 1
        assert bits_to_column(13, 4) == [1, 0, 1, 1]

    def test_value_2(self):
        # 2 = 0010_2 → 0, 1, 0, 0
        assert bits_to_column(2, 4) == [0, 1, 0, 0]

    def test_value_0(self):
        assert bits_to_column(0, 4) == [0, 0, 0, 0]

    def test_value_15(self):
        assert bits_to_column(15, 4) == [1, 1, 1, 1]

    def test_length(self):
        for n in [2, 4, 8]:
            assert len(bits_to_column(7, n)) == n

    def test_roundtrip(self):
        for n in [2, 4]:
            for val in range(1 << n):
                bits = bits_to_column(val, n)
                reconstructed = sum(b << i for i, b in enumerate(bits))
                assert reconstructed == val


class TestReconstructColumn:
    def test_col0_from_ch6(self, A4):
        # s_0 = σ(A,0) = 13; j=0 → rho = 13 → [1,0,1,1]
        col = reconstruct_column(13, 0, 4)
        assert col == [1, 0, 1, 1]

    def test_col1_from_ch6(self, A4):
        # s_1 = 18; j=1 → rho = 18-16=2 → [0,1,0,0]
        col = reconstruct_column(18, 1, 4)
        assert col == [0, 1, 0, 0]

    def test_col2_from_ch6(self, A4):
        col = reconstruct_column(43, 2, 4)
        assert col == [1, 1, 0, 1]

    def test_col3_from_ch6(self, A4):
        col = reconstruct_column(52, 3, 4)
        assert col == [0, 0, 1, 0]

    def test_roundtrip_all_columns(self, A4):
        n = A4.shape[0]
        sv = sigma_vector(A4)
        for j in range(n):
            col = reconstruct_column(sv[j], j, n)
            expected = list(A4[:, j])
            assert col == expected


# ---------------------------------------------------------------------------
# matrix_to_bits / bits_to_matrix
# ---------------------------------------------------------------------------

class TestMatrixBitsRoundtrip:
    def test_matrix_to_bits_length(self, A4):
        bits = matrix_to_bits(A4)
        assert len(bits) == 16

    def test_matrix_to_bits_content(self, A4):
        # Spaltenweise: Spalte0=(1,0,1,1), Spalte1=(0,1,0,0), etc.
        bits = matrix_to_bits(A4)
        expected = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        assert bits == expected

    def test_roundtrip(self, A4):
        bits = matrix_to_bits(A4)
        A_rec = bits_to_matrix(bits, 4)
        np.testing.assert_array_equal(A4, A_rec)

    def test_zero_matrix(self, A_zero):
        bits = matrix_to_bits(A_zero)
        assert all(b == 0 for b in bits)
        A_rec = bits_to_matrix(bits, 4)
        np.testing.assert_array_equal(A_zero, A_rec)

    def test_ones_matrix(self, A_ones):
        bits = matrix_to_bits(A_ones)
        assert all(b == 1 for b in bits)


# ---------------------------------------------------------------------------
# bytes_to_blocks / blocks_to_bytes
# ---------------------------------------------------------------------------

class TestBytesBlocksRoundtrip:
    def test_single_byte_roundtrip(self):
        data = bytes([0xAB])
        blocks = bytes_to_blocks(data, 4)
        result = blocks_to_bytes(blocks, 4, len(data))
        assert result == data

    def test_multiple_bytes(self):
        data = bytes(range(16))
        blocks = bytes_to_blocks(data, 8)
        result = blocks_to_bytes(blocks, 8, len(data))
        assert result == data

    def test_empty_data(self):
        data = b""
        blocks = bytes_to_blocks(data, 4)
        assert blocks == []

    def test_padding_stripped(self):
        # 1 Byte in n=8: 1 Block mit 64 Bits
        data = bytes([0xFF])
        blocks = bytes_to_blocks(data, 8)
        assert len(blocks) == 1
        result = blocks_to_bytes(blocks, 8, 1)
        assert result == data

    def test_block_shape(self):
        data = bytes([0x0F, 0xF0])
        blocks = bytes_to_blocks(data, 4)
        for b in blocks:
            assert b.shape == (4, 4)
