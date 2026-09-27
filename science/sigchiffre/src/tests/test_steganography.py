"""Tests für sigchiffre.steganography – Wasserzeichen und LSB-Steganographie."""

import numpy as np
import pytest

from sigchiffre.chiffre import SignatureChiffre
from sigchiffre.steganography import SignatureWatermark, StegaImage


@pytest.fixture
def sc():
    return SignatureChiffre(4, 7, 3, 1031)


@pytest.fixture
def A4():
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])


# ---------------------------------------------------------------------------
# SignatureWatermark
# ---------------------------------------------------------------------------

class TestSignatureWatermark:
    @pytest.fixture
    def wm(self, sc):
        return SignatureWatermark(sc)

    def test_embed_returns_list(self, wm, A4):
        C = wm.embed(A4, b"\xAB")
        assert isinstance(C, list)
        assert len(C) == 4

    def test_embed_extract_roundtrip(self, wm, A4):
        # 4 Chiffratwerte = 4 Bits Kapazitat; wir testen mit 0xF0 (obere 4 Bits)
        watermark = bytes([0xF0])
        C = wm.embed(A4, watermark)
        extracted = wm.extract(C, 1)
        assert extracted == bytes([0xF0])

    def test_embed_extract_multi_byte(self, wm, A4):
        # n=4 Chiffratwerte → 4 Bits Wasserzeichen → max 0 Bytes (4/8 = 0)
        # Testen mit 1-Bit-Wasserzeichen: 1 Bit
        watermark = bytes([0b10000000])  # MSB = 1, Rest 0
        C = wm.embed(A4, watermark)
        extracted = wm.extract(C, 1)
        # Nur erstes Bit relevant
        assert (extracted[0] & 0x80) == (watermark[0] & 0x80)

    def test_embed_zero_watermark(self, wm, A4):
        C = wm.embed(A4, b"\x00")
        extracted = wm.extract(C, 1)
        assert extracted == b"\x00"

    def test_embed_extract_no_watermark(self, wm, A4):
        C = wm.embed(A4, b"")
        extracted = wm.extract(C, 0)
        assert extracted == b""


# ---------------------------------------------------------------------------
# StegaImage
# ---------------------------------------------------------------------------

class TestStegaImage:
    def test_embed_extract_simple(self):
        carrier = bytes(range(256))
        si = StegaImage(carrier)
        msg = b"Hi"
        stego = si.embed(msg)
        extracted = StegaImage.extract(stego)
        assert extracted == msg

    def test_embed_extract_empty(self):
        carrier = bytes(range(64))
        si = StegaImage(carrier)
        stego = si.embed(b"")
        extracted = StegaImage.extract(stego)
        assert extracted == b""

    def test_embed_extract_max_length(self):
        carrier = bytes(range(256)) * 10
        si = StegaImage(carrier)
        msg = bytes(range(255))  # max 255 Bytes
        stego = si.embed(msg)
        extracted = StegaImage.extract(stego)
        assert extracted == msg

    def test_too_long_message_raises(self):
        carrier = bytes(range(256)) * 100
        si = StegaImage(carrier)
        with pytest.raises(ValueError, match="255"):
            si.embed(bytes(256))

    def test_carrier_too_small_raises(self):
        carrier = bytes(4)  # Viel zu klein
        si = StegaImage(carrier)
        with pytest.raises(ValueError, match="Träger"):
            si.embed(b"Hello World!")

    def test_stego_same_length_as_carrier(self):
        carrier = bytes(256)
        si = StegaImage(carrier)
        stego = si.embed(b"test")
        assert len(stego) == len(carrier)

    def test_stego_close_to_carrier(self):
        """Stego-Bild unterscheidet sich nur in LSBs."""
        carrier = bytes([0xFF] * 256)
        si = StegaImage(carrier)
        stego = si.embed(b"x")
        # Jedes Byte kann nur um 1 abweichen (LSB-Flip)
        for orig, mod in zip(carrier, stego):
            assert abs(orig - mod) <= 1

    def test_different_messages_different_stego(self):
        carrier = bytes(range(256))
        si = StegaImage(carrier)
        s1 = si.embed(b"A")
        s2 = si.embed(b"B")
        assert s1 != s2

    def test_extract_static(self):
        """Stego-Extraktion ohne Einbettungs-Objekt."""
        carrier = bytes(256)
        si = StegaImage(carrier)
        stego = si.embed(b"XY")
        extracted = StegaImage.extract(stego)
        assert extracted == b"XY"
