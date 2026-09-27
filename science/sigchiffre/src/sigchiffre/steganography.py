"""
sigchiffre.steganography
========================
Steganographie und digitale Wasserzeichen auf Basis der Signatur-Chiffre.

- SignatureWatermark: Einbetten von Wasserzeichen in binäre Matrizen via
  LSB-Substitution in den Signaturwerten.
- StegaImage: Pixel-LSB-Steganographie in Byte-Arrays.

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 18 (Steganographie).
          I.J. Cox et al., "Secure Spread Spectrum Watermarking", 1997.
"""
from __future__ import annotations

import numpy as np

from .chiffre import SignatureChiffre
from .signature import sigma_vector, reconstruct_column


class SignatureWatermark:
    """Wasserzeichen-Einbettung in Signaturwerte via LSB-Substitution.

    Ein k-Bit-Wasserzeichen wird in die k niedrigstwertigen Bits der
    ersten k Signaturwerte eingebettet. Die Extraktion ist ohne Schlüssel
    nicht möglich, da die Signaturwerte erst nach Entschlüsselung sichtbar sind.

    Parameters
    ----------
    chiffre : SignatureChiffre
    """

    def __init__(self, chiffre: SignatureChiffre) -> None:
        self.chiffre = chiffre

    def embed(self, A: np.ndarray, watermark: bytes) -> list[int]:
        """Bettet `watermark` in das Chiffrat ein.

        Die Wasserzeichenbits werden in die LSBs der Chiffratwerte eingebettet
        (nach Verschlüsselung, vor Übertragung).

        Parameters
        ----------
        A         : n×n Klartextmatrix
        watermark : Rohdaten (max. n//8 Bytes empfohlen)

        Returns
        -------
        list[int]  –  Chiffrat mit eingebettetem Wasserzeichen
        """
        C = self.chiffre.encrypt(A)
        wm_bits: list[int] = []
        for byte in watermark:
            for bit_idx in range(7, -1, -1):
                wm_bits.append((byte >> bit_idx) & 1)
        for k, bit in enumerate(wm_bits):
            if k >= len(C):
                break
            C[k] = (C[k] & ~1) | bit  # LSB ersetzen
        return C

    def extract(self, C: list[int], num_bytes: int) -> bytes:
        """Extrahiert `num_bytes` Bytes Wasserzeichen aus dem Chiffrat.

        Parameters
        ----------
        C         : Chiffrat (möglicherweise mit Wasserzeichen)
        num_bytes : Anzahl zu extrahierender Bytes

        Returns
        -------
        bytes
        """
        bits_needed = num_bytes * 8
        extracted_bits = [(C[k] & 1) for k in range(min(bits_needed, len(C)))]
        # Zu Bytes zusammensetzen
        result = bytearray()
        for byte_idx in range(num_bytes):
            byte_bits = extracted_bits[byte_idx * 8: (byte_idx + 1) * 8]
            if len(byte_bits) < 8:
                byte_bits += [0] * (8 - len(byte_bits))
            result.append(sum(b << (7 - i) for i, b in enumerate(byte_bits)))
        return bytes(result)


class StegaImage:
    """LSB-Steganographie in Byte-Arrays (z.B. Graustufen-Pixeldaten).

    Bettet eine Nachricht in die niedrigstwertigen Bits eines Trägers ein.
    Das erste Byte kodiert die Länge der Nachricht (max. 255 Bytes).

    Parameters
    ----------
    carrier : bytearray oder bytes – Trägermedium
    """

    def __init__(self, carrier: bytes | bytearray) -> None:
        self.carrier = bytearray(carrier)

    def embed(self, message: bytes) -> bytearray:
        """Bettet `message` in den Träger ein.

        Parameters
        ----------
        message : Rohdaten (max. 255 Bytes)

        Returns
        -------
        bytearray  –  Stego-Träger
        """
        if len(message) > 255:
            raise ValueError("Nachricht darf max. 255 Bytes haben")
        bits: list[int] = []
        # Länge zuerst (8 Bits)
        length = len(message)
        for bit_idx in range(7, -1, -1):
            bits.append((length >> bit_idx) & 1)
        # Nachrichtenbits
        for byte in message:
            for bit_idx in range(7, -1, -1):
                bits.append((byte >> bit_idx) & 1)
        if len(bits) > len(self.carrier):
            raise ValueError(
                f"Träger zu klein: {len(bits)} Bits benötigt, "
                f"{len(self.carrier)} verfügbar"
            )
        stego = bytearray(self.carrier)
        for k, bit in enumerate(bits):
            stego[k] = (stego[k] & ~1) | bit
        return stego

    @staticmethod
    def extract(stego: bytes | bytearray) -> bytes:
        """Extrahiert eine eingebettete Nachricht aus dem Stego-Träger.

        Parameters
        ----------
        stego : Stego-Träger

        Returns
        -------
        bytes  –  extrahierte Nachricht
        """
        stego = bytearray(stego)
        # Länge lesen (erste 8 LSBs)
        length_bits = [(stego[k] & 1) for k in range(8)]
        length = sum(b << (7 - i) for i, b in enumerate(length_bits))
        # Nachricht lesen
        msg_bits: list[int] = []
        for k in range(8, 8 + length * 8):
            if k >= len(stego):
                break
            msg_bits.append(stego[k] & 1)
        result = bytearray()
        for byte_idx in range(length):
            byte_bits = msg_bits[byte_idx * 8: (byte_idx + 1) * 8]
            if len(byte_bits) < 8:
                byte_bits += [0] * (8 - len(byte_bits))
            result.append(sum(b << (7 - i) for i, b in enumerate(byte_bits)))
        return bytes(result)
