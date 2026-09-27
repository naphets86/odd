"""Affine Cipher Implementation (a=15, b=29, m=26)

This module implements the affine cipher as described in the academic paper "Die Affine Chiffre".
The cipher uses the parameters:
- a = 15 (multiplicator)
- b = 29 (shift, effective b' = 3 since 29 mod 26 = 3)
- m = 26 (alphabet size)
- a_inverse = 7 (multiplicative inverse of 15 mod 26)

Encryption: E(x) = (15x + 3) mod 26
Decryption: D(y) = (7y + 5) mod 26
"""

import math
from typing import Tuple, Optional


def gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor using Euclidean algorithm.
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        int: The greatest common divisor of a and b
    """
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Calculate the extended greatest common divisor.
    
    Uses the extended Euclidean algorithm to find gcd(a, b) and coefficients x, y
    such that a*x + b*y = gcd(a, b).
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        Tuple[int, int, int]: (gcd, x, y) where a*x + b*y = gcd
    """
    if b == 0:
        return a, 1, 0
    
    gcd_val, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return gcd_val, x, y


def mod_inverse(a: int, m: int) -> Optional[int]:
    """
    Calculate the modular multiplicative inverse of a modulo m.
    
    Finds x such that (a * x) mod m == 1, if it exists.
    This requires gcd(a, m) == 1.
    
    Args:
        a: The number to find inverse for
        m: The modulus
    
    Returns:
        Optional[int]: The modular inverse if it exists, None otherwise
    """
    gcd_val, x, _ = extended_gcd(a, m)
    
    if gcd_val != 1:
        return None  # Modular inverse does not exist
    
    return x % m


def char_to_num(char: str) -> int:
    """
    Convert a character to its numeric position (0-25).
    
    Args:
        char: Single character (case-insensitive)
    
    Returns:
        int: Position in alphabet (0=A, 25=Z)
    
    Raises:
        ValueError: If character is not in A-Z range
    """
    upper_char = char.upper()
    if not upper_char.isalpha() or len(upper_char) != 1:
        raise ValueError(f"Invalid character: {char}. Must be a single letter (A-Z).")
    
    num = ord(upper_char) - ord('A')
    if not 0 <= num <= 25:
        raise ValueError(f"Character out of alphabet range: {char}")
    
    return num


def num_to_char(num: int, uppercase: bool = True) -> str:
    """
    Convert a numeric position (0-25) to its character.
    
    Args:
        num: Position in alphabet (0-25)
        uppercase: If True, return uppercase letter, else lowercase
    
    Returns:
        str: Character in alphabet
    
    Raises:
        ValueError: If num is not in 0-25 range
    """
    if not 0 <= num <= 25:
        raise ValueError(f"Invalid numeric position: {num}. Must be 0-25.")
    
    char = chr(ord('A') + num)
    return char if uppercase else char.lower()


def encrypt_char(char: str, a: int = 15, b: int = 29, m: int = 26) -> str:
    """
    Encrypt a single character using the affine cipher.
    
    E(x) = (a*x + b) mod m
    
    Args:
        char: Single character to encrypt (A-Z, case-insensitive)
        a: Multiplicator (default: 15)
        b: Shift value (default: 29)
        m: Modulus/alphabet size (default: 26)
    
    Returns:
        str: Encrypted character (uppercase)
    
    Raises:
        ValueError: If character is invalid or gcd(a, m) != 1
    """
    if gcd(a, m) != 1:
        raise ValueError(f"Invalid key: gcd({a}, {m}) != 1. Multiplicator must be coprime to modulus.")
    
    x = char_to_num(char)
    b_effective = b % m
    encrypted_num = (a * x + b_effective) % m
    return num_to_char(encrypted_num, uppercase=True)


def decrypt_char(char: str, a: int = 15, b: int = 29, m: int = 26) -> str:
    """
    Decrypt a single character using the affine cipher.
    
    D(y) = a_inverse * (y - b) mod m
    
    Args:
        char: Single character to decrypt (A-Z, case-insensitive)
        a: Multiplicator (default: 15)
        b: Shift value (default: 29)
        m: Modulus/alphabet size (default: 26)
    
    Returns:
        str: Decrypted character (uppercase)
    
    Raises:
        ValueError: If character is invalid or gcd(a, m) != 1
    """
    if gcd(a, m) != 1:
        raise ValueError(f"Invalid key: gcd({a}, {m}) != 1. Multiplicator must be coprime to modulus.")
    
    a_inv = mod_inverse(a, m)
    if a_inv is None:
        raise ValueError(f"Cannot find modular inverse for {a} mod {m}")
    
    y = char_to_num(char)
    b_effective = b % m
    decrypted_num = (a_inv * (y - b_effective)) % m
    return num_to_char(decrypted_num, uppercase=True)


def encrypt_text(plaintext: str, a: int = 15, b: int = 29, m: int = 26, keep_case: bool = False) -> str:
    """
    Encrypt text using the affine cipher.
    
    Only alphabetic characters are encrypted; non-alphabetic characters are preserved.
    
    Args:
        plaintext: Text to encrypt
        a: Multiplicator (default: 15)
        b: Shift value (default: 29)
        m: Modulus/alphabet size (default: 26)
        keep_case: If True, maintain original case; if False, convert to uppercase
    
    Returns:
        str: Encrypted text
    
    Raises:
        ValueError: If gcd(a, m) != 1
    """
    if gcd(a, m) != 1:
        raise ValueError(f"Invalid key: gcd({a}, {m}) != 1. Multiplicator must be coprime to modulus.")
    
    ciphertext = ""
    for char in plaintext:
        if char.isalpha():
            encrypted = encrypt_char(char, a, b, m)
            ciphertext += encrypted if not keep_case else (encrypted.lower() if char.islower() else encrypted)
        else:
            ciphertext += char
    
    return ciphertext


def decrypt_text(ciphertext: str, a: int = 15, b: int = 29, m: int = 26, keep_case: bool = False) -> str:
    """
    Decrypt text using the affine cipher.
    
    Only alphabetic characters are decrypted; non-alphabetic characters are preserved.
    
    Args:
        ciphertext: Text to decrypt
        a: Multiplicator (default: 15)
        b: Shift value (default: 29)
        m: Modulus/alphabet size (default: 26)
        keep_case: If True, maintain original case; if False, convert to uppercase
    
    Returns:
        str: Decrypted text
    
    Raises:
        ValueError: If gcd(a, m) != 1
    """
    if gcd(a, m) != 1:
        raise ValueError(f"Invalid key: gcd({a}, {m}) != 1. Multiplicator must be coprime to modulus.")
    
    plaintext = ""
    for char in ciphertext:
        if char.isalpha():
            decrypted = decrypt_char(char, a, b, m)
            plaintext += decrypted if not keep_case else (decrypted.lower() if char.islower() else decrypted)
        else:
            plaintext += char
    
    return plaintext


def validate_key(a: int, m: int = 26) -> bool:
    """
    Validate that a key is valid for affine cipher.
    
    Key is valid if gcd(a, m) == 1.
    
    Args:
        a: Multiplicator to validate
        m: Modulus (default: 26)
    
    Returns:
        bool: True if key is valid, False otherwise
    """
    return gcd(a, m) == 1


def get_key_space(m: int = 26) -> int:
    """
    Calculate the size of the key space for affine cipher.
    
    Key space = φ(m) * m, where φ is Euler's totient function.
    For m=26: φ(26) = 12, so key space = 312
    
    Args:
        m: Modulus/alphabet size (default: 26)
    
    Returns:
        int: Size of key space
    """
    # Count numbers coprime to m
    phi_m = sum(1 for i in range(1, m) if gcd(i, m) == 1)
    return phi_m * m
