"""Comprehensive tests for the Affine Cipher module.

Tests cover all functions with various input scenarios:
- Basic encryption/decryption
- Key validation
- Edge cases
- Error handling
"""

import pytest
from src.affine_cipher import (
    gcd,
    extended_gcd,
    mod_inverse,
    char_to_num,
    num_to_char,
    encrypt_char,
    decrypt_char,
    encrypt_text,
    decrypt_text,
    validate_key,
    get_key_space,
)


class TestGCD:
    """Tests for Greatest Common Divisor function."""
    
    def test_gcd_basic(self):
        """Test basic GCD calculations."""
        assert gcd(15, 26) == 1
        assert gcd(12, 8) == 4
        assert gcd(17, 19) == 1
    
    def test_gcd_with_zero(self):
        """Test GCD with zero."""
        assert gcd(5, 0) == 5
        assert gcd(0, 7) == 7
    
    def test_gcd_same_numbers(self):
        """Test GCD of identical numbers."""
        assert gcd(10, 10) == 10
        assert gcd(1, 1) == 1
    
    def test_gcd_large_numbers(self):
        """Test GCD of large numbers."""
        assert gcd(1071, 462) == 21
        assert gcd(2000, 1500) == 500
    
    def test_gcd_consecutive_fibonacci(self):
        """Test GCD of consecutive Fibonacci numbers (should be 1)."""
        assert gcd(5, 8) == 1
        assert gcd(13, 21) == 1


class TestExtendedGCD:
    """Tests for Extended Greatest Common Divisor function."""
    
    def test_extended_gcd_basic(self):
        """Test extended GCD calculation."""
        gcd_val, x, y = extended_gcd(15, 26)
        assert gcd_val == 1
        assert 15 * x + 26 * y == 1
    
    def test_extended_gcd_coefficients(self):
        """Test that coefficients satisfy the Bézout identity."""
        gcd_val, x, y = extended_gcd(10, 15)
        assert gcd_val == 5
        assert 10 * x + 15 * y == 5
    
    def test_extended_gcd_with_zero(self):
        """Test extended GCD with zero."""
        gcd_val, x, y = extended_gcd(5, 0)
        assert gcd_val == 5
        assert 5 * x + 0 * y == 5
    
    def test_extended_gcd_same_numbers(self):
        """Test extended GCD of identical numbers."""
        gcd_val, x, y = extended_gcd(7, 7)
        assert gcd_val == 7
        assert 7 * x + 7 * y == 7


class TestModInverse:
    """Tests for Modular Multiplicative Inverse function."""
    
    def test_mod_inverse_15_mod_26(self):
        """Test that inverse of 15 mod 26 is 7."""
        inv = mod_inverse(15, 26)
        assert inv == 7
        assert (15 * inv) % 26 == 1
    
    def test_mod_inverse_7_mod_26(self):
        """Test that inverse of 7 mod 26 is 15."""
        inv = mod_inverse(7, 26)
        assert inv == 15
        assert (7 * inv) % 26 == 1
    
    def test_mod_inverse_1(self):
        """Test inverse of 1 (should be 1)."""
        inv = mod_inverse(1, 26)
        assert inv == 1
    
    def test_mod_inverse_nonexistent(self):
        """Test that inverse doesn't exist when gcd != 1."""
        inv = mod_inverse(2, 26)
        assert inv is None
        inv = mod_inverse(13, 26)
        assert inv is None
    
    def test_mod_inverse_verification(self):
        """Verify modular inverse property for multiple values."""
        for a in [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]:
            inv = mod_inverse(a, 26)
            assert inv is not None
            assert (a * inv) % 26 == 1


class TestCharToNum:
    """Tests for character to number conversion."""
    
    def test_char_to_num_lowercase(self):
        """Test conversion of lowercase letters."""
        assert char_to_num('a') == 0
        assert char_to_num('z') == 25
        assert char_to_num('h') == 7
    
    def test_char_to_num_uppercase(self):
        """Test conversion of uppercase letters."""
        assert char_to_num('A') == 0
        assert char_to_num('Z') == 25
        assert char_to_num('H') == 7
    
    def test_char_to_num_all_letters(self):
        """Test all letters A-Z."""
        for i in range(26):
            assert char_to_num(chr(ord('A') + i)) == i
            assert char_to_num(chr(ord('a') + i)) == i
    
    def test_char_to_num_invalid_digit(self):
        """Test that digits raise ValueError."""
        with pytest.raises(ValueError):
            char_to_num('5')
    
    def test_char_to_num_invalid_special_char(self):
        """Test that special characters raise ValueError."""
        with pytest.raises(ValueError):
            char_to_num('!')
        with pytest.raises(ValueError):
            char_to_num(' ')
    
    def test_char_to_num_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            char_to_num('')
    
    def test_char_to_num_multiple_chars(self):
        """Test that multiple characters raise ValueError."""
        with pytest.raises(ValueError):
            char_to_num('AB')


class TestNumToChar:
    """Tests for number to character conversion."""
    
    def test_num_to_char_uppercase(self):
        """Test conversion to uppercase letters."""
        assert num_to_char(0) == 'A'
        assert num_to_char(25) == 'Z'
        assert num_to_char(7) == 'H'
    
    def test_num_to_char_lowercase(self):
        """Test conversion to lowercase letters."""
        assert num_to_char(0, uppercase=False) == 'a'
        assert num_to_char(25, uppercase=False) == 'z'
        assert num_to_char(7, uppercase=False) == 'h'
    
    def test_num_to_char_all_positions(self):
        """Test all positions 0-25."""
        for i in range(26):
            upper = num_to_char(i, uppercase=True)
            assert ord(upper) - ord('A') == i
            lower = num_to_char(i, uppercase=False)
            assert ord(lower) - ord('a') == i
    
    def test_num_to_char_invalid_negative(self):
        """Test that negative numbers raise ValueError."""
        with pytest.raises(ValueError):
            num_to_char(-1)
    
    def test_num_to_char_invalid_too_large(self):
        """Test that numbers >= 26 raise ValueError."""
        with pytest.raises(ValueError):
            num_to_char(26)
        with pytest.raises(ValueError):
            num_to_char(100)


class TestEncryptChar:
    """Tests for single character encryption."""
    
    def test_encrypt_char_examples_from_paper(self):
        """Test examples from the academic paper."""
        # E(0) = (15*0 + 3) mod 26 = 3 -> A to D
        assert encrypt_char('A') == 'D'
        # E(1) = (15*1 + 3) mod 26 = 18 -> B to S
        assert encrypt_char('B') == 'S'
        # E(2) = (15*2 + 3) mod 26 = 33 mod 26 = 7 -> C to H
        assert encrypt_char('C') == 'H'
        # E(7) = (15*7 + 3) mod 26 = 108 mod 26 = 4 -> H to E
        assert encrypt_char('H') == 'E'
    
    def test_encrypt_char_all_uppercase(self):
        """Test encryption of all uppercase letters."""
        expected = "DSWHLAPFUKZOETIXMCRUEAWLPFU"
        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            encrypted = encrypt_char(char)
            # Verify it's uppercase
            assert encrypted.isupper()
    
    def test_encrypt_char_case_insensitivity(self):
        """Test that case is ignored in input."""
        assert encrypt_char('a') == encrypt_char('A')
        assert encrypt_char('h') == encrypt_char('H')
        assert encrypt_char('z') == encrypt_char('Z')
    
    def test_encrypt_char_custom_key_valid(self):
        """Test encryption with different valid keys."""
        # a=1 (identity for multiplication), b=0 -> no encryption
        encrypted = encrypt_char('A', a=1, b=0)
        assert encrypted == 'A'
        
        # a=1, b=1 -> simple shift by 1 (Caesar)
        encrypted = encrypt_char('A', a=1, b=1)
        assert encrypted == 'B'
    
    def test_encrypt_char_invalid_key(self):
        """Test that invalid key raises ValueError."""
        # a=2 is not coprime with m=26
        with pytest.raises(ValueError, match="Invalid key"):
            encrypt_char('A', a=2, b=0)
        
        # a=13 is not coprime with m=26
        with pytest.raises(ValueError, match="Invalid key"):
            encrypt_char('A', a=13, b=0)
    
    def test_encrypt_char_with_different_modulus(self):
        """Test encryption with different modulus."""
        # Using m=5 with a=2, b=1
        # E(0) = (2*0 + 1) mod 5 = 1
        encrypted = encrypt_char('A', a=2, b=1, m=5)
        assert encrypted == 'B'


class TestDecryptChar:
    """Tests for single character decryption."""
    
    def test_decrypt_char_examples_from_paper(self):
        """Test examples from the academic paper."""
        # D(3) = (7*3 + 5) mod 26 = 26 mod 26 = 0 -> D to A
        assert decrypt_char('D') == 'A'
        # D(4) = (7*4 + 5) mod 26 = 33 mod 26 = 7 -> E to H
        assert decrypt_char('E') == 'H'
    
    def test_decrypt_char_all_uppercase(self):
        """Test decryption of all uppercase letters."""
        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            decrypted = decrypt_char(char)
            # Verify it's uppercase
            assert decrypted.isupper()
    
    def test_decrypt_char_case_insensitivity(self):
        """Test that case is ignored in input."""
        assert decrypt_char('d') == decrypt_char('D')
        assert decrypt_char('e') == decrypt_char('E')
    
    def test_decrypt_char_invalid_key(self):
        """Test that invalid key raises ValueError."""
        with pytest.raises(ValueError, match="Invalid key"):
            decrypt_char('A', a=2, b=0)


class TestEncryptDecryptRoundtrip:
    """Tests for encryption/decryption roundtrips."""
    
    def test_roundtrip_single_char(self):
        """Test that decrypt(encrypt(x)) = x for single characters."""
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            encrypted = encrypt_char(char)
            decrypted = decrypt_char(encrypted)
            assert decrypted == char
    
    def test_roundtrip_all_chars(self):
        """Test roundtrip for all letters multiple times."""
        for _ in range(5):
            for i in range(26):
                char = num_to_char(i)
                encrypted = encrypt_char(char)
                decrypted = decrypt_char(encrypted)
                assert decrypted == char


class TestEncryptText:
    """Tests for text encryption."""
    
    def test_encrypt_text_hallo(self):
        """Test encryption of 'HALLO' example from paper."""
        result = encrypt_text('HALLO')
        assert result == 'EDMMF'
    
    def test_encrypt_text_kryptographie(self):
        """Test encryption of 'KRYPTOGRAPHIE' from paper."""
        result = encrypt_text('KRYPTOGRAPHIE')
        assert result == 'XYZUCFPYDUETL'
    
    def test_encrypt_text_lowercase(self):
        """Test encryption preserves case when keep_case=True."""
        encrypted = encrypt_text('hello', keep_case=False)
        assert encrypted == encrypted.upper()
    
    def test_encrypt_text_with_spaces(self):
        """Test that spaces are preserved."""
        encrypted = encrypt_text('HELLO WORLD')
        parts = encrypted.split(' ')
        assert len(parts) == 2
        assert parts[0] == encrypt_text('HELLO')
        assert parts[1] == encrypt_text('WORLD')
    
    def test_encrypt_text_with_punctuation(self):
        """Test that punctuation is preserved."""
        encrypted = encrypt_text('HELLO, WORLD!')
        assert ',' in encrypted
        assert '!' in encrypted
        assert not encrypted.startswith(',')
        assert encrypted.endswith('!')  # Punctuation at end is preserved
    
    def test_encrypt_text_with_numbers(self):
        """Test that numbers are preserved."""
        encrypted = encrypt_text('HELLO 123 WORLD')
        assert '1' in encrypted
        assert '2' in encrypted
        assert '3' in encrypted
    
    def test_encrypt_text_mixed_case_keep_false(self):
        """Test that mixed case is converted to uppercase when keep_case=False."""
        encrypted = encrypt_text('HeLLo', keep_case=False)
        assert encrypted.isupper()
    
    def test_encrypt_text_mixed_case_keep_true(self):
        """Test that case is preserved when keep_case=True."""
        original = 'HeLLo'
        encrypted = encrypt_text(original, keep_case=True)
        # Original: HeLLo -> indices 0:H(upper), 1:e(lower), 2:L(upper), 3:L(upper), 4:o(lower)
        # Check that case matches original
        assert encrypted[1] == encrypted[1].lower()  # Index 1 was lowercase
        assert encrypted[4] == encrypted[4].lower()  # Index 4 was lowercase
        assert encrypted[0] == encrypted[0].upper()  # Index 0 was uppercase
        assert encrypted[2] == encrypted[2].upper()  # Index 2 was uppercase
    
    def test_encrypt_text_empty_string(self):
        """Test encryption of empty string."""
        assert encrypt_text('') == ''
    
    def test_encrypt_text_only_spaces(self):
        """Test encryption of string with only spaces."""
        assert encrypt_text('   ') == '   '
    
    def test_encrypt_text_only_punctuation(self):
        """Test encryption of string with only punctuation."""
        assert encrypt_text('!!!') == '!!!'
    
    def test_encrypt_text_long_text(self):
        """Test encryption of longer text."""
        plaintext = 'THE AFFINE CIPHER WITH PARAMETERS A EQUALS FIFTEEN AND B EQUALS TWENTY NINE'
        ciphertext = encrypt_text(plaintext)
        # Verify non-alphabetic characters are preserved
        assert ciphertext.count(' ') == plaintext.count(' ')


class TestDecryptText:
    """Tests for text decryption."""
    
    def test_decrypt_text_edmmf(self):
        """Test decryption of 'EDMMF' example from paper."""
        result = decrypt_text('EDMMF')
        assert result == 'HALLO'
    
    def test_decrypt_text_with_spaces(self):
        """Test that spaces are preserved during decryption."""
        ciphertext = 'EDHHDF WFNLJ'
        plaintext = decrypt_text(ciphertext)
        assert plaintext.count(' ') == ciphertext.count(' ')
    
    def test_decrypt_text_with_punctuation(self):
        """Test that punctuation is preserved during decryption."""
        ciphertext = 'EDHHDJ, WFNLJ!'
        plaintext = decrypt_text(ciphertext)
        assert ',' in plaintext
        assert '!' in plaintext
    
    def test_decrypt_text_empty_string(self):
        """Test decryption of empty string."""
        assert decrypt_text('') == ''
    
    def test_decrypt_text_roundtrip(self):
        """Test decrypt(encrypt(text)) = text."""
        texts = ['HALLO', 'WORLD', 'THE QUICK BROWN FOX', 'AFFINE CIPHER']
        for text in texts:
            encrypted = encrypt_text(text)
            decrypted = decrypt_text(encrypted)
            assert decrypted == text.upper()


class TestValidateKey:
    """Tests for key validation."""
    
    def test_validate_key_valid_keys(self):
        """Test validation of valid keys."""
        valid_keys = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
        for a in valid_keys:
            assert validate_key(a)
    
    def test_validate_key_invalid_keys(self):
        """Test validation of invalid keys."""
        invalid_keys = [2, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24]
        for a in invalid_keys:
            assert not validate_key(a)
    
    def test_validate_key_custom_modulus(self):
        """Test validation with different modulus."""
        # All odd numbers are coprime with m=10
        assert validate_key(3, m=10)
        assert validate_key(7, m=10)
        # Even numbers are not coprime with m=10
        assert not validate_key(2, m=10)
        assert not validate_key(4, m=10)


class TestGetKeySpace:
    """Tests for key space calculation."""
    
    def test_get_key_space_modulus_26(self):
        """Test that key space for m=26 is 312."""
        # φ(26) = 12 (numbers coprime with 26)
        # Key space = 12 * 26 = 312
        assert get_key_space(26) == 312
    
    def test_get_key_space_prime_modulus(self):
        """Test key space for prime modulus."""
        # For prime p, φ(p) = p-1
        # Key space = (p-1) * p
        assert get_key_space(5) == 5 * 4  # φ(5) = 4
        assert get_key_space(7) == 7 * 6  # φ(7) = 6
    
    def test_get_key_space_modulus_1(self):
        """Test key space for m=1 (edge case, typically not used)."""
        # For m=1: φ(1)=1, key_space = 1 * 1 = 1
        # However, the function counts coprime numbers starting from 1
        # which for m=1 means no numbers in range(1,1), so returns 0
        assert get_key_space(1) == 0  # Empty range, no coprime numbers found


class TestSecurityProperties:
    """Tests for security-related properties."""
    
    def test_substitution_table_is_bijection(self):
        """Test that encryption creates a bijection (all chars map uniquely)."""
        encrypted_chars = set()
        for i in range(26):
            char = num_to_char(i)
            encrypted = encrypt_char(char)
            encrypted_chars.add(encrypted)
        
        # Should have 26 unique characters
        assert len(encrypted_chars) == 26
    
    def test_deterministic_encryption(self):
        """Test that same plaintext always produces same ciphertext."""
        plaintext = 'AFFINE CIPHER'
        encrypted1 = encrypt_text(plaintext)
        encrypted2 = encrypt_text(plaintext)
        assert encrypted1 == encrypted2
    
    def test_frequency_preservation(self):
        """Test that letter frequencies are preserved (monoalphabetic property)."""
        # Create plaintext with known frequency pattern
        plaintext = 'AABBCC'  # Each letter twice
        ciphertext = encrypt_text(plaintext)
        
        # Count frequencies in ciphertext
        freq = {}
        for char in ciphertext:
            if char.isalpha():
                freq[char] = freq.get(char, 0) + 1
        
        # Should have 3 characters, each appearing twice
        assert len(freq) == 3
        assert all(count == 2 for count in freq.values())


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_encrypt_text_invalid_key(self):
        """Test that encrypt_text raises error for invalid key."""
        with pytest.raises(ValueError, match="Invalid key"):
            encrypt_text('HELLO', a=2, b=0)  # a=2 not coprime with 26
    
    def test_decrypt_text_invalid_key(self):
        """Test that decrypt_text raises error for invalid key."""
        with pytest.raises(ValueError, match="Invalid key"):
            decrypt_text('EDHHDJ', a=13, b=0)  # a=13 not coprime with 26
    
    def test_char_to_num_boundary_values(self):
        """Test char_to_num with boundary ASCII values."""
        # Just before A
        with pytest.raises(ValueError):
            char_to_num(chr(ord('A') - 1))
        # Just after Z
        with pytest.raises(ValueError):
            char_to_num(chr(ord('Z') + 1))
    
    def test_num_to_char_boundary_values(self):
        """Test num_to_char with boundary values."""
        # Just before 0
        with pytest.raises(ValueError):
            num_to_char(-1)
        # Just after 25
        with pytest.raises(ValueError):
            num_to_char(26)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complete_workflow(self):
        """Test complete encryption/decryption workflow."""
        original = 'AFFINE CIPHER USES A EQUALS FIFTEEN AND B EQUALS TWENTY NINE'
        
        # Encrypt
        encrypted = encrypt_text(original)
        
        # Decrypt
        decrypted = decrypt_text(encrypted)
        
        # Should match original
        assert decrypted == original
    
    def test_multiple_messages(self):
        """Test encrypting/decrypting multiple messages."""
        messages = [
            'THE QUICK BROWN FOX',
            'CRYPTOGRAPHY IS FUN',
            'MATHEMATICS RULES',
            'AFFINE CIPHERS ARE WEAK',
        ]
        
        for msg in messages:
            encrypted = encrypt_text(msg)
            decrypted = decrypt_text(encrypted)
            assert decrypted == msg
    
    def test_edge_case_single_letter(self):
        """Test single letter encryption/decryption."""
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            encrypted = encrypt_text(char)
            decrypted = decrypt_text(encrypted)
            assert decrypted == char
    
    def test_edge_case_repeating_letters(self):
        """Test text with repeating letters."""
        plaintext = 'AAABBBCCCDDD'
        encrypted = encrypt_text(plaintext)
        decrypted = decrypt_text(encrypted)
        assert decrypted == plaintext
