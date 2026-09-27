"""Tests for encryption and decryption"""

import pytest
from src.key_management import KeyManager
from src.encrypt import encrypt_message, decrypt_message


class TestEncryption:
    
    def setup_method(self):
        self.km = KeyManager()
        self.key_pair = self.km.generate_key_pair(key_size=2048)
    
    def test_encrypt_decrypt_simple_message(self):
        """Test encrypting and decrypting a simple message"""
        plaintext = "Hello, World!"
        
        ciphertext = encrypt_message(plaintext, self.key_pair['public_key'])
        decrypted = decrypt_message(ciphertext, self.key_pair['private_key'])
        
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_long_message(self):
        """Test encrypting and decrypting a long message"""
        plaintext = "This is a much longer message. " * 100
        
        ciphertext = encrypt_message(plaintext, self.key_pair['public_key'])
        decrypted = decrypt_message(ciphertext, self.key_pair['private_key'])
        
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_unicode(self):
        """Test encrypting and decrypting Unicode text"""
        plaintext = "Hello 世界! Привет мир! مرحبا العالم!"
        
        ciphertext = encrypt_message(plaintext, self.key_pair['public_key'])
        decrypted = decrypt_message(ciphertext, self.key_pair['private_key'])
        
        assert decrypted == plaintext
    
    def test_encrypt_with_compression(self):
        """Test encryption with compression"""
        plaintext = "A" * 1000  # Highly compressible
        
        ciphertext_compressed = encrypt_message(plaintext, self.key_pair['public_key'], compress=True)
        ciphertext_uncompressed = encrypt_message(plaintext, self.key_pair['public_key'], compress=False)
        
        # Compressed should be smaller
        assert len(ciphertext_compressed) < len(ciphertext_uncompressed)
        
        # Both should decrypt correctly
        assert decrypt_message(ciphertext_compressed, self.key_pair['private_key']) == plaintext
        assert decrypt_message(ciphertext_uncompressed, self.key_pair['private_key']) == plaintext
    
    def test_decrypt_with_wrong_key(self):
        """Test that decryption with wrong key fails"""
        plaintext = "Secret message"
        wrong_key_pair = self.km.generate_key_pair(key_size=2048)
        
        ciphertext = encrypt_message(plaintext, self.key_pair['public_key'])
        
        with pytest.raises(Exception):
            decrypt_message(ciphertext, wrong_key_pair['private_key'])
    
    def test_encrypt_bytes(self):
        """Test encrypting raw bytes"""
        plaintext = b"Binary data: \x00\x01\x02\x03"
        plaintext_str = plaintext.decode('utf-8', errors='replace')
        
        ciphertext = encrypt_message(plaintext_str, self.key_pair['public_key'])
        decrypted = decrypt_message(ciphertext, self.key_pair['private_key'])
        
        assert decrypted == plaintext_str
