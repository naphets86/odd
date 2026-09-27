"""Tests for digital signatures"""

import pytest
from src.key_management import KeyManager
from src.signing import sign_message, verify_signature


class TestSigning:
    
    def setup_method(self):
        self.km = KeyManager()
        self.key_pair = self.km.generate_key_pair(key_size=2048)
    
    def test_sign_and_verify(self):
        """Test signing and verifying a message"""
        message = "This is a signed message"
        
        signature = sign_message(message, self.key_pair['private_key'])
        is_valid = verify_signature(message, signature, self.key_pair['public_key'])
        
        assert is_valid is True
    
    def test_verify_modified_message(self):
        """Test that modified message fails verification"""
        message = "Original message"
        signature = sign_message(message, self.key_pair['private_key'])
        
        modified_message = "Modified message"
        is_valid = verify_signature(modified_message, signature, self.key_pair['public_key'])
        
        assert is_valid is False
    
    def test_verify_with_wrong_key(self):
        """Test that verification with wrong key fails"""
        message = "Signed message"
        wrong_key_pair = self.km.generate_key_pair(key_size=2048)
        
        signature = sign_message(message, self.key_pair['private_key'])
        is_valid = verify_signature(message, signature, wrong_key_pair['public_key'])
        
        assert is_valid is False
    
    def test_sign_long_message(self):
        """Test signing a long message"""
        message = "Long message " * 1000
        
        signature = sign_message(message, self.key_pair['private_key'])
        is_valid = verify_signature(message, signature, self.key_pair['public_key'])
        
        assert is_valid is True
    
    def test_sign_unicode_message(self):
        """Test signing Unicode message"""
        message = "Unicode message: 你好世界 🌍"
        
        signature = sign_message(message, self.key_pair['private_key'])
        is_valid = verify_signature(message, signature, self.key_pair['public_key'])
        
        assert is_valid is True
