"""Tests for key management"""

import pytest
from src.key_management import KeyManager


class TestKeyManagement:
    
    def setup_method(self):
        self.km = KeyManager()
    
    def test_generate_key_pair_2048(self):
        """Test generating 2048-bit key pair"""
        key_pair = self.km.generate_key_pair(key_size=2048, user_id="test@example.com")
        
        assert 'private_key' in key_pair
        assert 'public_key' in key_pair
        assert 'fingerprint' in key_pair
        assert key_pair['user_id'] == "test@example.com"
        assert key_pair['key_size'] == 2048
    
    def test_generate_key_pair_4096(self):
        """Test generating 4096-bit key pair"""
        key_pair = self.km.generate_key_pair(key_size=4096)
        assert key_pair['key_size'] == 4096
    
    def test_invalid_key_size(self):
        """Test that invalid key sizes raise ValueError"""
        with pytest.raises(ValueError):
            self.km.generate_key_pair(key_size=1024)
    
    def test_export_import_public_key(self):
        """Test exporting and importing public key"""
        key_pair = self.km.generate_key_pair(key_size=2048)
        
        # Export
        pem = self.km.export_public_key(key_pair['public_key'])
        assert "-----BEGIN PUBLIC KEY-----" in pem
        assert "-----END PUBLIC KEY-----" in pem
        
        # Import
        imported_key = self.km.import_public_key(pem)
        assert imported_key is not None
    
    def test_export_import_private_key_no_passphrase(self):
        """Test exporting and importing private key without passphrase"""
        key_pair = self.km.generate_key_pair(key_size=2048)
        
        # Export
        pem = self.km.export_private_key(key_pair['private_key'])
        assert "-----BEGIN PRIVATE KEY-----" in pem or "-----BEGIN ENCRYPTED PRIVATE KEY-----" in pem
        
        # Import
        imported_key = self.km.import_private_key(pem)
        assert imported_key is not None
    
    def test_export_import_private_key_with_passphrase(self):
        """Test exporting and importing private key with passphrase"""
        key_pair = self.km.generate_key_pair(key_size=2048)
        passphrase = "my_secure_passphrase"
        
        # Export with passphrase
        pem = self.km.export_private_key(key_pair['private_key'], passphrase=passphrase)
        assert "-----BEGIN ENCRYPTED PRIVATE KEY-----" in pem
        
        # Import with passphrase
        imported_key = self.km.import_private_key(pem, passphrase=passphrase)
        assert imported_key is not None
        
        # Import with wrong passphrase should fail
        with pytest.raises(Exception):
            self.km.import_private_key(pem, passphrase="wrong_passphrase")
    
    def test_fingerprint_generation(self):
        """Test fingerprint generation"""
        key_pair = self.km.generate_key_pair(key_size=2048)
        fingerprint = key_pair['fingerprint']
        
        # Fingerprint should be hex string with spaces
        assert len(fingerprint) > 0
        assert ' ' in fingerprint
        
        # Should be consistent
        fingerprint2 = self.km.get_fingerprint(key_pair['public_key'])
        assert fingerprint == fingerprint2
