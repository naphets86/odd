"""Tests for ASCII armor"""

import pytest
from src.armor import armor_message, dearmor_message, calculate_crc24


class TestArmor:
    
    def test_armor_dearmor_message(self):
        """Test armoring and dearmoring a message"""
        data = b"This is test data to be armored"
        
        armored = armor_message(data, block_type="MESSAGE")
        
        assert "-----BEGIN PGP MESSAGE-----" in armored
        assert "-----END PGP MESSAGE-----" in armored
        assert "Version: PGPI 1.0" in armored
        
        dearmored = dearmor_message(armored)
        assert dearmored == data
    
    def test_armor_different_types(self):
        """Test armoring different block types"""
        data = b"Test data"
        
        for block_type in ["MESSAGE", "SIGNATURE", "PUBLIC KEY", "PRIVATE KEY"]:
            armored = armor_message(data, block_type=block_type)
            assert f"-----BEGIN PGP {block_type}-----" in armored
            assert f"-----END PGP {block_type}-----" in armored
    
    def test_armor_long_data(self):
        """Test armoring long data (line wrapping)"""
        data = b"X" * 1000
        
        armored = armor_message(data)
        lines = armored.split('\n')
        
        # Check that lines are wrapped (64 chars max for data lines)
        for line in lines:
            if line and not line.startswith('-----') and not line.startswith('Version:') and not line.startswith('='):
                assert len(line) <= 64
    
    def test_crc24_checksum(self):
        """Test CRC24 checksum calculation"""
        data1 = b"Test data"
        data2 = b"Test data"
        data3 = b"Different data"
        
        crc1 = calculate_crc24(data1)
        crc2 = calculate_crc24(data2)
        crc3 = calculate_crc24(data3)
        
        # Same data should have same checksum
        assert crc1 == crc2
        
        # Different data should have different checksum (usually)
        assert crc1 != crc3
    
    def test_corrupted_armor(self):
        """Test that corrupted armored data is detected"""
        data = b"Original data"
        armored = armor_message(data)
        
        # Corrupt the base64 data (not just any 'A')
        lines = armored.split('\n')
        data_line_idx = None
        for i, line in enumerate(lines):
            if line and not line.startswith('-----') and not line.startswith('Version:') and not line.startswith('='):
                data_line_idx = i
                break
        
        if data_line_idx:
            # Replace a character in the data to corrupt it
            corrupted_line = lines[data_line_idx][:-1] + ('B' if lines[data_line_idx][-1] != 'B' else 'C')
            lines[data_line_idx] = corrupted_line
            corrupted = '\n'.join(lines)
            
            with pytest.raises(ValueError, match="Checksum verification failed"):
                dearmor_message(corrupted)
    
    def test_armor_binary_data(self):
        """Test armoring binary data"""
        import os
        data = os.urandom(256)
        
        armored = armor_message(data)
        dearmored = dearmor_message(armored)
        
        assert dearmored == data