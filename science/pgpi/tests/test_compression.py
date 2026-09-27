"""Tests for compression"""

import pytest
from src.compression import compress_data, decompress_data


class TestCompression:
    
    def test_compress_decompress(self):
        """Test basic compression and decompression"""
        data = b"This is some test data that should compress well. " * 10
        
        compressed = compress_data(data)
        decompressed = decompress_data(compressed)
        
        assert decompressed == data
        assert len(compressed) < len(data)
    
    def test_compress_random_data(self):
        """Test compression of random data (poor compression ratio)"""
        import os
        data = os.urandom(100)
        
        compressed = compress_data(data)
        decompressed = decompress_data(compressed)
        
        assert decompressed == data
    
    def test_compression_levels(self):
        """Test different compression levels"""
        data = b"Compressible data " * 100
        
        compressed_low = compress_data(data, level=1)
        compressed_high = compress_data(data, level=9)
        
        # Higher compression should result in smaller size
        assert len(compressed_high) <= len(compressed_low)
        
        # Both should decompress correctly
        assert decompress_data(compressed_low) == data
        assert decompress_data(compressed_high) == data
