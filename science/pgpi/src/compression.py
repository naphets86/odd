"""Data compression functions"""

import zlib


def compress_data(data, level=6):
    """Compress data using ZLIB
    
    Args:
        data: Bytes to compress
        level: Compression level (1-9, default 6)
        
    Returns:
        bytes: Compressed data
    """
    return zlib.compress(data, level=level)


def decompress_data(data):
    """Decompress ZLIB data
    
    Args:
        data: Compressed bytes
        
    Returns:
        bytes: Decompressed data
    """
    return zlib.decompress(data)
