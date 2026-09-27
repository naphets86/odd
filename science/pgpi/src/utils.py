"""Utility functions"""

import os

def secure_delete(data):
    """Securely overwrite data in memory (best effort)"""
    if isinstance(data, bytearray):
        for i in range(len(data)):
            data[i] = 0
    # Note: Python doesn't provide guaranteed memory overwriting


def generate_random_bytes(length):
    """Generate cryptographically secure random bytes"""
    return os.urandom(length)