"""ASCII armor encoding/decoding"""

import base64
import struct


def armor_message(data, block_type="MESSAGE"):
    """Encode binary data in ASCII armor format
    
    Args:
        data: Binary data to encode
        block_type: Type of block (MESSAGE, SIGNATURE, PUBLIC KEY, PRIVATE KEY)
        
    Returns:
        str: ASCII armored text
    """
    # Calculate CRC24 checksum
    crc = calculate_crc24(data)
    
    # Base64 encode the data
    encoded = base64.b64encode(data).decode('ascii')
    
    # Split into lines of 64 characters
    lines = [encoded[i:i+64] for i in range(0, len(encoded), 64)]
    
    # Format checksum
    checksum = base64.b64encode(struct.pack('>I', crc)[1:]).decode('ascii')
    
    # Build armored message
    result = f"-----BEGIN PGP {block_type}-----\n"
    result += "Version: PGPI 1.0\n\n"
    result += '\n'.join(lines)
    result += f"\n={checksum}\n"
    result += f"-----END PGP {block_type}-----\n"
    
    return result


def dearmor_message(armored):
    """Decode ASCII armored message
    
    Args:
        armored: ASCII armored string
        
    Returns:
        bytes: Decoded binary data
    """
    lines = armored.strip().split('\n')
    
    # Find data section (skip headers and footers)
    data_lines = []
    in_data = False
    checksum_line = None
    
    for line in lines:
        if line.startswith('-----BEGIN'):
            in_data = False
            continue
        elif line.startswith('-----END'):
            break
        elif line.startswith('='):
            checksum_line = line[1:]
            break
        elif line and not line.startswith('Version:') and not line.startswith('Comment:'):
            in_data = True
        
        if in_data and line:
            data_lines.append(line)
    
    # Decode base64
    encoded = ''.join(data_lines)
    try:
        data = base64.b64decode(encoded)
    except Exception:
        raise ValueError("Checksum verification failed")
    
    # Verify checksum if present
    if checksum_line:
        try:
            expected_crc = struct.unpack('>I', b'\x00' + base64.b64decode(checksum_line))[0]
            actual_crc = calculate_crc24(data)
            if expected_crc != actual_crc:
                raise ValueError(f"Checksum verification failed: expected {expected_crc:06X}, got {actual_crc:06X}")
        except Exception as e:
            if "Checksum verification failed" in str(e):
                raise
            raise ValueError(f"Failed to verify checksum: {e}")
    
    return data


def calculate_crc24(data):
    """Calculate CRC24 checksum (OpenPGP standard)"""
    crc = 0xB704CE
    
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    
    return crc & 0xFFFFFF