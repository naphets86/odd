"""Encryption and decryption functions"""

from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import secrets
import struct
from src.compression import compress_data, decompress_data


def encrypt_message(message, public_key, compress=True):
    """Encrypt a message using hybrid encryption (RSA + AES-256-GCM)
    
    Args:
        message: String or bytes to encrypt
        public_key: Recipient's public key
        compress: Whether to compress before encryption
        
    Returns:
        bytes: Encrypted message with format: [session_key_len][encrypted_session_key][nonce][tag][ciphertext]
    """
    if isinstance(message, str):
        message = message.encode('utf-8')
    
    # Compress if requested
    if compress:
        message = compress_data(message)
        compression_flag = b'\x01'
    else:
        compression_flag = b'\x00'
    
    # Generate random session key (256 bits for AES-256)
    session_key = secrets.token_bytes(32)
    
    # Generate nonce for GCM
    nonce = secrets.token_bytes(12)
    
    # Encrypt the message with AES-256-GCM
    cipher = Cipher(
        algorithms.AES(session_key),
        modes.GCM(nonce),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    encryptor.authenticate_additional_data(compression_flag)
    ciphertext = encryptor.update(message) + encryptor.finalize()
    tag = encryptor.tag
    
    # Encrypt the session key with RSA
    encrypted_session_key = public_key.encrypt(
        session_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Pack everything together
    session_key_len = len(encrypted_session_key)
    result = struct.pack('!I', session_key_len)
    result += encrypted_session_key
    result += compression_flag
    result += nonce
    result += tag
    result += ciphertext
    
    return result


def decrypt_message(ciphertext, private_key):
    """Decrypt a message
    
    Args:
        ciphertext: Encrypted bytes from encrypt_message
        private_key: Recipient's private key
        
    Returns:
        str: Decrypted message
    """
    # Unpack the structure
    offset = 0
    session_key_len = struct.unpack('!I', ciphertext[offset:offset+4])[0]
    offset += 4
    
    encrypted_session_key = ciphertext[offset:offset+session_key_len]
    offset += session_key_len
    
    compression_flag = ciphertext[offset:offset+1]
    offset += 1
    
    nonce = ciphertext[offset:offset+12]
    offset += 12
    
    tag = ciphertext[offset:offset+16]
    offset += 16
    
    encrypted_data = ciphertext[offset:]
    
    # Decrypt the session key with RSA
    session_key = private_key.decrypt(
        encrypted_session_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Decrypt the message with AES-256-GCM
    cipher = Cipher(
        algorithms.AES(session_key),
        modes.GCM(nonce, tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    decryptor.authenticate_additional_data(compression_flag)
    plaintext = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # Decompress if needed
    if compression_flag == b'\x01':
        plaintext = decompress_data(plaintext)
    
    return plaintext.decode('utf-8', errors='replace')

