
"""Digital signature functions"""

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature


def sign_message(message, private_key):
    """Sign a message with RSA-PSS
    
    Args:
        message: String or bytes to sign
        private_key: Signer's private key
        
    Returns:
        bytes: Digital signature
    """
    if isinstance(message, str):
        message = message.encode('utf-8')
    
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    return signature


def verify_signature(message, signature, public_key):
    """Verify a digital signature
    
    Args:
        message: Original message (string or bytes)
        signature: Signature bytes
        public_key: Signer's public key
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    if isinstance(message, str):
        message = message.encode('utf-8')
    
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False
