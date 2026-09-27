"""Key generation and management"""

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
import hashlib
from datetime import datetime


class KeyManager:
    """Manages PGP key pairs"""
    
    def __init__(self):
        self.backend = default_backend()
    
    def generate_key_pair(self, key_size=2048, user_id=""):
        """Generate RSA key pair
        
        Args:
            key_size: Key size in bits (2048, 3072, 4096)
            user_id: User identifier (email, name)
            
        Returns:
            dict with 'private_key', 'public_key', 'fingerprint', 'user_id', 'created'
        """
        if key_size not in [2048, 3072, 4096]:
            raise ValueError("Key size must be 2048, 3072, or 4096")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        public_key = private_key.public_key()
        
        fingerprint = self.get_fingerprint(public_key)
        
        return {
            'private_key': private_key,
            'public_key': public_key,
            'fingerprint': fingerprint,
            'user_id': user_id,
            'created': datetime.now().isoformat(),
            'key_size': key_size
        }
    
    def export_public_key(self, public_key):
        """Export public key to PEM format"""
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def export_private_key(self, private_key, passphrase=None):
        """Export private key to PEM format
        
        Args:
            private_key: Private key object
            passphrase: Optional passphrase for encryption (bytes or str)
        """
        if passphrase:
            if isinstance(passphrase, str):
                passphrase = passphrase.encode('utf-8')
            encryption = serialization.BestAvailableEncryption(passphrase)
        else:
            encryption = serialization.NoEncryption()
        
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )
        return pem.decode('utf-8')
    
    def import_public_key(self, pem_data):
        """Import public key from PEM format"""
        if isinstance(pem_data, str):
            pem_data = pem_data.encode('utf-8')
        
        public_key = serialization.load_pem_public_key(
            pem_data,
            backend=self.backend
        )
        return public_key
    
    def import_private_key(self, pem_data, passphrase=None):
        """Import private key from PEM format"""
        if isinstance(pem_data, str):
            pem_data = pem_data.encode('utf-8')
        if passphrase and isinstance(passphrase, str):
            passphrase = passphrase.encode('utf-8')
        
        private_key = serialization.load_pem_private_key(
            pem_data,
            password=passphrase,
            backend=self.backend
        )
        return private_key
    
    def get_fingerprint(self, public_key):
        """Generate key fingerprint (SHA-256 hash of public key)"""
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        fingerprint = hashlib.sha256(pem).hexdigest().upper()
        # Format as groups of 4 characters
        return ' '.join([fingerprint[i:i+4] for i in range(0, len(fingerprint), 4)])

