"""
Payment Integration Service
AES-CBC encryption and decryption routines for CCAvenue payment gateway.
"""
import hashlib
from Crypto.Cipher import AES

def encrypt(plain_text, working_key):
    """
    Encrypts plaintext using AES-CBC with MD5-derived 16-byte key and static IV.
    Returns hex-encoded ciphertext.
    """
    iv = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    key = working_key.encode('utf-8')
    byte_key = hashlib.md5(key).digest()
    cipher = AES.new(byte_key, AES.MODE_CBC, iv)
    pad_len = 16 - (len(plain_text) % 16)
    padded_text = plain_text + chr(pad_len) * pad_len
    encrypted_text = cipher.encrypt(padded_text.encode('utf-8'))
    return encrypted_text.hex()

def decrypt(cipher_text, working_key):
    """
    Decrypts hex-encoded ciphertext using AES-CBC with MD5-derived 16-byte key and static IV.
    Returns unpadded plaintext string.
    """
    iv = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    key = working_key.encode('utf-8')
    byte_key = hashlib.md5(key).digest()
    encrypted_text = bytes.fromhex(cipher_text)
    cipher = AES.new(byte_key, AES.MODE_CBC, iv)
    decrypted_text = cipher.decrypt(encrypted_text).decode('utf-8')
    pad_len = ord(decrypted_text[-1])
    return decrypted_text[:-pad_len]
