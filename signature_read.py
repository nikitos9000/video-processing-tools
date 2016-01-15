#!/usr/bin/python
import struct

def read_signature_short(filename):
    signature_file = open(filename, "r")
    signature_string = signature_file.read()
    signature_value = struct.unpack(str(len(signature_string) / 2) + "h", signature_string)
    signature_file.close()
    return signature_value

def read_signature_byte(filename):
    signature_file = open(filename, "r")
    signature_string = signature_file.read()
    signature_value = struct.unpack(str(len(signature_string)) + "B", signature_string)
    signature_file.close()
    return signature_value

def read_signature_64(filename):
    signature_file = open(filename, "rb")
    signature_string = signature_file.read()
    signature_value = struct.unpack(str(len(signature_string) / 8) + "Q", signature_string)
    signature_file.close()
    return signature_value
