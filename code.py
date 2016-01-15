#!/usr/bin/python
import sys

def main():
    filename = sys.argv[1]

    values = []
    for line in open(filename, "r").readlines():
        values.append(int(line.strip()))

    string = []
    for value in values:
        value += 2**15
        string.append((value >> 12) % 16)
        string.append((value >> 8) % 16)
        string.append((value >> 4) % 16)
        string.append(value % 16)

    result = []
    lastvalue = -1
    for value in string:
        if value != lastvalue:
            result.append(value)
            lastvalue = value

    print "".join(to_char(value) for value in result)

def to_char(value):
    return chr(ord('a') + value)

if __name__ == "__main__":
    main()
