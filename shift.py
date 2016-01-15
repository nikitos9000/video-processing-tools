#!/usr/bin/python
import sys
import math
import random
from collections import deque

def main():
    filename = sys.argv[1]
    count = int(sys.argv[2])

    values = []
    for line in open(filename, "r").readlines():
        values.append(int(line.strip()))

    minvalue = min(values)
    maxvalue = max(values)

    random.seed()

    lastvalue = random.randrange(minvalue, maxvalue + 1)
    threshold = 500
    for i in xrange(0, count):
        value = random.randrange(max(minvalue, lastvalue - threshold), min(maxvalue + 1, lastvalue + threshold))
        lastvalue = value
        print value

    for value in values[max(-count, 0):]:
        print value

if __name__ == "__main__":
    main()
