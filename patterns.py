#!/usr/bin/python
import sys
import math
import pickle
from collections import deque

def main():
    filenames = sys.argv[1:]

    patterns = []

    for filename in filenames:
        values = []
        for line in open(filename, "r").readlines():
            values.append(int(line.strip()))

        queue = deque()
        qlen = 100
        for v in values:
            if len(queue) >= qlen:
                queue.popleft()
            queue.append(v)

            if len(queue) < qlen:
                continue

            minvalue = min(queue)
            maxvalue = max(queue)
            pattern = [(qlen * (value - minvalue)) / (maxvalue - minvalue) for value in queue]
            patterns.append(pattern)

    print pickle.dumps(patterns)

if __name__ == "__main__":
    main()
