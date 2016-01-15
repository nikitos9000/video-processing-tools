#!/usr/bin/python
import sys
import pickle
import correlate
from collections import deque

def main():
    pattern_length = 100
    pattern_step = 50

    indexname = sys.argv[1]
    filename = sys.argv[2]

    patterns  = pickle.loads(open(indexname, "r").read())

    values = []
    for line in open(filename, "r").readlines():
        value = int(line.strip())
        values.append(value)

    result = []
    queue = deque()
    for i, v in enumerate(values):
        if len(queue) >= pattern_length:
            queue.popleft()
        queue.append(v)

        if len(queue) >= pattern_length and ((i + 1) % pattern_step) == 0:
            rank = []
            for pindex, pattern in enumerate(patterns):
                rv = correlate.pearson(pattern, queue)
                rank.append((rv, pindex))
            rank.sort(key = lambda x: -x[0])
            result.append(str(rank[0][1]))

    print "_".join(result)

if __name__ == "__main__":
    main()
