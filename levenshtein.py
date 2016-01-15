#!/usr/bin/python
import sys
import pickle
import time

def main():
    first = sys.argv[1]
    second = sys.argv[2]
    limit = None
    if len(sys.argv) >= 4:
        limit = sys.argv[3]

    print metric(first, second, limit)

def metric(first, second, limit = None):
    limit = limit or max(len(first), len(second))

    if len(first) > len(second):
        first, second = second, first

    if len(second) - len(first) > limit:
        return limit + 1

    previous_row = [i for i in xrange(0, len(first) + 1)]
    current_row = [0 for i in xrange(0, len(first) + 1)]

    for i in xrange(1, len(second) + 1):
        current_row[0] = i

        start = max(i - limit - 1, 1)
        end = min(i + limit + 1, len(first))
        for j in xrange(start, end + 1):
            cost = 1 if first[j - 1] != second[i - 1] else 0
            current_row[j] = min(current_row[j - 1] + 1, previous_row[j] + 1, previous_row[j - 1] + cost)

        previous_row, current_row = current_row, previous_row

    return previous_row[len(first)]


if __name__ == "__main__":
    main()
