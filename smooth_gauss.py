#!/usr/bin/python
import sys
import math
from collections import deque

def main():
    filename = sys.argv[1]

    values = []
    for line in open(filename, "r").readlines():
        values.append(int(line.strip()))
    result = gaussian_smooth(values)
    for value in result:
        print value

def gaussian_smooth(values):
    result = []
    queue = deque([])
    maxlength = 20
    sigma = 3
    for value in values:
        if len(queue) >= maxlength:
            queue.popleft()
        queue.append(value)
        cindex = (len(queue) - 1) / 2.0
        result_value = 0.0
        if len(queue) == maxlength:
            for index, current in enumerate(queue):
                x = abs(index - cindex) / cindex
                m = math.exp(-x**2 / (2 * sigma**2)) # / math.sqrt(2 * math.pi * sigma ** 2)
                result_value = result_value + current * m
            result.append(int(result_value / len(queue)))
    return result

if __name__ == "__main__":
    main()
