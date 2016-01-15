#!/usr/bin/python
import sys
from collections import deque

def main():
    filename = sys.argv[1]

    values = []
    for line in open(filename, "r").readlines():
        values.append(int(line.strip()))
    result = avg_smooth(values)
    for value in result:
        print value

def avg_smooth(values):
    step = 0
    maxstep = 20

    result = []
    average = 0
    queue = deque()
    for current in values:
        if step < maxstep:
            step += 1

        if len(queue) >= step:
            average = average - queue.popleft()
        average = average + current
        queue.append(current)
        result.append(average / step)
    return result

if __name__ == "__main__":
    main()
