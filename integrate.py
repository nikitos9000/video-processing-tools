#!/usr/bin/python
import sys
import math
import peakfind
from collections import deque

def main():
    filename = sys.argv[1]

    values = []
    for line in open(filename, "r").readlines():
        values.append(int(line.strip()))
    integrate_result = integrate(values)
    codes_result = integrate_codes(values, integrate_result)

print str(len(values))
    print "\n".join(str(sindex) + " " + str(eindex) + " " + str(value) for sindex, eindex, value in codes_result)

def integrate(values):
    result = []
    value_window = 100
    value_base = 4
    global_base = 4
    queue = deque(maxlen = value_window)
    queue2 = deque(maxlen = value_window * 8)

    minglobal, maxglobal = minmax(values)

    for i, v in enumerate(values):
        queue.append(v)
        queue2.append(v)

        minglobal, maxglobal = minmax(queue2)

        if len(queue) == value_window:
            minvalue = min(queue)
            maxvalue = max(queue)
            value = sum(float(value - minvalue) / (maxvalue - minvalue + 1) for value in queue) / len(queue)
            globa = float(v - minglobal) / (maxglobal - minglobal + 1)

            base_value = int(max(math.ceil(value * value_base) - 1, 0))
            base_global = int(max(math.ceil(globa * global_base) - 1, 0))
            result.append(base_value + value_base * base_global)

    return result

def integrate_codes(values, integrate_values):
    levels_window = 4

    result = []

    levels = []
    for (last_index, last_value), (index, value) in adjacent(enumerate(integrate_values)):
        if value != last_value:
            levels.append((last_index, last_value))
    if len(integrate_values) > 0:
        levels.append((len(integrate_values) - 1, integrate_values[-1]))

    levels_queue = deque(maxlen = levels_window)
    for i, v in levels:
        levels_queue.append((i, v))

        if len(levels_queue) == levels_window:
            mindiff, maxdiff = minmax(index - lindex for (lindex, lvalue), (index, value) in adjacent(levels_queue))
            result_value = []

            for index, value in levels_queue:
                result_value.append(value)
            firstindex, firstvalue = levels_queue[0]
            lastindex, lastvalue = levels_queue[-1]
            result.append((firstindex, lastindex, ".".join(str(value) for value in result_value)))

    return result

def adjacent(iterator, start = None):
    iterator = iter(iterator)
    lastvalue = start
    if not lastvalue:
        lastvalue = next(iterator, None)
    for value in iterator:
        yield (lastvalue, value)
        lastvalue = value

def minmax(iterator):
    iterator = iter(iterator)
    minvalue = maxvalue = next(iterator)
    for value in iterator:
        if value > maxvalue:
            maxvalue = value
        elif value < minvalue:
            minvalue = value
    return (minvalue, maxvalue)

if __name__ == "__main__":
    main()
