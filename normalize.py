#!/usr/bin/python
import sys
import math
import peakfind
from collections import deque

def main():
    filename = sys.argv[1]
    import signature_read

    values = signature_read.read_signature_short(filename)

#    result = slide_normalize(values)
#    result = window_normalize(values)
#    result = line_normalize(values)
    result = bline_normalize(values)

    for value in result:
        print str(value)

def line_normalize(values, min_value = 0.0, max_value = 1.0):
    minvalue = min(values)
    maxvalue = max(values)
    if minvalue == maxvalue:
        return [value - minvalue + min_value for value in values]
    return [float(value - minvalue) / float(maxvalue - minvalue) * (max_value - min_value) + min_value for value in values]

def wline_normalize(values):
    block_length = len(values) / 4
    avg_length = len(values) / 32

    avg_length = max(avg_length, 1)
    block_length = max(block_length, avg_length)

    if len(values) < block_length:
        return values

    avg_offset = (block_length - avg_length) / 2 + avg_length
    block_offset = 0

    avg_queue = deque(values[avg_offset:avg_offset + avg_length], maxlen = avg_length)
    block_queue = deque(values[block_offset:block_offset + block_length], maxlen = block_length)

    result = []

    for i in xrange(len(values)):
        block_queue.append(values[i])
        if i >= avg_offset:
            avg_queue.append(values[i - avg_offset])

        if i >= block_length:
            avg_value = sum(avg_queue) / float(len(avg_queue))
            block_min, block_max = min(block_queue), max(block_queue)
            result.append((avg_value - block_min) / (block_max - block_min + 1))

    return result

def bline_normalize(values):
    n = 8
    step = int(math.ceil(len(values) / float(n)))

    result = []
    for i in xrange(n):
        block = values[i * step : (i+1) * step]
        block_min, block_max = min(block), max(block)
        result.extend([float(value - block_min) / (block_max - block_min + 1) for value in block])

    return result

def window_normalize(values, window = 200):
    import smooth_avg
    values = smooth_avg.avg_smooth(values)

    forward_avg, backward_avg = [], []

    queue = deque(maxlen = window / 2)
    for value in values:
        forward_avg.append((sum(queue), len(queue)))
        queue.append(value)

    queue = deque(maxlen = window / 2)
    for value in reversed(values):
        backward_avg.append((sum(queue), len(queue)))
        queue.append(value)

    avg = [float(fs + value + bs) / float(fl + 1 + bl) for value, (fs, fl), (bs, bl) in zip(values, forward_avg, backward_avg)]

    return [value / average if average else 0 for value, average in zip(values, avg)]

def slide_normalize(values):
    import smooth_avg
    values = smooth_avg.avg_smooth(values)
    values = [value + 2**15 for value in values]

    forward_avg = []
    backward_avg = []

    alpha = 0.5

    average = 0.0

    for value in values:
        average = average * alpha + value * (1 - alpha)
        forward_avg.append(average)

    average = 0.0

    for value in reversed(values):
        average = average * alpha + value * (1 - alpha)
        backward_avg.append(average)

    backward_avg.reverse()

    bi_avg = [(forward + backward) / 2 for forward, backward in zip(forward_avg, backward_avg)]

    result = []

    for value, average in zip(values, forward_avg):
        if value and average:
            result_value = value / average
        else:
            result_value = 0
        result.append(result_value)

    return result[int(len(result) * 0.1):-int(len(result) * 0.1)]

if __name__ == "__main__":
    main()
