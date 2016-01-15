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

    peaks, centers = peakfind.peakfind(values)
    indexes, codes = makecode(peaks, centers, 10)
    print str(len(values))
    for (sindex, eindex), code in zip(indexes, codes):
        print str(sindex) + " " + str(eindex) + " " + code

def makecode(peaks, centers, threshold):
    indexes = []
    result = []
    window = 4
    queue = deque(maxlen = window)

    condition = lambda peak, last_peak, threshold: abs(peak) > threshold and abs(peak - last_peak) > threshold

    mintotalcenter, maxtotalcenter = minmax(centers[index] for (last_index, last_peak), (index, peak) \
                                                           in adjacent(enumerate(peaks)) \
                                                           if condition(peak, last_peak, threshold))

    mintotaltop, maxtotaltop = minmax(centers[index] + peak / 2 for (last_index, last_peak), (index, peak) \
                                                                in adjacent(enumerate(peaks)) \
                                                                if condition(peak, last_peak, threshold))

    last_peak = 0

    for index, peak in enumerate(peaks):
        if condition(peak, last_peak, threshold):
            queue.append((index, peak, centers[index]))

            if len(queue) == window:
                minvalue, maxvalue = minmax(value for i, value, center in queue)
                mincenter, maxcenter = minmax(center for i, value, center in queue)
                mintop, maxtop = minmax(center + value / 2 for i, value, center in queue)

                mindiff = min(i - last_i for last_i, i in adjacent(i for i, value, center in queue))

                nqueue = [queue[0]]
                for (last_i, last_value, last_center), (i, value, center) in adjacent(queue):
                    nblocks = int(round(math.sqrt(float(i - last_i) / mindiff)))
                    for k in xrange(1, nblocks):
                        ni = int(last_i + float(i - last_i) * k / nblocks)
                        nqueue.append((ni, 0, 0))
                    nqueue.append((i, value, center))

                result_value = []
                base = 4
                prefix_base = 8

                prefix_ncenter = (float(maxtop + mintop) / 2 - mintotaltop) / (maxtotaltop - mintotaltop)
                prefix_icenter = math.ceil(prefix_ncenter * prefix_base)

                result_value.append(str(int(prefix_icenter)))

                for i, value, center in nqueue:
                    if value != 0:
                        top = center + value / 2
                        ntop = float(top - mintop) / (maxtop - mintop)
                        itop = max(math.ceil(ntop * base), 1.0)
                        result_value.append(str(int(itop)))
                    else:
                        result_value.append(str(0))
                indexes.append((nqueue[0], nqueue[-1]))
                result.append(".".join(result_value))
        last_peak = peak
    return (indexes, result)

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
