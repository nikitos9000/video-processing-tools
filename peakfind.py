#!/usr/bin/python
import sys
import math
from collections import deque

def main():
    filename = sys.argv[1]

    values = []
    for line in open(filename, "r").readlines():
        values.append(int(line.strip()))
    peaks, centers = peakfind(values)
    for peak in peaks:
         print peak

def peakfind(values):
    result = []
    centers = []
    threshold = 6
    for index, value in enumerate(values):
        lmin = findmin(values, index, xrange(index - 1, 0, -1), threshold)
        rmin = findmin(values, index, xrange(index + 1, len(values)), threshold)
        lmax = findmax(values, index, xrange(index - 1, 0, -1), threshold)
        rmax = findmax(values, index, xrange(index + 1, len(values)), threshold)
        minsize = value - max(values[lmin], values[rmin])
        maxsize = min(values[lmax], values[rmax]) - value
        if abs(minsize - maxsize) > threshold:
            result.append(minsize - maxsize)
        else:
            result.append(0)
        if minsize > maxsize:
            centers.append(value - minsize / 2)
        else:
            centers.append(value + maxsize / 2)
    return (result, centers)

def findmin(values, cindex, indexes, maxthreshold):
    lastminvalue = values[cindex]
    threshold = maxthreshold
    result_index = cindex
    for index in indexes:
        value = values[index]
        if value >= lastminvalue + threshold:
            return result_index

        if value < lastminvalue:
            lastminvalue = value
            threshold = maxthreshold

        result_index = index
    return result_index

def findmax(values, cindex, indexes, maxthreshold):
    lastmaxvalue = values[cindex]
    threshold = maxthreshold
    result_index = cindex
    for index in indexes:
        value = values[index]
        if value <= lastmaxvalue - threshold:
            return result_index

        if value > lastmaxvalue:
            lastmaxvalue = value
            threshold = maxthreshold

        result_index = index
    return result_index

if __name__ == "__main__":
    main()
