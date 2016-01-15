#!/usr/bin/python
import sys
from collections import deque

def main():
    index_group = 31.0
    diff_group = 20

    firstfile = sys.argv[1]
    secondfile = sys.argv[2]

    result = dict()

    firstlen = 0

    firstvalues = []
    for line in open(firstfile, "r").readlines():
        linevalues = line.strip().split()
        if len(linevalues) == 1:
            firstlen = int(linevalues[0])
            continue
        first = int(linevalues[0])
        first2 = int(linevalues[1])
        second = linevalues[2]
        firstvalues.append((first, second))
        if second not in result:
            result[second] = ([], [])
        result[second][0].append((first, first2))

    secondlen = 0

    secondvalues = []
    for line in open(secondfile, "r").readlines():
        linevalues = line.strip().split()
        if len(linevalues) == 1:
            secondlen = int(linevalues[0])
            continue
        first = int(linevalues[0])
        first2 = int(linevalues[1])
        second = linevalues[2]
        secondvalues.append((first, second))
        if second not in result:
            result[second] = ([], [])
        result[second][1].append((first, first2))

    print str(len(firstvalues)) + " " + str(len(secondvalues))

    pairs = []
    for key, value in result.iteritems():
        first, second = value
        for v1 in first:
            for v2 in second:
                pairs.append((v1, v2))

    pairs.sort(key = lambda p: p[0])

    stream = []

    diffs = dict()
    for v1, v2 in pairs:
        key = int((v2[1] - v1[1] + index_group / 2) / index_group)
        if key not in diffs:
            diffs[key] = []
        diffs[key].append((v1, v2))

        stream.append((v2[1] - v1[1], (v1, v2)))

    stream.sort(key = lambda x: x[0])

    maxsrange = (0, 0, 0)
    for index, (key, value) in enumerate(stream):
        eindex = index
        average_shift = 0.0
        while eindex < len(stream) and stream[eindex][0] <= key + diff_group:
            average_shift += float(stream[eindex][0])
            eindex += 1
        average_shift /= eindex - index

        if eindex - index > maxsrange[1] - maxsrange[0]:
            maxsrange = (index, eindex, average_shift)

    if len(stream) > 0:
        maxirange = (maxsrange[1] - maxsrange[0], stream[maxsrange[0]][0], stream[maxsrange[1] - 1][0], int(maxsrange[2]))
        print "max range: " + str(maxirange)

    diffs = sorted(diffs.items(), key = lambda v: -len(v[1]))

    for key, value in diffs:
        minrange = sys.maxint
        maxrange = -sys.maxint
        spacerange = 0.0
        last_evalue = 0.0
        for (first_svalue, first_evalue), (second_svalue, second_evalue) in value:
            spacerange += max(first_svalue - last_evalue - 1, 0)
            minrange = min(minrange, first_svalue)
            maxrange = max(maxrange, first_evalue)
            last_evalue = first_evalue

        if firstlen != 0:
            minrange = 0
            maxrange = firstlen

        spacerange += max(maxrange - last_evalue - 1, 0)

        print "range: " + str(minrange) + " " + str(maxrange)

        coverage = 1.0 - float(spacerange) / (maxrange - minrange)

        print str(len(value)) + ": " + str(value) + " coverage = " + str(int(coverage * 100.0)) + "%"

if __name__ == "__main__":
    main()
