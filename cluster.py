#!/usr/bin/python
import sys
import pickle
import correlate
from collections import deque

def main():
    threshold = float(sys.argv[2])

    indexname = sys.argv[1]

    patterns = pickle.loads(open(indexname, "r").read())
    patterns = patterns[::int(sys.argv[3])]

    values = []
    matches = []

    for i in xrange(0, len(patterns)):
        matches.append([i])
        for k in xrange(i + 1, len(patterns)):
            value = correlate.pearson(patterns[i], patterns[k])
            values.append((i, k, value))
            if value > threshold:
                matches[i].append(k)

    firstresult = []
    secondresult = []

    rpatterns = []

    for ps in matches:
        if len(ps) <= 1:
            continue

        plen = int(sys.argv[4])
        presult = []
        for k in xrange(0, plen):
            average = 0
            for x in ps:
                average += patterns[x][k]
            average /= len(ps)
            presult.append(average)
        rpatterns.append(presult)

    for i, k, value in values:
        if value > threshold:
            firstresult.extend(patterns[i])
            secondresult.extend(patterns[k])
            print "(" + str(i) + " x " + str(k) + " = " + str(value) + ")"

    print pickle.dumps(rpatterns)

    ff = open(indexname + ".first", "w")
    sf = open(indexname + ".second", "w")
    rf = open(indexname + ".average", "w")

    for rpattern in rpatterns:
        for rvalue in rpattern:
            rf.write(str(rvalue) + "\n")
    rf.close()

    for value in firstresult[:]:
        ff.write(str(value) + "\n")
    ff.close()

    for value in secondresult[:]:
        sf.write(str(value) + "\n")
    sf.close()

if __name__ == "__main__":
    main()
