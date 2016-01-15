#!/usr/bin/python
import sys
import correlate

def main():
    listfile = sys.argv[1]
    listfilea = sys.argv[2]
    files = load_listfile(listfile)
    filesa = load_listfile(listfilea)

    data = []
    dataa = []

    for filename in files:
        values = []
        for line in open(filename, "r").readlines():
            vs = line.strip().split()
            value = int(vs[1]) + 2**15
            values.append(value)
        data.append(frozenset(values))

    for filename in filesa:
        values = []
        for line in open(filename, "r").readlines():
            vs = line.strip().split()
            value = int(vs[1]) + 2**15
            values.append(float(value))
        dataa.append(values)

    n = 20
    sarray = [0 for i in xrange(0, n)]
    narray = [0 for i in xrange(0, n)]
    for i in xrange(0, len(data)):
        for j in xrange(i + 1, len(data)):
            similarity_p = correlate.pearson(dataa[i], dataa[j])
            similarity = 2.0 * float(len(data[i].intersection(data[j]))) / float(len(data[i]) + len(data[j]))
            index = min(max(int(n * similarity), 0), n - 1)
            if similarity_p > 0.8:
                sarray[index] += 1
            else:
                narray[index] += 1

    index = 0
    for svalue, nvalue in zip(sarray, narray):
        print str(index) + " " + str(svalue) + " " + str(nvalue)
        index += 1

def load_listfile(filename):
    return [v.strip() for v in open(filename, "r").readlines()]

if __name__ == "__main__":
    main()
