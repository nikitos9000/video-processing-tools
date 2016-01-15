#!/usr/bin/python
import sys
import pickle
import search
import graph
import os.path
import metavideo

def main():
    data = []
    for line in open(sys.argv[1], "r").readlines():
        data.append(tuple(line.split("\t")))

    result = pickle.load(open(sys.argv[2], "r"))

    result_set = []

    for entries in result:
        e_set = set(entries)
        for entries_set in result_set:
            if e_set & entries_set:
                entries_set |= e_set
                break
        else:
            result_set.append(e_set)

    result = result_set

    value_dict = dict()
    for index, values in enumerate(result):
        for value in values:
            value_dict[value] = index

    for name, url, vid in data:
        if vid.strip() in value_dict:
            print "\t".join((name, url, vid.strip(), str(value_dict[vid.strip()])))

if __name__ == "__main__":
    main()
