#!/usr/bin/python

import sys
import pickle
import signature_read
import signature_make

def main():
#    c = float(sys.argv[1])
    level_filelist = sys.argv[1]
    feature_filelist = sys.argv[2]

    level_filenames = open_filelist(level_filelist)
    feature_filenames = open_filelist(feature_filelist)

    signatures = make_signatures(level_filenames, feature_filenames)

#    signature_dict = dict()
#    for signature_index, signature in enumerate(signatures):
#        for position, value in signature:
#            if value not in signature_dict:
#                signature_dict[value] = (set(), dict())
#            value_indexes, value_positions = signature_dict[value]
#            value_indexes.add(signature_index)
#            value_positions[signature_index] = position

#    signature_position_dict = dict()
#    for value in signature_dict.iterkeys(): # remove positions
#        value_indexes, value_positions = signature_dict[value]
#        signature_dict[value] = value_indexes
#        signature_position_dict[value] = value_positions

#    result = signature_m_n(signatures, signature_dict, c)
    result = signature_cliques(signatures)

    print pickle.dumps((result, level_filenames))



# def signature_clique(signatures, signature_dict):
#     result = [set() for i in xrange(len(signatures))]
#     mn_result = signature_m_n_all(signature_dict, 3)

#     for signature_indexes in mn_result.itervalues():
#         signature_indexes = set(signature_indexes)
#         for signature_index in signature_indexes:
#             result[signature_index] |= signature_indexes

#     for signature_index in xrange(len(result)): # add fake compare value
#         result[signature_index] = zip(result[signature_index], [1] * len(result[signature_index]))

#     return result


# def signature_m_n_all(signature_dict, n):
#     from heapq import heapify, heappop, heappush
#     from math import ceil
#     from itertools import combinations

#     heap_value = lambda h_index, h_list, h_list_value: (h_list[h_index], h_index, h_list, h_list_value)

#     heap = [heap_value(0, sorted(signature_dict[signature_value]), signature_value) for signature_value in signature_dict.iterkeys()]
#     heapify(heap)

#     result = dict(
#         if len(lists) >= n:
#             key_values = sorted(value_list_value for value, value_index, value_list, value_list_value in lists)
#             for key_value in combinations(key_values, n):
#                 key = tuple(keyvalue)
#                 if key not in result:
#                     result[key] = set()
#                 result[key].add(min_value)

#         for value, value_index, value_list, value_list_value in lists:
#             if value_index + 1 < len(value_list):
#                 heappush(heap, heap_value(value_index + 1, value_list, value_list_value))

#     return result


def signature_cliques(signatures):
    signature_map = dict()
    for signature_index, signature in enumerate(signatures):
        for signature_value, signature_position in signature:
            if signature_position not in signature_map:
                signature_map[signature_position] = dict()
            if signature_value not in signature_map[signature_position]:
                signature_map[signature_position][signature_value] = set()
            signature_map[signature_position][signature_value].add(signature_index)

    signature_position_delta = 100

    signature_position_map = dict()
    for signature_position in signature_map.iterkeys(): # replace with fast iteration
        signature_position_map[signature_position] = set(position for position in signature_map.iterkeys() if position - signature_position <= signature_position_delta and position > signature_position)

    sresult = dict()

    for signature_position_i in signature_map.iterkeys():
        signature_map_end = dict()
        for signature_position in signature_position_map[signature_position_i]:
            for signature_value in signature_map[signature_position]:
                if signature_value not in signature_map_end:
                    signature_map_end[signature_value] = set()
                signature_map_end[signature_value] |= signature_map[signature_position][signature_value]

        signature_map_start = signature_map[signature_position_i]
        for signature_value_s, signature_indexes_s in signature_map_start.iteritems():
            for signature_value_e, signature_indexes_e in signature_map_end.iteritems():
                sresult[tuple(sorted((signature_value_s, signature_value_e)))] = signature_indexes_s & signature_indexes_e

    result = [set() for i in xrange(len(signatures))]

    for signature_indexes in sresult.itervalues():
        for signature_index in signature_indexes:
            result[signature_index] |= signature_indexes

    for signature_index in xrange(len(result)): # add fake compare value
        result[signature_index] = zip(result[signature_index], [1] * len(result[signature_index]))

    return result

# def signature_m_n(signatures, signature_dict, c):
#     result = []

#     signature_lengths = [len(signature) for signature in signatures]

#     for signature in signatures:
#         signature_set = set(value for position, value in signature)
#         signature_result = signature_search_intersect(signature_dict, signature_set, signature_lengths, c)
#         result.append(signature_result)

#     return result

# def signature_search_intersect(signature_dict, signature_set, signature_lengths, c):
#     from heapq import heapify, heappop, heappush
#     from math import ceil

#     heap_value = lambda h_index, h_list: (h_list[h_index], h_index, h_list)

#     value_lists = [sorted(signature_dict[signature_value]) for signature_value in signature_set]

#     heap = [heap_value(0, value_list) for value_list in value_lists]
#     heapify(heap)

#     result = dict()

#     while len(heap) > 0:
#         lists = [heappop(heap)]
#         min_value, min_index, min_list = lists[0]

#         while len(heap) > 0 and heap[0][0] == min_value:
#             lists.append(heappop(heap))

#         avg_length = float(len(value_lists) + signature_lengths[min_value]) / 2

#         n = max(int(ceil(c * avg_length)), 1)

#         if len(lists) >= n:
#             if min_value not in result:
#                 result[min_value] = float(len(lists)) / avg_length

#         for value, value_index, value_list in lists:
#             if value_index + 1 < len(value_list):
#                 heappush(heap, heap_value(value_index + 1, value_list))

#     return sorted(result.items(), key = lambda (value, compare): -compare)

def make_signature(level_filename, feature_filename):
    level_signature = signature_read.read_signature_short(level_filename)
    feature_signature = signature_read.read_signature_byte(feature_filename)
    return signature_make.signature_make_i(level_signature, feature_signature)

def make_signatures(level_filenames, feature_filenames):
    signatures = []
    for level_filename, feature_filename in zip(level_filenames, feature_filenames):
        signature = make_signature(level_filename, feature_filename)
        signatures.append(signature)
    return signatures

def adjacent(iterator, start = None):
    iterator = iter(iterator)
    lastvalue = start
    if not lastvalue:
        lastvalue = next(iterator, None)
    for value in iterator:
        yield (lastvalue, value)
        lastvalue = value

def open_filelist(filelist):
    return [line.strip() for line in open(filelist, "r").readlines()]

if __name__ == "__main__":
    main()
