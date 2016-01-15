#!/usr/bin/python

def signature_make_i(level_signature, feature_signature):
    result = []

    level_range = 2**16
    feature_range = 2**8

    for index, ((last_level, last_feature), (level, feature)) in enumerate(adjacent(zip(level_signature, feature_signature))):
        if abs(level - last_level) > level_range / 8:
            feature_value = last_feature * feature_range + feature
            result.append((index, feature_value))
    return result

def signature_make(level_signature, feature_signature):
    result = []

    level_range = 2**16
    feature_range = 2**8

    import smooth_avg
    level_signature = smooth_avg.avg_smooth(level_signature)

    for (last_level, last_feature), (level, feature) in adjacent(zip(level_signature, feature_signature)):
        if abs(level - last_level) > level_range / 8:
            feature_value = last_feature * feature_range + feature
            result.append(feature_value)

    return result

def avg(block):
    return float(sum(block)) / float(len(block))

def avgm(lblock, cblock, rblock, i, n):
    return (avg(lblock) + avg(cblock) + avg(rblock)) / 3

def signature_matrix(level_signature):
    import math
    n = 16
    m = min(max(len(level_signature) / 64, 8), 64) # 32
    result = [0 for i in xrange(n)]

    level_min = -2**15
    level_range = 2**16

    step = int(math.ceil(len(level_signature) / float(m)))
    block_level_signature = [level_signature[(i * step):((i + 1) * step)] for i in xrange(m)]
    block_level_signature_r = [level_signature[(i * step + step / 2):((i + 1) * step + step / 2)] for i in xrange(m)]
    block_level_signature_l = [level_signature[max(0, ((i - 1) * step + step / 2)):(i * step + step / 2)] for i in xrange(m)]
#    avg_level_signature = [float(sum(block)) / len(block) for block in block_level_signature if len(block) > 0]
    avg_level_signature = [avgm(lb, cb, rb, i, m) for i, (lb, cb, rb) in enumerate(zip(block_level_signature_l, block_level_signature, block_level_signature_r)) if len(lb) > 0 and len(cb) > 0 and len(rb) > 0]

    for index, level in enumerate(avg_level_signature):
        level_i = int((n * (level - level_min)) / level_range)
        if level_i > 0:
            result[level_i - 1] |= 1 << index
        if level_i + 1 < len(result):
            result[level_i + 1] |= 1 << index
        result[level_i] |= 1 << index

    return result

def bitcount(value):
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count

def distance(first_value, second_value, n):
    return n - bitcount(first_value ^ second_value)

def similarity(first_value, second_value):
    return bitcount(first_value & second_value)

def signature_matrix_compare(first_matrix, second_matrix):
    result = 0.0

    count = 0
    for index, (first_value, second_value) in enumerate(zip(first_matrix, second_matrix)):
        value, match = signature_row_compare(first_value, second_value)
        count += 1 if match else 0
        result += value
    if count > 0:
        return result / float(count)
    return 0

def signature_row_compare(first_value, second_value):
    first_count = bitcount(first_value)
    second_count = bitcount(second_value)
    both_count = bitcount(first_value & second_value)

    if first_count and second_count:
        value = 2 * float(both_count) / float(first_count + second_count)
        return value, True

    return 0, False

def signature_compare(first_signature, second_signature):
    matches = 0
    first_index = 0
    second_index = 0

    while first_index < len(first_signature) and second_index < len(second_signature):
        if first_signature[first_index] < second_signature[second_index]:
            first_index += 1
        elif second_signature[second_index] < first_signature[first_index]:
            second_index += 1
        else:
            first_index += 1
            second_index += 1
            matches += 1

    length = len(first_signature) + len(second_signature)
    return 2.0 * float(matches) / float(length) if length > 0 else 0.0

def signature_index(signatures):
    index = dict()
    lengths = map(lambda signature: len(set(signature)), signatures) # set(signature)

    for i, signature in enumerate(signatures):
        for value in signature:
            value_index = index.setdefault(value, set([]))
            value_index.add(i)

    for value_key in index.iterkeys():
        value_index = list(index[value_key])
        value_index.sort()
        index[value_key] = value_index

    return (index, lengths)

def signature_search_intersect(data, signature):
    import heapq
    import math

    c = 0.1

    index, lengths = data

    value_lists = [index[value] for value in set(signature) if value in index] # set(signature)

    heap = [(value_list[0], 0, value_list) for value_list in value_lists if len(value_list) > 0]
    heapq.heapify(heap)

    result = []

    while len(heap) > 0:
        lists = [heapq.heappop(heap)]
        min_value, min_index, min_list = lists[0]

        while len(heap) > 0 and heap[0][0] == min_value:
            lists.append(heapq.heappop(heap))

        avg_length = float(len(value_lists) + lengths[min_value]) / 2

        n = int(math.ceil(c * avg_length))

        if len(lists) >= max(n, 1):
            if min_value not in map(lambda (i, compare): i, result):
                result.append((min_value, float(len(lists)) / avg_length))

        for value, value_index, value_list in lists:
            if value_index + 1 < len(value_list):
                heapq.heappush(heap, (value_list[value_index + 1], value_index + 1, value_list))

    result.sort(key = lambda (i, compare): -compare)
    return result

def signature_search(data, signature):
    index, lengths = data
    i_values = dict()

    for value in set(signature):
        if value in index:
            for i in index[value]:
                i_values.setdefault(i, 0)
                i_values[i] += 1

    result = []
    for i, count in i_values.iteritems():
        length = len(set(signature)) + lengths[i]
        compare = 2.0 * float(count) / float(length) if length > 0 else 0.0
        result.append((i, compare))

    result.sort(key = lambda (i, compare): -compare)
    return result

def adjacent(iterator, start = None):
    iterator = iter(iterator)
    lastvalue = start
    if not lastvalue:
        lastvalue = next(iterator, None)
    for value in iterator:
        yield (lastvalue, value)
        lastvalue = value
