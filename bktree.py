#!/usr/bin/python
import sys
import pickle
import time

def main():
    action = sys.argv[1]

    if action == "index":
        filename = sys.argv[2]
        indexname = sys.argv[3]
        values = []
        for line in open(filename, "r").readlines():
            value = line.strip()
            if len(value) > 0:
                values.append(value)
        index = bktree_index(values)
        indexfile = open(indexname, "w")
        indexfile.write(pickle.dumps(index))
        indexfile.close()

    if action == "search":
        indexname = sys.argv[2]
        index = pickle.loads(open(indexname, "r").read())
        search_value = sys.argv[3].strip()
        search_distance = int(sys.argv[4].strip())
        start = time.clock()
        result = bktree_search(index, search_value, search_distance)
        end = time.clock()
        print str(result) + " time: " + str(end - start)

def bktree_index(values):
    average_length = int(sum(len(value) for value in values) / float(len(values)))
    root_value = min((value for value in values), key = lambda value: abs(len(value) - average_length))

    root_node = (root_value, {})

    for value in values:
        if value != root_value:
            bktree_add(root_node, value)

    return root_node

def bktree_add(node, value):
    node_value, node_children = node
    distance = metric(node_value, value)
    if distance in node_children:
        bktree_add(node_children[distance], value)
    else:
        node_children[distance] = (value, {})

def bktree_search(root_node, value, maxdistance):
    return bktree_search_value(root_node, value, maxdistance, [])

def bktree_search_value(node, value, maxdistance, result):
    node_value, node_children = node

    limitdistance = maxdistance + max(node_children.keys() or [0])
    distance = metric(node_value, value, limitdistance)
    if distance <= maxdistance:
        result.append(node_value)

    if distance != maxdistance:
        minscore = max(distance - maxdistance, 0)
        maxscore = min(distance + maxdistance, max(node_children.keys() or [0]))
        for score in xrange(minscore, maxscore + 1):
            if score in node_children:
                bktree_search_value(node_children[score], value, maxdistance, result)
    return result

def metric(first, second, limit = None):
    limit = limit or max(len(first), len(second))

    if len(first) > len(second):
        first, second = second, first

    if len(second) - len(first) > limit:
        return limit + 1

    previous_row = [i for i in xrange(0, len(first) + 1)]
    current_row = [0 for i in xrange(0, len(first) + 1)]

    for i in xrange(1, len(second) + 1):
        current_row[0] = i

        start = max(i - limit - 1, 1)
        end = min(i + limit + 1, len(first))
        for j in xrange(start, end + 1):
            cost = 1 if first[j - 1] != second[i - 1] else 0
            current_row[j] = min(current_row[j - 1] + 1, previous_row[j] + 1, previous_row[j - 1] + cost)

        previous_row, current_row = current_row, previous_row

    return previous_row[len(first)]

if __name__ == "__main__":
    main()
