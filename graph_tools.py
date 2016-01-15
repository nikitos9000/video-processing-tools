#!/usr/bin/python
import sys
import pickle
import math
import search
import showsig
import correlate

def main():
    action = sys.argv[1]

    if action == "create":
        indexfile = sys.argv[2]
        filelist = load_listfile(sys.argv[3])
        create_graph(indexfile, filelist)

    if action == "create_feature":
        indexfile = sys.argv[2]
        level_filelist = load_listfile(sys.argv[3])
        feature_filelist = load_listfile(sys.argv[4])
        create_graph_feature(indexfile, level_filelist, feature_filelist)

    if action == "create2":
        indexfile = sys.argv[2]
        filelist = load_listfile(sys.argv[3])
        create_graph2(indexfile, filelist)

    if action == "create_correlation":
        indexfile = sys.argv[2]
        filelist = load_listfile(sys.argv[3])
        create_graph_correlation(indexfile, filelist)

    if action == "print":
        print_graph(sys.argv[2])

    if action == "correlate":
        find_duplicates(sys.argv[2], float(sys.argv[3]))

    if action == "dot":
        print_dot(sys.argv[2])

    if action == "rebase_log":
        graph, filenames = load_graph(sys.argv[2])
        n = float(sys.argv[3])

        for s_index in xrange(len(graph)):
            for e_i in xrange(len(graph[s_index])):
                e_index, correlation = graph[s_index][e_i]
                graph[s_index][e_i] = (e_index, rebase_value_log(n, correlation))

        print pickle.dumps((graph, filenames))

    if action == "rebase_exp":
        graph, filenames = load_graph(sys.argv[2])
        n = float(sys.argv[3])

        for s_index in xrange(len(graph)):
            for e_i in xrange(len(graph[s_index])):
                e_index, correlation = graph[s_index][e_i]
                graph[s_index][e_i] = (e_index, rebase_value_exp(n, correlation))

        print pickle.dumps((graph, filenames))

    if action == "adjust":
        graph, filenames = load_graph(sys.argv[2])
        values = [float(v) for v in open(sys.argv[3], "r").readlines()]
        value_index = 0
        for i in xrange(len(filenames)):
            for j in xrange(len(filenames)):
                if i != j:
                    edges = graph[i]
                    for k, (e_index, correlation) in enumerate(edges):
                        if e_index == j:
                            edges[k] = (e_index, values[value_index])
                            break
                    value_index += 1

        print pickle.dumps((graph, filenames))

    if action == "extend":
        graph, filenames = load_graph(sys.argv[2])
        graph = fill_graph(graph, float(sys.argv[3]))
        print pickle.dumps((graph, filenames))

    if action == "level_update":
        graph, filenames = load_graph(sys.argv[2])
        level_signatures = pickle.loads(open(sys.argv[3], "r").read())
        c_value = float(sys.argv[4])
        l_value = float(sys.argv[5])
        import signature_make

        result_graph = []
        for s_index, edges in enumerate(graph):
            result_edges = []
            for e_index, correlation in edges:
                l_correlation = signature_make.signature_matrix_compare(level_signatures[s_index], level_signatures[e_index])
                if l_correlation  >= l_value and correlation >= c_value:
                    result_edges.append((e_index, l_correlation))
            result_graph.append(result_edges)

        print pickle.dumps((result_graph, filenames))

    if action == "clique":
        graph, filenames = load_graph(sys.argv[2])

        r_sets = find_all_cliques([[e_index for e_index, correlation in edges] for edges in graph])
        r_sets.sort(key = lambda x: -len(x))

        for r_set in r_sets:
            print str(list(r_set))
            print str([filenames[i] for i in r_set])

    if action == "auto_clique":
        graph, filenames = load_graph(sys.argv[2])
        weak_threshold = float(sys.argv[3])
        strong_threshold = float(sys.argv[4])
        weak_graph = [[e_index for e_index, correlation in edges if correlation >= weak_threshold] for edges in graph]
        strong_graph = [[(e_index, correlation) for e_index, correlation in edges if correlation >= strong_threshold] for edges in graph]

        print pickle.dumps((strong_graph, filenames))


def find_connected_components(graph):
    components = [[] for i in xrange(len(graph))]
    mark = [0 for i in xrange(len(graph))]
    for s_index, edges in enumerate(graph):
        stack = []
        stack.append(s_index)
        while stack:
            index = stack.pop()
            if not mark[index]:
                mark[index] = s_index + 1
                components[s_index].append(index)
                stack.extend(graph[index])

    components = [sorted(component) for component in components if component]
    component_maps = []
    for component in components:
        component_map = [0 for i in xrange(len(graph))]
        for component_index, index in enumerate(component):
            component_map[index] = component_index
        component_maps.append(component_map)

    graphs = []
    # for component, component_map in zip(components, component_maps):
    #         graph = [[component_map[e_index] for e_index in graph[s_index]] for s_index in component]
    #         graphs.append(graph)

    return (graphs, components)

# def find_all_x(graph):
#         r_sets = []
#         r_set_all = set()
#         for s_index, edges in enumerate(graph):
#             s_set = set(edges)
#             s_set.add(s_index)

#             for e_index, correlation in edges:
#                 e_edges = graph[e_index]
#                 e_set = set(e_edges)
#                 e_set.add(e_index)
#                 s_set &= e_set

#             r_sets.append(s_set)

#         r_sets.sort(key = lambda x: -len(x))

#         for r_set in r_sets:
#             r_set -= r_set_all
#             r_set_all |= r_set

#         r_sets = [r_set for r_set in r_sets if len(r_set) > 0]
#         r_sets.sort(key = lambda x: -len(x))
#         return r_sets


def find_all_cliques(graph):
    cliques = []
    stack = []
    stack.append((set(), set(xrange(len(graph))), set(), None, len(graph)))
    while stack:
        compsub, candidates, nodes_not, node, num = stack.pop()
        if not candidates and not nodes_not and len(compsub) > 0:
            cliques.append(compsub)
        else:
            for candidate in list(candidates):
                if node == None or candidate not in graph[node]:
                    candidates.remove(candidate)
                    adjacent = set(graph[candidate])
                    new_compsub = set(compsub)
                    new_compsub.add(candidate)
                    new_candidates = candidates & adjacent
                    new_nodes_not = nodes_not & adjacent
                    if node != None:
                        if node in new_nodes_not:
                            new_num = num - 1
                            if new_num > 0:
                                stack.append((new_compsub, new_candidates, new_nodes_not, node, new_num))
                        else:
                            new_num = len(graph)
                            new_node = node
                            for c_node in new_nodes_not:
                                c_num = len(new_candidates) - len(new_candidates & set(graph[c_node]))
                                if c_num < new_num:
                                    new_num = c_num
                                    new_node = c_node
                            stack.append((new_compsub, new_candidates, new_nodes_not, new_node, new_num))
                    else:
                        stack.append((new_compsub, new_candidates, new_nodes_not, node, num))
                    nodes_not.add(candidate)

                    new_num = sum(1 for c in candidates if c not in graph[candidate])
                    if 0 < new_num < num:
                        stack.append((compsub, candidates, nodes_not, candidate, new_num))
                    else:
                        stack.append((compsub, candidates, nodes_not, node, num))
    return cliques

def rebase_value_log(n, value):
    base = (1.0 - 2.0 * n) / (n * n)
    return min(max(math.log(value * base + 1.0, base + 1.0), 0.0), 1.0)

def rebase_value_exp(n, value):
    base = 0.001
    return (math.pow(base, value) - 1) / (base - 1)

def print_dot(graphfile):
    graph, filenames = pickle.loads(open(graphfile, "r").read())
    print "digraph video {"
    for index, filename in enumerate(filenames):
        print "\ta%d [label=\"%s\"];" % (index, filename[6:filename.index("m450x334") - 1].replace("/", "."))

    for s_index, edges in enumerate(graph):
        for e_index, correlation in edges:
            print "\ta%d -> a%d [weight=%d];" % (s_index, e_index, int((1.0 - correlation) * 99 + 1))

    print "}"

def print_graph(graphfile):
    graph, filenames = pickle.loads(open(graphfile, "r").read())
    for s_index, edges in enumerate(graph):
        for e_index, correlation in edges:
            print "%d %d %f" % (s_index, e_index, correlation)

def find_duplicates(graphfile, min_correlation):
    graph, filenames = pickle.loads(open(graphfile, "r").read())

    deep_graph = fill_graph(graph, min_correlation, True)
    graph = fill_graph(graph, min_correlation, False)

    correlation_value = sum(len(edges) for edges in graph) / float(len(graph))
    deep_correlation_value = sum(len(edges) for edges in deep_graph) / float(len(deep_graph))
    print "source: %d%%, deep: %d%%" % (int(correlation_value * 100), int(deep_correlation_value * 100))

def fill_graph(graph, min_correlation, deep = True):
    result_graph = []
    for s_index, edges in enumerate(graph):
        marked = set([s_index])
        result_graph.append([edge for edge in get_links(graph, s_index, min_correlation, marked, deep)])
    return result_graph

def get_links(graph, s_index, min_correlation, marked, deep):
    for e_index, correlation in graph[s_index]:
        if correlation > min_correlation and e_index not in marked:
            marked.add(e_index)
            yield (e_index, correlation)
            if deep:
                for result in get_links(graph, e_index, min_correlation, marked, deep):
                    yield result

def create_graph_feature(indexfile, level_filelist, feature_filelist):
    import signature_tool
    result = signature_tool.search_signatures(indexfile, level_filelist, feature_filelist)

    graph = []
    for search_result in result:
        edges = []
        for search_i, search_compare in search_result:
            edges.append((search_i, search_compare))
        graph.append(edges)

    print pickle.dumps((graph, level_filelist))

def create_graph2(indexfile, filelist):
    index = pickle.loads(open(indexfile, "r").read())
    db, filenames, filevalues = index

    sigs = []
    for filename in filelist:
        sigs.append(showsig.minisig(filename.rstrip(".txt")))

    graph = []
    for index1, filename1 in enumerate(filelist):
        edges = []
        for index2, filename2 in enumerate(filelist):
            if filename1 != filename2:
                correlation = showsig.compare(sigs[index1], sigs[index2])
                edges.append((index2, correlation))
        graph.append(edges)

    data = (graph, [filename for filename, filesize in filenames])
    print pickle.dumps(data)

def create_graph_correlation(indexfile, filelist):
    index = pickle.loads(open(indexfile, "r").read())
    db, filenames, filevalues = index

    graph = []
    for index1, filename1 in enumerate(filenames):
        edges = []
        for index2, filename2 in enumerate(filenames):
            if filename1 != filename2:
                correlation = correlate.pearson(filevalues[index1], filevalues[index2])
                edges.append((index2, correlation))
        graph.append(edges)

    data = (graph, [filename for filename, filesize in filenames])
    print pickle.dumps(data)

def create_graph(indexfile, filelist):
    graph = []
    for sfileindex, sfilename in enumerate(filelist):
        result = search.search(index, sfilename)
        result.sort(key = lambda (fileindex, filename, correlation): -correlation)
        result = result[1:]
        edges = []
        for fileindex, filename, correlation in result:
            edges.append((fileindex, correlation))
        graph.append(edges)

    data = (graph, [filename for filename, filesize in filenames])
    print pickle.dumps(data)

def load_listfile(filename):
    return [v.strip() for v in open(filename, "r").readlines()]

def complement_graph(graph):
    v = set(xrange(len(graph)))
    return [list(v - set(edges) - set([s_index])) for s_index, edges in enumerate(graph)]

def random_graph_fast(n, seed = 1234):
    import random
    import sys
    import itertools
    random.seed(seed)

    init = [int(random.randrange(n)) for i in xrange(n)]

    graph = [[] for i in xrange(n)]
    for s_index, v in enumerate(init):
        for e_index in xrange(s_index):
            if (v * init[e_index]) % (2*n) == 0:
                graph[s_index].append(e_index)

    for s_index, edges in enumerate(graph):
        for e_index in edges:
            graph[e_index].append(s_index)

    return graph

def random_graph(n):
    check = new_check()

    graph = [[k for k in xrange(n) if k < i and check(n)] for i in xrange(n)]
    for s_index, edges in enumerate(graph):
        for e_index in edges:
            graph[e_index].append(s_index)

    return graph

def maximum_clique_luby(v, e, check):
    from itertools import groupby, chain, product

    i = []
    v = sort(v)
    # e = sort(e, lambda (s_index, e_index): s_index)
    e = sort(e)

    while v:
        d = map(lambda (s_index, e_indexes): (s_index, count(e_indexes)), groupby(e, lambda (s_index, e_index): s_index)) # calc D

        d = left(join(d, v, first_key = lambda (s_index, e_index): s_index))
        d = sort(d) # write, not sort

        if d:
            max_d = max(sd_value for sd_index, sd_value in d)

        x = (sd_index for sd_index, sd_value in d if sd_value == max_d or check(2 * (max_d - sd_value))) # compute list of marked verticles
        x = sort(x) # write, not sort

        print "v: " + str(v)
        print "e: " + str(e)
        print "d: " + str(d)
        print "x: " + str(x)

        em = difference(product(x, x), e)
        em = filter(lambda (s_index, e_index): s_index != e_index, em)

        print "em: " + str(em)

        em_sd = join(em, d, lambda (s_index, e_index): s_index, lambda (sd_index, sd_value): sd_index)
        em_sd = map(lambda ((s_index, e_index), (sd_index, sd_value)): (s_index, e_index, sd_value), em_sd)
        em_sd = sort(em_sd, lambda (s_index, e_index, sd_value): e_index) # write and sort

        em_sd_ed = join(em_sd, d, lambda (s_index, e_index, sd_value): e_index, lambda (ed_index, ed_value): ed_index)
        em_sd_ed = map(lambda ((s_index, e_index, sd_value), (ed_index, ed_value)): (s_index, e_index, sd_value, ed_value), em_sd_ed)

        # ux = map(lambda (s_index, e_index, sd_value, ed_value): min(s_index, e_index) if sd_value == ed_value else (s_index if sd_value < ed_value else e_index), em_sd_ed)
        ux = filter(lambda (s_index, e_index, sd_value, ed_value): s_index < e_index, em_sd_ed)
        ux = map(lambda (s_index, e_index, sd_value, ed_value): s_index if sd_value < ed_value else e_index, ux)
        ux = sort(ux) # write and sort
        ux = unique(ux)
        ux = sort(ux) # write, not sort
        # ux = sort(ux) # write and sort

        # print "x: " + str(x)
 #        print "ux: " + str(ux)

        x = difference(x, ux)
        x = sort(x) # write, not sort

        i = union(i, x)
        i = sort(i) # write, not sort

        print "xx: " + str(x)

        ye = left(join(e, x, lambda (s_index, e_index): s_index))
        ye = sort(ye, lambda (s_index, e_index): e_index)

        print "ye: " + str(ye)

        y = map(lambda (e_index, s_indexes): e_index, filter(lambda (e_index, s_indexes): count(s_indexes) >= len(x), groupby(ye, lambda (s_index, e_index): e_index)))
        # y = map(lambda (s_index, e_index): s_index, ye)
        y = sort(y)

        print "y: " + str(y)
        print "i: " + str(i)

        v = difference(y, x)
        v = sort(v) # write, not sort

        # y = map(lambda (s_index, e_index): e_index, left(join(e, x, first_key = lambda (s_index, e_index): s_index)))
        # y = sort(y) # write and sort

        # y = chain(x, difference(v, y))
        # y = sort(y) # write and sort

        # v = difference(v, y)
        # v = sort(v) # write, not sort

        e = difference(e, v, first_key = lambda (s_index, e_index): s_index)
        e = sort(e, lambda (s_index, e_index): e_index) # write and sort
        e = difference(e, v, first_key = lambda (s_index, e_index): e_index)
        e = sort(e, lambda (s_index, e_index): s_index) # write and sort

        # print ""
    return i

def cliques_luby(graph):
    v = range(len(graph))
    e = [(s_index, e_index) for s_index, edges in enumerate(graph) for e_index in edges]
    check = new_check()

    print "%d / %d" % (len(v), len(e))

    from itertools import groupby
    e = sort(e, lambda (s_index, e_index): s_index)
    ee = filter(lambda (s_index, e_indexes): count(e_indexes) > 1, groupby(e, lambda (s_index, e_index): s_index))
    v = map(lambda (s_index, e_indexes): s_index, ee)

    e = difference(e, v, lambda (s_index, e_index): s_index)
    e = sort(e, lambda (s_index, e_index): e_index)
    e = difference(e, v, lambda (s_index, e_index): e_index)
    e = sort(e, lambda (s_index, e_index): e_index)

    # vv = map(lambda (s_index, e_index): s_index, unique(ee, lambda (s_index, e_index): s_index))

    # ee = sort(ee, lambda (s_index, e_index): e_index)
    # ee = filter(lambda (e_index, s_indexes): count(s_indexes) > 1, groupby(ee, lambda (s_index, e_index): e_index))
    # e = sort(ee, lambda (s_index, e_index): s_index)

    print "e: " + str(e)

    # v = map(lambda (s_index, e_index): s_index, unique(e, lambda (s_index, e_index): s_index))
    print "v: " + str(v)

    print "%d / %d" % (len(v), len(e))

    index = 0
    while v:
        i = maximum_clique_luby(v, e, check)
        if not i: break
        yield i
        print str((index, len(i)))
        index += 1

        v = difference(v, i)
        v = sort(v)

        e = difference(e, i, first_key = lambda (s_index, e_index): s_index)
        e = sort(e, lambda (s_index, e_index): e_index) # write and sort
        e = difference(e, i, first_key = lambda (s_index, e_index): e_index)
        e = sort(e, lambda (s_index, e_index): s_index) # write and sort

def check_clique(graph, clique):
    for v in clique:
        if any(k not in graph[v] for k in clique if k != v):
            return False
    return True

def count(values):
    return sum(1 for i in values)

def check_independent_set(graph, independent_set):
    for v in independent_set:
        if any(k in graph[v] for k in independent_set):
            return False
    return True

def check_independent_sets(graph, independent_sets):
    return list(check_independent_set(graph, independent_set) for independent_set in independent_sets)

def maximum_independent_set_luby(v, e, check):
    from itertools import groupby, chain

    i = []
    v = sort(v)
    e = sort(e, lambda (s_index, e_index): s_index)

    while v:
        d = map(lambda (s_index, e_indexes): (s_index, len(list(e_indexes))), groupby(e, lambda (s_index, e_index): s_index)) # calc D

        d = left(join(d, v, first_key = lambda (s_index, e_index): s_index))
        # d = union(d, map(lambda s_index: (s_index, 0), v), first_key = lambda (sd_index, sd_value): sd_index, second_key = lambda (sd_index, sd_value): sd_index)
        d = sort(d) # write, not sort

        # x = (sd_index for sd_index, sd_value in d if not sd_value or check(2 * sd_value)) # compute list of marked verticles
        dx = union(d, map(lambda s_index: (s_index, 0), v), lambda (sd_index, sd_value): sd_index, lambda (sd_index, sd_value): sd_index)
        # x = (sd_index for sd_index, sd_value in dx if not sd_value or check(2 * sd_value)) # compute list of marked verticles
        x = map(lambda (sd_index, sd_value): sd_index, filter(lambda (sd_index, sd_value): not sd_value or check(2 * sd_value), dx)) # compute list of marked verticles
        x = sort(x) # write, not sort

        em_sx = left(join(e, x, first_key = lambda (s_index, e_index): s_index))
        em_sx_sd = join(em_sx, d, first_key = lambda (s_index, e_index): s_index, second_key = lambda (sd_index, sd_value): sd_index)
        em_sx_sd = map(lambda ((s_index, e_index), (sd_index, sd_value)): (s_index, e_index, sd_value), em_sx_sd)

        em_sx_sd = sort(em_sx_sd, lambda (s_index, e_index, sd_value): e_index) # write and sort

        em_sx_sd_ex = left(join(em_sx_sd, x, first_key = lambda (s_index, e_index, sd_value): e_index))
        em_sx_sd_ex_ed = join(em_sx_sd_ex, d, first_key = lambda (s_index, e_index, sd_value): e_index, second_key = lambda (ed_index, ed_value): ed_index)
        em_sx_sd_ex_ed = map(lambda ((s_index, e_index, sd_value), (ed_index, ed_value)): (s_index, e_index, sd_value, ed_value), em_sx_sd_ex_ed)

        # ux = map(lambda (s_index, e_index, sd_value, ed_value): min(s_index, e_index) if sd_value == ed_value else (s_index if sd_value < ed_value else e_index), em_sx_sd_ex_ed)
        ux = filter(lambda (s_index, e_index, sd_value, ed_value): s_index < e_index, em_sx_sd_ex_ed)
        ux = map(lambda (s_index, e_index, sd_value, ed_value): s_index if sd_value <= ed_value else e_index, ux)
        ux = sort(ux) # write and sort
        ux = unique(ux)
        ux = sort(ux) # write, not sort

        print "px: " + str(x)

        x = difference(x, ux)
        x = sort(x) # write, not sort

        i = union(i, x)
        i = sort(i) # write, not sort

        y = chain(x, map(lambda (s_index, e_index): e_index, left(join(e, x, first_key = lambda (s_index, e_index): s_index))))
        y = sort(y) # write and sort

        print "i: " + str(i)
        print "v: " + str(v)
        print "e: "  + str(e)
        print "em_sx_sd_ex_ed: "  + str(em_sx_sd_ex_ed)
        print "x: " + str(x)
        print "ux: " + str(ux)
        print "y: " + str(y)
        print

        v = difference(v, y)
        v = sort(v) # write, not sort

        e = difference(e, y, first_key = lambda (s_index, e_index): s_index)
        e = sort(e, lambda (s_index, e_index): e_index) # write and sort
        e = difference(e, y, first_key = lambda (s_index, e_index): e_index)
        e = sort(e, lambda (s_index, e_index): s_index) # write and sort
    # print ""
    return i

def independent_sets_luby(graph):
    v = range(len(graph))
    e = [(s_index, e_index) for s_index, edges in enumerate(graph) for e_index in edges]
    check = new_check()

    while v:
        i = maximum_independent_set_luby(v, e, check)
        yield i

        v = difference(v, i)
        v = sort(v)

        e = difference(e, i, first_key = lambda (s_index, e_index): s_index)
        e = sort(e, lambda (s_index, e_index): e_index) # write and sort
        e = difference(e, i, first_key = lambda (s_index, e_index): e_index)
        e = sort(e, lambda (s_index, e_index): s_index) # write and sort


def connected(graph):
    g = graph

    xs = set()
    result = dict()
    for x, e in enumerate(g):
        stack = [x]
        key = x

        while stack:
            x = stack.pop()
            if x not in xs:
                xs.add(x)
                if key not in result:
                    result[key] = 0
                result[key] += 1

                stack.extend(g[x])
    return result


def connected_3cliques(graph):
    v = range(len(graph))
    e = [(s_index, e_index) for s_index, edges in enumerate(graph) for e_index in edges]

    sk = lambda (s_index, e_index): s_index
    ek = lambda (s_index, e_index): e_index
    skk = lambda (s_index, e_index): (s_index, e_index)
    ekk = lambda (s_index, e_index): (e_index, s_index)
    se = sort(e, skk)
    ee = sort(e, ekk)

    se_ee = list(join(se, ee, sk, ek))
    se_ee_k = lambda ((ss_index, se_index), (es_index, ee_index)): (se_index, es_index)

    se_ee = sort(se_ee, se_ee_k)

    s3s = intersect(se_ee, se, se_ee_k, skk)

    s3e = intersect(se_ee, ee, se_ee_k, ekk)

    sts = map(lambda ((ss_index, se_index), (es_index, ee_index)): tuple(sorted((ss_index, se_index, es_index))), s3s + s3e)
    sts = list(unique(sort(sts)))

    stss = map(lambda (index, (s_index, se_index, es_index)): (s_index, se_index, index), enumerate(sts))
    stss += map(lambda (index, (s_index, se_index, es_index)): (se_index, s_index, index), enumerate(sts))
    stss += map(lambda (index, (s_index, se_index, es_index)): (s_index, es_index, index), enumerate(sts))
    stss += map(lambda (index, (s_index, se_index, es_index)): (es_index, s_index, index), enumerate(sts))
    stss += map(lambda (index, (s_index, se_index, es_index)): (es_index, se_index, index), enumerate(sts))
    stss += map(lambda (index, (s_index, se_index, es_index)): (se_index, es_index, index), enumerate(sts))

    stss = sort(stss)

    adjs = []

    from itertools import groupby
    for key, values in groupby(stss, lambda (s_index, e_index, i): (s_index, e_index)):
        v = set(values)
        for (vs_index, ve_index, v_i) in v:
                for (xs_index, xe_index, x_i) in v:
                        if v_i != x_i:
                                adjs.append((v_i, x_i))


    mx = max(max(map(lambda (x, y): x, adjs)), max(map(lambda (x, y): y, adjs))) + 1

    g = [set() for i in xrange(mx)]

    for x, y in adjs:
        g[x].add(y)

    xs = set()
    result = dict()
    for x, e in enumerate(g):
        stack = [x]
        key = x

        while stack:
            x = stack.pop()
            if x not in xs:
                xs.add(x)
                if key not in result:
                    result[key] = 0
                result[key] += 1

                stack.extend(g[x])

    return result

 def find_component(g, x, xs, key, result):
     if x in xs: return
     xs.add(x)

     if key not in result:
         result[key] = 0
     result[key] += 1

     for xx in g[x]:
         find_component(g, xx, xs, key, result)

def sort(values, key = None):
    return sorted(values, key = key)

def unique(values, key = None):
    from itertools import groupby, imap
    return imap(lambda (key, value): next(iter(value)), groupby(values, key))

def uniquekey(values, key = None):
    from itertools import groupby, imap
    return imap(lambda (key, value): key, groupby(values, key))

def union(first_values, second_values, first_key = None, second_key = None):
    first_values = iter(first_values)
    second_values = iter(second_values)
    try:
        first_next, second_next = True, True
        first_next_done, second_next_done = True, True

        while True:
            if first_next: first_value = next(first_values)
            first_next_done = False
            if second_next: second_value = next(second_values)
            second_next_done = False

            first_kvalue = first_key(first_value) if first_key else first_value
            second_kvalue = second_key(second_value) if second_key else second_value

            first_less_second = first_kvalue < second_kvalue
              second_less_first = second_kvalue < first_kvalue

              if first_less_second:
                  first_next_done = True
                  yield first_value
              elif second_less_first:
                  second_next_done = True
                  yield second_value
              else:
                  first_next_done = True
                  second_next_done = True
                  yield first_value

            first_next = not second_less_first
            second_next = not first_less_second

    except StopIteration:
        if not first_next_done and not second_next_done:
            yield min(first_value, second_value)
            yield max(first_value, second_value)
        elif not first_next_done:
            yield first_value
        elif not second_next_done:
            yield second_value
        for first_value in first_values:
            yield first_value
        for second_value in second_values:
            yield second_value

def difference(first_values, second_values, first_key = None, second_key = None):
    first_values = iter(first_values)
    second_values = iter(second_values)
    try:
        first_next, second_next = True, True
        last_first_kvalue = None

        while True:
            if second_next: second_value = next(second_values)
            if first_next: first_value = next(first_values)

            first_kvalue = first_key(first_value) if first_key else first_value
            second_kvalue = second_key(second_value) if second_key else second_value

            first_less_second = first_kvalue < second_kvalue
            second_less_first = second_kvalue < first_kvalue

            if first_less_second and not (last_first_kvalue == first_kvalue and second_next and first_next): #
                yield first_value

            first_next = not second_less_first
            second_next = not first_less_second

            if first_next: last_first_kvalue = first_kvalue #

    except StopIteration:
        if not first_next:
            yield first_value

        for first_value in first_values:
            first_kvalue = first_key(first_value) if first_key else first_value

            if not (last_first_kvalue == first_kvalue and second_next and first_next):
                yield first_value

def join(first_values, second_values, first_key = None, second_key = None, first_default = None, second_default = None):
    first_values = iter(first_values)
    second_values = iter(second_values)
    try:
        first_value = next(first_values)
        second_value = next(second_values)
        while True:
            first_kvalue = first_key(first_value) if first_key else first_value
            second_kvalue = second_key(second_value) if second_key else second_value

            if first_kvalue < second_kvalue:
                if second_default:
                    yield (first_value, second_default(first_value))
                first_value = next(first_values)
            elif first_kvalue > second_kvalue:
                if first_default:
                    yield (first_default(second_value), second_value)
                second_value = next(second_values)
            else:
                last_first_kvalue = first_kvalue
                while first_kvalue == last_first_kvalue:
                    yield (first_value, second_value)
                    first_value = next(first_values)
                    first_kvalue = first_key(first_value) if first_key else first_value
                second_value = next(second_values)

    except StopIteration:
        pass


    # first_index = 0
    # second_index = 0
    # while first_index < len(first_values) and second_index < len(second_values):
    #         first_value = first_values[first_index]
    #         second_value = second_values[second_index]
    #         if first_key: first_value = first_key(first_value)
    #         if second_key: second_value = second_key(second_value)

    #         if first_value < second_value:
    #             if default is not None:
    #                 yield (first_values[first_index], default(first_value))
    #             first_index += 1
    #         elif first_value > second_value:
    #             if default is not None:
    #                 yield (default(second_value), second_values[second_index])
    #             second_index += 1
    #         else:
    #             yield (first_values[first_index], second_values[second_index])
    #             first_index += 1
    #             second_index += 1

def left(values):
    return map(lambda (left, right): left, values)

def right(values):
    return map(lambda (left, right): right, values)

def intersect(first_values, second_values, first_key = None, second_key = None):
    return map(lambda (first, second): first, join(first_values, second_values, first_key, second_key))

# def intersect(first_values, second_values, first_key = None, second_key = None):
#     first_index = 0
#     second_index = 0
#     while first_index < len(first_values) and second_index < len(second_values):
#         first_value = first_values[first_index]
#         second_value = second_values[second_index]
#         if first_key: first_value = first_key(first_value)
#         if second_key: second_value = second_key(second_value)

#         if first_value < second_value:
#             first_index += 1
#         elif first_value > second_value:
#             second_index += 1
#         else:
#             yield first_values[first_index]
#             first_index += 1
#             second_index += 1

# def difference(first_values, second_values, first_key = None, second_key = None):
#     first_index = 0
#     second_index = 0
#     while first_index < len(first_values) and second_index < len(second_values):
#         first_value = first_values[first_index]
#         second_value = second_values[second_index]
#         if first_key: first_value = first_key(first_value)
#         if second_key: second_value = second_key(second_value)

#         if first_value < second_value:
#             yield first_values[first_index]
#             first_index += 1
#         elif first_value > second_value:
#             second_index += 1
#         else:
#             first_index += 1
#             second_index += 1

def mis_luby(graph):
    iss = set()
    vs = set(xrange(len(graph)))
    # es = set((min(s_index, e_index), max(s_index, e_index)) for s_index, edges in enumerate(graph) for e_index in edges)
    es = set((s_index, e_index) for s_index, edges in enumerate(graph) for e_index in edges)
    check = new_check()

    while vs:
        ds = dict((v, sum(1 for s_index, e_index in es if v == s_index or v == e_index)) for v in vs)
        mark = dict((v, not ds[v] or check(2 * ds[v])) for v in vs)

        print "e: " + str(sorted(es))
        print "x: " + str([v for v, x in mark.iteritems() if x])

        for s_index, e_index in es:
            if mark[s_index] and mark[e_index]:
                if ds[s_index] <= ds[e_index]:
                    mark[s_index] = False
                else:
                    mark[e_index] = False

        ist = set(v for v in vs if mark[v])
        iss |= ist
        ys = ist | set(v for v in vs for u in ist if (v, u) in es or (u, v) in es)
        vs -= ys
        es = [(u, v) for u, v in es if u not in ys and v not in ys]
        # es -= ys # !fixit

        print ""
    return sorted(iss)

def new_check():
    import random
    random.seed(1234)
    return lambda x: random.randrange(x) == 0

def load_graph(filename):
    return pickle.loads(open(filename, "r").read())


if __name__ == "__main__":
    main()
