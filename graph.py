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

    if action == "pagerank":
            import graph_reweight as rw
            graph, filenames = load_graph(sys.argv[2])
            # graph = rw.make_agraph(rw.reweight(rw.make_graph(threshold_graph(graph, 0.1))))
            graph = rw.make_agraph(rw.reweight(rw.make_graph(graph)))
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

def load_graph(filename):
    return pickle.loads(open(filename, "r").read())

def threshold_graph(graph, threshold):
    return [[(vv, w) for vv, w in vs if w >= threshold] for vs in graph]

if __name__ == "__main__":
    main()
