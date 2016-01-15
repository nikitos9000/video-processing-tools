import sys

def reweight(graph):
	result_graph = {}
	for v in graph.iterkeys():
		print >>sys.stderr, v, "/", len(graph)
		subgraph = make_subgraph(graph, v)
		subgraph_weights = calc_weights(subgraph, v)
		result_graph[v] = zip((vv for vv, w in subgraph[v]), (subgraph_weights[vv] for vv, w in subgraph[v]))
	return result_graph

def make_graph(agraph):
	return dict(enumerate(agraph))

def make_agraph(graph):
	agraph = []
	for i in xrange(max(graph.iterkeys()) + 1):
		if i in graph:
			agraph.append(graph[i])
		else:
			agraph.append([])
	return agraph

def make_subgraph(graph, v):
	subnodes = set(vv for vv, w in graph[v])
	subnodes |= set([v])
	subgraph = {}
	subgraph[v] = graph[v]
	for vv, w in graph[v]:
		subgraph[vv] = [(vvv, ww) for vvv, ww in graph[vv] if vvv in subnodes and ww >= 0.01]
	return subgraph

def calc_weights(subgraph, sv):
	from math import fsum

	d = 0.85
	n = len(subgraph)
	wsubgraph = dict(zip(subgraph.keys(), [0] * n))
	wsubgraph[sv] = 1.0 / n

	while True:
		nwsubgraph = dict()

		for v, vs in subgraph.iteritems():
			nwsubgraph[v] = ((1.0 - d) / n + d * fsum(w * wsubgraph[vv] / len(subgraph[vv]) for vv, w in vs if subgraph[vv])) # * ww

		for v in subgraph.iterkeys():
			if abs(nwsubgraph[v] - wsubgraph[v]) > 1e-5:
				break
		else: break

		wsubgraph = nwsubgraph

	del nwsubgraph[sv]

	wf = lambda vv: next((w for v, w in subgraph[vv] if v == sv), 0)

	nwsubgraph = dict((k, wf(k) * v) for k, v in nwsubgraph.iteritems())

	max_norm = max(nwsubgraph.itervalues())
	min_norm = min(nwsubgraph.itervalues())
	nwsubgraph = dict((k, (v - min_norm) / (max_norm - min_norm) if max_norm > min_norm else 0) for k, v in nwsubgraph.iteritems())
	return nwsubgraph


# def graph_reweight(subgraph, sv):
# 	import numpy

# 	numpy.
	# import math

	# -1 * x0 + w_x1_x0 * x1 + w_x2_x0 * x2 + w_x3_x0 * x3 = y1
	# w_x0_x1 * x0 + -1 * x1 + w_x2_x1 * x2 + w_x3_x1 * x3 = y2
	# w_x0_x2 * x0 + w_x1_x2 * x1 + -1 * x2 + w_x3_x2 * x3 = y3
	# w_x0_x3 * x0 + w_x1_x3 * x1 + w_x2_x3 * x2 + -1 * x3 = y4

	# w_x1_x0 * x1 + w_x2_x0 * x2 + w_x3_x0 * x3 = x0
	# w_x0_x1 * x0 + w_x2_x1 * x2 + w_x3_x1 * x3 = x1
	# w_x0_x2 * x0 + w_x1_x2 * x1 + w_x3_x2 * x3 = x2
	# w_x0_x3 * x0 + w_x1_x3 * x1 + w_x2_x3 * x2 = x3

	# w_xi_xj - known, xi - unknown, y1-y4 ??? == 0?
