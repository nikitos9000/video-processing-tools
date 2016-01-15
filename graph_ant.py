# -*- coding: utf8 -*-

import time

# def weight_exact(graph, s_v):
#         import math

#         def make_edge(sv, ev):
#                 return min(sv, ev), max(sv, ev)

#         def make_weight(ws):
#                 return math.fsum(ws) # / len(ws)

#         vv = dict()

#         stack = [(s_v, 1.0, s_v)]
#         while stack:
#                 sv, sw, ssv = stack.pop()

#                 for ev, ew in graph[sv]:
#                         w = sw * ew

#                         if ev == ssv or w < 0.1: continue

#                         vv.setdefault(ev, []).append(w)
#                         stack.append((ev, w, sv))

#         return { v: make_weight(ws) for v, ws in vv.iteritems() }

i_random_choice = 32112
i_random_value = 9887

def weight_distribution(graph, s_v):
        import random; random.seed(1234)
        import math


        def make_edge(sv, ev):
                return min(sv, ev), max(sv, ev)

        max_ws_len = max(len(x) for x in graph.itervalues())

        def make_weight(ws):
                return math.fsum(ws.itervalues()) / max_ws_len # / len(ws)

        def make_result_weight(ws):
                return math.fsum(ws.itervalues())

        vv = { s_v: { s_v: 1.0 } }

        n = 200
        e = 0.01

        k = 0
        j = 0

        # vsum = None
        # for i in xrange(n):
        #         for sv, edges in graph.iteritems():
        #                 if sv not in vv:
        #                         continue

        #                 sw = make_weight(vv[sv])

        #                 for ev, ew in edges:
        #                         if ev not in vv:
        #                                 vv[ev] = dict()

        #                         vv[ev][sv] = sw * ew

        #         pvsum = vsum
        #         vsum = map(make_weight, vv.itervalues())

        #         if pvsum and vsum and len(vsum) == len(pvsum):
        #                 if all(abs(x - y) < e for x, y in zip(pvsum, vsum)):
        #                         break


        def make_rweight(v):
                return math.fsum(vv[v].itervalues()) / math.fsum(make_weight(ws) for ws in vv.itervalues()) # / len(ws)


        def random_choice(keys):
                return random.choice(keys)

                # global i_random_choice
                # x = keys[(i_random_choice ** 2 + 311) % len(keys)]
                # i_random_choice += 1
                # return x


        def random_value(n):
                return random.randrange(n)

                # global i_random_value
                # x = (i_random_value ** 2 * 431234) % n
                # i_random_value += 1
                # return x

        import heapq

        vsum = None
        for i in xrange(n):
                ee = set()
                # qq = [random.choice(vv.keys())]
                qq = [(0, random_choice(vv.keys()))]

                while qq:
                        k += 1

                        # random.shuffle(qq)
                        # sv = qq.pop()
                        random_key, sv = heapq.heappop(qq)
                        # sw = make_weight(vv[sv])
                        sw = make_rweight(sv)

                        for ev, ew in graph[sv]:
                                j += 1
                                sv_ev = make_edge(sv, ev)

                                if sv_ev not in ee:
                                        if ev not in vv:
                                                vv[ev] = dict()

                                        vv[ev][sv] = sw * ew

                                        heapq.heappush(qq, (-random_value(n), ev))
                                        # qq.insert(random.randrange(n + 1), ev)
                                        # qq.append(ev)
                                        ee.add(sv_ev)

                pvsum = vsum
                # vsum = map(make_weight, vv.itervalues())
                vsum = map(make_rweight, vv.iterkeys())

                if pvsum and vsum and len(vsum) == len(pvsum):
                        # if all(abs(x - y) / abs(min(x, y)) < e for x, y in zip(pvsum, vsum)):
                        if all(abs(x / y - 1) < e and abs(y / x - 1) < e for x, y in zip(pvsum, vsum)):
                                break

        # return { v: make_weight(ws) * len(ws) for v, ws in vv.iteritems() if v != s_v }, i

        # vx = { v: make_weight(ws) * len(ws) for v, ws in vv.iteritems() if v != s_v }
        # vx = { v: make_result_weight(ws) for v, ws in vv.iteritems() if v != s_v }
        vx = { v: make_rweight(v) for v, ws in vv.iteritems() if v != s_v }
        # vx_max = max(vx.itervalues())

        # return { v: pfloat(w / vx_max) for v, w in vx.iteritems() }, i
        return vx, 0

class pfloat(float):
        def __repr__(self):
                return "%0.5f" % self

def reweight(graph, sv):
        def make_edge(sv, ev):
                return min(sv, ev), max(sv, ev)

        from collections import deque
        queue = deque([sv])

        edges = set()
        verticles = dict((v, []) for v in graph.iterkeys())

        while queue:
                sv = queue.popleft()

                for ev, w in graph[sv]:
                        sev = make_edge(sv, ev)
                        if sev not in edges:
                                verticles[ev].append((sv, w))

                                queue.append(ev)
                                edges.add(sev)


        tree = dict()

        for sv, deps in verticles.iteritems():
                parents, siblings = tree[sv] = [], []

                for ev, w in deps:
                        print sv, ev, verticles[ev]
                        if (sv, w) in verticles[ev]:
                                siblings.append((ev, w))
                        else:
                                parents.append((ev, w))


        def compute_node(factor, v, tree, cache):
                if v in cache:
                        return cache[v]

                parents, siblings = tree[v]

                if not parents and not siblings:
                        weight = 1
                else:
                        w_parents = 0
                        for ev, w in parents:
                                w_parents += compute_node(factor, ev, tree, cache) * w / len(parents)

                        w_siblings = 0
                        for ev, w in siblings:
                                w_siblings += compute_node(factor, ev, tree, cache) * w / len(siblings)

                        weight = w_parents

                        if siblings:
                                weight = (1 - factor) * weight + factor * w_siblings

                cache[v] = weight
                return weight

        factor = 0.5

        return dict(map(lambda v: (v, compute_node(factor, v, tree, {})), graph.iterkeys()))


def mincost_ant(graph, sv):
        import math
        import random
        random.seed(1234)

        n = len(graph)

        Q = 2
        a = 1
        b = 3
        p = 0.1
        p_best = 0.5
        t0 = 1000000
        n_ants = n
        n_iters = 200

        r_step = 0.001

        arcs = [(i, j) for i, edges in graph.iteritems() for j, w in edges]
        costs = dict(((i, j), w) for i, edges in graph.iteritems() for j, w in edges)


        ts = dict((arc, t0) for arc in arcs)
        ns = dict((arc, 1 / cij) for arc, cij in costs.iteritems())

        def make_t_max(S):
                return 1 / ((1 - p) * make_g(S))

        def make_t_min(t_max, n):
                n_p_best = math.pow(p_best, 1 / n)
                return t_max * (1 / n_p_best - 1) / (n / 2 - 1)

        def make_tij(arc, S):
                return (1 - p) * ts[arc] + (Q / make_g(S) if arc in S else 0)

        def make_p_all():
                return sum(ns[arc]**a * ts[arc]**b for arc in arcs)

        def make_pij(p_all, arc):
                return ns[arc]**a * ts[arc]**b / p_all

        def make_g(S):
                return sum(costs[arc] for arc in S)

        def make_ant_solution(p_all):
                s_nodes = set([sv])
                solution_arcs = []

                while len(s_nodes) < len(graph):
                        s_arcs = []
                        for s_node in s_nodes:
                                for e_node, w in graph[s_node]:
                                        if e_node not in s_nodes:
                                                s_arcs.append((s_node, e_node))


                        for arc in s_arcs:
                                p_arc = make_pij(p_all, arc)

                                if random.randrange(1.0 / r_step) * r_step <= p_arc:
                                        solution_arcs.append(arc)
                                        s_node, e_node = arc
                                        s_nodes.add(s_node)
                                        s_nodes.add(e_node)

                return solution_arcs


        for it in xrange(n_iters):
                p_all = make_p_all()

                g, G = min((make_g(s), s) for s in (make_ant_solution(p_all) for ant in xrange(n_ants)))

                t_max = make_t_max(G)
                t_min = make_t_min(t_max, n)

                for arc in arcs:
                        ts[arc] = max(min(make_tij(arc, G), t_max), t_min)

        return make_g(G), G

def make_random_graph(n, seed = 1234):
        import random
        random.seed(seed)

        graph = {}

        init = [int(random.randrange(n)) for i in xrange(n)]

        for s_index, s_random_value in enumerate(init):
                for e_index, e_random_value in enumerate(init):
                        if e_index >= s_index:
                                break

                        if (s_random_value * e_random_value) % (2 * n) == 0:
                                weight = (float(s_random_value) + float(e_random_value)) / (2 * n)

                                graph.setdefault(s_index, []).append((e_index, weight))
                                graph.setdefault(e_index, []).append((s_index, weight))

        return graph



def make_graph(edges):
        graph = {}

        for sv, ev, w in edges:
                if sv not in graph:
                        graph[sv] = []

                graph[sv].append((ev, w))

                if ev not in graph:
                        graph[ev] = []

                graph[ev].append((sv, w))

        return graph


def measure(a, b):
        import math

        result = []
        for ix, x in enumerate(a):
                for iy, y in enumerate(b):
                        if x == y:
                                result.append(abs(ix - iy) / (float(len(a) + len(b)) / 4))

        return 1 - math.fsum(result) / len(result)


def solve_weight_distribution(graph, s):
        import numpy as np
        import graph_solve

        pt = max(graph.iterkeys()) + 1
        m = np.empty((pt, pt), np.double)
        m.fill(0)

        for s_index, edges in graph.iteritems():
                for e_index, w in edges:
                        m[s_index, e_index] = w
                        m[e_index, s_index] = w

        m = graph_solve.prepare_matrix(m)

        x = { i: graph_solve.find_distance(m, 0, i) for i in xrange(1, len(m)) }

        x_max = max(x.itervalues())

        return x, 0

import contextlib

@contextlib.contextmanager
def time_me(name):
        import time
        start = time.time()
        yield
        end = time.time()
        print "Time %s: " % name, (end - start) * 1000, "ms"

n = 100
graph = make_random_graph(n)

s = next(iter(graph.iterkeys()))

with time_me("MY"):
        r1, i = weight_distribution(graph, s)

with time_me("SOLVE"):
        r2, i = solve_weight_distribution(graph, s)

rr1 = sorted(r1.items(), key = lambda (x, y): -y)
rr2 = sorted(r2.items(), key = lambda (x, y): -y)

rrr1 = map(lambda (x, y): x, rr1)
rrr2 = map(lambda (x, y): x, rr2)

import random
rrr3 = list(rrr2)
random.shuffle(rrr3)

with open("graph_order_compare.txt", "w") as f:
        f.write(" ".join(str(x) for x in rr1))
        f.write("\n\n")
        f.write(" ".join(str(x) for x in rr2))
        f.write("\n")

print measure(rrr1, rrr2)

import pprint
with open("graph_c.txt", "w") as f:
        f.write("START NODE: ")
        f.write(pprint.pformat(s))
        f.write("\n\n\nGRAPH:\n")
        f.write(pprint.pformat(graph))
