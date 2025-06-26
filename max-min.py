#!/usr/bin/env python3
#
#
# Copyright (c) 2024, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: q-coarse.py,v 1.3 2024/06/27 09:26:19 ohsaki Exp ohsaki $
#

from functools import lru_cache
import collections
import random
import sys

import graph_tools
from perlcompat import die, warn, getopts
import tbdump

INFINITY = 10000
DEFAULT_BW = 1000
DELTA = .1
FLOW_RATIO = 1
# 一致と判定する許容誤差 (例: 0.1は10%の誤差を許容)
ERROR_THRESHOLD = 0.1

def usage():
    die(f"""\
usage: {sys.argv[0]} [-v] [file...]
  -v    verbose mode
""")

@lru_cache
def shortest_path(g, s, t):
    # Return one of the shortest paths.
    paths = g.shortest_paths(s, t)
    path = list(paths)[0]
    return path

def find_super_node(g, v):
    if g.has_vertex(v):
        return v
    #super 属性の中に v が含まれていないかを探す
    for u in g.vertices():
        super_ = g.get_vertex_attribute(u, 'super') or ''
        for w in super_.split('+'):
            if w != '' and v == int(w):
                return u
    raise

def allocate_delta(g, link_bw, flows, allocation):
    starved = []
    n_allocated = 0
    for s0, t0 in flows:
        s = find_super_node(g, s0)
        t = find_super_node(g, t0)
        if s == t:
            allocation[s0, t0] = DEFAULT_BW
            continue

        # Find the shortest path from node s to node t.
        path = shortest_path(g, s, t)
        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        # Is bandwidth available all the path?
        is_allocatable = True
        for u, v in links:
            if link_bw[u, v] < DELTA:
                is_allocatable = False
                break

        # Try with the next flow.
        if not is_allocatable:
            continue

        # If possible, allocate the bandwidth to the flow.
        allocation[s0, t0] += DELTA
        for u, v in links:
            link_bw[u, v] -= DELTA
            if link_bw[u, v] < DELTA:
                starved.append((u, v))
        n_allocated += 1
    return n_allocated, starved

def max_min_allocation(g, flows):
    # Initialize all link bandwidth.
    link_bw = {}
    for u, v in g.edges():
        link_bw[u, v] = link_bw[v, u] = DEFAULT_BW

    # Gradually allocate bandwidth to every flow to find the max-min bandwidth sharing.
    allocation = collections.defaultdict(float)
    congestion_order = []
    while True:
        n_allocated, starved = allocate_delta(g, link_bw, flows, allocation)
        if starved:
            congestion_order.extend(starved)
        if not n_allocated:
            break

    return allocation, congestion_order


def main():
    opt = getopts('vs:a:l') or usage()
    verbose = opt.v
    seed = int(opt.s) if opt.s else None
    if seed:
        random.seed(seed)
    alpha = float(opt.a) if opt.a else .5
    list_only = opt.l

    # Load graph.
    dot_file = sys.argv[1]
    g = graph_tools.Graph(directed=False)
    with open(dot_file) as f:
        lines = f.readlines()
        g.import_graph('dot', lines)
        
    # Load coarse graph.
    dot_file = sys.argv[2]
    gc = graph_tools.Graph(directed=False)
    with open(dot_file) as f:
        lines = f.readlines()
        gc.import_graph('dot', lines)        

    # # Load flows.
    flows = []
    flow_file = sys.argv[3]
    with open(flow_file) as f:
        for line in f:
            line = line.rstrip()
            u, v = line.split(' ')
            u, v = int(u), int(v)
            flows.append((u, v))

    g_allocation, g_congestion_order = max_min_allocation(g, flows) #allocation = 最終的な各フローへの割り当て量, congestion_order = リンクが枯渇した順序
    gc_allocation, gc_congestion_order = max_min_allocation(gc, flows) #allocation = 最終的な各フローへの割り当て量, congestion_order = リンクが枯渇した順序    
    for s, t in flows:
        print(g_allocation[s, t], gc_allocation[s, t])
    #print(congestion_order)

if __name__ == "__main__":
    main()
