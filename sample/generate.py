#!/usr/bin/env python3
#
#
# Copyright (c) 2023, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: generate.py,v 1.3 2023/04/08 09:15:17 ohsaki Exp $
#

import math
import sys

from perlcompat import die, warn, getopts
import graph_tools
import tbdump

GRAPH_TYPES = 'random ba barandom ring tree btree lattice voronoi db 3-regular 4-regular'.split(
)

def usage():
    die(f"""\
usage: {sys.argv[0]} [-v] [file...]
  -v    verbose mode
""")

def create_graph(type_, n=100, k=3.):
    """Randomly generate a graph instance using network generation model TYPE_.
    If possible, a graph with N vertices and the average degree of K is
    generated."""
    m = int(n * k / 2)
    g = graph_tools.Graph(directed=False)
    g.set_graph_attribute('name', type_)
    if type_ == 'random':
        return g.create_random_graph(n, m)
    if type_ == 'ba':
        return g.create_graph('ba', n, 10, int(k))
    if type_ == 'barandom':
        return g.create_graph('barandom', n, m)
    if type_ == 'ring':
        return g.create_graph('ring', n)
    if type_ == 'tree':
        return g.create_graph('tree', n)
    if type_ == 'btree':
        return g.create_graph('btree', n)
    if type_ == 'lattice':
        return g.create_graph('lattice', 2, int(math.sqrt(n)))
    if type_ == 'voronoi':
        return g.create_graph('voronoi', n // 2)
    if type_ == 'db':
        # NOTE: average degree must be divisable by 2.
        return g.create_graph('db', n, n * 2)
    if type_ == '3-regular':
        return g.create_random_regular_graph(n, 3)
    if type_ == '4-regular':
        return g.create_random_regular_graph(n, 4)
    if type_ == 'li_maini':
        # NOTE: 5 clusters, 5% of vertices in each cluster, other vertices are
        # added with preferential attachment.
        return g.create_graph('li_maini', int(n * .75), 5, int(n * .25 / 5))
    # FIXMME: support treeba, general_ba, and latent.
    assert False

def save_graph(g, path):
    with open(path, 'w') as f:
        f.write(g.export_dot())

def main():
    opt = getopts('N:n:k:') or usage()
    n_graphs = int(opt.N) if opt.N else 100
    n_desired = int(opt.n) if opt.n else 100
    k_desired = float(opt.k) if opt.k else 2.5
    for type_ in GRAPH_TYPES:
        for n in range(n_graphs):
            g = create_graph(type_, n_desired, k_desired)
            base = f'{type_}-{n}.dot'
            save_graph(g, f'data-original/{base}')

            h = g.copy_graph()
            h.minimum_degree_coarsening(2)
            save_graph(h, f'data-coarse/{base}')

if __name__ == "__main__":
    main()
