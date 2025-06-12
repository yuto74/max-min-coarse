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
import random


GRAPH_TYPES = 'random ba barandom db tree li_maini ring 4-regular voronoi'.split()
#GRAPH_TYPES = 'random ba barandom ring tree btree lattice voronoi db 3-regular 4-regular'.split()
#GRAPH_TYPES = 'random'.split() 
#GRAPH_TYPES = 'ba'.split() 
#GRAPH_TYPES = 'barandom'.split() 
#GRAPH_TYPES = 'db'.split() 
#GRAPH_TYPES = 'tree'.split() 
#GRAPH_TYPES = 'li_maini'.split() 
#GRAPH_TYPES = 'ring'.split() 
#GRAPH_TYPES = '4-regular'.split() 
#GRAPH_TYPES = 'voronoi'.split() 
#GRAPH_TYPES = 'er'.split() 

def usage():
    die(f"""\
usage: {sys.argv[0]} [-v] [file...]
  -v    verbose mode
""")


def save_graph_coarse(g, path, degree=False):
    with open(path, 'w') as f:
        if degree:
            txt = '// original degree distribution, ' + str(degree) + '\n'
            f.write(txt)
        f.write(g.export_dot())
        

def read_graph(g, path):
    with open(path, 'r') as f:
        line = f.readlines()
        g.import_dot(line)

def main():
    opt = getopts('N:g:m:') or usage()

    coarsening_method = opt.m if opt.m else 'COARSENET'
    n_graphs = int(opt.N) if opt.N else 1
    graph_types = [opt.g] if opt.g else GRAPH_TYPES
    
    for type_ in graph_types:
        for n in range(n_graphs):

            base = f'{type_}-{n}.dot'
            g = graph_tools.Graph(directed=False)
            read_graph(g, f'data-original/{base}')
            original_degree = {degree:0 for degree in g.degrees()} # 正解ラベル用
            for degree in g.degrees():
                original_degree[degree] += 1
            for degree in original_degree.keys():
                original_degree[degree] /= g.nvertices()

            h = g.copy_graph()
            
            #h.minimum_degree_coarsening(2)
            #alpha = random.choice(rand_value)
            
            alpha = 0.5
            #h.MGC_coarsening(alpha=alpha)
            
            if coarsening_method == 'MGC':
                h.MGC_coarsening(alpha=alpha)
            elif coarsening_method == 'RM':
                h.RM_coarsening(alpha=alpha)                
            elif coarsening_method == 'COARSENET':
                h.COARSENET_coarsening(alpha=alpha)
            elif coarsening_method == 'LVN':
                h.LVN_coarsening(alpha=alpha)
            elif coarsening_method == 'LVE':
                h.LVE_coarsening(alpha=alpha)
            elif coarsening_method == 'kron':
                h.kron_coarsening(alpha=alpha)
            elif coarsening_method == 'HEM':
                h.HEM_coarsening(alpha=alpha)                

            save_graph_coarse(h, f'data-coarse/{base}', original_degree)

if __name__ == "__main__":
    main()
