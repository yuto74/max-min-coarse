#!/usr/bin/python3

import graph_tools
import sys
from collections import defaultdict

degree_distribs = defaultdict(int)
files = sys.argv[1:]
nfile = len(files)
total_vertices = 0  # 総ノード数を保持する変数

for file in files:
    g = graph_tools.Graph(directed=False)
    with open(file) as f:
        lines = f.readlines()
        g.import_dot(lines)

    for v in g.vertices():
        degree_distribs[g.degree(v)] += 1
        total_vertices += 1  # ノード数をカウント

degree_distribs = sorted(degree_distribs.items())
print('name: input coarse')
for d, c in degree_distribs:
    probability = c / total_vertices  # 確率を計算
    print(f"{d} {probability:.10f}")  # 小数点形式に変更