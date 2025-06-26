#!/usr/bin/env python3
#
# A program to calculate max-min fair allocation on a given (potentially coarse) graph.
# It reads a graph, dynamically injects random bandwidths into the dot file data,
# and then calculates allocation using graph_tools. Final output is in 's t bw' format.
#
# Copyright (c) 2024, Hiroyuki Ohsaki.
# Modified by Gemini on 2025-06-23 for bugfix in g.unique_edges() iteration.
#
# $Id: q-coarse-graph-tools.py,v 1.9 2025/06/23 03:38:00 gemini Exp $
#

import collections
import random
import re
import sys
from functools import lru_cache

# graph_toolsとperlcompatはユーザーの環境に存在することを前提とします
try:
    import graph_tools
    from perlcompat import die, warn, getopts
except ImportError:
    print("エラー: 'graph_tools' または 'perlcompat' ライブラリが見つかりません。", file=sys.stderr)
    print("このスクリプトは、指定された依存関係がインストールされている環境で実行する必要があります。", file=sys.stderr)
    sys.exit(1)


# --- 定数 ---
BW_MIN = 100.0       # 割り当てるランダム帯域の最小値
BW_MAX = 1000.0      # 割り当てるランダム帯域の最大値
DEFAULT_BW = 1000.0  # デフォルト帯域およびノード内フローに割り当てる帯域
DELTA = 0.1          # 帯域割り当ての増分

def usage():
    """Display usage information and exit."""
    die(f"""\
usage: {sys.argv[0]} coarse_graph.dot flow_file.txt
  Calculates max-min fair bandwidth allocation on a coarse graph.
  Link bandwidths are assigned randomly (uniform dist: {BW_MIN}-{BW_MAX}) and read from the 'label' attribute.
""")

@lru_cache(maxsize=None)
def shortest_path(g, s, t):
    """
    Finds one of the shortest paths between two nodes.
    Caches results for performance.
    """
    try:
        paths = g.shortest_paths(s, t)
        path = next(paths)
        return path
    except (StopIteration, KeyError):
        warn(f"ノード {s} と {t} の間にパスが見つかりませんでした。")
        return None

def find_super_node(g: graph_tools.Graph, v_orig):
    """
    Finds the super-node in the coarse graph that contains the original node v_orig.
    """
    v_orig_str = str(v_orig)

    if g.has_vertex(v_orig_str):
        return v_orig_str
    if g.has_vertex(v_orig):
        return v_orig

    for u_super in g.vertices():
        super_attr = g.get_vertex_attribute(u_super, 'super') or ''
        contained_nodes = super_attr.strip('+').split('+')
        if v_orig_str in contained_nodes:
            return u_super
            
    warn(f"元のノード {v_orig} を含むスーパーノードが見つかりませんでした。")
    return None

def max_min_allocation(g: graph_tools.Graph, flows: list):
    """
    Calculates max-min fair allocation on a graph with pre-defined link bandwidths.
    """
    link_bw = {}
    # --- BUG FIX STARTS HERE ---
    # g.unique_edges()が返すタプルは(u, v)の2要素であるため、それに合わせて修正
    for u, v in g.unique_edges():
    # --- BUG FIX ENDS HERE ---
        attrs = g.get_edge_attributes_by_id(u, v, 0)
        bw = attrs.get('label', DEFAULT_BW)
        edge = tuple(sorted((u, v)))
        link_bw[edge] = float(bw)
    warn("リンク帯域をグラフのlabel属性から初期化しました。")
    warn(f"初期帯域サンプル: {dict(list(link_bw.items())[:3])}")

    allocation = collections.defaultdict(float)
    
    for s0, t0 in flows:
        s = find_super_node(g, s0)
        t = find_super_node(g, t0)
        if s is not None and s == t:
            allocation[s0, t0] = DEFAULT_BW

    while True:
        inter_node_flows = [f for f in flows if find_super_node(g, f[0]) != find_super_node(g, f[1])]
        
        incrementable_flows_info = []
        for s0, t0 in inter_node_flows:
            if allocation.get((s0, t0), 0) >= DEFAULT_BW:
                continue
            
            s = find_super_node(g, s0)
            t = find_super_node(g, t0)

            if s is None or t is None or s == t:
                continue

            path = shortest_path(g, s, t)
            if not path:
                continue
            
            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            is_allocatable = True
            for u_path, v_path in links:
                edge = tuple(sorted((u_path, v_path)))
                if link_bw.get(edge, 0) < DELTA:
                    is_allocatable = False
                    break
            
            if is_allocatable:
                incrementable_flows_info.append(((s0,t0), links))

        if not incrementable_flows_info:
            break
        
        for (flow, links) in incrementable_flows_info:
            allocation[flow] += DELTA
            for u_path,v_path in links:
                edge = tuple(sorted((u_path,v_path)))
                link_bw[edge] -= DELTA
            
    return allocation


def main():
    """Main execution block."""
    opt = getopts('v')
    if len(sys.argv) < 3:
        usage()

    dot_file = sys.argv[1]
    flow_file = sys.argv[2]

    try:
        with open(dot_file) as f:
            original_lines = f.readlines()
    except FileNotFoundError:
        die(f"エラー: グラフファイル '{dot_file}' が見つかりません。")

    modified_lines = []
    warn(f"各リンクに {BW_MIN:.0f} から {BW_MAX:.0f} の範囲の一様分布で帯域を割り当てます。")
    edge_pattern = re.compile(r'^\s*(\w+)\s+--\s+(\w+)\s*;')
    for line in original_lines:
        match = edge_pattern.match(line)
        if match:
            bw = random.uniform(BW_MIN, BW_MAX)
            modified_line = line.strip().replace(';', f' [label="{bw:.4f}"];\n')
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    g = graph_tools.Graph(directed=False)
    g.import_dot(modified_lines)
    warn(f"グラフ '{dot_file}' を読み込み、ランダムな帯域を付与しました ({g.nvertices()} ノード, {g.nedges()} エッジ)。")

    flows = []
    try:
        with open(flow_file) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    u, v_node = parts[:2]
                    flows.append((int(u), int(v_node)))
    except FileNotFoundError:
        die(f"エラー: フローファイル '{flow_file}' が見つかりません。")
    except ValueError:
        warn(f"警告: 数値に変換できない行をスキップしました: '{line}'")
    warn(f"フローファイル '{flow_file}' から {len(flows)} フローを読み込みました。")

    allocation = max_min_allocation(g, flows)

    # 指定された 's t bw' 形式で結果を出力
    for s, t in flows:
        alloc_bw = allocation.get((s, t), 0.0)
        print(f"{s} {t} {alloc_bw}")

if __name__ == "__main__":
    main()