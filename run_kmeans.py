#!/usr/bin/env python3
#
# A graph coarsening program based on an iterative clustering and
# merging algorithm. In each step, it generates GCN embeddings,
# clusters nodes with k-means, and merges adjacent nodes that
# fall into the same cluster.
#
# This file has been modified to use graph_tools for input/output and
# feature extraction, and to incorporate flow data as features.
#
import argparse
import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import defaultdict
from datetime import datetime
import graph_tools  # Use graph_tools instead of pydot

def import_graph(file_path: str) -> graph_tools.Graph:
    """
    Load a graph in DOT format from FILE using graph_tools.
    It also normalizes node IDs to be consecutive integers starting from 0,
    while preserving original names in a vertex attribute.
    """
    print(f"Step 0: 入力グラフ {file_path} を graph_tools で読み込み中...")
    g = graph_tools.Graph(directed=False)
    with open(file_path) as f:
        lines = f.readlines()
        g.import_dot(lines)
    
    # Map original node names to consecutive integer IDs for stable processing
    # Note: Node names are converted to strings to ensure consistent sorting
    sorted_vertices = sorted(g.vertices(), key=str)
    node_map = {name: i for i, name in enumerate(sorted_vertices)}
    
    # Create a new graph with integer IDs
    h = graph_tools.Graph(directed=False)
    for orig_name, new_name in node_map.items():
        h.add_vertex(new_name)
        h.set_vertex_attribute(new_name, 'original_name', orig_name)

    for u_orig, v_orig in g.edges():
        h.add_edge(node_map[u_orig], node_map[v_orig])
        
    print(f"  グラフの読み込みとノードIDの正規化が完了。ノード数: {h.nvertices()}")
    return h


def graph_tools_to_networkx(g_tools: graph_tools.Graph) -> nx.Graph:
    """
    Convert a graph_tools.Graph object to a networkx.Graph object.
    """
    g_nx = nx.Graph()
    # Node names in g_tools are integers, which networkx will use directly.
    g_nx.add_nodes_from(g_tools.vertices())
    g_nx.add_edges_from(g_tools.edges())
    return g_nx


def create_node_features(g: graph_tools.Graph, g_orig_for_names: graph_tools.Graph, flow_file: str, flow_data_cache: dict) -> torch.Tensor:
    """
    Create node features using graph metrics and optional flow data.
    For supernodes, flow data is aggregated from all constituent original nodes.
    """
    # === 1. Calculate structural features for each vertex on the CURRENT graph ===
    for v in g.vertices():
        g.set_vertex_attribute(v, 'degree', g.degree(v))
        g.set_vertex_attribute(v, 'betweenness', g.betweenness_centrality(v, normalize=False))

    # === 2. Calculate flow-based features if flow_file is provided ===
    if flow_file and not flow_data_cache:
        # Cache flow data to avoid re-reading the file in every iteration
        print(f"    フローデータ {flow_file} を初回読み込み中...")
        flow_out_bw = defaultdict(float)
        flow_in_bw = defaultdict(float)
        flow_out_count = defaultdict(int)
        flow_in_count = defaultdict(int)

        with open(flow_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    src, dst, bw = parts
                    bw = float(bw)
                    flow_out_bw[src] += bw
                    flow_in_bw[dst] += bw
                    flow_out_count[src] += 1
                    flow_in_count[dst] += 1
        flow_data_cache['out_bw'] = flow_out_bw
        flow_data_cache['in_bw'] = flow_in_bw
        flow_data_cache['out_count'] = flow_out_count
        flow_data_cache['in_count'] = flow_in_count

    if flow_file:
        for v in g.vertices():
            # Identify all original nodes that constitute the current supernode 'v'
            original_names_in_supernode = []
            
            # The representative node 'v' is an integer ID from the original graph
            rep_orig_name = str(g_orig_for_names.get_vertex_attribute(v, 'original_name'))
            original_names_in_supernode.append(rep_orig_name)
            
            super_attr = g.get_vertex_attribute(v, 'super') # e.g., "+5+12"
            if super_attr:
                # The 'super' attribute contains the integer IDs of merged nodes
                merged_node_ids_str = super_attr.strip('+').split('+')
                merged_node_ids = [int(i) for i in merged_node_ids_str if i]
                for merged_id in merged_node_ids:
                    merged_orig_name = str(g_orig_for_names.get_vertex_attribute(merged_id, 'original_name'))
                    original_names_in_supernode.append(merged_orig_name)
            
            # Aggregate flow metrics from all original nodes using the cache
            g.set_vertex_attribute(v, 'flow_out_bw', sum(flow_data_cache['out_bw'].get(name, 0) for name in original_names_in_supernode))
            g.set_vertex_attribute(v, 'flow_in_bw', sum(flow_data_cache['in_bw'].get(name, 0) for name in original_names_in_supernode))
            g.set_vertex_attribute(v, 'flow_out_count', sum(flow_data_cache['out_count'].get(name, 0) for name in original_names_in_supernode))
            g.set_vertex_attribute(v, 'flow_in_count', sum(flow_data_cache['in_count'].get(name, 0) for name in original_names_in_supernode))

    # === 3. Compose the final feature vector ===
    features = []
    feature_names = ['degree', 'betweenness']
    if flow_file:
        flow_feature_names = ['flow_out_bw', 'flow_in_bw', 'flow_out_count', 'flow_in_count']
        feature_names.extend(flow_feature_names)

    # Ensure a consistent order for feature creation
    for v in sorted(g.vertices()):
        attrs = g.get_vertex_attributes(v)
        vec = [attrs.get(name, 0) for name in feature_names]
        features.append(vec)
        
    features_tensor = torch.FloatTensor(features)
    
    # Normalize each feature column
    max_vals = features_tensor.max(axis=0).values
    max_vals[max_vals == 0] = 1.0
    features_tensor = features_tensor / max_vals
    
    return features_tensor


class GCNEmbedder(nn.Module):
    """A simple 2-layer GCN model for generating node embeddings."""
    def __init__(self, in_feats: int, h_feats: int, out_feats: int):
        super(GCNEmbedder, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, out_feats)

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor) -> torch.Tensor:
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def generate_node_embeddings(g_nx: nx.Graph, features: torch.Tensor, embedding_dim: int, hidden_dim: int) -> np.ndarray:
    """Generate node embeddings using the GCN model."""
    g_dgl = dgl.from_networkx(g_nx)
    g_dgl = dgl.add_self_loop(g_dgl)
    in_dim = features.shape[1]
    
    if in_dim == 0:
        print("    特徴量ベクトルが空です。埋め込みをスキップします。")
        return np.array([])

    model = GCNEmbedder(in_dim, hidden_dim, embedding_dim)
    model.eval()
    with torch.no_grad():
        embeddings = model(g_dgl, features)
    return embeddings.numpy()

def merge_nodes_and_propagate_super(g: graph_tools.Graph, u: int, v: int):
    """
    Merges node v into node u, correctly propagating the 'super' attribute.
    This replaces the standard g.merge_vertices to handle transitive merges.
    u: the node to keep (representative).
    v: the node to remove.
    """
    if not g.has_vertex(u) or not g.has_vertex(v) or u == v:
        return

    # 1. Consolidate 'super' attributes BEFORE merging.
    # The new 'super' for 'u' will contain its old children, node 'v', and v's children.
    u_super = g.get_vertex_attribute(u, 'super') or ''
    v_super = g.get_vertex_attribute(v, 'super') or ''
    # Combine them all.
    new_super_for_u = u_super + f'+{v}' + v_super
    
    # 2. Use the library's merge function for the complex edge re-wiring.
    g.merge_vertices(u, v)
    
    # 3. After the merge, overwrite the 'super' attribute with our consolidated one.
    # The library call would have set u's super to u_super + f'+{v}'. We replace it.
    g.set_vertex_attribute(u, 'super', new_super_for_u)

def iterative_coarsening(g_orig: graph_tools.Graph, args: argparse.Namespace, flow_data_cache: dict) -> graph_tools.Graph:
    """
    Iteratively coarsens the graph until the target number of supernodes is reached.
    """
    g = g_orig.copy_graph()  # Work on a copy
    print(f"🔷 反復的粗視化プロセス開始 🔷")
    print(f"   開始ノード数: {g.nvertices()}, 目標スーパーノード数: {args.supernodes}")

    iteration = 1
    max_iterations = 100 # Safety break
    while g.nvertices() > args.supernodes and iteration <= max_iterations:
        print(f"\n--- イテレーション {iteration}: 現在のノード数 {g.nvertices()} ---")
        
        # 1. Create features and embeddings for the CURRENT graph state
        features = create_node_features(g, g_orig, args.flows, flow_data_cache)
        g_nx = graph_tools_to_networkx(g)
        embeddings = generate_node_embeddings(g_nx, features, args.embedding_dim, args.hidden_dim)

        if embeddings.size == 0:
            print("  埋め込みが生成できなかったため、処理を終了します。")
            break

        # 2. Cluster nodes using k-means
        n_clusters = args.supernodes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(embeddings)
        
        # Map cluster IDs back to the graph's node IDs
        sorted_nodes = sorted(g.vertices())
        node_to_cluster_map = {node_id: cluster_id for node_id, cluster_id in zip(sorted_nodes, clusters)}

        # 3. Find adjacent node pairs within the same cluster to merge
        merged_in_this_iteration = set()
        nodes_to_merge = []
        # Iterating over a copy of edges as the graph will be modified
        for u, v in list(g.edges()):
            if u in merged_in_this_iteration or v in merged_in_this_iteration:
                continue
            
            if node_to_cluster_map.get(u) == node_to_cluster_map.get(v):
                nodes_to_merge.append((u, v))
                merged_in_this_iteration.add(u)
                merged_in_this_iteration.add(v)

        # 4. Perform the merge operation
        if not nodes_to_merge:
            print("  マージ対象のノードペアが見つかりませんでした。これ以上粗視化できないため、処理を終了します。")
            print("  ヒント: クラスタリングの結果、隣接ノードが同じクラスタに属さなかった可能性があります。")
            print("  --embedding_dim や --hidden_dim を調整すると結果が変わる場合があります。")
            break

        for u, v in nodes_to_merge:
            # merge_vertices(A, B) merges B into A.
            # Keep the node with the smaller ID as the representative.
            node_to_keep, node_to_remove = (u, v) if u < v else (v, u)
            merge_nodes_and_propagate_super(g, node_to_keep, node_to_remove)
        
        print(f"  イテレーション {iteration} 完了、マージ後のノード数: {g.nvertices()}")
        iteration += 1

    if iteration > max_iterations:
        print("最大イテレーション回数に達しました。")

    print("\n✅ 粗視化ループ完了。")
    return g


def write_coarse_graph_dot(g_coarse: graph_tools.Graph, g_orig_for_names: graph_tools.Graph, output_path: str):
    """
    Builds a final graph with original names and writes it to a DOT file
    in the user-specified format, with nodes and edges numerically sorted.
    """
    print(f"\n✅ 処理完了！ 粗視化グラフを {output_path} に保存中...")

    # Verification step to ensure all original nodes are accounted for
    all_orig_ids = set(g_orig_for_names.vertices())
    accounted_ids_in_coarse = set()
    for v_int in g_coarse.vertices():
        accounted_ids_in_coarse.add(v_int)
        super_attr = g_coarse.get_vertex_attribute(v_int, 'super')
        if super_attr:
            children = {int(i) for i in super_attr.strip('+').split('+') if i}
            accounted_ids_in_coarse.update(children)

    if all_orig_ids != accounted_ids_in_coarse:
        print("\n警告: ノードの消失または重複が検出されました。")
        missing = all_orig_ids - accounted_ids_in_coarse
        extra = accounted_ids_in_coarse - all_orig_ids
        if missing:
            print(f"  消失した元のノードID: {sorted(list(missing))}")
        if extra:
            print(f"  余分なノードID: {sorted(list(extra))}")
    else:
        print("  全ノードの整合性を確認しました。")


    # 1. Create a new graph that will use original node names for output.
    g_final = graph_tools.Graph(directed=False)
    int_to_orig_name = {}

    # 2. Populate the final graph with nodes (using original names) and their 'super' attributes.
    for v_int in sorted(g_coarse.vertices()):
        # Get the original name for the representative node of the supernode
        rep_orig_name = str(g_orig_for_names.get_vertex_attribute(v_int, 'original_name'))
        int_to_orig_name[v_int] = rep_orig_name
        g_final.add_vertex(rep_orig_name)
        
        # Get the 'super' attribute which contains merged integer IDs
        super_attr_int = g_coarse.get_vertex_attribute(v_int, 'super')
        if super_attr_int:
            # Convert merged integer IDs back to their original names
            merged_node_ids_str = super_attr_int.strip('+').split('+')
            merged_orig_names = []
            for merged_id_str in merged_node_ids_str:
                if merged_id_str:
                    merged_id = int(merged_id_str)
                    merged_orig_name = str(g_orig_for_names.get_vertex_attribute(merged_id, 'original_name'))
                    merged_orig_names.append(merged_orig_name)
            
            if merged_orig_names:
                # Create the new 'super' attribute string with original names
                final_super_attr_str = "".join(sorted([f"+{n}" for n in merged_orig_names], key=str))
                g_final.set_vertex_attribute(rep_orig_name, 'super', final_super_attr_str)

    # 3. Add edges to the final graph using the newly mapped original names.
    for u_int, v_int in g_coarse.edges():
        u_orig = int_to_orig_name[u_int]
        v_orig = int_to_orig_name[v_int]
        if not g_final.has_edge(u_orig, v_orig):
             g_final.add_edge(u_orig, v_orig)

    # 4. Generate the DOT string using the format from the user's example.
    dot_lines = []
    dot_lines.append('graph export_dot {')
    dot_lines.append('  graph [start="1"];')
    dot_lines.append('  node [color=gray90,style=filled];')

    # Add node definitions, sorted numerically.
    # Assumes node names can be cast to int for sorting.
    try:
        sorted_nodes = sorted(g_final.vertices(), key=int)
    except ValueError:
        # Fallback to string sort if node names are not purely numeric
        sorted_nodes = sorted(g_final.vertices(), key=str)

    for v_orig in sorted_nodes:
        super_attr = g_final.get_vertex_attribute(v_orig, 'super')
        if super_attr:
            # The 'super' attribute should be quoted as it contains '+'
            dot_lines.append(f'  {v_orig} [super="{super_attr}"];')
        else:
            dot_lines.append(f'  {v_orig};')
    
    # Add edge definitions, sorted numerically.
    # Creates tuples of integers (u, v) with u < v for consistent sorting.
    edge_list_for_sorting = []
    for u, v in g_final.unique_edges():
        try:
            u_int, v_int = int(u), int(v)
            edge_list_for_sorting.append(tuple(sorted((u_int, v_int))))
        except ValueError:
            # Fallback for non-numeric node names
            edge_list_for_sorting.append(tuple(sorted((u, v))))
            
    sorted_edges = sorted(list(set(edge_list_for_sorting)))

    for u, v in sorted_edges:
        dot_lines.append(f'  {u} -- {v};')

    dot_lines.append('}')
    dot_body = "\n".join(dot_lines)

    # 5. Write the final string to the output file
    with open(output_path, 'w') as f:
        now = datetime.now()
        datestr = now.strftime("%Y/%m/%d %H:%M:%S")
        header = f"// Generated by run_kmeans.py (iterative version) at {datestr}\n"
        header += f"// undirected, {g_final.nvertices()} vertices, {g_final.nedges()} edges\n"
        f.write(header)
        f.write(dot_body)


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="GCNとk-meansを用いた反復的グラフ粗視化プログラム")
    parser.add_argument('-i', '--input', required=True, help='入力グラフのDOTファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力する粗視化グラフのDOTファイルパス')
    parser.add_argument('-f', '--flows', help='フロー帯域データのファイルパス (オプション)')
    parser.add_argument('-k', '--supernodes', type=int, default=10, help='目標とするスーパーノード数')
    parser.add_argument('--embedding_dim', type=int, default=16, help='GCNが生成する埋め込みベクトルの次元数')
    parser.add_argument('--hidden_dim', type=int, default=32, help='GCN中間層の次元数')
    args = parser.parse_args()

    print("🔷 反復的 k-meansクラスタリング粗視化プログラム開始 🔷")
    
    g_tools_orig = import_graph(args.input)
    
    # Cache for flow data to avoid re-reading the file in each iteration
    flow_data_cache = {} 
    
    g_coarse = iterative_coarsening(g_tools_orig, args, flow_data_cache)
    
    write_coarse_graph_dot(g_coarse, g_tools_orig, args.output)
    
    print("-" * 50)
    print("サマリー:")
    print(f"  元のノード数: {g_tools_orig.nvertices()}")
    print(f"  最終的なスーパーノード数: {g_coarse.nvertices()}")
    print(f"  ファイル '{args.output}' が作成されました。")
    print("-" * 50)

if __name__ == "__main__":
    main()
