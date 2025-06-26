#!/usr/bin/env python3
#
# A graph coarsening program based on a 1-pass clustering and merging
# algorithm. In each step, it generates GCN embeddings, clusters nodes
# with k-means, and merges all nodes within each cluster into a supernode.
# This version includes features for passing-through flows.
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
import graph_tools

def import_graph(file_path: str) -> tuple[graph_tools.Graph, dict]:
    """
    Load a graph in DOT format from FILE using graph_tools.
    It also normalizes node IDs to be consecutive integers starting from 0,
    while preserving original names in a vertex attribute.
    Returns the graph and a map from original names to new integer IDs.
    """
    print(f"Step 0: 入力グラフ {file_path} を graph_tools で読み込み中...")
    g = graph_tools.Graph(directed=False)
    with open(file_path) as f:
        lines = f.readlines()
        g.import_dot(lines)
    
    # Map original node names to consecutive integer IDs for stable processing
    sorted_vertices = sorted(g.vertices(), key=str)
    # { "1": 0, "2": 1, ... }
    node_map = {str(name): i for i, name in enumerate(sorted_vertices)}
    
    # Create a new graph with integer IDs
    h = graph_tools.Graph(directed=False)
    for orig_name, new_name in node_map.items():
        h.add_vertex(new_name)
        h.set_vertex_attribute(new_name, 'original_name', orig_name)

    for u_orig, v_orig in g.edges():
        h.add_edge(node_map[str(u_orig)], node_map[str(v_orig)])
        
    print(f"  グラフの読み込みとノードIDの正規化が完了。ノード数: {h.nvertices()}")
    return h, node_map

def graph_tools_to_networkx(g_tools: graph_tools.Graph) -> nx.Graph:
    """
    Convert a graph_tools.Graph object to a networkx.Graph object.
    """
    g_nx = nx.Graph()
    g_nx.add_nodes_from(g_tools.vertices())
    g_nx.add_edges_from(g_tools.edges())
    return g_nx

def create_node_features(g: graph_tools.Graph, node_map: dict, flow_file: str) -> torch.Tensor:
    """
    Create node features using graph metrics and optional flow data for the original graph.
    Includes features for passing-through traffic.
    """
    print("Step 1: ノード特徴量を生成中...")
    # === 1. Calculate structural features for each vertex ===
    for v in g.vertices():
        g.set_vertex_attribute(v, 'degree', g.degree(v))
        g.set_vertex_attribute(v, 'betweenness', g.betweenness_centrality(v, normalize=False))

    # === 2. Calculate flow-based features if flow_file is provided ===
    if flow_file:
        print(f"    フローデータ {flow_file} を読み込み中...")
        # --- End-point flow features ---
        flow_out_bw = defaultdict(float)
        flow_in_bw = defaultdict(float)
        flow_out_count = defaultdict(int)
        flow_in_count = defaultdict(int)
        
        # --- NEW: Passing-through flow features ---
        passing_through_bw = defaultdict(float)
        passing_through_count = defaultdict(int)

        flows = []
        with open(flow_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    src, dst, bw = parts
                    bw = float(bw)
                    flows.append((src, dst, bw))
                    
                    # Aggregate endpoint features
                    flow_out_bw[src] += bw
                    flow_in_bw[dst] += bw
                    flow_out_count[src] += 1
                    flow_in_count[dst] += 1
        
        # Aggregate passing-through features
        print("    通過フローの特徴量を計算中...")
        for src, dst, bw in flows:
            # Convert original string names to internal integer IDs for path calculation
            try:
                src_id = node_map[src]
                dst_id = node_map[dst]
            except KeyError:
                print(f"      警告: フロー ({src}, {dst}) のノードがグラフに存在しません。スキップします。")
                continue

            # Find the shortest path for the flow
            # The graph 'g' uses integer IDs, so we use src_id and dst_id
            paths = list(g.shortest_paths(src_id, dst_id))
            if not paths:
                continue
            path = paths[0]
            
            # Identify intermediate nodes
            intermediate_nodes = path[1:-1]
            for node_id in intermediate_nodes:
                passing_through_bw[node_id] += bw
                passing_through_count[node_id] += 1

        # Assign all flow data to each node attribute
        for v in g.vertices():
            orig_name = str(g.get_vertex_attribute(v, 'original_name'))
            # Endpoint features
            g.set_vertex_attribute(v, 'flow_out_bw', flow_out_bw.get(orig_name, 0))
            g.set_vertex_attribute(v, 'flow_in_bw', flow_in_bw.get(orig_name, 0))
            g.set_vertex_attribute(v, 'flow_out_count', flow_out_count.get(orig_name, 0))
            g.set_vertex_attribute(v, 'flow_in_count', flow_in_count.get(orig_name, 0))
            # Passing-through features (key is integer ID `v`)
            g.set_vertex_attribute(v, 'passing_through_bw', passing_through_bw.get(v, 0))
            g.set_vertex_attribute(v, 'passing_through_count', passing_through_count.get(v, 0))


    # === 3. Compose the final feature vector ===
    features = []
    feature_names = [#'degree', 'betweenness'
        ]
    if flow_file:
        feature_names.extend([
            'flow_out_bw', 'flow_in_bw', 'flow_out_count', 'flow_in_count',
            'passing_through_bw', 'passing_through_count' # Add new features
        ])
    
    print(f"    使用する特徴量: {feature_names}")

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
    print("Step 2: GCNによるノード埋め込みを生成中...")
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
    print(f"    {embeddings.shape[0]}個のノードに対し、{embeddings.shape[1]}次元の埋め込みを生成しました。")
    return embeddings.numpy()

def coarsen_graph_by_clustering(g_orig: graph_tools.Graph, node_map: dict, args: argparse.Namespace) -> graph_tools.Graph:
    """
    Coarsens the graph in a single pass by clustering nodes and merging them.
    """
    # 1. Create features and embeddings for the ORIGINAL graph
    features = create_node_features(g_orig, node_map, args.flows)
    g_nx = graph_tools_to_networkx(g_orig)
    embeddings = generate_node_embeddings(g_nx, features, args.embedding_dim, args.hidden_dim)

    if embeddings.size == 0:
        print("  埋め込みが生成できなかったため、処理を終了します。")
        return g_orig # Return original graph if coarsening fails

    # 2. Cluster nodes using k-means
    print(f"Step 3: k-meansクラスタリングを実行中 (k={args.supernodes})...")
    kmeans = KMeans(n_clusters=args.supernodes, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(embeddings)
    
    # Map each original node to its cluster ID
    sorted_nodes = sorted(g_orig.vertices())
    node_to_cluster_map = {node_id: cluster_id for node_id, cluster_id in zip(sorted_nodes, cluster_ids)}
    
    # Group nodes by their assigned cluster
    cluster_to_nodes_map = defaultdict(list)
    for node_id, cluster_id in node_to_cluster_map.items():
        cluster_to_nodes_map[cluster_id].append(node_id)
    print(f"    クラスタリング完了。{len(cluster_to_nodes_map)}個のクラスタが生成されました。")

    # 3. Build the new coarse graph
    print("Step 4: 粗視化グラフを構築中...")
    g_coarse = graph_tools.Graph(directed=False)
    
    # Add supernodes to the coarse graph
    for cluster_id, orig_nodes in sorted(cluster_to_nodes_map.items()):
        g_coarse.add_vertex(cluster_id)
        
        rep_node_id = min(orig_nodes)
        child_node_ids = [nid for nid in sorted(orig_nodes) if nid != rep_node_id]
        
        g_coarse.set_vertex_attribute(cluster_id, 'rep_node_id', rep_node_id)
        if child_node_ids:
            super_attr = "".join([f"+{nid}" for nid in child_node_ids])
            g_coarse.set_vertex_attribute(cluster_id, 'super_ids', super_attr)

    # Aggregate edges between supernodes
    edge_weights = defaultdict(int)
    for u_orig, v_orig in g_orig.edges():
        cluster_u = node_to_cluster_map[u_orig]
        cluster_v = node_to_cluster_map[v_orig]
        
        if cluster_u != cluster_v:
            super_edge = tuple(sorted((cluster_u, cluster_v)))
            edge_weights[super_edge] += 1
    
    for (u_coarse, v_coarse), weight in edge_weights.items():
        g_coarse.add_edge(u_coarse, v_coarse)
        g_coarse.set_edge_weight(u_coarse, v_coarse, weight)

    print("    粗視化グラフの構築完了。")
    return g_coarse

def write_coarse_graph_dot(g_coarse: graph_tools.Graph, g_orig_for_names: graph_tools.Graph, output_path: str):
    """
    Builds a final graph with original names and writes it to a DOT file.
    """
    print(f"\n✅ 処理完了！ 粗視化グラフを {output_path} に保存中...")

    g_final = graph_tools.Graph(directed=False)
    coarse_id_to_orig_name = {}

    for v_coarse in sorted(g_coarse.vertices()):
        rep_node_id = g_coarse.get_vertex_attribute(v_coarse, 'rep_node_id')
        rep_orig_name = str(g_orig_for_names.get_vertex_attribute(rep_node_id, 'original_name'))
        coarse_id_to_orig_name[v_coarse] = rep_orig_name
        g_final.add_vertex(rep_orig_name)
        
        super_ids_attr = g_coarse.get_vertex_attribute(v_coarse, 'super_ids')
        if super_ids_attr:
            child_node_ids_str = super_ids_attr.strip('+').split('+')
            child_orig_names = []
            for child_id_str in child_node_ids_str:
                if child_id_str:
                    child_id = int(child_id_str)
                    child_orig_name = str(g_orig_for_names.get_vertex_attribute(child_id, 'original_name'))
                    child_orig_names.append(child_orig_name)
            
            if child_orig_names:
                final_super_attr_str = "".join(sorted([f"+{n}" for n in child_orig_names], key=str))
                g_final.set_vertex_attribute(rep_orig_name, 'super', final_super_attr_str)

    for u_coarse, v_coarse in g_coarse.edges():
        u_orig = coarse_id_to_orig_name[u_coarse]
        v_orig = coarse_id_to_orig_name[v_coarse]
        if not g_final.has_edge(u_orig, v_orig):
             g_final.add_edge(u_orig, v_orig)
             weight = g_coarse.get_edge_weight(u_coarse, v_coarse)
             if weight:
                 g_final.set_edge_attribute_by_id(u_orig, v_orig, 0, 'label', int(weight))

    dot_lines = []
    dot_lines.append('graph export_dot {')
    dot_lines.append('  graph [start="1"];')
    dot_lines.append('  node [color=gray90,style=filled];')

    try:
        sorted_nodes = sorted(g_final.vertices(), key=int)
    except ValueError:
        sorted_nodes = sorted(g_final.vertices(), key=str)

    for v_orig in sorted_nodes:
        super_attr = g_final.get_vertex_attribute(v_orig, 'super')
        if super_attr:
            dot_lines.append(f'  {v_orig} [super="{super_attr}"];')
        else:
            dot_lines.append(f'  {v_orig};')
    
    edge_list_for_sorting = []
    for u, v in g_final.unique_edges():
        try:
            u_int, v_int = int(u), int(v)
            edge_list_for_sorting.append(tuple(sorted((u_int, v_int))))
        except ValueError:
            edge_list_for_sorting.append(tuple(sorted((u, v), key=str)))
            
    sorted_edges = sorted(list(set(edge_list_for_sorting)))

    for u_key, v_key in sorted_edges:
        u, v = str(u_key), str(v_key)
        attrs = g_final.get_edge_attributes_by_id(u,v,0)
        label_str = ""
        if attrs and 'label' in attrs:
            label_str = f' [label="{attrs["label"]}"]'
        dot_lines.append(f'  {u} -- {v}{label_str};')

    dot_lines.append('}')
    dot_body = "\n".join(dot_lines)

    with open(output_path, 'w') as f:
        now = datetime.now()
        datestr = now.strftime("%Y/%m/%d %H:%M:%S")
        header = f"// Generated by run_kmeans.py (1-pass cluster version) at {datestr}\\n"
        header += f"// undirected, {g_final.nvertices()} vertices, {g_final.nedges()} edges\\n"
        f.write(header)
        f.write(dot_body)

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="GCNとk-meansを用いた1-passクラスタリングによるグラフ粗視化プログラム")
    parser.add_argument('-i', '--input', required=True, help='入力グラフのDOTファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力する粗視化グラフのDOTファイルパス')
    parser.add_argument('-f', '--flows', help='フロー帯域データのファイルパス (オプション)')
    parser.add_argument('-k', '--supernodes', type=int, default=10, help='目標とするスーパーノード数 (k-meansのk)')
    parser.add_argument('--embedding_dim', type=int, default=16, help='GCNが生成する埋め込みベクトルの次元数')
    parser.add_argument('--hidden_dim', type=int, default=32, help='GCN中間層の次元数')
    args = parser.parse_args()

    print("🔷 1-pass k-meansクラスタリング粗視化プログラム開始 🔷")
    
    g_tools_orig, node_map = import_graph(args.input)
    
    g_coarse = coarsen_graph_by_clustering(g_tools_orig, node_map, args)
    
    write_coarse_graph_dot(g_coarse, g_tools_orig, args.output)
    
    print("-" * 50)
    print("サマリー:")
    print(f"  元のノード数: {g_tools_orig.nvertices()}")
    print(f"  最終的なスーパーノード数: {g_coarse.nvertices()}")
    print(f"  ファイル '{args.output}' が作成されました。")
    print("-" * 50)

if __name__ == "__main__":
    main()
