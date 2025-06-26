#!/usr/bin/env python3
#
# Copyright (c) 2024, BW-COARSE implementation by Gemini
# Based on run_kmeans_cluster_coarse.py
#
# BW-COARSE: A graph coarsening program that preserves max-min fair bandwidth allocation.
# This program implements the concepts described in the provided research paper.
#

import argparse
import math
from collections import defaultdict, deque
import heapq # heapqのインポートを追加

import dgl
import graph_tools
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# s_t_flow.pyで使用されているデフォルトのリンク帯域を定数として定義
DEFAULT_LINK_BANDWIDTH = 1000

def import_graph(file_path: str) -> tuple[graph_tools.Graph, dict]:
    """
    Load a graph in DOT format from a file using graph_tools.
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
    try:
        # 数値のノード名を整数としてソート
        sorted_vertices = sorted(g.vertices(), key=int)
    except ValueError:
        # 文字列のノード名を文字列としてソート
        sorted_vertices = sorted(g.vertices(), key=str)
        
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
    Create node features using graph metrics and flow data.
    These features are designed to capture the bandwidth structure as described in the paper.
    This version includes additional graph-theoretic features.
    """
    print("Step 1: ノード特徴量を生成中...")

    # === 1. Calculate structural and graph-theoretic features ===
    print("    構造的特徴量およびグラフ理論的特徴量を計算中...")
    for v in g.vertices():
        # 次数 (既存)
        degree = g.degree(v)
        g.set_vertex_attribute(v, 'degree', degree)
        
        # ★新規追加: 介在中心性 (Betweenness Centrality)
        #    正規化された値を計算
        betweenness = g.betweenness_centrality(v, normalize=True)
        g.set_vertex_attribute(v, 'betweenness', betweenness)

        # ★新規追加: 局所クラスタ係数 (Local Clustering Coefficient)
        #    次数が2未満のノードは計算できないため0とする
        if degree > 1:
            # ntriadsはノードvの近傍間に存在するエッジ数を数える
            local_clustering = g.ntriads(v) / ((degree * (degree - 1)) / 2)
        else:
            local_clustering = 0.0
        g.set_vertex_attribute(v, 'local_clustering', local_clustering)


    # === 2. Calculate flow-based features (既存の処理) ===
    print(f"    フローデータ {flow_file} を読み込み中...")
    flow_out_bw = defaultdict(float)
    flow_in_bw = defaultdict(float)
    flow_out_count = defaultdict(int)
    flow_in_count = defaultdict(int)
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
    
    print("    通過フローの特徴量を計算中...")
    for src, dst, bw in flows:
        try:
            src_id = node_map[src]
            dst_id = node_map[dst]
        except KeyError:
            print(f"      警告: フロー ({src}, {dst}) のノードがグラフに存在しません。スキップします。")
            continue

        paths = list(g.shortest_paths(src_id, dst_id))
        if not paths:
            continue
        path = paths[0]
        
        intermediate_nodes = path[1:-1]
        for node_id in intermediate_nodes:
            passing_through_bw[node_id] += bw
            passing_through_count[node_id] += 1

    # Assign all flow data and derived features to each node attribute
    for v in g.vertices():
        orig_name = str(g.get_vertex_attribute(v, 'original_name'))
        f_out_bw = flow_out_bw.get(orig_name, 0)
        f_in_bw = flow_in_bw.get(orig_name, 0)
        p_thr_bw = passing_through_bw.get(v, 0)

        g.set_vertex_attribute(v, 'flow_out_bw', f_out_bw)
        g.set_vertex_attribute(v, 'flow_in_bw', f_in_bw)
        g.set_vertex_attribute(v, 'flow_out_count', flow_out_count.get(orig_name, 0))
        g.set_vertex_attribute(v, 'flow_in_count', flow_in_count.get(orig_name, 0))
        g.set_vertex_attribute(v, 'passing_through_bw', p_thr_bw)
        g.set_vertex_attribute(v, 'passing_through_count', passing_through_count.get(v, 0))

        # ★新規追加: 中継フロー率 (Transit Flow Ratio)
        total_bw = f_in_bw + f_out_bw + p_thr_bw
        if total_bw > 0:
            transit_ratio = p_thr_bw / total_bw
        else:
            transit_ratio = 0.0
        g.set_vertex_attribute(v, 'transit_flow_ratio', transit_ratio)

    # === 3. Compose the final feature vector ===
    features = []
    # ★特徴量リストを更新
    feature_names = [
        # 構造的・グラフ理論的特徴量
        'degree', 
        'betweenness',
        'local_clustering',
        # フローベース特徴量
        'flow_out_bw', 'flow_in_bw', 
        'flow_out_count', 'flow_in_count',
        'passing_through_bw', 'passing_through_count',
        #'transit_flow_ratio',
    ]
    
    print(f"    使用する特徴量: {feature_names}")

    for v in sorted(g.vertices()):
        attrs = g.get_vertex_attributes(v)
        vec = [attrs.get(name, 0) for name in feature_names]
        features.append(vec)
        
    features_tensor = torch.FloatTensor(features)
    
    # Normalize each feature column for stable GCN training
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

def _group_nodes_by_connectivity_and_cluster(g: graph_tools.Graph, node_to_cluster_map: dict) -> tuple[dict, dict]:
    """
    同じクラスタに属し、かつ接続されているノードをグループ化する。
    """
    print("    Step 3a: リンク接続とクラスタに基づき、ノードを初期グループ化中...")
    visited = set()
    groups = {}
    node_to_group_map = {}
    group_id_counter = 0

    for node in g.vertices():
        if node not in visited:
            component = []
            q = deque([node])
            visited.add(node)
            base_cluster_id = node_to_cluster_map[node]
            
            while q:
                u = q.popleft()
                component.append(u)
                for v in g.neighbors(u):
                    if v not in visited and node_to_cluster_map.get(v) == base_cluster_id:
                        visited.add(v)
                        q.append(v)

            groups[group_id_counter] = component
            for n in component:
                node_to_group_map[n] = group_id_counter
            group_id_counter += 1
            
    print(f"      初期グループ数: {len(groups)}")
    return node_to_group_map, groups

def _merge_groups_to_target_count(g_orig_for_names: graph_tools.Graph, groups: dict, node_to_group_map: dict, embeddings: np.ndarray, target_count: int, args: argparse.Namespace) -> dict:
    """
    初期グループを、目標ノード数に達するまでマージする。
    ★ボトルネックに対するペナルティを考慮する機能を追加。
    """
    num_groups = len(groups)
    print(f"    Step 3b: 初期グループ ({num_groups}個) を目標 ({target_count}個) まで階層的にマージ中...")
    
    # (Union-Find, group_embeddingsの計算は変更なし)
    # ...
    parent = {i: i for i in groups.keys()}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if root_i < root_j:
                parent[root_i] = root_j
            else:
                parent[root_j] = root_i
            return True
        return False

    group_embeddings = {}
    for gid, nodes in groups.items():
        if nodes:
            group_embeddings[gid] = embeddings[nodes].mean(axis=0)


    # グループ間のエッジを基にマージ候補を評価
    merge_candidates = []
    processed_pairs = set()
    for u, v in g_orig_for_names.edges():
        g1 = node_to_group_map[u]
        g2 = node_to_group_map[v]
        if g1 != g2:
            pair = tuple(sorted((g1, g2)))
            if pair not in processed_pairs:
                # 元の距離を計算
                dist = np.linalg.norm(group_embeddings[g1] - group_embeddings[g2])
                
                # ★ペナルティの計算
                b_u = g_orig_for_names.get_vertex_attribute(u, 'betweenness') or 0
                b_v = g_orig_for_names.get_vertex_attribute(v, 'betweenness') or 0
                penalty = b_u * b_v
                penalized_dist = dist * (1 + args.bottleneck_penalty * penalty)

                # ★ペナルティ適用後の距離を優先度として使用
                heapq.heappush(merge_candidates, (penalized_dist, dist, g1, g2)) # 元の距離もログ表示用に保持
                processed_pairs.add(pair)
    
    # (マージ状況の表示準備は変更なし)
    # ...
    id_to_orig_name = {i: g_orig_for_names.get_vertex_attribute(i, 'original_name') for i in g_orig_for_names.vertices()}
    supergroup_contents = {gid: {id_to_orig_name[node_id] for node_id in nodes} for gid, nodes in groups.items()}

    num_merges_needed = num_groups - target_count
    merges_done = 0
    print(f"      --- マージ開始 (計 {num_merges_needed} 回) ---")
    while merges_done < num_merges_needed and merge_candidates:
        # ★penalized_distとdistを取得
        penalized_dist, dist, g1, g2 = heapq.heappop(merge_candidates)
        
        root1 = find(g1)
        root2 = find(g2)

        if root1 != root2:
            nodes_in_sg1 = sorted(list(supergroup_contents[root1]), key=lambda x: int(x) if x.isdigit() else x)
            nodes_in_sg2 = sorted(list(supergroup_contents[root2]), key=lambda x: int(x) if x.isdigit() else x)
            
            # ★ログ表示を修正
            print(f"      - ステップ {merges_done + 1}: 以下の2グループをマージ (ペナルティ後距離: {penalized_dist:.4f}, 元距離: {dist:.4f})")
            print(f"        - グループA: {nodes_in_sg1}")
            print(f"        - グループB: {nodes_in_sg2}")

            union(g1, g2)
            new_root = find(g1)
            old_root = root1 if new_root == root2 else root2
            
            if new_root in supergroup_contents and old_root in supergroup_contents:
                 supergroup_contents[new_root].update(supergroup_contents[old_root])
                 del supergroup_contents[old_root]

            merges_done += 1
    print("      --- マージ完了 ---")
    
    # (最終的なマッピング作成は変更なし)
    # ...
    final_map = {}
    supernode_roots = {find(i) for i in groups.keys()}
    root_to_final_id = {root: i for i, root in enumerate(supernode_roots)}

    for node in g_orig_for_names.vertices():
        group_id = node_to_group_map[node]
        root = find(group_id)
        final_map[node] = root_to_final_id[root]
        
    print(f"      マージ後の最終的なスーパーノード数: {len(supernode_roots)}")
    return final_map


def coarsen_graph_by_clustering(g_orig: graph_tools.Graph, node_map: dict, args: argparse.Namespace) -> graph_tools.Graph:
    """
    Coarsens the graph by generating embeddings, clustering nodes, and merging them.
    This version ensures that only connected nodes are merged.
    """
    # 1. Create features and embeddings
    features = create_node_features(g_orig, node_map, args.flows)
    g_nx = graph_tools_to_networkx(g_orig)
    embeddings = generate_node_embeddings(g_nx, features, args.embedding_dim, args.hidden_dim)

    if embeddings.size == 0:
        print("  埋め込みが生成できなかったため、処理を終了します。")
        return g_orig

    # --- Display generated embeddings for verification ---
    print("\n--- 生成された埋め込みベクトル (最初の5件) ---")
    print(embeddings[:5])
    print("-" * 50)

    # 2. Cluster nodes using k-means
    num_nodes = g_orig.nvertices()
    k = int(math.ceil(args.alpha * num_nodes))
    print(f"Step 3: k-meansクラスタリングを実行中 (α={args.alpha}, 元ノード数={num_nodes} -> 目標クラスタ数 k={k})...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(embeddings)
    node_to_cluster_map = {node_id: cluster_id for node_id, cluster_id in zip(sorted(g_orig.vertices()), cluster_ids)}
    
    # ★新規追加: k-meansクラスタリングの直接の結果を表示するブロック
    cluster_to_nodes_map = defaultdict(list)
    for node_id, cluster_id in node_to_cluster_map.items():
        orig_name = g_orig.get_vertex_attribute(node_id, 'original_name')
        cluster_to_nodes_map[cluster_id].append(orig_name)
    
    print("\n--- k-meansクラスタリングの直接の結果 ---")
    for cid, nodes in sorted(cluster_to_nodes_map.items()):
        try:
            sorted_nodes = sorted(nodes, key=int)
        except ValueError:
            sorted_nodes = sorted(nodes, key=str)
        print(f"  Cluster {cid}: {sorted_nodes}")
    print("-" * 50)
    # ★ここまでが新規追加部分

    # 3. Perform link-based merging
    # Step 3a: Group nodes based on connectivity within the same cluster
    node_to_group_map, groups = _group_nodes_by_connectivity_and_cluster(g_orig, node_to_cluster_map) #
    
    # Step 3b: Merge groups to meet the target coarse ratio
    target_node_count = int(math.ceil(args.alpha * num_nodes))
    if len(groups) > target_node_count:
        final_node_to_supernode_map = _merge_groups_to_target_count(g_orig, groups, node_to_group_map, embeddings, target_node_count, args) #
    else:
        print("    初期グループ数が目標数以下であるため、追加マージは行いません。")
        final_node_to_supernode_map = node_to_group_map #

    # --- Display FINAL grouping results for verification ---
    supernode_to_nodes_map = defaultdict(list)
    for node_id, supernode_id in final_node_to_supernode_map.items():
        orig_name = g_orig.get_vertex_attribute(node_id, 'original_name')
        supernode_to_nodes_map[supernode_id].append(orig_name)
    
    print("\n--- 最終的なグループ化の結果 ---")
    for sid, nodes in sorted(supernode_to_nodes_map.items()):
        try:
            sorted_nodes = sorted(nodes, key=int)
        except ValueError:
            sorted_nodes = sorted(nodes, key=str)
        print(f"  Supernode {sid}: {sorted_nodes}")
    print("-" * 50)

    # 4. Build the new coarse graph
    print("Step 4: 粗視化グラフを構築中...")
    g_coarse = graph_tools.Graph(directed=False)
    
    # Add supernodes to the coarse graph
    for supernode_id, orig_nodes in sorted(supernode_to_nodes_map.items()):
        g_coarse.add_vertex(supernode_id)
        
        # Determine the representative node (e.g., the one with the smallest ID)
        rep_node_name = min(orig_nodes, key=lambda x: int(x) if x.isdigit() else x)
        rep_node_id = node_map[str(rep_node_name)]
        
        child_nodes = [n for n in sorted(orig_nodes) if n != rep_node_name]
        child_node_ids = [node_map[str(n)] for n in child_nodes]
        
        g_coarse.set_vertex_attribute(supernode_id, 'rep_node_id', rep_node_id)
        if child_node_ids:
            super_attr = "".join([f"+{nid}" for nid in sorted(child_node_ids)])
            g_coarse.set_vertex_attribute(supernode_id, 'super_ids', super_attr)

    # Aggregate edges between supernodes based on the paper's rule
    print("    スーパーノード間のリンクと帯域を集約中...")
    edge_bandwidths = defaultdict(float)
    for u_orig, v_orig in g_orig.edges():
        super_u = final_node_to_supernode_map[u_orig]
        super_v = final_node_to_supernode_map[v_orig]
        
        if super_u != super_v:
            super_edge = tuple(sorted((super_u, super_v)))
            # Rule: Sum of bandwidths of original links
            edge_bandwidths[super_edge] += DEFAULT_LINK_BANDWIDTH
    
    for (u_coarse, v_coarse), bandwidth in edge_bandwidths.items():
        g_coarse.add_edge(u_coarse, v_coarse)
        g_coarse.set_edge_weight(u_coarse, v_coarse, bandwidth)

    print("    粗視化グラフの構築完了。")
    return g_coarse


def write_coarse_graph_dot(g_coarse: graph_tools.Graph, g_orig_for_names: graph_tools.Graph, output_path: str):
    """
    最終的なグラフを元のノード名で構築し、DOTファイルに書き込む。
    エッジラベルは集約された帯域幅を表す。
    'random-100.dot' のスタイルに合わせて、不要な引用符を避けるために、可能な場合は整数ノードIDを使用するように出力形式を調整する。
    """
    print(f"\n✅ 処理完了！ 粗視化グラフを {output_path} に保存中...")

    g_final = graph_tools.Graph(directed=False)
    coarse_id_to_final_name = {}

    def _to_int_if_possible(name_str):
        """文字列が有効な整数を表す場合、文字列を整数に変換する。"""
        s = str(name_str)
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        return s

    for v_coarse in sorted(g_coarse.vertices()):
        rep_node_id = g_coarse.get_vertex_attribute(v_coarse, 'rep_node_id')
        rep_orig_name = str(g_orig_for_names.get_vertex_attribute(rep_node_id, 'original_name'))
        
        final_name = _to_int_if_possible(rep_orig_name)
        coarse_id_to_final_name[v_coarse] = final_name
        g_final.add_vertex(final_name)
        
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
                # 'super'属性の内容を数値的にソートする
                try:
                    sorted_names = sorted(child_orig_names, key=int)
                except ValueError:
                    sorted_names = sorted(child_orig_names)
                
                final_super_attr_str = "".join([f"+{n}" for n in sorted_names])
                g_final.set_vertex_attribute(final_name, 'super', final_super_attr_str)

    for u_coarse, v_coarse in g_coarse.edges():
        u_final = coarse_id_to_final_name[u_coarse]
        v_final = coarse_id_to_final_name[v_coarse]
        if not g_final.has_edge(u_final, v_final):
             g_final.add_edge(u_final, v_final)
             bandwidth = g_coarse.get_edge_weight(u_coarse, v_coarse)
             if bandwidth:
                 g_final.set_edge_attribute_by_id(u_final, v_final, 0, 'label', int(bandwidth))

    with open(output_path, 'w') as f:
        # graph_toolsにDOTファイル全体の生成を任せる
        # g_finalのノードIDは可能な限り整数になっているため、
        # export_dotはそれらを引用符で囲まず、期待される形式に一致する
        f.write(g_final.export_dot())

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description="BW-COARSE: Max-Min公平性を考慮した帯域構造保存型グラフ粗視化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True, help='入力グラフのDOTファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力する粗視化グラフのDOTファイルパス')
    parser.add_argument('-f', '--flows', required=True, help='フロー帯域データのファイルパス')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='粗視化率 (0 < alpha <= 1)')
    parser.add_argument('--embedding_dim', type=int, default=16, help='GCNが生成する埋め込みベクトルの次元数')
    parser.add_argument('--hidden_dim', type=int, default=32, help='GCN中間層の次元数')
    parser.add_argument('--bottleneck_penalty', type=float, default=10000, help='ボトルネックリンクの集約を防ぐためのペナルティ係数。大きいほど効果が強い。')
    args = parser.parse_args()    

    if not (0 < args.alpha <= 1):
        parser.error("粗視化率 alpha は 0 より大きく 1 以下の値でなければなりません。")

    print("🔷 BW-COARSE プログラム開始 🔷")
    
    g_tools_orig, node_map = import_graph(args.input)
    
    g_coarse = coarsen_graph_by_clustering(g_tools_orig, node_map, args)
    
    write_coarse_graph_dot(g_coarse, g_tools_orig, args.output)
    
    print("-" * 50)
    print("サマリー:")
    print(f"  元のノード数: {g_tools_orig.nvertices()}")
    print(f"  粗視化後のノード数: {g_coarse.nvertices()}")
    print(f"  ファイル '{args.output}' が作成されました。")
    print("-" * 50)

if __name__ == "__main__":
    main()