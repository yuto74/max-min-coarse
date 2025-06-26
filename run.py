import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- GCNモデルの定義 ---
# 論文[1]で言及されている2層GCNを実装
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # GCNの畳み込み層を定義
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 1層目: GCN -> ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 2層目: GCN
        x = self.conv2(x, edge_index)
        return x

# --- 特徴量計算 ---
def calculate_node_features(graph, flows):
    """
    論文[1]で定義された6次元の特徴量を計算する関数
    """
    features = {node: {
        'flow_out_bw': 0, 'flow_in_bw': 0,
        'flow_out_count': 0, 'flow_in_count': 0,
        'passing_through_bw': 0, 'passing_through_count': 0
    } for node in graph.nodes()}

    for flow in flows:
        src, dst, demand = flow['src'], flow['dst'], flow['demand']
        
        # 最短経路を計算
        try:
            path = nx.shortest_path(graph, source=src, target=dst)
        except nx.NetworkXNoPath:
            continue

        # 特徴量の割り当て
        # 始点ノード
        features[src]['flow_out_count'] += 1
        features[src]['flow_out_bw'] += demand
        # 終点ノード
        features[dst]['flow_in_count'] += 1
        features[dst]['flow_in_bw'] += demand
        
        # 中間ノード
        for i in range(1, len(path) - 1):
            node = path[i]
            features[node]['passing_through_count'] += 1
            features[node]['passing_through_bw'] += demand
            
    df = pd.DataFrame.from_dict(features, orient='index')
    
    # 特徴量ごとに最大値で正規化
    # ゼロ除算を避けるため、最大値が0の場合は1に置き換える
    normalized_df = df.copy()
    for col in df.columns:
        max_val = df[col].max()
        if max_val > 0:
            normalized_df[col] = df[col] / max_val
        else:
            normalized_df[col] = 0 # 全て0の場合は0のまま
            
    return normalized_df

# --- 可視化関数群 ---
def display_feature_matrix(features_df):
    """初期特徴量マトリックスを整形して表示"""
    print("--- Initial Node Features (Normalized) ---")
    print(features_df.to_string())
    print("-" * 40)

def plot_embeddings_scatterplot(embeddings, labels, title="Node Embeddings"):
    """GCNによって生成された埋め込みベクトルを2D散布図で可視化"""
    plt.figure(figsize=(8, 8))
    palette = sns.color_palette("bright", n_colors=len(set(labels)))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette=palette, s=200, legend='full')
    for i, label in enumerate(range(embeddings.shape)):
        plt.text(embeddings[i, 0] + 0.01, embeddings[i, 1] + 0.01, str(label))
    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True)
    plt.show()

def plot_graph_with_clusters(graph, cluster_labels, title="Graph with Clusters"):
    """クラスタリング結果をグラフ上で可視化"""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42) # レイアウトを固定
    
    # ノードの色をクラスタごとに設定
    node_colors = [cluster_labels[node] for node in graph.nodes()]
    
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.jet, node_size=800, font_size=12, font_color='white')
    
    # リンク容量をエッジラベルとして表示
    edge_labels = nx.get_edge_attributes(graph, 'capacity')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title(title)
    plt.show()

# --- メイン分析関数 ---
def analyze_gcn_fcc(graph, flows, config):
    """
    GCN-FCCプロセス全体を実行し、中間結果を返すモジュール化された関数
    """
    print(f"--- Starting Analysis for: {config['name']} ---")
    
    # 1. 特徴量計算
    features_df = calculate_node_features(graph, flows)
    
    # 2. GCNによる埋め込みベクトル生成
    # PyTorch Geometric用のデータ形式に変換
    data = from_networkx(graph)
    data.x = torch.tensor(features_df.values, dtype=torch.float)
    
    # モデルの初期化と実行
    model = GCN(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels']
    )
    model.eval() # 訓練はしないので評価モード
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index).numpy()
        
    # 3. k-meansクラスタリング
    kmeans = KMeans(n_clusters=config['k'], random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(node_embeddings)
    
    # 結果を辞書形式で整理
    cluster_labels = {node: label for node, label in zip(graph.nodes(), cluster_ids)}
    
    # 4. 粗視化グラフの構築（本分析では可視化が主目的なので省略可）
    #...
    
    # 分析結果を返す
    analysis_results = {
        "initial_features": features_df,
        "node_embeddings": node_embeddings,
        "cluster_labels": cluster_labels,
        "original_graph": graph
    }
    
    # --- 結果の表示 ---
    print(f"\n[Analysis for {config['name']}]")
    display_feature_matrix(analysis_results['initial_features'])
    
    print(f"\nNode Embeddings (Shape: {analysis_results['node_embeddings'].shape}):")
    print(analysis_results['node_embeddings'])
    
    print(f"\nCluster Assignments:")
    print(analysis_results['cluster_labels'])
    
    plot_embeddings_scatterplot(
        analysis_results['node_embeddings'],
        list(analysis_results['cluster_labels'].values()),
        title=f"Node Embeddings for {config['name']} (k={config['k']})"
    )
    
    plot_graph_with_clusters(
        analysis_results['original_graph'],
        analysis_results['cluster_labels'],
        title=f"Clustering Result for {config['name']} (k={config['k']})"
    )
    
    return analysis_results

# --- メイン実行ブロック ---
if __name__ == '__main__':
    # --- テストケース1: ハブ＆スポークグラフ ---
    G_hub_spoke = nx.Graph()
    hub_spoke_edges = [
        (0,1,1000), (0,2,1000), (0,3,1000), (0,4,1000),
        (5,6,1000), (5,7,1000), (5,8,1000), (5,9,1000),
        (0,5,1000)
    ]
    for u, v, cap in hub_spoke_edges:
        G_hub_spoke.add_edge(u, v, capacity=cap)
        
    hub_spoke_flows = [
        {'src': 1, 'dst': 6, 'demand': 100},
        {'src': 2, 'dst': 7, 'demand': 100},
        {'src': 3, 'dst': 8, 'demand': 100},
        {'src': 4, 'dst': 9, 'demand': 100},
    ]
    
    hub_spoke_config = {
        'name': "Hub-and-Spoke",
        'in_channels': 6,
        'hidden_channels': 4, # 小規模グラフ用に縮小
        'out_channels': 2,    # 2次元で可視化するため
        'k': 3                # ハブクラスタ + 2つのスポーククラスタ
    }
    
    analyze_gcn_fcc(G_hub_spoke, hub_spoke_flows, hub_spoke_config)

    # --- テストケース2: バーベル・ボトルネックグラフ ---
    G_barbell = nx.Graph()
    barbell_edges = [
        (0,1,1000), (0,2,1000), (1,3,1000), (2,3,1000),
        (6,7,1000), (6,8,1000), (7,9,1000), (8,9,1000),
        (3,4,1000), (4,5,10), (5,6,1000) # ボトルネックリンク
    ]
    for u, v, cap in barbell_edges:
        G_barbell.add_edge(u, v, capacity=cap)

    barbell_flows = [
        {'src': 0, 'dst': 7, 'demand': 100},
        {'src': 1, 'dst': 8, 'demand': 100},
        {'src': 2, 'dst': 9, 'demand': 100},
    ]
    
    barbell_config = {
        'name': "Barbell-Bottleneck",
        'in_channels': 6,
        'hidden_channels': 4,
        'out_channels': 2,
        'k': 3 # 2つのクラスタ + ボトルネック部分でどうなるか
    }
    
    analyze_gcn_fcc(G_barbell, barbell_flows, barbell_config)