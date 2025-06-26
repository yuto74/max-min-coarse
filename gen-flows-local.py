#!/usr/bin/env python3
#
# gen-flows-local.py: トポロジー的に局所的なフロー要求を生成する
#
# このスクリプトは、グラフ構造に基づき、近距離のノードペア間のフロー要求を生成します。
# ランダムウォークを用いて、始点ノードから数ホップ内のノードを終点とすることで、
# トポロジーと整合性の取れた、より現実的なトラフィックパターンを作成します。
#
# 使い方:
# ./gen-flows-local.py [グラフファイル.dot] [生成するフロー数] > flows.txt
#
import sys
import random
import argparse

# ユーザーがアップロードした graph_tools をインポート
import graph_tools

def random_walk(graph, start_node, max_length):
    """
    指定された開始ノードからランダムウォークを実行し、終点を返す。

    Args:
        graph (graph_tools.Graph): グラフオブジェクト。
        start_node: ウォークを開始するノード。
        max_length (int): ウォークの最大長。

    Returns:
        ウォークの終点ノード。
    """
    current_node = start_node
    # 1からmax_lengthまでのランダムなウォーク長を決定
    walk_length = random.randint(1, max_length)

    for _ in range(walk_length):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            # 行き止まりの場合はそこでウォークを終了
            break
        # 次のノードを隣人からランダムに選択
        current_node = random.choice(neighbors)
    
    return current_node

def generate_local_flows(graph, num_flows, max_walk_length):
    """
    ランダムウォークに基づいて局所的なフロー要求を生成する。

    Args:
        graph (graph_tools.Graph): グラフオブジェクト。
        num_flows (int): 生成するフローの総数。
        max_walk_length (int): ランダムウォークの最大長。

    Returns:
        (送信元, 受信先) のタプルのリスト。
    """
    flows = []
    all_nodes = list(graph.vertices())
    
    if not all_nodes:
        return []

    print(f"# {num_flows}個の局所的なフロー要求を生成中 (最大ウォーク長: {max_walk_length})...", file=sys.stderr)

    while len(flows) < num_flows:
        # ランダムに始点を選択
        start_node = random.choice(all_nodes)
        
        # ランダムウォークを実行して終点を決定
        end_node = random_walk(graph, start_node, max_walk_length)

        # 始点と終点が同じ場合は再試行
        if start_node == end_node:
            continue
        
        flows.append((start_node, end_node))
    
    print(f"# 生成完了。", file=sys.stderr)
    return flows

def main():
    """メインの実行ブロック"""
    parser = argparse.ArgumentParser(
        description="ランダムウォークに基づき、グラフのトポロジーを考慮した局所的なフロー要求を生成します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "graph_file",
        help="入力グラフのDOTファイルパス。"
    )
    parser.add_argument(
        "num_flows",
        type=int,
        help="生成するフロー要求の総数。"
    )
    parser.add_argument(
        "-l", "--max_length",
        type=int,
        default=5,
        help="ランダムウォークの最大長（ホップ数）。デフォルト: 5"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="乱数シードを整数で指定します。"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # graph_tools を使ってグラフを読み込む
    try:
        g = graph_tools.Graph(directed=False)
        with open(args.graph_file) as f:
            lines = f.readlines()
            g.import_dot(lines)
    except FileNotFoundError:
        print(f"エラー: グラフファイル '{args.graph_file}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    if g.nvertices() == 0:
        print("エラー: グラフにノードが存在しません。", file=sys.stderr)
        sys.exit(1)

    # フローを生成
    generated_flows = generate_local_flows(g, args.num_flows, args.max_length)

    # 結果を標準出力に表示 (s_t_flow.py の入力として使える形式)
    for s, t in generated_flows:
        print(s, t)

if __name__ == "__main__":
    main()
