#!/usr/bin/env python3
import sys
import numpy as np

def analyze_differences(filename):
    """
    指定されたファイル（各行に2つの数値）を読み込み、
    数値ペアの差の平均値、最大値、最小値を計算する。

    Args:
        filename (str): 入力ファイルのパス。

    Returns:
        tuple: (平均差、最大差、最小差) のタプル。
               有効なデータがない場合は (None, None, None) を返す。
    """
    differences = []
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f, 1):
                # enumerateを使用して行番号を取得
                try:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        num1 = float(parts[0])
                        num2 = float(parts[1])
                        differences.append(abs(num1 - num2))
                    elif parts: # 空行は無視し、不正な形式の行のみ警告
                        print(f"警告: {i}行目は不正な形式のためスキップします: {line.strip()}", file=sys.stderr)
                except ValueError:
                    print(f"警告: {i}行目に数値変換できないデータが含まれています: {line.strip()}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。", file=sys.stderr)
        return None, None, None
    except IOError as e:
        print(f"エラー: ファイル '{filename}' の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return None, None, None


    if not differences:
        print("警告: ファイル内に有効なデータが見つかりませんでした。", file=sys.stderr)
        return None, None, None

    # numpyを使用して統計量を計算
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    min_diff = np.min(differences)

    return avg_diff, max_diff, min_diff

# スクリプトとして直接実行された場合のみ以下の処理を行う
if __name__ == "__main__":
    # コマンドライン引数のチェック
    if len(sys.argv) != 2:
        # sys.argv[0] にはスクリプト名が入る
        print(f"使用法: python {sys.argv[0]} <入力ファイル名>", file=sys.stderr)
        sys.exit(1)  # エラーコード 1 を返して終了

    # コマンドラインからファイル名を取得
    input_file = sys.argv[1]

    # 関数を呼び出して分析を実行
    average, maximum, minimum = analyze_differences(input_file)

    # # 結果を出力
    if average is not None:
        print(f"ファイル '{input_file}' の分析結果:")
        print(f"  差の平均値: {average}")
        print(f"  差の最大値: {maximum}")
        print(f"  差の最小値: {minimum}")