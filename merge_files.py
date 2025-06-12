#!/usr/bin/python3

import sys

def merge_files(output_file, *input_files):
    with open(output_file, 'w') as outfile:
        for i, fname in enumerate(input_files):
            try:
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())
                    # 最後のファイルでなければ改行を追加
                    if i < len(input_files) - 1:
                        outfile.write("\n\n")  # 各ファイルの間に2つの改行を追加
            except FileNotFoundError:
                print(f"File not found: {fname}")
            except Exception as e:
                print(f"An error occurred while reading {fname}: {e}")

    print(f"Files merged into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_files.py <output_file> <input_file1> <input_file2> [... <input_fileN>]")
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    merge_files(output_file, *input_files)