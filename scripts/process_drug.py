#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# 确保可以导入 src
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from util import preprocess_medical_text, render_query_prompt  # noqa: E402


def main():
    # 固定输入/输出路径
    in_path = "/home/lx/drug-ragLLM/data/CDrugRed_test-A.jsonl"
    out_path = "/home/lx/drug-ragLLM/outputs/queries.txt"

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"未找到输入文件: {in_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    ok = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                prompt_text = preprocess_medical_text(line)
                query = render_query_prompt(prompt_text)
                fout.write((query or "") + "\n")
                ok += 1
            except Exception:
                fout.write("\n")

    print(f"完成：共处理 {total} 行，成功生成 {ok} 条query。输出: {out_path}")


if __name__ == "__main__":
    main()
