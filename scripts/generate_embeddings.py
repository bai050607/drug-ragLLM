
#用于调用函数生成嵌入向量


import os
import sys

# 从 src 目录导入 DrugGraph
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from raggraph import DrugGraph


def main():
    # 可通过环境变量覆盖，或按需改成你的连接参数
    url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")

    # 千问 API Key：QianwenEmbedding 会读取 DASHSCOPE_API_KEY
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[WARN] 未检测到 DASHSCOPE_API_KEY，嵌入请求将失败。请先 export DASHSCOPE_API_KEY=...")

    try:
        dg = DrugGraph(url=url, username=username, password=password)
    except Exception as e:
        print(f"[ERR] 连接 Neo4j 失败: {e}")
        return

    try:
        print("[RUN] 开始为图数据库节点生成/写回嵌入...")
        dg.add_embedding_for_graph()
        print("[DONE] 向量生成流程完成。")
    except Exception as e:
        print(f"[ERR] 生成嵌入过程中出错: {e}")


if __name__ == "__main__":
    main()
