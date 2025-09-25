"""
用于从已存储的向量数据库中检索信息的脚本。
"""
import sys
from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from wap.vector_retriver import VectorDatabaseFactory

# 确保可以从父目录导入模块
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))


def search_drug_info(query: str, top_k: int = 3):
    """
    加载向量数据库并根据给定的查询进行搜索。

    Args:
        query (str): 要搜索的查询字符串。
        top_k (int): 要返回的最相关结果的数量。
    """
    # 1. 设置与存储时相同的参数
    VECTOR_DB_PATH = "./chroma_store"
    COLLECTION_NAME = "drug_info"
    EMBEDDING_MODEL = "bge-m3"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # 2. 初始化嵌入模型
    print("初始化嵌入模型...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        print(f"❌ 初始化嵌入模型失败: {e}")
        print("请确保Ollama服务正在运行并且可以访问。")
        return

    # 3. 创建向量数据库实例以加载现有数据库
    print(f"正在加载向量数据库 from {VECTOR_DB_PATH}...")
    vector_db = VectorDatabaseFactory.create(
        embeddings=embeddings,
        vector_db_path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    # 检查数据库是否成功加载
    stats = vector_db.get_stats()
    if stats.get("status") != "initialized" or stats.get("documents_count", 0) == 0:
        print("❌ 数据库未初始化或为空。请先运行脚本将数据存入数据库。")
        return
    
    print(f"数据库加载成功。包含 {stats['documents_count']} 个文档。")

    # 4. 执行搜索
    print(f"\n--- 开始搜索 ---")
    print(f"查询: {query}")
    results = vector_db.search(query, top_k=top_k)

    # 5. 显示结果
    if not results:
        print("未找到相关结果。")
        return

    print(f"找到 {len(results)} 个相关结果:")
    for i, node in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"相似度得分: {node.score:.4f}")
        print("内容:")
        # 打印完整的文档内容
        print(node.node.get_content())


if __name__ == "__main__":
    # 定义您想问的问题
    search_query = "BTK抑制剂适用于哪些病症?"
    
    # 您也可以从命令行接收查询
    # if len(sys.argv) > 1:
    #     search_query = " ".join(sys.argv[1:])
    
    if not search_query:
        print("错误: 请提供一个查询。")
        print("用法: python search_vector_db.py '您的问题'")
    else:
        search_drug_info(search_query)
