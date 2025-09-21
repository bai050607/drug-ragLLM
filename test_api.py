import sys
import traceback
# 确保能导入 src 包
sys.path.insert(0, "/home/lx/drug-ragLLM")

from src.qianwen_class import QianwenEmbedding, QianwenLLM

def test_embedding():
    e = QianwenEmbedding()
    try:
        emb = e._get_query_embedding("测试文本用于生成嵌入")
        if isinstance(emb, list):
            print("Embedding 向量长度：", len(emb))
            print("Embedding 向量前10维：", emb[:10])
        else:
            print("Embedding 返回：", emb)
    except Exception as ex:
        print("Embedding 请求出错：", type(ex), ex)
        traceback.print_exc()

def test_llm():
    llm = QianwenLLM()
    try:
        resp = llm.chat([{"role":"user", "content":"请用一句话介绍一下你自己。"}])
        # 尝试打印主要内容
        try:
            content = resp.message.content
        except Exception:
            content = str(resp)
        print("LLM 返回：", content)
    except Exception as ex:
        print("LLM 请求出错：", type(ex), ex)
        traceback.print_exc()

def main():
    print("使用 qianwen_class 测试嵌入与 LLM")
    test_embedding()
    print()
    test_llm()

if __name__ == "__main__":
    main()