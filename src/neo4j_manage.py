from neo4j import GraphDatabase
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from qianwen_class import QianwenEmbedding, QianwenLLM


class Neo4jManager:
    """Neo4j数据库管理器"""
    def __init__(self, url: str, username: str, password: str):
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        self.url = url
        self.username = username
        self.password = password
        self._query_engine = None
        
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

    def _get_query_engine(self):
        """获取或创建查询引擎（基于Neo4j混合搜索，延迟初始化）。"""
        if self._query_engine is None:
            print("🔄 正在初始化Neo4j混合检索引擎...")
            llm, embed_model = QianwenLLM(), QianwenEmbedding()
            neo4j_vector = Neo4jVectorStore(
                url=self.url, username=self.username, password=self.password,
                embedding_dimension=1024, text_node_property="name", embedding_node_property="embedding",
                node_label=["Check","Disease","Drug","Symptom","Food","Department","Cure"], hybrid_search=True
            )
            storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
            try:
                print("💾 正在从 Neo4j 混合向量存储构建索引（免重建）...")
                index = VectorStoreIndex.from_vector_store(neo4j_vector, storage_context, embed_model)
                print("✅ 成功从 Neo4j 混合向量存储构建索引")
            except Exception as e:
                print(f"⚠️ 基于向量存储构建索引失败，将回退全量构建：{e}")
            if index is None:
                print("📊 从 Neo4j 分页拉取全部节点并构建索引...")
                docs, batch_size, total, skip = [], 5000, 0, 0
                with self.driver.session() as session:
                    while True:
                        cypher = (
                            "MATCH (n) WHERE n.name IS NOT NULL "
                            "RETURN n SKIP $skip LIMIT $limit"
                        )
                        records = list(session.run(cypher, skip=skip, limit=batch_size))
                        if not records:
                            break
                        for rec in records:
                            node, properties = rec["n"], dict(node)
                            docs.append(Document(
                                text=f"{properties.get('name','')}，{list(node.labels)}。{properties.get('desc','')}",
                                metadata={"name": properties.get('name'), "labels": list(node.labels)}
                            ))
                        total, skip = total + len(records), skip + batch_size
                        print(f"📥 已加载 {total} 条节点为文档...")
                
                if not docs: raise RuntimeError("未从 Neo4j 拉取到任何节点，无法构建向量索引")
                print(f"📚 全量节点文档数：{len(docs)}，开始构建混合向量索引...")
                index = VectorStoreIndex.from_documents(docs, storage_context, embed_model)
            
            self._query_engine = index.as_query_engine(llm=llm, similarity_top_k=10)
            print("✅ Neo4j混合检索引擎初始化完成")
        return self._query_engine

    def retrieve_medical_info(self, medical_text: str) -> str:
        """基于Neo4j混合搜索检索相关医疗信息"""
        try:
            # 使用Neo4j内置混合搜索（向量+全文+图搜索）
            import re
            cleaned_text = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?()（）【】""''、，。；：！？]', '', str(medical_text))
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            print("length:",len(cleaned_text))
            # 使用Neo4j内置混合搜索（向量+全文+图搜索）
            query_engine = self._get_query_engine()
            retrieved_nodes = query_engine.retriever.retrieve(cleaned_text)
            # 直接返回所有检索到的节点
            result = []
            for node in retrieved_nodes:
                result.append(f"{node.text}")
            print("检索结果:", result)
            return "、".join(result) if result else "未找到相关信息"
        except Exception as e:
            print(f"❌ 检索过程中出错: {e}")
            return f"检索过程中出现错误: {e}"