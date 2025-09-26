from neo4j import GraphDatabase
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from qianwen_class import QianwenEmbedding, QianwenLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from wap.vector_retriver import VectorDatabaseFactory


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
        """获取或创建查询引擎（基于Neo4j混合搜索，延迟初始化）"""
        if self._query_engine is None:
            print("🔄 正在构建向量索引...")
            llm, embed_model = QianwenLLM(), QianwenEmbedding()
            index = None
            try:
                print("💾 尝试从 Neo4j 混合向量存储构建索引...")
                neo4j_vector = Neo4jVectorStore(
                    url=self.url, username=self.username, password=self.password,
                    embedding_dimension=1024, text_node_property="name", embedding_node_property="embedding",
                    hybrid_search=True
                )
                storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
                print("💾 正在从 Neo4j 混合向量存储构建索引（免重建）...")
                index = VectorStoreIndex.from_vector_store(neo4j_vector, storage_context, embed_model)
                print("✅ 成功从 Neo4j 混合向量存储构建索引")
            except Exception as e:
                print(f"⚠️ 基于向量存储构建索引失败，将回退全量构建：{e}")
            
            # 如果Neo4jVectorStore失败，回退到直接构建
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
                
                if not docs: 
                    raise RuntimeError("未从 Neo4j 拉取到任何节点，无法构建向量索引")
                print(f"📚 全量节点文档数：{len(docs)}，开始构建混合向量索引...")
                index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            
            self._query_engine = index
            print("✅ Neo4j混合检索引擎初始化完成")
        return self._query_engine
        
    def retrieve_medical_info(self, medical_text: str) -> str:
        """基于外部向量库（Chroma + LlamaIndex）检索相关医疗信息（直接使用 medical_text 作为查询）。"""
        try:
            # 1) 直接将 medical_text 作为查询语句
            query_text = str(medical_text).strip()
            if not query_text:
                return ""
            # 2) 初始化外部向量数据库（需与入库时配置一致）
            VECTOR_DB_PATH = "./chroma_store"
            COLLECTION_NAME = "drug_info"
            EMBEDDING_MODEL = "bge-m3"
            OLLAMA_BASE_URL = "http://localhost:11434"
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vector_db = VectorDatabaseFactory.create(
                embeddings=embeddings,
                vector_db_path=VECTOR_DB_PATH,
                collection_name=COLLECTION_NAME,
            )
            stats = vector_db.get_stats()
            if stats.get("status") != "initialized" or stats.get("documents_count", 0) == 0:
                return "向量库为空或未初始化"
            # 3) 执行检索
            nodes = vector_db.search(query_text, top_k=10)
            if not nodes:
                return "未找到相关信息"
            results = [n.node.get_content() for n in nodes if getattr(n, 'node', None)]
            return "\n".join(results)
        except Exception as e:
            print(f"❌ 检索过程中出错: {e}")
            return f"检索过程中出现错误: {e}"