"""
向量数据库管理模块

提供向量数据库的统一管理接口，支持多种向量数据库后端。
"""

from pathlib import Path
import sys
import os
from typing import List, Optional, Dict, Any, Union
import asyncio

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


from llama_index.core import Document
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class VectorDatabase:
    """向量数据库管理类"""
    
    def __init__(self, 
                 embeddings=None,
                 vector_db_path: str = None,
                 collection_name: str = None):
        self.embeddings = embeddings 
        self.vector_db_path = vector_db_path 
        self.collection_name = collection_name 
        self.client = None
        self.vector_store = None
        self.index = None
        
        # 初始化向量数据库
        self._init_vector_database()
    
    def _init_vector_database(self):
        """初始化向量数据库"""
        try:
            # 创建ChromaDB客户端
            self.client = chromadb.PersistentClient(path=self.vector_db_path)
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 创建向量存储
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            # 创建存储上下文
            try:
                self.storage_context = StorageContext.from_defaults(
                    persist_dir=self.vector_db_path,
                    vector_store=self.vector_store
                )
            except FileNotFoundError:
                # 如果持久化目录不存在，创建新的存储上下文
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
                self.storage_context.persist(persist_dir=self.vector_db_path)
            
            # 创建向量索引
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embeddings
            )
            
            print(f"✅ 成功连接到向量数据库: {self.vector_db_path}")
            print(f"📊 集合 '{self.collection_name}' 包含 {self.collection.count()} 个条目")
            
        except Exception as e:
            print(f"❌ 初始化向量数据库失败: {e}")
            self.client = None
            self.vector_store = None
            self.index = None
    
    def add_documents(self, documents: List[Document]) -> bool:
        """添加文档到向量数据库"""
        if not self.index:
            print("❌ 向量数据库未初始化")
            return False
        
        try:
            # 将文档添加到索引
            for doc in documents:
                self.index.insert(doc)
            
            # 持久化存储
            self.storage_context.persist(persist_dir=self.vector_db_path)
            
            print(f"✅ 成功添加 {len(documents)} 个文档到向量数据库")
            return True
            
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """从向量数据库删除文档"""
        if not self.index:
            print("❌ 向量数据库未初始化")
            return False
        
        try:
            # 删除文档
            for doc_id in doc_ids:
                self.index.delete_ref_doc(doc_id)
            
            # 持久化存储
            self.storage_context.persist(persist_dir=self.vector_db_path)
            
            print(f"✅ 成功删除 {len(doc_ids)} 个文档")
            return True
            
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[NodeWithScore]:
        """在向量数据库中搜索"""
        if not self.index:
            print("❌ 向量数据库未初始化")
            return []
        
        top_k = top_k 
        
        try:
            # 创建检索器
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            
            # 执行检索
            nodes = retriever.retrieve(query)
            
            return nodes
            
        except Exception as e:
            print(f"❌ 向量数据库搜索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        if not self.client:
            return {"status": "not_initialized"}
        
        try:
            return {
                "status": "initialized",
                "vector_db_path": self.vector_db_path,
                "collection_name": self.collection_name,
                "documents_count": self.collection.count(),
                "embeddings_model": self.embeddings.__class__.__name__
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_database(self) -> bool:
        """清空向量数据库"""
        if not self.client:
            print("❌ 向量数据库未初始化")
            return False
        
        try:
            # 删除集合
            self.client.delete_collection(name=self.collection_name)
            
            # 重新创建集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 重新初始化索引
            self._init_vector_database()
            
            print("✅ 成功清空向量数据库")
            return True
            
        except Exception as e:
            print(f"❌ 清空数据库失败: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """备份向量数据库"""
        if not self.client:
            print("❌ 向量数据库未初始化")
            return False
        
        try:
            import shutil
            shutil.copytree(self.vector_db_path, backup_path)
            print(f"✅ 成功备份数据库到: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ 备份数据库失败: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """从备份恢复向量数据库"""
        try:
            import shutil
            # 清空当前数据库
            if os.path.exists(self.vector_db_path):
                shutil.rmtree(self.vector_db_path)
            
            # 恢复备份
            shutil.copytree(backup_path, self.vector_db_path)
            
            # 重新初始化
            self._init_vector_database()
            
            print(f"✅ 成功从备份恢复数据库: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ 恢复数据库失败: {e}")
            return False


class VectorDatabaseFactory:
    """向量数据库工厂"""
    
    @staticmethod
    def create(embeddings=None,
               vector_db_path: str = None,
               collection_name: str = None) -> VectorDatabase:
        """创建向量数据库实例"""
        return VectorDatabase(
            embeddings=embeddings,
            vector_db_path=vector_db_path,
            collection_name=collection_name
        )


# 使用示例
if __name__ == "__main__":
    import json
    from llama_index.core import Document
    from langchain_ollama.embeddings import OllamaEmbeddings

    # 1. 设置参数
    JSON_FILE_PATH = "/home/lx/drug-ragLLM/merged_20250923_195353.json"
    VECTOR_DB_PATH = "./chroma_store"
    COLLECTION_NAME = "drug_info"
    EMBEDDING_MODEL = "bge-m3"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # 2. 创建向量数据库实例
    print("初始化嵌入模型...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    print(f"创建或连接到向量数据库 at {VECTOR_DB_PATH} with collection {COLLECTION_NAME}...")
    vector_db = VectorDatabaseFactory.create(
        embeddings=embeddings,
        vector_db_path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    # 3. 清空数据库 (可选, 确保从一个干净的状态开始)
    print("清空数据库...")
    vector_db.clear_database()

    # 4. 加载并处理JSON数据
    print(f"从 {JSON_FILE_PATH} 加载数据...")
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条记录。")
    except Exception as e:
        print(f"❌ 加载JSON文件失败: {e}")
        sys.exit(1)

    # 5. 将JSON数据转换为Document对象
    print("正在将数据转换为Document对象...")
    documents = []
    # 扩展后的字段中文名称映射
    field_map = {
        "drug_name": "药物名称",
        "suitable_for": "适用人群",
        "contraindications": "禁忌症",
        "treats": "治疗病症",
        "symptoms": "相关症状",
        "common_adverse_reactions": "常见不良反应",
        "serious_adverse_reactions": "严重不良反应",
        "side_effects": "副作用",
        "drug_interactions": "药物相互作用",
        "interactions": "相互作用",
        "precautions": "注意事项",
        "pharmacological_effects": "药理作用",
        "dosage_and_administration": "用法用量",
        "dosage": "剂量",
        "special_populations": "特殊人群",
        "storage": "储存方法"
    }

    for item in data:
        # 构建描述性段落
        text_parts = []
        # 使用 field_map 的顺序来构建文本，以获得更一致的输出
        for key, readable_name in field_map.items():
            value = item.get(key)
            if value:
                # 如果值是列表，用逗号连接
                if isinstance(value, list):
                    value_str = "、".join(map(str, value))
                else:
                    value_str = str(value)
                
                if value_str:
                    text_parts.append(f"{readable_name}为“{value_str}”")

        text_content = "；".join(text_parts) + "。"
        
        # 使用 drug_name 作为文档ID
        doc_id = item.get("drug_name")
        
        # 创建Document时不再传递metadata
        documents.append(Document(text=text_content, doc_id=doc_id))
    
    print(f"成功创建 {len(documents)} 个Document对象。")

    # 6. 添加文档到向量数据库
    print("添加文档到向量数据库...")
    vector_db.add_documents(documents)
    
    # 7. 获取统计信息并验证
    stats = vector_db.get_stats()
    print(f"向量数据库统计: {stats}")
    
    # 8. 测试搜索
    if stats.get("documents_count", 0) > 0:
        print("\n--- 测试搜索 ---")
        query = "BTK抑制剂适用于哪些病症?"
        print(f"查询: {query}")
        results = vector_db.search(query, top_k=3)
        
        print(f"搜索到 {len(results)} 个结果:")
        for i, node in enumerate(results):
            print(f"\n结果 {i+1} (得分: {node.score:.3f}):")
            print(node.node.get_content()[:300] + "...")
    else:
        print("数据库为空，跳过测试搜索。")