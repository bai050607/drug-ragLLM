from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any
import os
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.query_engine.knowledge_graph_rag_query_engine import KnowledgeGraphRAGQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

class DrugGraph:
    def __init__(
        self,
        llm: OpenAI,  # 语言模型实例
        embed_model: OpenAIEmbedding,  # 嵌入模型实例
        url: str = "bolt://localhost:7687",  # Neo4j数据库的URL
        username: str = "neo4j",  # Neo4j数据库的用户名
        password: str = "12345678",  # Neo4j数据库的密码
    ):
        self.driver = GraphDatabase.driver(
            url, auth=(username, password)
        )  # 创建Neo4j数据库驱动
        self.url = url
        self.username = username
        self.password = password
        self.llm = llm
        self.embed_model = embed_model
        self._query_engine = None  # 延迟初始化 


    
    def _get_query_engine(self):
        """获取或创建查询引擎（延迟初始化）"""
        if self._query_engine is None:
            graph_store = Neo4jGraphStore(
                username=self.username,
                password=self.password,
                url=self.url
            )
            self._query_engine = KnowledgeGraphRAGQueryEngine(
                graph_store=graph_store,
                llm=self.llm,
                embed_model=self.embed_model
            )
        return self._query_engine

    def retrieve_medical_info(self, medical_text: str) -> str:
        """基于病历文本检索相关医疗信息"""
        try:
            query_engine = self._get_query_engine()
            query = f"根据以下病历信息，检索相关的医疗实体和关系：{medical_text}"
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            print(f"❌ 检索过程中出错: {e}")
            return f"检索过程中出现错误: {e}"

    def query_medical_advice(self, medical_text: str) -> str:
        """基于病历文本生成医疗建议"""
        try:
            query_engine = self._get_query_engine()
            query = f"根据以下病历信息，推荐合适的药物和治疗方案：{medical_text}"
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            print(f"❌ 查询过程中出错: {e}")
            return f"抱歉，查询过程中出现错误: {e}"

    # def add_embedding_for_graph(self):
    #     """分层嵌入策略：为不同实体类型生成不同质量的嵌入向量"""
    #     disease_query = """
    #     MATCH (n:Disease)
    #     WHERE n.desc IS NOT NULL AND n.desc <> ''
    #     RETURN n
    #     """
    #     with self.driver.session() as session:
    #         result = session.run(disease_query) 
    #         for record in result:
    #             try:
    #                 node = record["n"]
    #                 node_name = node.get("name", "")
    #                 description_parts = []  # 构建丰富的描述文本
    #                 description_parts.append(f"疾病描述：{node['desc']}")
    #                 description_parts.append(f"病因：{node['cause']}")
    #                 description_parts.append(f"预防：{node['prevent']}")
    #                 description_parts.append(f"治愈时间：{node['cure_lasttime']}")
    #                 description_parts.append(f"治愈概率：{node['cured_prob']}")
    #                 description_parts.append(f"易感人群：{node['easy_get']}")
    #                 full_description = "；".join(description_parts)
    #                 embedding = self.embedding.get_emb(full_description)  # 生成嵌入向量
    #                 update_query = """
    #                 MATCH (n:Disease {name: $name})
    #                 SET n.embedding = $embedding,
    #                     n.full_description = $full_description
    #                 """
    #                 session.run(update_query, 
    #                            name=node_name, 
    #                            embedding=embedding,
    #                            full_description=full_description)                        
    #             except Exception as e:
    #                 print(f"处理Disease实体时出错: {e}")
    #                 continue
        
    #     remaining_entities = ['Drug', 'Symptom', 'Food', 'Check', 'Cure', 'Producer', 'Department']
    #     for entity_type in remaining_entities:
    #         query = f"""
    #         MATCH (n:{entity_type})
    #         RETURN n
    #         """
    #         with self.driver.session() as session:
    #             result = session.run(query)
    #             for record in result:
    #                 try:
    #                     node = record["n"]
    #                     node_name = node.get("name", "")
    #                     simple_description = f"{entity_type}：{node_name}"
    #                     embedding = self.embedding.get_emb(simple_description)
    #                     update_query = f"""
    #                     MATCH (n:{entity_type} {{name: $name}})
    #                     SET n.embedding = $embedding,
    #                         n.simple_description = $simple_description
    #                     """
    #                     session.run(update_query, 
    #                                name=node_name, 
    #                                embedding=embedding,
    #                                simple_description=simple_description)
    #                 except Exception as e:
    #                     print(f"处理{entity_type}实体时出错: {e}")
    #                     continue