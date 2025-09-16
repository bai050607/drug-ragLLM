from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any
import os
from llama_index.core import Document, VectorStoreIndex
from qianwen_class import QianwenEmbedding, QianwenLLM


class DrugGraph:
    def __init__(
        self,
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
        self._query_engine = None  # 延迟初始化 


    def _get_query_engine(self):
        """获取或创建查询引擎（基于向量索引，延迟初始化）。"""
        if self._query_engine is None:
            qianwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            qianwen_api_key = os.getenv("DASHSCOPE_API_KEY")
            llm = QianwenLLM(
                api_key=qianwen_api_key,
                api_base=qianwen_api_base,
                model="qwen-turbo",
                temperature=0.0,
            )
            embed_model = QianwenEmbedding(
                api_key=qianwen_api_key,
                api_base=qianwen_api_base,
                embed_dim=256,
            )

            # 从 Neo4j 拉取节点，构造文档
            docs: List[Document] = []
            with self.driver.session() as session:
                # 可根据需要调整 LIMIT 或拼接更多属性
                cypher = (
                    "MATCH (n) WHERE n.name IS NOT NULL "
                    "RETURN labels(n) AS labels, n.name AS name LIMIT 1000"
                )
                for rec in session.run(cypher):
                    labels = rec["labels"] or []
                    name = rec["name"]
                    text = f"名称：{name}；标签：{', '.join(labels)}"
                    docs.append(Document(text=text, metadata={"name": name, "labels": labels}))

            if not docs:
                raise RuntimeError("未从 Neo4j 拉取到任何节点，无法构建向量索引")

            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            self._query_engine = index.as_query_engine(llm=llm)
        return self._query_engine

    def retrieve_medical_info(self, medical_text: str) -> str:
        """基于病历文本检索相关医疗信息"""
        try:
            query_engine = self._get_query_engine()
            query = f"{medical_text}"
            print(query)
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
    #     """分层嵌入策略：为不同实体类型生成不同质量的嵌入向量（写回节点属性）。"""
    #     qianwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #     qianwen_api_key = os.getenv("DASHSCOPE_API_KEY")
    #     embedder = QianwenEmbedding(api_key=qianwen_api_key, api_base=qianwen_api_base, embed_dim=1024)

    #     # 1) Disease：用完整描述生成较丰富的向量并写回 full_description
    #     disease_query = (
    #         """
    #     MATCH (n:Disease)
    #     WHERE n.desc IS NOT NULL AND n.desc <> '' AND n.embedding IS NULL
    #     RETURN n
    #         """
    #     )
    #     updated = 0
    #     skipped = 0
    #     with self.driver.session() as session:
    #         disease_records = session.run(disease_query).data()
    #         total = len(disease_records)
    #         print(f"[EMB] Disease 需要处理: {total} 个（跳过已有嵌入的节点）")
    #         for idx, record in enumerate(disease_records, start=1):
    #             try:
    #                 node = record["n"]
    #                 node_name = node.get("name", "")
                    
    #                 # 再次检查是否已有嵌入（双重保险）
    #                 if node.get("embedding") is not None:
    #                     skipped += 1
    #                     print(f"[EMB] Disease 跳过已有嵌入: {node_name}")
    #                     continue
                    
    #                 description_parts: List[str] = []
    #                 if "desc" in node:
    #                     description_parts.append(f"疾病描述：{node['desc']}")
    #                 if "cause" in node:
    #                     description_parts.append(f"病因：{node['cause']}")
    #                 if "prevent" in node:
    #                     description_parts.append(f"预防：{node['prevent']}")
    #                 if "cure_lasttime" in node:
    #                     description_parts.append(f"治愈时间：{node['cure_lasttime']}")
    #                 if "cured_prob" in node:
    #                     description_parts.append(f"治愈概率：{node['cured_prob']}")
    #                 if "easy_get" in node:
    #                     description_parts.append(f"易感人群：{node['easy_get']}")
    #                 full_description = "；".join(description_parts) if description_parts else node_name

    #                 vector = embedder.get_text_embedding(full_description)
    #                 update_query = (
    #                     """
    #                 MATCH (n:Disease {name: $name})
    #                 SET n.embedding = $embedding,
    #                     n.full_description = $full_description
    #                     """
    #                 )
    #                 session.run(update_query, name=node_name, embedding=vector, full_description=full_description)
    #                 updated += 1
    #                 print(f"[EMB] Disease 进度: {idx}/{total}")
    #             except Exception as e:
    #                 print(f"处理 Disease 实体时出错: {e}")
    #                 continue
    #     print(f"[EMB] Disease 向量写回完成，共 {updated} 个，跳过 {skipped} 个")

    #     # 2) 其他实体：用较简单的文本
    #     remaining_entities = ["Drug", "Symptom", "Food", "Check", "Cure", "Producer", "Department"]
    #     for entity_type in remaining_entities:
    #         with self.driver.session() as session:
    #             records = session.run(f"MATCH (n:{entity_type}) WHERE n.embedding IS NULL RETURN n").data()
    #             total = len(records)
    #             print(f"[EMB] {entity_type} 需要处理: {total} 个（跳过已有嵌入的节点）")
    #             updated = 0
    #             skipped = 0
    #             for idx, record in enumerate(records, start=1):
    #                 try:
    #                     node = record["n"]
    #                     node_name = node.get("name", "")
                        
    #                     # 再次检查是否已有嵌入（双重保险）
    #                     if node.get("embedding") is not None:
    #                         skipped += 1
    #                         print(f"[EMB] {entity_type} 跳过已有嵌入: {node_name}")
    #                         continue
                        
    #                     simple_description = f"{entity_type}：{node_name}"
    #                     vector = embedder.get_text_embedding(simple_description)
    #                     update_query = (
    #                         f"MATCH (n:{entity_type}) WHERE n.name = $name "
    #                         "SET n.embedding = $embedding, n.simple_description = $simple_description"
    #                     )
    #                     session.run(update_query, name=node_name, embedding=vector, simple_description=simple_description)
    #                     updated += 1
    #                     print(f"[EMB] {entity_type} 进度: {idx}/{total}")
    #                 except Exception as e:
    #                     print(f"处理 {entity_type} 实体时出错: {e}")
    #                     continue
    #         print(f"[EMB] {entity_type} 向量写回完成，共 {updated} 个，跳过 {skipped} 个")
        
