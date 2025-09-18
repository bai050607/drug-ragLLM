from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any, Set
import os
import json
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
        self._candidate_names: Optional[Set[str]] = None


    def _get_query_engine(self):
        """获取或创建查询引擎（基于向量索引，延迟初始化）。"""
        if self._query_engine is None:
            print("🔄 正在初始化向量检索引擎...")
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
            print("📊 正在从 Neo4j 拉取节点...")
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
            
            print(f"📚 已拉取 {len(docs)} 个节点，正在构建向量索引...")

            if not docs:
                raise RuntimeError("未从 Neo4j 拉取到任何节点，无法构建向量索引")

            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            self._query_engine = index.as_query_engine(llm=llm)
            print("✅ 向量检索引擎初始化完成")
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

    def query_medical_advice(self, medical_text: str, retrieved_info: Optional[str] = None) -> str:
        """基于病历文本生成医疗建议"""
        try:
            query_engine = self._get_query_engine()
            # 加载（缓存）候选药物集合
            candidate_names = self._load_candidate_names()

            # 极简提示词（减少 token）：仅约束输出与必要上下文
            query = "仅输出候选药物内的中文通用名JSON数组。无则返回[]。\n"
            query += f"病历：{medical_text}\n"
            if retrieved_info:
                query += f"检索：{retrieved_info}\n"
            response = query_engine.query(query)
            text = str(response)

            # 解析模型输出为列表
            drugs: List[str] = []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    drugs = [str(x).strip() for x in parsed if isinstance(x, (str, int, float))]
            except Exception:
                drugs = []

            # 过滤至候选集合（若候选集合可用）
            if candidate_names:
                filtered: List[str] = []
                seen: Set[str] = set()
                for name in drugs:
                    if name in candidate_names and name not in seen:
                        filtered.append(name)
                        seen.add(name)
                return json.dumps(filtered, ensure_ascii=False)
            else:
                # 若没有候选集合可用，直接回传原始 JSON（或空数组）
                return json.dumps(drugs, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 查询过程中出错: {e}")
            return f"抱歉，查询过程中出现错误: {e}"

    def _load_candidate_names(self, force_reload: bool = False) -> Set[str]:
        """加载并缓存候选药物集合。"""
        if self._candidate_names is not None and not force_reload:
            return self._candidate_names
        candidates_path = os.getenv(
            "CANDIDATE_DRUGS_JSON",
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data", "候选药物列表.json"),
        )
        names: Set[str] = set()
        if os.path.isfile(candidates_path):
            try:
                with open(candidates_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        names = {str(x).strip() for x in data if str(x).strip()}
            except Exception:
                names = set()
        self._candidate_names = names
        return names


    def prime_model_with_rules(self, include_full_list: bool = False, max_names: int = 200) -> None:
        """在批量开始前，先发送一次提示词，告知规则与候选集合。

        include_full_list: 是否在提示中包含完整候选清单（可能较长）。
        max_names: 若不包含全量，则示例前 N 项，控制 token。
        """
        print("🔄 正在预热模型...")
        query_engine = self._get_query_engine()
        names = list(self._load_candidate_names())
        names.sort()
        if include_full_list:
            listing = "、".join(names)
        else:
            listing = "、".join(names[:max_names])
        prompt = (
            "你是一个专业的医疗用药推荐系统。接下来的每个问题我都会给你一段病历描述和检索出的相关医疗信息，"
            "你需要根据这些内容，从以下候选药物集合中仔细选择需要使用的药物。\n\n"
            "重要要求：\n"
            "1. 只能从候选药物集合中选择，不能推荐集合外的任何药物\n"
            "2. 必须仔细分析病历中的症状、疾病、检查结果等信息\n"
            "3. 必须结合检索出的相关医疗信息进行综合判断\n"
            "4. 只返回药物名称的列表，格式为JSON数组，如：[\"药物1\", \"药物2\"]\n"
            "5. 如果根据病历和检索信息无法确定需要任何药物，则返回空数组：[]\n"
            "6. 不要输出任何解释、剂量、用法、适应症等额外信息\n"
            "7. 不要输出任何非药物名称的内容\n"
            "8. 药物名称必须与候选集合中的名称完全一致\n\n"
            f"候选药物集合（共{len(names)}种药物）：{listing}\n\n"
            "请严格按照以上要求执行，确保输出的准确性和一致性。"
        )
        try:
            _ = query_engine.query(prompt)
            print("✅ 模型预热完成")
        except Exception as e:
            print(f"⚠️ 模型预热失败: {e}")

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
        
