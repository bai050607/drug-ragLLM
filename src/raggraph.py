from typing import List, Optional, Set
import os
import json
from qianwen_class import QianwenLLM
from prompt import recommend_prompt as PROMPT
from util import remove_think_blocks as chunk_text
from neo4j_manage import Neo4jManager

class DrugGraph:
    def __init__(
        self,
        url: str = "bolt://localhost:7687",  # Neo4j数据库的URL
        username: str = "neo4j",  # Neo4j数据库的用户名
        password: str = "12345678",  # Neo4j数据库的密码
    ):
        # 使用Neo4jManager来管理数据库连接和查询
        self.neo4j_manager = Neo4jManager(url, username, password)
        self.url = url
        self.username = username
        self.password = password
        self.llm = QianwenLLM()
        self.candidate_names = self._load_candidate_names()

    def query_medical_advice(self, medical_text: str, retrieved_info: Optional[str] = None) -> str:
        """基于病历文本生成医疗建议"""
        try:
            query = "请根据以下信息做出药物推荐，返回值仅为由药物组成的列表。\n"
            # 极简提示词（减少 token）：仅约束输出与必要上下文
            query += "病例信息：{medical_text}\n"
            if retrieved_info:
                query += f"知识库检索可能节点：{retrieved_info}\n"
            response = self.llm.complete(query)
            text = str(response)
            text = chunk_text(text)
            print(text)
            # 解析模型输出为列表
            drugs: List[str] = []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    drugs = [str(x).strip() for x in parsed if isinstance(x, (str, int, float))]
            except Exception:
                drugs = []
            # 过滤至候选集合（若候选集合可用）
            if self.candidate_names:
                filtered: List[str] = []
                seen: Set[str] = set()
                for name in drugs:
                    if name in self.candidate_names and name not in seen:
                        filtered.append(name)
                        seen.add(name)
                return json.dumps(filtered, ensure_ascii=False)
            else:
                # 若没有候选集合可用，直接回传原始 JSON（或空数组）
                return json.dumps(drugs, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 查询过程中出错: {e}")
            return f"抱歉，查询过程中出现错误: {e}"

    def _load_candidate_names(self) -> Set[str]:
        """加载候选药物集合。"""
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
        return names


    def prime_model_with_rules(self, include_full_list: bool = False, max_names: int = 200) -> None:
        """在批量开始前，先发送一次提示词，告知规则与候选集合。

        include_full_list: 是否在提示中包含完整候选清单（可能较长）。
        max_names: 若不包含全量，则示例前 N 项，控制 token。
        """
        print("🔄 正在预热模型...")
        names = list(self.candidate_names)
        names.sort()
        if include_full_list:
            listing = "、".join(names)
        else:
            listing = "、".join(names[:max_names])
        prompt = PROMPT.replace("{list}", listing)
        resp = self.llm.complete(prompt)
        tt = str(resp)
        tt = chunk_text(tt)
        print(tt)
        print("✅ 模型预热完成")
        # except Exception as e:
        #     print(f"⚠️ 模型预热失败: {e}")

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
        
