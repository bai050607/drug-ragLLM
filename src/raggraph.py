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
                query += f"知识库检索内容：{retrieved_info}\n"
            response = self.llm.complete(query)
            text = str(response)
            text = chunk_text(text)
            text = self.filter_to_candidates(text)
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

    def filter_to_candidates(self, text: str) -> str:
        """检测并过滤：将 text（应为JSON数组字符串）中的药物名限制在候选集合内，去重后返回JSON数组字符串。"""
        try:
            parsed = json.loads(text)
        except Exception:
            return json.dumps([], ensure_ascii=False)
        if not isinstance(parsed, list):
            return json.dumps([], ensure_ascii=False)
        filtered: List[str] = []
        seen: Set[str] = set()
        for x in parsed:
            name = str(x).strip()
            if not name:
                continue
            if self.candidate_names and name in self.candidate_names and name not in seen:
                filtered.append(name)
                seen.add(name)
        return json.dumps(filtered, ensure_ascii=False)

    def prime_model_with_rules(self, include_full_list: bool = False, max_names: int = 200) -> None:
        """在批量开始前，先发送一次提示词，告知规则与候选集合。"""
        print("🔄 正在预热模型...")
        # 直接使用 PROMPT，不再做占位符替换
        resp = self.llm.complete(PROMPT)
        tt = str(resp)
        tt = chunk_text(tt)
        print(tt)
        print("✅ 模型预热完成")
        
