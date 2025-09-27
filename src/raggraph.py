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

    def ask_query_prompt(self, content: str) -> str:
        """使用单一字符串 content 填充 query_prompt 中的全部占位符并询问 LLM，返回清洗后的回答。"""
        from prompt import query_prompt  # 按需导入
        tpl = str(query_prompt)
        val = "" if content is None else str(content)
        keys = [
            "gender", "bmi", "treatment_process", "admission_status",
            "current_illness", "past_history", "chief_complaint", "discharge_diagnosis",
        ]
        for k in keys:
            tpl = tpl.replace(f"{{{k}}}", val)
            tpl = tpl.replace(f"{{{{{k}}}}}", val)
        resp = self.llm.complete(tpl)
        return chunk_text(str(resp))

    def query_medical_advice(self, medical_text: str, retrieved_info: Optional[str] = None) -> str:
        """基于病历文本生成医疗建议：将 PROMPT 内嵌并用 text1/text2 填充。"""
        try:
            prompt_text = str(PROMPT)
            t1 = "" if medical_text is None else str(medical_text)
            t2 = "" if retrieved_info is None else str(retrieved_info)
            # 支持 {text1}/{text2} 与 {{text1}}/{{text2}}
            for k, v in (("text1", t1), ("text2", t2)):
                prompt_text = prompt_text.replace(f"{{{k}}}", v)
                prompt_text = prompt_text.replace(f"{{{{{k}}}}}", v)
            response = self.llm.complete(prompt_text)
            text = chunk_text(str(response))
            # 解析模型输出为列表
            drugs: List[str] = []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "药物推荐" in parsed:
                    # 处理 {"药物推荐": ["药物1", "药物2"]} 格式
                    drug_list = parsed["药物推荐"]
                    if isinstance(drug_list, list):
                        drugs = [str(x).strip() for x in drug_list if isinstance(x, (str, int, float))]
                elif isinstance(parsed, list):
                    # 处理 ["药物1", "药物2"] 格式
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

    def retrieve_medical_info(self, query_text: str) -> str:
        """调用 Neo4j 管理器的检索函数获取相关医疗信息。"""
        return self.neo4j_manager.retrieve_medical_info(query_text)
