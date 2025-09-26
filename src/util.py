import re
from prompt import query_prompt


def remove_think_blocks(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 去除所有 <think>...</think>（跨行、非贪婪）
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # 去除首尾空白与回车
    return cleaned.strip()


def preprocess_medical_text(medical_text: str) -> str:
    """将 medical_text(一行JSON或原文) 解析并填充到 query_prompt 模板，返回完整提示词。"""
    import json
    text = str(medical_text)
    try:
        obj = json.loads(text)
    except Exception:
        obj = {}
    def g(k: str) -> str:
        v = obj.get(k)
        if isinstance(v, list):
            return "、".join(map(str, v))
        return "" if v is None else str(v)
    filled = (
        query_prompt
        .replace("{{gender}}", g("性别"))
        .replace("{{bmi}}", g("BMI"))
        .replace("{{treatment_process}}", g("诊疗过程描述"))
        .replace("{{admission_status}}", g("入院情况"))
        .replace("{{current_illness}}", g("现病史"))
        .replace("{{past_history}}", g("既往史"))
        .replace("{{chief_complaint}}", g("主诉"))
        .replace("{{discharge_diagnosis}}", g("出院诊断"))
    )
    return re.sub(r"\s+", " ", filled).strip()

def render_query_prompt(prompt_text: str) -> str:
    """将参数 prompt_text 交给 QianwenLLM 并返回生成结果（去首尾空白）。"""
    try:
        from qianwen_class import QianwenLLM
        llm = QianwenLLM()
        return (llm.complete(prompt_text).text or "").strip()
    except Exception:
        return ""