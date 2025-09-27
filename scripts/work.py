# 标准库导入
import os
import sys
import json
from typing import TypedDict

# 项目内部导入
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from raggraph import DrugGraph  # noqa: E402
from langgraph.graph import StateGraph, END  # noqa: E402


class MedicalState(TypedDict):
    # TXT 中的一行（作为 ask 的输入）
    query_src: str
    # JSONL 中的一行（作为 advice 的输入）
    json_text: str
    # 生成的 query
    query_text: str
    # 检索到的医疗信息
    retrieved_info: str
    # 生成的建议（JSON字符串）
    advice_json: str


dg = DrugGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")


def ask_query(state: MedicalState) -> dict:
    """使用 TXT 的内容调用 ask 函数，生成 query_text。"""
    src = state.get("query_src", "")
    query_text = dg.ask_query_prompt(src)
    return {"query_text": query_text}


def retrieve_info(state: MedicalState) -> dict:
    """使用生成的 query_text 进行向量检索。"""
    query_text = state.get("query_text", "")
    retrieved_info = dg.retrieve_medical_info(query_text)
    return {"retrieved_info": retrieved_info}


def gen_advice(state: MedicalState) -> dict:
    """使用 JSON 的内容和检索信息生成建议。"""
    json_text = state.get("json_text", "")
    retrieved_info = state.get("retrieved_info", "")
    advice_json = dg.query_medical_advice(json_text, retrieved_info=retrieved_info)
    return {"advice_json": advice_json}


def build_graph() -> StateGraph:
    g = StateGraph(MedicalState)
    g.add_node("ask_query", ask_query)
    g.add_node("retrieve_info", retrieve_info)
    g.add_node("gen_advice", gen_advice)
    g.set_entry_point("ask_query")
    g.add_edge("ask_query", "retrieve_info")
    g.add_edge("retrieve_info", "gen_advice")
    g.add_edge("gen_advice", END)
    return g.compile()


def main():
    txt_path = "/home/lx/drug-ragLLM/outputs/queries.txt"  # TXT：每行作为 ask 的输入
    jsonl_path = "/home/lx/drug-ragLLM/data/CDrugRed_test-A.jsonl"  # JSONL：每行作为 advice 的输入
    out_path = "/home/lx/drug-ragLLM/outputs/results.txt"

    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"未找到输入文件: {txt_path}")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"未找到输入文件: {jsonl_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    graph = build_graph()

    total = 0
    ok = 0
    results = []
    with open(txt_path, "r", encoding="utf-8") as ft, open(jsonl_path, "r", encoding="utf-8") as fj:
        for idx, (t_line, j_line) in enumerate(zip(ft, fj), start=1):
            t_line = t_line.strip()
            j_line = j_line.strip()
            if not t_line or not j_line:
                continue
            total += 1
            initial_state: MedicalState = {
                "query_src": t_line,
                "json_text": j_line,
                "query_text": "",
                "retrieved_info": "",
                "advice_json": "",
            }
            try:
                result = graph.invoke(initial_state)
                advice_json = result.get("advice_json", "[]")
                # 解析药物列表
                try:
                    drugs = json.loads(advice_json)
                    if not isinstance(drugs, list):
                        drugs = []
                except Exception:
                    drugs = []
            except Exception:
                drugs = []
            
            # 从JSONL中提取就诊标识作为case_id
            try:
                json_data = json.loads(j_line)
                case_id = json_data.get("就诊标识", f"line-{idx}")
            except Exception:
                case_id = f"line-{idx}"
            results.append({
                "ID": case_id,
                "prediction": drugs
            })
            ok += 1
            print(f"# 处理 {case_id}: {len(drugs)} 个药物")

    # 输出为JSON格式
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成：共处理 {total} 行，已写出 {ok} 行到 {out_path}")


if __name__ == "__main__":
    main()