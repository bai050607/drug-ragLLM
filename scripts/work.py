# 标准库导入
import os
import sys
import json
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# 项目内部导入
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.environ.setdefault("OPENAI_API_KEY", "dummy_key")

class MedicalState(TypedDict):
    """医疗状态类型定义"""
    messages: Annotated[List[HumanMessage], add_messages]
    medical_text: str
    retrieved_info: str
    medical_advice: str
from raggraph import DrugGraph

def extract_medical_text(state: MedicalState):
    print("[WORK] node: extract_medical_text")
    last_message = state["messages"][-1].content
    return {"medical_text": last_message}

def retrieve_medical_info(state: MedicalState):
    print("[WORK] node: retrieve_medical_info")
    if drug_graph is None:
        return {"retrieved_info": "DrugGraph未正确初始化"}
    medical_text = state["medical_text"]
    return {"retrieved_info": drug_graph.neo4j_manager.retrieve_medical_info(medical_text)}

def generate_medical_advice(state: MedicalState):
    print("[WORK] node: generate_medical_advice")
    if drug_graph is None:
        return {"medical_advice": "DrugGraph未正确初始化"}
    medical_text = state["medical_text"]
    retrieved_info = state.get("retrieved_info")
    return {"medical_advice": drug_graph.query_medical_advice(medical_text, retrieved_info)}

def format_response(state: MedicalState):
    print("[WORK] node: format_response")
    retrieved_info = state.get("retrieved_info", "无检索信息")
    medical_advice = state.get("medical_advice", "无建议信息")
    response = f"""
    检索到的医疗信息:
    {retrieved_info}
    
    医疗建议:
    {medical_advice}
    """
    return {"messages": [HumanMessage(content=response)]}


def create_medical_graph() -> StateGraph:
    """创建医疗处理图"""
    graph_builder = StateGraph(MedicalState)
    
    # 添加节点
    graph_builder.add_node("extract_medical_text", extract_medical_text)
    graph_builder.add_node("retrieve_medical_info", retrieve_medical_info)
    graph_builder.add_node("generate_medical_advice", generate_medical_advice)
    graph_builder.add_node("format_response", format_response)
    
    # 添加边
    graph_builder.set_entry_point("extract_medical_text")
    graph_builder.add_edge("extract_medical_text", "retrieve_medical_info")
    graph_builder.add_edge("retrieve_medical_info", "generate_medical_advice")
    graph_builder.add_edge("generate_medical_advice", "format_response")
    graph_builder.add_edge("format_response", END)
    
    return graph_builder.compile()


# 全局变量初始化
drug_graph = DrugGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
drug_graph.prime_model_with_rules(include_full_list=False, max_names=200)
medical_graph = create_medical_graph()
if __name__ == "__main__":
    input_path = "/home/lx/drug-ragLLM/data/CDrugRed_test-A.jsonl"
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
    
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            # 解析JSON行获取ID和病历文本
            try:
                obj = json.loads(line)
                case_id = obj.get("就诊标识", f"unknown-{idx}")
                
                # 优先使用常见字段，其次回退为整行文本
                message_text = line
                for key in ("medical_text", "text", "query", "input", "诊疗过程描述"):
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        message_text = obj[key].strip()
                        break
            except Exception:
                case_id = f"unknown-{idx}"
                message_text = line
            
            initial_state = {"messages": [HumanMessage(content=message_text)]}
            result = medical_graph.invoke(initial_state)
            
            # 解析医疗建议为药物列表
            medical_advice = result.get("medical_advice", "[]")
            try:
                drugs = json.loads(medical_advice)
                if not isinstance(drugs, list):
                    drugs = []
            except Exception:
                drugs = []
            
            # 添加到结果列表
            results.append({
                "ID": case_id,
                "prediction": drugs
            })
            
            print(f"# 处理 {case_id}: {len(drugs)} 个药物")
    
    # 输出为submit_pred_ex.json格式
    output_path = "/home/lx/drug-ragLLM/outputs/submit_pred.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 处理完成，共 {len(results)} 条记录")
    print(f"结果已保存到: {output_path}")
    drug_graph.neo4j_manager.close()
    print("🔒 数据库连接已关闭")