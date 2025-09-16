from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import os
from neo4j import GraphDatabase
import os

# 导入DrugGraph类
from raggraph import DrugGraph
os.environ.setdefault("OPENAI_API_KEY", "dummy_key")

class MedicalState(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    medical_text: str
    retrieved_info: str
    medical_advice: str


def extract_medical_text(state: MedicalState):
    print("[WORK] node: extract_medical_text")
    last_message = state["messages"][-1].content
    return {"medical_text": last_message}

def retrieve_medical_info(state: MedicalState):
    print("[WORK] node: retrieve_medical_info")
    if drug_graph is None:
        return {"retrieved_info": "DrugGraph未正确初始化"}
    medical_text = state["medical_text"]
    return {"retrieved_info": drug_graph.retrieve_medical_info(medical_text)}

def generate_medical_advice(state: MedicalState):
    print("[WORK] node: generate_medical_advice")
    if drug_graph is None:
        return {"medical_advice": "DrugGraph未正确初始化"}
    medical_text = state["medical_text"]
    return {"medical_advice": drug_graph.query_medical_advice(medical_text)}

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

# 不再直接使用 langchain_openai / openai 客户端；DrugGraph 内部已集成千问 LLM 与 Embedding
drug_graph = DrugGraph(url="bolt://localhost:7687",username="neo4j",password="12345678")

graph_builder = StateGraph(MedicalState)

graph_builder.add_node("extract_medical_text", extract_medical_text)
graph_builder.add_node("retrieve_medical_info", retrieve_medical_info)
graph_builder.add_node("generate_medical_advice", generate_medical_advice)
graph_builder.add_node("format_response", format_response)

graph_builder.set_entry_point("extract_medical_text")
graph_builder.add_edge("extract_medical_text", "retrieve_medical_info")
graph_builder.add_edge("retrieve_medical_info", "generate_medical_advice")
graph_builder.add_edge("generate_medical_advice", "format_response")
graph_builder.add_edge("format_response", END)
medical_graph = graph_builder.compile()
if __name__ == "__main__":
    user_input = "患者有高血压和糖尿病，需要用药建议"
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = medical_graph.invoke(initial_state)
    print(result["messages"][-1].content)