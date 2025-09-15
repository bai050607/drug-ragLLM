


#暂时没有用



from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from openai import OpenAI
from raggraph import DrugGraph
import os


class MedicalState(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    medical_text: str
    retrieved_info: str
    medical_advice: str

qianwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 替换为千问的实际API端点
qianwen_api_key = os.getenv("DASHSCOPE_API_KEY")  # 替换为您的千问API密钥
# 初始化LangChain的ChatOpenAI客户端（用于LangGraph）
llm = ChatOpenAI(
    model="qwen-plus",  # 或千问的模型名称
    openai_api_base=qianwen_api_base,
    openai_api_key=qianwen_api_key
)
openai_client = OpenAI(
    api_key=qianwen_api_key,
    base_url=qianwen_api_base
)
drug_graph = DrugGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="12345678",
    llm=OpenAI(
        model="qwen-plus",  # 或千问的模型名称
        api_key=qianwen_api_key,
        api_base=qianwen_api_base
    ),
    embed_model=OpenAIEmbedding(
        api_key=qianwen_api_key,
        api_base=qianwen_api_base
    )
)
graph_builder = StateGraph(MedicalState)

def extract_medical_text(state: MedicalState):
    """提取医疗文本"""
    last_message = state["messages"][-1].content
    return {"medical_text": last_message}

def retrieve_medical_info(state: MedicalState):
    """检索医疗信息"""
    medical_text = state["medical_text"]
    retrieved_info = drug_graph.retrieve_medical_info(medical_text)
    return {"retrieved_info": retrieved_info}

def generate_medical_advice(state: MedicalState):
    """生成医疗建议"""
    medical_text = state["medical_text"]
    medical_advice = drug_graph.query_medical_advice(medical_text)
    return {"medical_advice": medical_advice}

def format_response(state: MedicalState):
    """格式化响应"""
    retrieved_info = state.get("retrieved_info", "无检索信息")
    medical_advice = state.get("medical_advice", "无建议信息")
    
    response = f"""
    检索到的医疗信息:
    {retrieved_info}
    
    医疗建议:
    {medical_advice}
    """
    return {"messages": [HumanMessage(content=response)]}

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

# 使用示例
if __name__ == "__main__":
    # 模拟用户输入
    user_input = "患者有高血压和糖尿病，需要用药建议"
    
    # 初始状态
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # 执行图
    result = medical_graph.invoke(initial_state)
    
    # 输出结果
    print("最终响应:")
    print(result["messages"][-1].content)