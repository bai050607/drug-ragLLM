import os
import openai
from langgraph.graph import StateGraph, END, START, Send, State

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    #生成查询
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # 使用 OpenAI API 生成查询
    current_date = get_current_date()
    formatted_prompt = "...待填充"
    
    # 设置 OpenAI API Key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # 调用 OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=1.0,
        max_tokens=1000
    )
    
    result_text = response.choices[0].message.content
    # 解析结果并返回查询列表
    return {"search_query": parse_query_result(result_text)}

def continue_to_web_research(state: QueryGenerationState):
    #继续思考
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]

def web_research(state: OverallState):
    # 网络研究
    search_query = state.get("search_query", "未知查询")
    research_result = perform_web_search(search_query)
    return {"web_research_result": [research_result]}

def reflection(state: OverallState):
    # 反思和评估
    research_results = state.get("web_research_result", [])
    reflection_result = evaluate_research_sufficiency(research_results)
    return reflection_result

def finalize_answer(state: OverallState):
    # 生成最终答案
    research_results = state.get("web_research_result", [])
    final_answer = generate_final_answer(research_results, state)
    return {"final_answer": final_answer}



builder = StateGraph(OverallState, config_schema=Configuration)


builder.add_node("generate_query",generate_query)
builder.add_node("web_research",web_research)
builder.add_node("reflection",reflection)
builder.add_node("finalize_answer",finalize_answer)


builder.add_edge(START,"generate_query")
builder.add_conditional_edges("generate_query",continue_to_web_research,["web_research"])
builder.add_edge("web_research","reflection")
builder.add_edge("reflection","finalize_answer")
builder.add_edge("finalize_answer", END)
graph=builder.compile(name="drug-ragLLM")