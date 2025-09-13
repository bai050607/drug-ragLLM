import os
import requests
import json
from typing import List, Dict, Any

# 这是一个由ai简单写出的代码
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-08fe07510ee848c2a0c85e1d3fc705b1')
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

def send_message(message: str, model: str = "qwen-turbo") -> str:
    """
    发送单条消息给大模型
    
    Args:
        message: 输入消息
        model: 模型名称，默认qwen-turbo
    
    Returns:
        模型回复的文本
    """
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "input": {
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['output']['text']
    
    except Exception as e:
        print(f"API调用失败: {e}")
        return ""

def chat_with_context(messages: List[Dict[str, str]], model: str = "qwen-turbo") -> str:
    """
    带上下文的对话
    
    Args:
        messages: 消息列表，格式 [{"role": "user", "content": "..."}]
        model: 模型名称
    
    Returns:
        模型回复的文本
    """
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "input": {
            "messages": messages
        },
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['output']['text']
    
    except Exception as e:
        print(f"API调用失败: {e}")
        return ""

def get_drug_recommendation(medical_record: str) -> str:
    """
    基于病历推荐药物
    
    Args:
        medical_record: 病历文本
    
    Returns:
        推荐的药物列表
    """
    prompt = f"""
    请根据以下病历信息，推荐患者出院时应携带的药物：
    
    病历信息：
    {medical_record}
    
    请以JSON格式返回推荐结果，包含药物名称和推荐理由。
    """
    
    return send_message(prompt)

# 测试函数
if __name__ == "__main__":    
    prompt = "现在你是一个医生，之后我会提供一些患者的信息，请根据患者信息提供一些治疗糖尿病的药物"
    response = send_message(prompt)
    print(response)
    
    medical_record = "患者，男，65岁，主诉多饮多尿3个月。既往有高血压病史5年。入院检查：空腹血糖12.5mmol/L，餐后2小时血糖18.2mmol/L。诊断：2型糖尿病，高血压病。"
    recommendation = get_drug_recommendation(medical_record)
    print("药物推荐：", recommendation)