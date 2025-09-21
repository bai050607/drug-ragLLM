import re

def remove_think_blocks(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 去除所有 <think>...</think>（跨行、非贪婪）
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # 去除首尾空白与回车
    return cleaned.strip()