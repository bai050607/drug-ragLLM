#这个是ai生成的用于使用llamaindex的类

import asyncio
from typing import List, Optional, Generator, Any
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    LLMMetadata,
    ChatResponse,
    ChatMessage,
    MessageRole,
)
from openai import OpenAI as OpenAIClient
import asyncio
import requests
from typing import List, Optional, Generator, Any

class QianwenEmbedding(BaseEmbedding):
    """基于 vLLM bge-m3 嵌入服务的自定义嵌入类。"""
    embed_dim: int = 1024  # bge-m3 的嵌入维度
    api_key: str = "sk-dummy"
    api_base: str = "http://localhost:11434"  # 保留主机:端口

    def __init__(self, api_key: str = "sk-dummy", api_base: str = "http://localhost:11434", embed_dim: int = 1024, **kwargs):
        super().__init__(embed_dim=embed_dim, api_key=api_key, api_base=api_base, **kwargs)

    def _new_client(self) -> OpenAIClient:
        return OpenAIClient(api_key=self.api_key, base_url=self.api_base)

    def _get_query_embedding(self, query: str) -> List[float]:
        # 直接调用完整 embeddings 接口，避免 OpenAIClient 拼接路径带来 404
        url = f"{self.api_base.rstrip('/')}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": "bge-m3", "input": query}
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict) and "data" in j and j["data"]:
            item = j["data"][0]
            return item.get("embedding") or item.get("vector") or item
        if isinstance(j, list) and j and isinstance(j[0], (list, float)):
            return j[0]
        raise ValueError("无法解析嵌入响应: " + str(j))

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.gather(*(self._aget_text_embedding(t) for t in texts))


class QianwenLLM(LLM):
    """基于 vLLM carebot-llama3 服务的自定义 LLM 适配类。"""

    api_key: str = ""  # vLLM 不需要真实的 API key
    api_base: str = "http://localhost:11434/v1"  # 仅主机:端口（不要包含 /v1 或具体路径）
    model: str = "qwq:latest"  # 你的模型名称
    temperature: float = 0.0

    def _new_client(self) -> OpenAIClient:
        return OpenAIClient(api_key=self.api_key, base_url=self.api_base)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=1024,
            is_chat_model=True,
            model_name=self.model,
        )

    # ---- Completion API ----
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        client = self._new_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content
        cr = CompletionResponse(text=text)
        # cr.message = ChatMessage(role=MessageRole.ASSISTANT, content=text)
        return cr

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        cr = await asyncio.to_thread(self.complete, prompt, **kwargs)
        return cr

    # ---- Chat API ----
    def chat(self, messages: List[Any], **kwargs) -> ChatResponse:
        client = self._new_client()
        openai_msgs = []
        for m in messages:
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                role = role or m.get("role")
                content = m.get("content")
            if content:
                openai_msgs.append({"role": role or "user", "content": content})
        if not openai_msgs:
            openai_msgs = [{"role": "user", "content": ""}]
        resp = client.chat.completions.create(
            model=self.model,
            messages=openai_msgs,
            temperature=self.temperature,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    async def achat(self, messages: List[Any], **kwargs) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    # ---- Streaming APIs (简化为非流式fallback) ----
    def stream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        yield await self.acomplete(prompt, **kwargs)

    def stream_chat(self, messages: List[Any], **kwargs) -> Generator[ChatResponse, None, None]:
        yield self.chat(messages, **kwargs)

    async def astream_chat(self, messages: List[Any], **kwargs) -> Generator[ChatResponse, None, None]:
        yield await self.achat(messages, **kwargs)
