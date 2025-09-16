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


class QianwenEmbedding(BaseEmbedding):
    """基于阿里云 DashScope OpenAI 兼容接口的自定义嵌入类。
    使用 text-embedding-v4，支持 dimensions（默认 256）。
    同步与异步接口均实现，兼容 LlamaIndex BaseEmbedding 抽象。
    """

    embed_dim: int = 256
    api_key: str
    api_base: str

    def __init__(self, api_key: str, api_base: str, embed_dim: int = 256, **kwargs):
        super().__init__(embed_dim=embed_dim, api_key=api_key, api_base=api_base, **kwargs)

    def _new_client(self) -> OpenAIClient:
        return OpenAIClient(api_key=self.api_key, base_url=self.api_base)

    def _get_query_embedding(self, query: str) -> List[float]:
        client = self._new_client()
        resp = client.embeddings.create(
            model="text-embedding-v3",
            input=query,
            dimensions=self.embed_dim,
            encoding_format="float",
        )
        return resp.data[0].embedding

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
    """基于阿里云 DashScope OpenAI 兼容接口的自定义 LLM 适配类。"""

    api_key: str
    api_base: str
    model: str = "qwen-plus"
    temperature: float = 0.0

    def _new_client(self) -> OpenAIClient:
        return OpenAIClient(api_key=self.api_key, base_url=self.api_base)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=256,
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
            max_tokens=256,
        )
        text = resp.choices[0].message.content
        cr = CompletionResponse(text=text)
        cr.message = ChatMessage(role=MessageRole.ASSISTANT, content=text)
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
            max_tokens=256,
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
