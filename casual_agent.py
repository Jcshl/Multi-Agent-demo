# casual_agent.py

import os

from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]


class CasualAgent:
    """闲聊 Agent：无业务工具，仅自然对话。"""

    def __init__(self, model_name: str, api_key: str):
        _timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        _retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0.7,
            request_timeout=_timeout,
            max_retries=_retries,
        )
        self.messages: list = [
            SystemMessage(
                content=(
                    "你是一个友好、轻松的闲聊伙伴。可以聊日常、心情、趣味话题；"
                    "不要冒充游戏数据库或攻略权威；若用户问起专业攻略或账号数据，"
                    "简短说明这类问题更适合对应的专门助手即可，不必展开编造。"
                )
            ),
        ]

    def clear_history(self) -> None:
        self.messages = [
            SystemMessage(
                content=(
                    "你是一个友好、轻松的闲聊伙伴。可以聊日常、心情、趣味话题；"
                    "不要冒充游戏数据库或攻略权威；若用户问起专业攻略或账号数据，"
                    "简短说明这类问题更适合对应的专门助手即可，不必展开编造。"
                )
            ),
        ]

    def trim_messages(self, max_len: int = 40):
        if len(self.messages) > max_len:
            self.messages = [self.messages[0]] + self.messages[-max_len + 1 :]

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()
        response = self.llm.invoke(self.messages)
        if isinstance(response, AIMessage):
            self.messages.append(response)
            return (response.content or "").strip()
        self.messages.append(response)
        return getattr(response, "content", "") or ""
