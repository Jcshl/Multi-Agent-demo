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
        self._long_term_memory = ""
        self._base_system_content = (
            "你是一个友好、轻松的闲聊伙伴。可以聊日常、心情、趣味话题；"
            "不要冒充游戏数据库或攻略权威；若用户问起专业攻略或账号数据，"
            "简短说明这类问题更适合对应的专门助手即可，不必展开编造。\n"
            "若用户消息中包含「本会话此前轮次」，那是本对话中真实发生过的跨助手记录，"
            "用户追问「刚才问过什么」时应依据该段如实概括，不要说看不到历史。"
        )
        self.messages: list = [
            SystemMessage(content=self._base_system_content),
        ]

    def set_long_term_memory(self, text: str | None) -> None:
        self._long_term_memory = (text or "").strip()

    def clear_history(self) -> None:
        self.messages = [SystemMessage(content=self._base_system_content)]
        if self._long_term_memory:
            self.messages.append(
                SystemMessage(
                    content=(
                        "【过往会话摘要（提纲，仅供参考）】\n"
                        f"{self._long_term_memory}"
                    )
                )
            )

    def trim_messages(self, max_len: int = 40):
        prefix_n = 2 if self._long_term_memory else 1
        if len(self.messages) <= max_len:
            return
        head = self.messages[:prefix_n]
        tail_n = max_len - prefix_n
        if tail_n < 1:
            tail_n = 1
        tail = self.messages[-tail_n:]
        self.messages = head + tail

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()
        response = self.llm.invoke(self.messages)
        if isinstance(response, AIMessage):
            self.messages.append(response)
            return (response.content or "").strip()
        self.messages.append(response)
        return getattr(response, "content", "") or ""
