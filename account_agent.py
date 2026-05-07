# account_agent.py

import os
from typing import Any

from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import (  # pyright: ignore[reportMissingImports]
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from account_tools import account_lc_tools


class AccountAgent:
    """账号数据 Agent：多轮对话 + 仅 MySQL 只读工具。"""

    def __init__(self, model_name: str, api_key: str):
        _timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        _retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0,
            request_timeout=_timeout,
            max_retries=_retries,
        )
        self.llm_tools = self.llm.bind_tools(account_lc_tools)
        self._tool_by_name = {t.name: t for t in account_lc_tools}
        try:
            self._max_agent_steps = int((os.getenv("AGENT_MAX_ITERATIONS") or "12").strip())
        except ValueError:
            self._max_agent_steps = 12
        self._max_agent_steps = max(1, min(self._max_agent_steps, 50))

        self._long_term_memory = ""
        # 最近一次工具调用中解析到的 UID（供编排器跨轮 composite，无需依赖助手复述）
        self.last_resolved_player_uid: str | None = None
        self.messages: list = [
            SystemMessage(content=self.build_system_prompt()),
        ]

    def set_long_term_memory(self, text: str | None) -> None:
        self._long_term_memory = (text or "").strip()

    def clear_history(self) -> None:
        self.last_resolved_player_uid = None
        self.messages = [SystemMessage(content=self.build_system_prompt())]
        if self._long_term_memory:
            self.messages.append(
                SystemMessage(
                    content=(
                        "【过往会话摘要（提纲，仅供参考）】\n"
                        "以下为以往对话压缩后的要点；查询数据库时仍以工具返回为准。\n"
                        f"{self._long_term_memory}"
                    )
                )
            )

    def build_system_prompt(self) -> str:
        default_uid = (os.getenv("DEFAULT_PLAYER_UID") or "").strip()
        uid_hint = (
            f"当前默认游戏 UID（users.uid / characters.uid，用户未显式提供时可使用）：{default_uid}"
            if default_uid
            else "未配置 DEFAULT_PLAYER_UID：请向用户询问或让其提供游戏 UID（users.uid）。"
        )
        lines = [
            "你是游戏「账号与角色养成数据」查询助手。回答简洁、结构化；只能依据工具返回的数据陈述事实，禁止编造库存、体力或角色信息。",
            "",
            uid_hint,
            "",
            "【工具】",
        ]
        for t in account_lc_tools:
            desc = (t.description or "").strip().replace("\n", " ")
            lines.append(f"- {t.name}: {desc}")
        lines.extend(
            [
                "",
                "【原则】",
                "1. 需要某玩家数据时，优先使用 get_player_profile；需要角色列表与培养细节时用 list_player_characters。",
                "2. 仅有 UID 或培养目标片段时用 search_players_by_name_keyword，再根据返回的 uid 继续查询。",
                "3. 若数据库未配置或查询为空，如实说明，不要臆测。",
                "4. 若用户只想做纯算术、不涉及账号或数据库，说明此类请求应由攻略助手（计算器工具）处理，不要编造玩家数据。",
            ]
        )
        return "\n".join(lines)

    def trim_messages(self, max_len: int = 24):
        prefix_n = 2 if self._long_term_memory else 1
        if len(self.messages) <= max_len:
            return
        head = self.messages[:prefix_n]
        tail_n = max_len - prefix_n
        if tail_n < 1:
            tail_n = 1
        tail = self.messages[-tail_n:]
        self.messages = head + tail

    def _invoke_tool(self, name: str, args: dict[str, Any]) -> str:
        print(f"[账号Tool] {name} 参数: {args}")
        if name in ("get_player_profile", "list_player_characters"):
            pu = (args.get("player_uid") or "").strip()
            if pu:
                self.last_resolved_player_uid = pu
        tool = self._tool_by_name.get(name)
        if tool is None:
            return f"未知工具: {name}"
        try:
            out = tool.invoke(args)
        except Exception as e:
            out = f"工具执行失败: {e}"
        print(f"[账号Tool返回] {out}")
        return str(out)

    def _run_tool_loop(self) -> str:
        steps = 0
        while steps < self._max_agent_steps:
            steps += 1
            print(f"[账号LLM] 第 {steps} 步...")
            response = self.llm_tools.invoke(self.messages)
            print("[账号LLM] 响应完成")

            if not isinstance(response, AIMessage):
                self.messages.append(response)
                return getattr(response, "content", "") or ""

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                self.messages.append(response)
                return (response.content or "").strip()

            self.messages.append(response)

            for tc in tool_calls:
                if isinstance(tc, dict):
                    tid = tc.get("id") or ""
                    name = tc.get("name") or ""
                    args = tc.get("args") or {}
                else:
                    tid = getattr(tc, "id", "") or ""
                    name = getattr(tc, "name", "") or ""
                    args = getattr(tc, "args", None) or {}

                payload = self._invoke_tool(name, args)
                self.messages.append(ToolMessage(content=payload, tool_call_id=tid))

            self.trim_messages()

        print("[账号Agent] 已达步数上限，强制收束...")
        self.messages.append(
            HumanMessage(
                content="工具调用次数已达上限。请仅根据已有内容输出最终中文回答，不要再调用工具。"
            )
        )
        final = self.llm.invoke(self.messages)
        self.messages.append(final)
        return (final.content or "").strip()

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()
        return self._run_tool_loop()
