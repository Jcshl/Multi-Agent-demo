# chatbot.py

import os
from typing import Any

from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import (  # pyright: ignore[reportMissingImports]
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tools import lc_tools


class ChatBot:
    """
    ChatBot：多轮对话 + 原生 tool calling + 工具循环（ReAct 风格上限）
    """

    def __init__(self, model_name: str, api_key: str):
        """
        初始化对话机器人与工具绑定。

        参数:
            model_name: 大模型名称（从 .env 的 MODEL_NAME 读取后传入）。
            api_key: 模型服务 API Key（从 .env 的 SILICONFLOW_API_KEY 读取后传入）。
        """
        # LLM 调用超时（秒），默认 120，可被环境变量覆盖。
        _timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        # LLM 网络重试次数，默认 2。
        _retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        # 基础模型客户端（不含工具执行逻辑）。
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0,
            request_timeout=_timeout,
            max_retries=_retries,
        )
        # 绑定工具后的模型客户端：可返回 tool_calls。
        self.llm_tools = self.llm.bind_tools(lc_tools)
        # 按工具名建立索引，便于根据模型返回的 name 快速调用工具。
        self._tool_by_name = {t.name: t for t in lc_tools}
        try:
            # Agent 工具循环最大步数，避免无限循环。
            self._max_agent_steps = int((os.getenv("AGENT_MAX_ITERATIONS") or "12").strip())
        except ValueError:
            self._max_agent_steps = 12
        self._max_agent_steps = max(1, min(self._max_agent_steps, 50))

        # 跨会话注入的「提纲式」长期记忆（由编排器从 SQLite 加载）。
        self._long_term_memory = ""
        # 多轮对话历史；首条固定为系统提示词。
        self.messages: list = [
            SystemMessage(content=self.build_system_prompt()),
        ]

    def set_long_term_memory(self, text: str | None) -> None:
        """设置提纲摘要文本；空字符串表示不注入。"""
        self._long_term_memory = (text or "").strip()

    def clear_history(self) -> None:
        """清空多轮对话，仅保留 system prompt（用于网页端「新对话」）。"""
        self.messages = [SystemMessage(content=self.build_system_prompt())]
        if self._long_term_memory:
            self.messages.append(
                SystemMessage(
                    content=(
                        "【过往会话摘要（提纲，仅供参考）】\n"
                        "以下为以往对话压缩后的要点，细节可能已省略；本轮请以用户原话与当前检索为准。\n"
                        f"{self._long_term_memory}"
                    )
                )
            )

    def build_system_prompt(self) -> str:
        """构造系统提示词，并动态附加当前可用工具说明。"""
        lines = [
            "你是「原神深渊与养成」方向的助手，语气清晰、友好，回答尽量结构化（可分点、小标题），避免无根据的臆测。",
            "",
            "【任务与工具使用原则】",
            "1. 深渊相关（含深境螺旋、12层、配队、敌人机制、打法要点、环境Buff、版本环境等）：以本地知识库为准。请先使用 rag_search，用简短、可检索的中文关键词构造 query；若结果不足可换关键词再次 rag_search；不得编造知识库中未出现的关键机制或数值。",
            "2. 凡用户要求**得出具体数值**的算式（含四则、幂、括号、百分数转算、伤害/词条等「脱手算式」），**必须先调用 calculator** 再作答；将用户式子整理为仅含数字与 + - * / // % ** 和括号的合法表达式传入，禁止心算或口答替代工具结果。",
            "3. 泛化搜索：在 rag_search 仍不足且用户需要更广信息时可使用 search；若与深渊攻略冲突，以 rag_search 为准。",
            "4. 玩家「当前账号」信息（角色持有、库存等）：若下列工具列表中无对应工具，请直接说明暂不支持查库，严禁捏造数据。",
            "",
            "【当前可用工具】",
        ]
        for t in lc_tools:
            desc = (t.description or "").strip().replace("\n", " ")
            lines.append(f"- {t.name}: {desc}")
        return "\n".join(lines)

    def trim_messages(self, max_len: int = 24):
        """保留 system（及可选摘要条）+ 最近若干条消息（工具循环会较快占满上下文）。"""
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
        """
        按工具名执行一次工具调用并返回文本结果。

        参数:
            name: 工具名称（由模型 tool_calls 提供）。
            args: 工具参数字典。
        """
        print(f"[Tool调用] {name} 参数: {args}")
        tool = self._tool_by_name.get(name)
        if tool is None:
            return f"未知工具: {name}"
        try:
            out = tool.invoke(args)
        except Exception as e:
            out = f"工具执行失败: {e}"
        print(f"[Tool返回] {out}")
        return str(out)

    def _run_tool_loop(self) -> str:
        """在已有 self.messages 末尾为用户轮次的前提下，多轮 tool 直至无 tool_calls 或达上限。"""
        # 当前轮已执行的模型步骤数（每次 llm_tools.invoke 算一步）。
        steps = 0
        while steps < self._max_agent_steps:
            steps += 1
            print(f"[LLM] 正在请求模型（第 {steps} 步）...")
            # response 可能是普通文本回复，也可能包含 tool_calls。
            response = self.llm_tools.invoke(self.messages)
            print("[LLM] 响应完成")

            if not isinstance(response, AIMessage):
                self.messages.append(response)
                return getattr(response, "content", "") or ""

            # 从 AIMessage 中提取工具调用列表；无工具调用时为空列表。
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                self.messages.append(response)
                return (response.content or "").strip()

            self.messages.append(response)

            for tc in tool_calls:
                if isinstance(tc, dict):
                    # 工具调用唯一 ID：用于把 ToolMessage 回填给对应调用。
                    tid = tc.get("id") or ""
                    # 工具名称，例如 rag_search / calculator。
                    name = tc.get("name") or ""
                    # 工具参数字典。
                    args = tc.get("args") or {}
                else:
                    tid = getattr(tc, "id", "") or ""
                    name = getattr(tc, "name", "") or ""
                    args = getattr(tc, "args", None) or {}

                # 执行工具并把结果以 ToolMessage 形式回填到上下文。
                payload = self._invoke_tool(name, args)
                self.messages.append(ToolMessage(content=payload, tool_call_id=tid))

            self.trim_messages()

        # 步数用尽：在不绑定工具的情况下强制收束为自然语言回答
        print("[Agent] 已达 AGENT_MAX_ITERATIONS，请求最终回答（不再调用工具）...")
        self.messages.append(
            HumanMessage(
                content="工具调用次数已达上限。请仅根据已有对话与工具返回，直接给出最终中文回答，不要再调用任何工具。"
            )
        )
        # 达到上限后改用不带工具的 llm 强制收束答案。
        final = self.llm.invoke(self.messages)
        self.messages.append(final)
        return (final.content or "").strip()

    def chat(self, user_input: str) -> str:
        """
        对外主入口：接收用户输入并返回最终回复文本。

        参数:
            user_input: 用户本轮输入内容。
        """
        # 把用户输入写入历史，供模型读取上下文。
        self.messages.append(HumanMessage(content=user_input))
        self.trim_messages()

        reply = self._run_tool_loop()
        return reply
