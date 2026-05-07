# orchestrator.py
"""多 Agent 编排：路由后委托「攻略」「账号库」「闲聊」 specialist；支持复合问题多路调用与合并。"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]

from account_agent import AccountAgent
from casual_agent import CasualAgent
from chatbot import ChatBot

# 路由决策的四种结果：game 攻略 | account 查库 | casual 闲聊 | composite 攻略+查库 双路
Intent = Literal["game", "account", "casual", "composite"]


class MultiAgentOrchestrator:
    """
    对外接口与原先 ChatBot.chat 一致；内部按意图分发。
    原「原神深渊与养成」能力仍由 ChatBot 承担，逻辑未改。
    """

    def __init__(self, model_name: str, api_key: str):
        # 与三个 specialist 共用的请求参数（SiliconFlow OpenAI 兼容接口）
        _timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        _retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self._model_name = model_name
        self._api_key = api_key
        # 专用「编排大脑」：意图分类、复合问题拆分、最终合并回答（temperature=0 求稳）
        self.router_llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0,
            request_timeout=_timeout,
            max_retries=_retries,
        )
        # 三个专家实例：各自维护多轮消息历史；composite 时会并行调用其中两个
        self.game_bot = ChatBot(model_name=model_name, api_key=api_key)
        self.account_bot = AccountAgent(model_name=model_name, api_key=api_key)
        self.casual_bot = CasualAgent(model_name=model_name, api_key=api_key)
        # 本会话最近解析到的游戏 UID（用于跨轮：下文只说「他/上文」仍可走账号库 + 攻略）
        self._last_uid: str | None = None

    def clear_history(self) -> None:
        # 网页「新对话」等场景：清空三个 Agent 的对话，避免串上下文
        self.game_bot.clear_history()
        self.account_bot.clear_history()
        self.casual_bot.clear_history()
        self._last_uid = None

    def _signals_game(self, t: str) -> bool:
        # 浅层关键词命中：用于启发式路由与 LLM 路由结果的纠偏（非穷尽游戏语义，只求常见攻略问法）
        game_kw = (
            "深渊",
            "配队",
            "攻略",
            "12层",
            "十一层",
            "十二层",
            "深境",
            "螺旋",
            "boss",
            "Boss",
            "BOSS",
            "机制",
            "伤害",
            "蕴光",
            "守宫",
            "层",
            "打法",
            "弱点",
            "抗性",
            "元素",
            "圣遗物",
            "武器",
            "天赋",
            "命座",
        )
        return any(k in t for k in game_kw)

    def _has_uid(self, t: str) -> bool:
        """是否出现游戏 UID（显式 uid 或长数字）。"""
        return self._extract_uid(t) is not None

    def _extract_uid(self, t: str) -> str | None:
        """从用户句中提取 UID：`uid12345` / `UID: 10001` / 独立长数字串。"""
        uid_m = re.search(r"(?:uid|UID)[为：:\s]*(\d{4,})", t)
        if uid_m:
            return uid_m.group(1)
        num_m = re.search(r"(?<![\d.])(\d{9,})(?!\d)", t)
        if num_m:
            return num_m.group(1)
        return None

    def _cross_turn_composite_hint(self, text: str) -> bool:
        """
        当前句未写 UID，但明显要结合「上文提到的账号/角色」与攻略时，
        若会话里已有 _last_uid，则应抬升为 composite（避免只走 ChatBot 丢库侧上下文）。
        """
        if not self._last_uid:
            return False
        t = text.strip()
        if not self._signals_game(t):
            return False
        refs = (
            "他",
            "她",
            "该用户",
            "该玩家",
            "上文",
            "刚才",
            "之前",
            "前面",
            "这个玩家",
            "此人",
            "其角色",
            "其账号",
            "用户信息",
            "角色培养",
            "培养列表",
            "账号信息",
            "拥有的角色",
            "持有角色",
            "结合账号",
            "结合用户",
        )
        return any(r in t for r in refs)

    def _game_only_fallback(self, t: str) -> str:
        """复合句里攻略侧子问为空时，去掉明显账号侧措辞再交给 ChatBot。"""
        s = t.strip()
        for phrase in (
            "根据他的用户信息",
            "根据她的用户信息",
            "结合用户信息和",
            "结合角色培养和",
            "结合账号",
            "角色培养列表详情",
            "角色培养列表",
            "用户信息和",
            "账号数据",
        ):
            s = s.replace(phrase, "")
        s = re.sub(r"[，,、]{2,}", "，", s)
        s = re.sub(r"\s+", " ", s).strip(" ，、")
        return s if s else "请根据用户描述的关卡或 Boss 给出攻略要点。"

    def _signals_account_strict(self, t: str) -> bool:
        """查库/账号侧（避免「深渊带什么角色」等泛问误判为账号）。"""
        if self._has_uid(t):
            return True
        if any(
            k in t
            for k in (
                "数据库",
                "mysql",
                "MySQL",
                "树脂",
                "培养目标",
                "我的数据",
                "玩家信息",
                "拥有角色",
                "角色列表",
            )
        ):
            return True
        # 「有什么/哪些角色」需配合用户/账号语境或 UID，避免纯攻略向泛问
        if ("用户有什么角色" in t) or (
            any(p in t for p in ("有什么角色", "哪些角色"))
            and ("用户" in t or "账号" in t)
        ):
            return True
        return False

    def _route(self, user_input: str) -> Intent:
        # MULTI_AGENT_ROUTE_MODE：可强制单一路径或切到纯启发式，默认 llm 分类
        raw = (os.getenv("MULTI_AGENT_ROUTE_MODE") or "llm").strip().lower()
        if raw == "game":
            return "game"
        if raw == "account":
            return "account"
        if raw == "casual":
            return "casual"
        if raw == "heuristic":
            return self._route_heuristic(user_input)
        return self._route_llm(user_input)

    def _route_heuristic(self, text: str) -> Intent:
        t = text.strip()
        if not t:
            return "casual"
        casual_kw = (
            "你好",
            "您好",
            "谢谢",
            "再见",
            "拜拜",
            "早上好",
            "晚上好",
            "在吗",
            "闲聊",
            "聊天",
            "讲个笑话",
            "心情",
        )
        # 短句且以寒暄开头 → 直接闲聊，减少误判
        if any(t.startswith(k) for k in casual_kw) and len(t) < 40:
            return "casual"

        g = self._signals_game(t)
        a = self._signals_account_strict(t)

        # 攻略 +（UID 或明确查库/角色列表）→ 复合，两路 Agent 都跑
        if g and a:
            return "composite"

        if a and not g:
            return "account"
        if g:
            return "game"
        return "casual"

    def _route_llm(self, user_input: str) -> Intent:
        # 让模型只吐 JSON intent；失败或解析异常则回退到 _route_heuristic
        sys = SystemMessage(
            content=(
                "你是意图分类器。仅输出一个 JSON 对象，不要 markdown 围栏，不要其它文字。\n"
                '格式：{"intent":"game"|"account"|"casual"|"composite"}\n'
                "- game：仅原神攻略/深渊/boss/机制/配队/伤害计算/知识库相关。\n"
                "- account：仅查询玩家账号数据（UID、树脂、角色列表、MySQL），不涉及当期关卡打法。\n"
                "- casual：日常寒暄、与游戏无关的闲聊。\n"
                "- composite：同一轮提问里**既要**攻略或 boss 机制/打法**又要**查某 UID 的账号或角色数据"
                "（例如：先问 12 层 boss 怎么打，同时问 uid10001 有哪些角色能用来打）；"
                "或「结合上文/刚才提到的 UID」同时需要角色列表与当期 Boss 攻略。\n"
                "判断不清时倾向 game；若明确只要库表数据则 account。"
            )
        )
        try:
            msg = self.router_llm.invoke([sys, HumanMessage(content=user_input)])
            text = (msg.content or "").strip()
            # 模型有时会在 JSON 外包一层废话，用正则抠出第一个 {...}
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)
            data = json.loads(text)
            v = (data.get("intent") or "").strip().lower()
            if v in ("game", "account", "casual", "composite"):
                # 修正：模型常把「攻略 + UID」误标为 account
                if (
                    v == "account"
                    and self._signals_game(user_input)
                    and self._has_uid(user_input)
                ):
                    return "composite"
                # 修正：仅有 UID 且模型标 game 但句子里确有查库语境 → 仍走 composite
                if (
                    v == "game"
                    and self._has_uid(user_input)
                    and self._signals_account_strict(user_input)
                ):
                    return "composite"
                return v  # type: ignore[return-value]
        except Exception as e:
            print(f"[路由] LLM 分类失败，回退 heuristic: {e}")
        return self._route_heuristic(user_input)

    def _decompose_composite(self, user_input: str) -> tuple[str, str]:
        """拆成 (攻略子问题, 账号子问题)，均可独立交给对应 Agent。"""
        sys = SystemMessage(
            content=(
                "你是任务拆分器。用户一句话里可能同时需要：\n"
                "1) 攻略侧：深渊层数、boss 名、机制与打法、配队思路（交给 RAG 知识库）。\n"
                "2) 账号侧：某 UID 的树脂、培养目标、拥有角色及等级/天赋（查 MySQL）。\n\n"
                "请输出唯一一个 JSON，不要其它文字：\n"
                '{"game_query":"字符串","account_query":"字符串"}\n'
                "规则：\n"
                "- game_query：只保留攻略相关表述，须自洽完整；若本侧不需要则为空字符串。\n"
                "- account_query：只保留查库所需表述，必须保留用户给出的 UID 数字；若本侧不需要则为空字符串。\n"
                "- 若用户问「哪些角色能攻克」，攻略部分写 boss/关卡打法；账号部分写「列出该 UID 下全部角色及培养信息」。\n"
            )
        )
        try:
            msg = self.router_llm.invoke([sys, HumanMessage(content=user_input)])
            text = (msg.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)
            data = json.loads(text)
            # 兼容键名 game / account（少数模型会擅自改名）
            gq = (data.get("game_query") or data.get("game") or "").strip()
            aq = (data.get("account_query") or data.get("account") or "").strip()
            # 至少一侧非空才采纳 LLM 拆分；两边都空则走启发式回退
            if gq or aq:
                return gq, aq
        except Exception as e:
            print(f"[编排] 复合问题拆分失败，使用启发式回退: {e}")
        return self._fallback_decompose(user_input)

    def _ensure_composite_queries(
        self, original: str, gq: str, aq: str
    ) -> tuple[str, str]:
        """保证 composite 两路都有可执行子问；跨轮时补上会话中的 last_uid。"""
        uid_in_msg = self._extract_uid(original)
        uid = uid_in_msg or self._last_uid
        if uid:
            self._last_uid = uid

        gq = gq.strip()
        aq = aq.strip()

        if not gq and self._signals_game(original):
            gq = self._game_only_fallback(original)
        if not aq and uid:
            aq = (
                f"请查询 UID 为 {uid} 的玩家档案、树脂与培养目标；"
                f"并列出该玩家全部角色及培养详情（等级、天赋、武器、圣遗物等）。"
            )
        elif aq and uid and uid not in aq and not uid_in_msg:
            aq = f"{aq}（目标 UID：{uid}）"
        return gq, aq

    def _fallback_decompose(self, user_input: str) -> tuple[str, str]:
        """LLM 拆分失败时的简单回退。"""
        t = user_input.strip()
        uid = self._extract_uid(t) or ""
        # 账号侧：有 uid 则模板化查询；无 uid 时把整句交给 AccountAgent（最后一搏）
        account_q = (
            f"请查询 uid 为 {uid} 的用户在数据库中的信息：拥有哪些角色及等级、天赋等培养情况；树脂与培养目标。"
            if uid
            else "请根据用户描述查询其账号下的角色与培养数据。"
        )
        # 攻略侧：从原文里删掉 UID 片段，减轻 RAG 噪声；删光则用兜底问法
        game_q = t
        uid_m = re.search(r"(?:uid|UID)[为：:\s]*(\d{4,})", t)
        num_m = re.search(r"(?<![\d.])(\d{9,})(?!\d)", t)
        if uid_m:
            game_q = (t[: uid_m.start()] + t[uid_m.end() :]).strip()
        elif num_m:
            game_q = (t[: num_m.start()] + t[num_m.end() :]).strip()
        game_q = re.sub(r"[，,、]\s*$", "", game_q).strip()
        if not game_q:
            game_q = "请根据用户上文提供的关卡或 boss 名称给出攻略要点。"
        # 无 uid 时 account_q 用整句 t，避免账号侧什么也得不到
        return game_q, account_q if uid else t

    def _synthesize(self, original: str, game_reply: str, account_reply: str) -> str:
        # 第三次调用 router_llm：只做编辑合并，不查库、不跑 RAG
        sys = SystemMessage(
            content=(
                "你是回答整合员。用户提了一个组合问题，下面两段分别是「攻略助手」与「账号数据助手」的独立输出。\n"
                "请合并成一条结构清晰的中文回复：\n"
                "1) 先简要给出 boss/关卡机制与打法要点（严格基于攻略段落，不编造）。\n"
                "2) 再列出该 UID 下的角色与培养情况（严格基于账号段落）。\n"
                "3) 若合适，用一两句话把「账号中的角色」与「攻略中的配队/属性需求」衔接起来；若账号段落未包含可用信息则说明无法从库中判断配队。\n"
                "不要重复两段中的废话标题，不要编造两段均未出现的事实。"
            )
        )
        human = HumanMessage(
            content=(
                f"【用户原问题】\n{original}\n\n"
                f"【攻略助手输出】\n{game_reply}\n\n"
                f"【账号助手输出】\n{account_reply}"
            )
        )
        out = self.router_llm.invoke([sys, human])
        return (out.content or "").strip()

    def chat(self, user_input: str) -> tuple[str, Intent]:
        uid_hit = self._extract_uid(user_input)
        if uid_hit:
            self._last_uid = uid_hit

        intent = self._route(user_input)
        # 跨轮：上文查过 UID，本轮只说「他/用户信息/角色培养」+ 攻略 → 抬升为 composite
        if intent == "game" and self._cross_turn_composite_hint(user_input):
            intent = "composite"

        if intent == "composite":
            # 1) 拆成两个子问 → 2) 攻略 Bot / 账号 Bot 各答一段 → 3) 有双份结果再合并
            gq, aq = self._decompose_composite(user_input)
            gq, aq = self._ensure_composite_queries(user_input, gq, aq)
            game_reply = self.game_bot.chat(gq) if gq.strip() else ""
            account_reply = self.account_bot.chat(aq) if aq.strip() else ""
            if game_reply and account_reply:
                reply = self._synthesize(user_input, game_reply, account_reply)
            elif game_reply:
                reply = game_reply
            elif account_reply:
                reply = account_reply
            else:
                # 两路都空：退回用整句问攻略（至少不给用户空白）
                reply = self.game_bot.chat(user_input)
            return reply, "composite"

        if intent == "account":
            reply = self.account_bot.chat(user_input)
        elif intent == "casual":
            reply = self.casual_bot.chat(user_input)
        else:
            # intent == "game" 及未列出的情况：默认走原 ChatBot（攻略 + 工具）
            reply = self.game_bot.chat(user_input)
        return reply, intent
