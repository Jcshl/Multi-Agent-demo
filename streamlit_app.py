# streamlit_app.py
# 运行：streamlit run streamlit_app.py
# 需先在 .env 配置 MODEL_NAME、SILICONFLOW_API_KEY（与 main.py 相同）

import os
import uuid

import streamlit as st
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

from orchestrator import MultiAgentOrchestrator

load_dotenv()


def _ensure_bot() -> MultiAgentOrchestrator:
    """确保当前会话有多 Agent 编排器；首次访问时按 .env 创建实例。"""
    if st.session_state.get("memory_session_key") is None:
        st.session_state.memory_session_key = str(uuid.uuid4())
    if st.session_state.get("bot") is None:
        # model/key：模型配置与鉴权信息，来自环境变量。
        model = (os.getenv("MODEL_NAME") or "").strip()
        key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
        if not model or not key:
            st.error("请在项目根目录 `.env` 中配置 `MODEL_NAME` 与 `SILICONFLOW_API_KEY`。")
            st.stop()
        st.session_state.bot = MultiAgentOrchestrator(
            model_name=model,
            api_key=key,
            session_key=st.session_state.memory_session_key,
        )
    return st.session_state.bot


st.set_page_config(page_title="对话", layout="centered")
st.title("多 Agent 协同 · 原神助手")

with st.sidebar:
    st.caption("攻略（原 ChatBot）· 账号 MySQL · 闲聊；自动路由。")
    if st.button("新对话"):
        # 页面展示层的聊天记录（仅用于 UI 回显）。
        st.session_state.messages = []
        # 业务层：先将本会话链摘要入库并加载提纲，再换新 session_key 与编排器实例。
        bot = st.session_state.get("bot")
        if bot is not None:
            bot.clear_history()
        st.session_state.memory_session_key = str(uuid.uuid4())
        st.session_state.bot = None
        st.rerun()

# 首次进入页面时初始化消息列表状态。
if "messages" not in st.session_state:
    st.session_state.messages = []

# 当前页面会话对应的编排器实例。
bot = _ensure_bot()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入你的问题…"):
    # prompt：用户在输入框提交的本轮问题。
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            intent = None
            try:
                # reply：最终回答；intent：路由 specialist。
                reply, intent = bot.chat(prompt)
            except Exception as e:
                reply = f"请求出错：{e}"
        st.markdown(reply)
        if intent is not None:
            st.caption(f"路由：{intent}")

    st.session_state.messages.append({"role": "assistant", "content": reply})
