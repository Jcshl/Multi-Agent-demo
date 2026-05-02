# api_app.py
# 运行：uvicorn api_app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000/ （与 streamlit_app 共用 .env）

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]

from orchestrator import MultiAgentOrchestrator

# 加载 .env 到进程环境变量，供 API 初始化编排器使用。
load_dotenv()

# 当前文件所在目录（项目根目录）。
_ROOT = Path(__file__).resolve().parent
# 静态资源目录，包含 index.html。
_STATIC = _ROOT / "static"

# 会话存储：session_id -> MultiAgentOrchestrator 实例。
_sessions: dict[str, MultiAgentOrchestrator] = {}
# 保护 _sessions 的线程锁，避免并发写入竞态。
_sessions_lock = threading.Lock()


def _env_credentials() -> tuple[str, str] | None:
    """从环境变量读取模型名与 API Key；任一缺失则返回 None。"""
    model = (os.getenv("MODEL_NAME") or "").strip()
    key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
    if not model or not key:
        return None
    return model, key


def _get_or_create_bot(session_id: str | None) -> tuple[str, MultiAgentOrchestrator]:
    """
    按 session_id 获取（或创建）一个 MultiAgentOrchestrator 实例。

    - 未传 session_id 时自动生成 UUID；
    - 使用进程内字典保存会话，配合锁保证并发安全；
    - 若 .env 凭证缺失，抛出 500 提示。
    """
    cred = _env_credentials()
    if cred is None:
        raise HTTPException(
            status_code=500,
            detail="请在项目根目录 `.env` 中配置 MODEL_NAME 与 SILICONFLOW_API_KEY。",
        )
    model, api_key = cred
    sid = (session_id or "").strip() or str(uuid.uuid4())
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = MultiAgentOrchestrator(model_name=model, api_key=api_key)
        return sid, _sessions[sid]


class ChatRequest(BaseModel):
    """聊天请求体。"""

    # 用户本轮输入文本。
    message: str = Field(..., min_length=1)
    # 会话 ID；为空时后端会创建新会话。
    session_id: str | None = None


class ChatResponse(BaseModel):
    """聊天响应体。"""

    # 助手本轮回复内容。
    reply: str
    # 实际使用的会话 ID（前端需持久化并复用）。
    session_id: str
    # 路由到的 specialist：game / account / casual。
    intent: str | None = None


class ResetRequest(BaseModel):
    """重置会话请求体。"""

    # 要重置的会话 ID。
    session_id: str = Field(..., min_length=1)


# FastAPI 应用实例；title/version 会显示在 OpenAPI 文档中。
app = FastAPI(title="LangChain Agent Demo", version="0.1.0")


@app.get("/")
def index_page():
    """返回聊天页面入口（static/index.html）。"""
    return FileResponse(_STATIC / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(body: ChatRequest):
    """
    聊天接口：接收用户消息并返回模型回复。

    行为：
    - 校验 message 非空；
    - 根据 session_id 复用同一会话上下文；
    - 调用 bot.chat() 生成回复并回传 session_id。
    """
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message 不能为空")
    sid, bot = _get_or_create_bot(body.session_id)
    reply, intent = bot.chat(msg)
    return ChatResponse(reply=reply, session_id=sid, intent=intent)


@app.post("/api/chat/reset")
def api_reset(body: ResetRequest):
    """
    重置会话接口：移除对应 session 的编排器实例。

    删除后该会话历史不再保留；前端下次聊天会创建新会话实例。
    """
    sid = body.session_id.strip()
    with _sessions_lock:
        # 删除会话对应的编排器，让下次请求重新创建实例。
        _sessions.pop(sid, None)
    return {"ok": True}


app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC)),
    name="static",
)
