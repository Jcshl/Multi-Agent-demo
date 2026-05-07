# memory_store.py
"""长期记忆（提纲摘要）持久化：SQLite，与会话 key 关联便于排查。"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _db_path() -> Path:
    raw = (os.getenv("MEMORY_DB_PATH") or "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "data" / "session_memory.sqlite"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def append_summary(session_key: str, summary_text: str) -> None:
    """写入一条会话结束摘要。"""
    sk = (session_key or "default").strip() or "default"
    text = (summary_text or "").strip()
    if not text:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO session_summaries (session_key, summary, created_at) VALUES (?, ?, ?)",
            (sk, text, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_summaries(limit: int) -> list[str]:
    """
    取最近若干条摘要（按创建时间），返回列表为**时间正序**（旧 → 新），
    便于拼进提示词。
    """
    n = max(1, min(int(limit), 50))
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT summary FROM session_summaries ORDER BY created_at DESC LIMIT ?",
            (n,),
        )
        rows = [r[0] for r in cur.fetchall() if r and r[0]]
    finally:
        conn.close()
    rows.reverse()
    return rows


def format_recent_for_prompt(limit: int) -> str:
    """拼成一段注入模型用的「过往提纲」文本；无记录则空串。"""
    parts = get_recent_summaries(limit)
    if not parts:
        return ""
    lines: list[str] = []
    for i, p in enumerate(parts, 1):
        lines.append(f"【历史摘要{i}】\n{p.strip()}")
    return "\n\n".join(lines)
