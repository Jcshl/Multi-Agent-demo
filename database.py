# database.py
"""MySQL 只读连接：供账号信息 Agent 的工具层使用。"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

try:
    import pymysql  # pyright: ignore[reportMissingImports]
    from pymysql.cursors import DictCursor as _DictCursor  # pyright: ignore[reportMissingImports]
except ImportError as _e:  # pragma: no cover
    pymysql = None  # type: ignore[assignment]
    _DictCursor = None  # type: ignore[assignment]
    print(
        f"[database] pymysql 导入失败，MySQL 工具不可用。解释器: {sys.executable}\n"
        f"[database] 原因: {_e}",
        file=sys.stderr,
    )


def mysql_configured() -> bool:
    """是否已在环境中配置完整的数据库连接参数。"""
    host = (os.getenv("MYSQL_HOST") or "").strip()
    user = (os.getenv("MYSQL_USER") or "").strip()
    password = os.getenv("MYSQL_PASSWORD")
    database = (os.getenv("MYSQL_DATABASE") or "").strip()
    return bool(host and user and password is not None and database)


def _connect_kwargs() -> dict[str, Any]:
    port_raw = (os.getenv("MYSQL_PORT") or "3306").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 3306
    return {
        "host": (os.getenv("MYSQL_HOST") or "").strip(),
        "port": port,
        "user": (os.getenv("MYSQL_USER") or "").strip(),
        "password": os.getenv("MYSQL_PASSWORD") or "",
        "database": (os.getenv("MYSQL_DATABASE") or "").strip(),
        "charset": "utf8mb4",
        "cursorclass": _DictCursor,
        "connect_timeout": int((os.getenv("MYSQL_CONNECT_TIMEOUT") or "10").strip() or "10"),
    }


@contextmanager
def mysql_connection() -> Iterator[Any]:
    """上下文管理器：获取一条 DictCursor 连接，用完关闭。"""
    if pymysql is None:
        raise RuntimeError(
            "未安装 pymysql 或导入失败。请在**当前运行服务的同一 Python** 中安装：pip install pymysql。"
            f" 当前解释器: {sys.executable}"
        )
    if not mysql_configured():
        raise RuntimeError("MySQL 未配置：请设置 MYSQL_HOST、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE。")
    conn = pymysql.connect(**_connect_kwargs())
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    """执行只读查询，返回字典行列表。"""
    normalized = (sql or "").lstrip()
    if not normalized.upper().startswith("SELECT"):
        raise ValueError("仅允许 SELECT 查询")
    with mysql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
    return list(rows)
