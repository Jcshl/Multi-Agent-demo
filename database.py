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

# MySQL 8 默认 caching_sha2_password：PyMySQL 连接时必须能导入 cryptography，否则会报含糊错误。
try:
    import cryptography  # noqa: F401  # pyright: ignore[reportMissingImports]
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False


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
    if not _CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError(
            "当前 Python 环境未安装 cryptography，无法使用 MySQL 8 默认认证（caching_sha2_password）。\n"
            f"解释器路径: {sys.executable}\n"
            "请确认：① 终端 `cd` 的目录是**正在开发的那份**含 `api_app.py` 的仓库根（看该目录的 `pyproject.toml` 里是否包含 `cryptography` 依赖）。\n"
            "② 在该根目录执行 `uv sync` 后，用 `uv run python -c \"import cryptography\"` 应无报错，再 `uv run uvicorn ...`。\n"
            "若 `pyproject.toml` 里本来没有 `cryptography`，需先加入依赖并 `uv lock` / `uv sync`。"
        )
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
