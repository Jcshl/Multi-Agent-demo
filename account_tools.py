# account_tools.py
"""账号数据查询工具：默认对齐 genshin_ai.users / genshin_ai.characters（通过 uid 关联）。"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool  # pyright: ignore[reportMissingImports]

from database import fetch_all, mysql_configured


def _table_users() -> str:
    return (os.getenv("MYSQL_TABLE_PLAYERS") or "users").strip() or "users"


def _table_characters() -> str:
    return (os.getenv("MYSQL_TABLE_CHARACTERS") or "characters").strip() or "characters"


def _col_uid() -> str:
    return (os.getenv("MYSQL_COL_PLAYER_UID") or "uid").strip() or "uid"


def _col_resin() -> str:
    return (os.getenv("MYSQL_COL_RESIN") or "resin").strip() or "resin"


def _col_goal() -> str:
    return (os.getenv("MYSQL_COL_GOAL") or "goal").strip() or "goal"


def _col_char_name() -> str:
    return (os.getenv("MYSQL_COL_CHARACTER_NAME") or "name").strip() or "name"


def _safe_ident(name: str, fallback: str) -> str:
    """仅允许字母数字下划线，防止标识符注入。"""
    n = (name or "").strip()
    if n and all(c.isalnum() or c == "_" for c in n):
        return n
    return fallback


@tool
def get_player_profile(player_uid: str) -> str:
    """按游戏 UID（users.uid）查询树脂 resin、培养目标 goal 等。player_uid 必填。"""

    if not mysql_configured():
        return "数据库未配置：请在 .env 中设置 MYSQL_HOST、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE。"
    uid = (player_uid or "").strip()
    if not uid:
        return "错误：player_uid 不能为空。"
    t = _safe_ident(_table_users(), "users")
    c = _safe_ident(_col_uid(), "uid")
    sql = f"SELECT * FROM `{t}` WHERE `{c}` = %s LIMIT 1"
    try:
        rows = fetch_all(sql, (uid,))
    except Exception as e:
        return f"查询失败: {e}"
    if not rows:
        return f"未找到玩家：{uid}"
    return json.dumps(rows[0], ensure_ascii=False, default=str)


@tool
def list_player_characters(player_uid: str) -> str:
    """列出某 UID 下 characters 表中的角色：姓名 name、等级 level、天赋 talent_level 等。关联字段与 users.uid 相同（characters.uid）。"""

    if not mysql_configured():
        return "数据库未配置：请在 .env 中设置 MYSQL_HOST、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE。"
    uid = (player_uid or "").strip()
    if not uid:
        return "错误：player_uid 不能为空。"
    ct = _safe_ident(_table_characters(), "characters")
    uc = _safe_ident(_col_uid(), "uid")
    cn = _safe_ident(_col_char_name(), "name")
    sql = f"SELECT * FROM `{ct}` WHERE `{uc}` = %s ORDER BY `{cn}`"
    try:
        rows = fetch_all(sql, (uid,))
    except Exception as e:
        return f"查询失败: {e}"
    if not rows:
        return f"UID {uid} 暂无角色记录或该 UID 在 users 中不存在。"
    return json.dumps(rows, ensure_ascii=False, default=str)


@tool
def search_players_by_name_keyword(keyword: str) -> str:
    """在 users 表中按 uid 或培养目标 goal 模糊查找（最多 20 条）；适合只记得片段时使用。"""

    if not mysql_configured():
        return "数据库未配置：请在 .env 中设置 MYSQL_HOST、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DATABASE。"
    kw = (keyword or "").strip()
    if not kw:
        return "错误：keyword 不能为空。"
    t = _safe_ident(_table_users(), "users")
    uc = _safe_ident(_col_uid(), "uid")
    gr = _safe_ident(_col_goal(), "goal")
    rs = _safe_ident(_col_resin(), "resin")
    pat = f"%{kw}%"
    sql = (
        f"SELECT id, `{uc}` AS uid, `{rs}` AS resin, `{gr}` AS goal "
        f"FROM `{t}` WHERE `{uc}` LIKE %s OR `{gr}` LIKE %s LIMIT 20"
    )
    try:
        rows = fetch_all(sql, (pat, pat))
    except Exception as e:
        return f"查询失败: {e}"
    if not rows:
        return "未匹配到任何 uid 或培养目标。"
    return json.dumps(rows, ensure_ascii=False, default=str)


account_lc_tools: list[Any] = [get_player_profile, list_player_characters, search_players_by_name_keyword]
