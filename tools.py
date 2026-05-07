# tools.py
"""Agent 可调用的工具：LangChain @tool + 懒加载 RAG。"""

from __future__ import annotations

import ast
import operator
from langchain_core.tools import tool  # pyright: ignore[reportMissingImports]

from rag import RAG

# 懒加载 RAG 实例（首次调用 rag_search 时创建，后续复用）。
_rag: RAG | None = None

# 允许的二元运算符映射：AST 节点类型 -> Python 运算函数。
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算符映射：正号/负号。
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> str:
    """
    仅允许数字与 + - * / // % **、括号；禁止函数调用、属性、变量名等。
    """
    # 去掉首尾空白后的表达式文本。
    expr = (expression or "").strip()
    if not expr:
        return "计算错误: 空表达式"

    try:
        # 将字符串解析为表达式 AST（只允许 eval 模式的单表达式）。
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"计算错误: 语法无效 ({e})"

    try:
        # 递归计算 AST 根节点。
        val = _eval_ast_node(tree.body)
    except ValueError as e:
        return f"计算错误: {e}"

    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def _eval_ast_node(node: ast.AST) -> float:
    """
    递归计算 AST 节点并返回浮点结果。

    参数:
        node: 当前要计算的 AST 节点。
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("不允许布尔参与运算")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("仅支持数值常量")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        # 计算一元表达式，如 -3。
        return float(_ALLOWED_UNARY[type(node.op)](_eval_ast_node(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        # 计算二元表达式，如 1 + 2。
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("除数不能为 0")
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))

    raise ValueError(f"不允许的表达式成分: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    对用户算式求数值；用户说「算一下」「等于多少」、或需精确数字时优先调用。
    表达式仅允许数字与 +、-、*、/、//、%、**、英文括号；例：1.5 * (800 + 200)
    """

    # 统一走安全计算逻辑，避免执行任意代码。
    return safe_calculate(expression)


@tool
def search(query: str) -> str:
    """泛化搜索或百科式补充（当前为占位实现）。若与本地攻略冲突，以 rag_search 为准。"""

    # 当前为示例实现，后续可替换为真实搜索 API。
    return f"搜索结果：关于『{query}』的相关信息..."


@tool
def rag_search(query: str) -> str:
    """从本地向量知识库检索原神深渊与养成相关内容；可用中文关键词并多次换词检索。返回条数由环境变量 RAG_TOP_K（默认 8）控制；切块由 RAG_CHUNK_SIZE 等控制，修改后需重建索引。"""

    global _rag
    if _rag is None:
        # 首次调用时构建向量检索实例（包含索引加载/构建）。
        print("RAG 初始化中...")
        _rag = RAG()
        print("RAG 初始化完成")
    # 在向量库中执行相似度检索并返回拼接文本。
    return _rag.search(query)


# 供 ChatOpenAI.bind_tools() 使用（顺序影响部分模型偏好；calculator 靠前便于数值类提问）
lc_tools = [rag_search, calculator, search]
