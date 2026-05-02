# main.py

"""
程序入口：多 Agent 协同对话（自动路由至攻略 / 账号库 / 闲聊）。
"""

import os
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from orchestrator import MultiAgentOrchestrator


# 加载环境变量
load_dotenv()

# 多 Agent 编排（内含原 ChatBot 攻略能力）
bot = MultiAgentOrchestrator(
    model_name=os.getenv("MODEL_NAME"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
)


# CLI 循环
while True:
    # user_input：用户在命令行输入的本轮问题。
    user_input = input("用户输入：")

    if user_input.lower() in ["exit", "quit"]:
        break

    # reply：模型本轮最终回复文本；intent：路由到的 specialist。
    reply, intent = bot.chat(user_input)
    print(f"AI（{intent}）：", reply)