"""
my_minimax_first_call.py — lesson 04 的 MiniMax 对照练习。

参照：
  - phases/00-setup-and-tooling/04-apis-and-keys/docs/en.md
  - phases/00-setup-and-tooling/04-apis-and-keys/code/first_api_call.py
  - MiniMax API: https://api.minimaxi.com/v1/text/chatcompletion_v2

这个文件刻意保留 first_api_call.py 的两层结构（SDK / raw HTTP），
方便对照两个 LLM 服务商在同样模式下的差异。差异点：
  - 端点: api.minimaxi.com（不是 api.anthropic.com）
  - 鉴权: Authorization: Bearer <key>（不是 x-api-key）
  - 环境变量: MINIMAX_API_KEY（不是 ANTHROPIC_API_KEY）
  - 模型: MiniMax-M3（不是 claude-sonnet-4-20250514）
  - 响应字段: choices[0].message.content（不是 content[0].text）
  - SDK 可选：未安装 minimax 包时，raw HTTP 段仍然能跑
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    """极简 .env 解析器——只依赖 stdlib（python-dotenv 不在项目依赖白名单里）。"""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def call_with_sdk() -> bool:
    """如果 SDK 装了，用 SDK 调 MiniMax。返回 True 表示成功。"""
    try:
        import minimax  # type: ignore
    except ImportError:
        print("Install the SDK: pip install minimax")
        print("(This is optional — call_raw_http() below needs no SDK.)")
        return False

    client = minimax.MiniMax(api_key=os.environ["MINIMAX_API_KEY"])
    response = client.chat.completions.create(
        model="MiniMax-M3",
        messages=[{"role": "user", "content": "What is a neural network in one sentence?"}],
    )
    print(f"SDK response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
    return True


def call_raw_http() -> None:
    """同样的调用，用 raw HTTP——只依赖 stdlib。这条路径是永远能跑的。"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("Set MINIMAX_API_KEY in .env or your shell first")
        return

    url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps({
        "model": "MiniMax-M3",
        "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"Raw HTTP response: {result['choices'][0]['message']['content']}")
            print(f"Tokens used: {result['usage']['prompt_tokens']} in, {result['usage']['completion_tokens']} out")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")


if __name__ == "__main__":
    env_file = Path(__file__).parent / ".env"
    load_env_file(env_file)
    if "MINIMAX_API_KEY" not in os.environ:
        print("Set MINIMAX_API_KEY in .env first (see .env.example)")
        sys.exit(1)
    print("=== MiniMax API Calls ===\n")
    print("1. Using the SDK:")
    call_with_sdk()
    print("\n2. Using raw HTTP:")
    call_raw_http()
