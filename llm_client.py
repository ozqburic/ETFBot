"""大模型调用相关封装。

目标：让 UI/数据处理逻辑与 LLM 调用解耦，方便后续替换模型或调整鉴权。

实现：OpenAI-compatible（`/v1/chat/completions`）模式。
- 默认指向 Gemini 的 OpenAI-compatible 端点（可在本文件中修改）
- 仅 API Key 从环境变量读取（也可用入参覆盖）
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, TYPE_CHECKING

try:
    from openai import AuthenticationError, OpenAI
except ModuleNotFoundError:  # pragma: no cover
    AuthenticationError = Exception  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI as OpenAIType


# 默认仍使用 Gemini（OpenAI-compatible 端点）
DEFAULT_MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# 环境变量：只读取 API Key
ENV_API_KEY = "GEMINI_API_KEY"


def _get_env_first(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def make_openai_compatible_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional["OpenAIType"]:
    """创建 OpenAI-compatible 客户端。

    取值优先级：
    - api_key 参数
    - 环境变量：LLM_API_KEY
    - 兼容变量：GEMINI_API_KEY / OPENAI_API_KEY

    base_url 不从环境变量读取：
    - base_url 参数（可选）
    - 默认值：BASE_URL（Gemini OpenAI-compatible）
    """

    if OpenAI is None:
        return None

    key = (
        api_key or _get_env_first(ENV_API_KEY, "GEMINI_API_KEY", "OPENAI_API_KEY")
    ).strip()
    if not key:
        return None

    use_base_url = (base_url or BASE_URL).strip()
    return OpenAI(api_key=key, base_url=use_base_url)  # type: ignore[misc]


def make_gemini_openai_client(api_key: Optional[str] = None) -> Optional["OpenAIType"]:
    """兼容旧接口：创建 Gemini OpenAI-compatible 客户端。"""

    return make_openai_compatible_client(api_key=api_key, base_url=BASE_URL)


def missing_api_key_message() -> str:
    """未配置 API Key 时，给用户的提示文案。"""

    if OpenAI is None:
        return (
            "⚠️ 未安装依赖 openai，无法调用大模型。\n\n"
            "请先安装：pip install openai\n\n"
            f"然后设置环境变量：{ENV_API_KEY}（或 GEMINI_API_KEY / OPENAI_API_KEY）。"
        )

    return (
        "⚠️ 未检测到大模型 API Key。\n\n"
        "请先在终端设置后再试：\n"
        f"- PowerShell：$env:{ENV_API_KEY}=\"你的key\"\n"
        f"- CMD：set {ENV_API_KEY}=你的key\n\n"
        "（兼容变量：GEMINI_API_KEY / OPENAI_API_KEY 也可）"
    )


def chat_completion(
    client: Optional["OpenAIType"],
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 1.0,
) -> str:
    """基于 chat messages 调用模型并返回文本结果。

    - 若 client 为空（未配置 key），返回提示文案
    - 捕获鉴权失败与其他异常，返回可读的错误信息
    """

    if client is None:
        return missing_api_key_message()

    use_model = (model or DEFAULT_MODEL).strip()
    try:
        response = client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()
    except AuthenticationError as e:
        return f"❌ 鉴权失败（401）：{e}"
    except Exception as e:
        return f"❌ 调用大模型时出错：{e}"
