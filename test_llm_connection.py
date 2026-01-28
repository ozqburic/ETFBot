# test_llm_connection.py
"""
用于测试大模型（如GPT等）连接的脚本。
"""


from llm_client import make_openai_compatible_client, chat_completion


def test_llm():
    try:
        client = make_openai_compatible_client()
        messages = [{"role": "user", "content": "你好，大模型！"}]
        response = chat_completion(client, messages)
        print("模型响应:", response)
    except Exception as e:
        print("连接大模型失败:", e)


if __name__ == "__main__":
    test_llm()
