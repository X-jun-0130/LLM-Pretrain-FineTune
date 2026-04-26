# -*- coding: utf-8 -*-
"""
API 配置模块 - 指令多样化合成
复用项目现有的 DeepSeek API 端点
"""
import httpx
import json

# ============ 全局异步客户端管理 ============
async_client = None

def get_async_client():
    """获取或创建异步客户端"""
    global async_client
    if async_client is None or async_client.is_closed:
        async_client = httpx.AsyncClient(timeout=httpx.Timeout(960.0))
    return async_client

async def close_async_client():
    """关闭异步客户端"""
    global async_client
    if async_client is not None and not async_client.is_closed:
        await async_client.aclose()
        async_client = None

# ============ 请求配置 ============
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

REQ_URL = "http://172.16.0.95:5053/v1/chat/completions"


async def async_text_generate_think(messages, url=REQ_URL, temperature=0.6):
    """异步文本生成（带思考链，用于验证）"""
    payload = {
        "model": "deepseek32",
        "messages": messages,
        "temperature": temperature,
        "chat_template_kwargs": {"thinking": True}
    }
    try:
        client = get_async_client()
        response = await client.post(url, json=payload, headers=HEADERS, timeout=960)
        answer = json.loads(response.text)["choices"][0]["message"]["content"]
        reply = answer.split('</think>')[-1].strip()
    except Exception as e:
        print(f"[API ERROR] async_text_generate_think: {e}")
        reply = ""
    return reply
