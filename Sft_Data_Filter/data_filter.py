#-*- coding: utf-8 -*-
import re
import random
from flask.sansio.scaffold import F
import httpx
import json
import os
import sys
import asyncio
import aiofiles
from tqdm.asyncio import tqdm as atqdm


# ============ 模型配置 ============
REQ_URL = "http://xxxx/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}


async def text_generate_async(client: httpx.AsyncClient, messages: list, url: str = REQ_URL) -> tuple[str, str]:
    """异步调用 WiNGPT 模型，返回 (reply)"""
    payload = {
        "model": "Qwen3.5",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    try:
        resp = await client.post(url, json=payload, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        print(f"[HTTP ERROR] status={e.response.status_code}, err={e}")
        reply = ""
    except Exception as e:
        print(f"[ERROR] {e}")
        reply = ""
    return reply

JUDGE_URL = "http://xxxx/v1/chat/completions"
async def judge_generate_async(client: httpx.AsyncClient, messages: list, url: str = JUDGE_URL) -> tuple[str, str]:
    """异步调用 deepseek32 模型，返回 (reply, reasoning_content)"""
    payload = {
        "model": "deepseek32",
        "messages": messages,
        "chat_template_kwargs": {"thinking": True}
    }
    try:
        resp = await client.post(url, json=payload, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        answer= data["choices"][0]["message"]["content"]
        reply  = answer.split('</think>')[-1].strip()
        reasoning_content = answer.split('</think>')[0].strip()
    except httpx.HTTPStatusError as e:
        print(f"[HTTP ERROR] status={e.response.status_code}, err={e}")
        reply = ""
        reasoning_content = ""
    except Exception as e:
        print(f"[ERROR] {e}")
        reply = ""
        reasoning_content = ""
    return reply, reasoning_content


from Verifier_Prompts import SELECTION_PROMPT_TEMPLATE_WITH_SOLUTION


def get_selection_prompt(
    message: list[dict],
    candidates: list[str],
) -> str:
    """构建 GenRM 选择模式的 prompt
    
    Args:
        message: 原始对话消息列表（raw_prompt）
        candidates: 候选答案列表（n 个 response）
    Returns:
        构建好的选择模式 prompt
    """
    prompt = SELECTION_PROMPT_TEMPLATE_WITH_SOLUTION
    
    # 对话历史和问题
    if len(message) >= 3:
        chathistory = message[:-1]
        history = '\n'.join([k['role'] + '：' + k['content'].strip() for k in chathistory])
        prompt = prompt.replace('{插入对话历史}', history)
        question = 'user：' + message[-1]['content'].strip()
    else:
        prompt = prompt.replace('{插入对话历史}', '无')
        question = '\n'.join([k['role'] + '：' + k['content'].strip() for k in message])
    
    answer = message[-1]['content'].strip()
    
    prompt = prompt.replace('{插入原始问题}', question.strip())
    prompt = prompt.replace('{插入优秀答案}', answer.strip())
    prompt = prompt.replace('{插入核心知识}', '无')
    # 候选答案块
    prompt = prompt.replace('{n_candidates}', str(len(candidates)))
    candidates_parts = []
    for i, cand in enumerate(candidates, 1):
        # 截断过长的候选答案以控制 prompt 长度
        cand_text = cand.strip()
        candidates_parts.append(f"--- 候选答案 {i} ---\n```\n{cand_text}\n```")
    prompt = prompt.replace('{candidates_block}', '\n\n'.join(candidates_parts))
    return prompt



INPUTFILE = "/data/synth_answer_output.jsonl"
OUTPUTFILE = "/data/final_data.jsonl"


# 并发控制
CONCURRENCY = 16
# 每条数据生成候选答案数量
N_CANDIDATES = 7


def parse_selection_result(response_text: str) -> list[int]:
    """从模型响应中提取选中的答案序号"""
    pattern = r'\\boxed\{\[(.*?)\]\}'
    match = re.search(pattern, response_text)
    if match:
        content = match.group(1).strip()
        if content:
            try:
                indices = [int(x.strip()) for x in content.split(',')]
                return [i for i in indices if 1 <= i <= N_CANDIDATES]
            except ValueError:
                return []
        return []
    return []


def load_processed_ids(output_file: str) -> set:
    """读取已有输出文件，收集已处理的id集合"""
    processed = set()
    if not os.path.exists(output_file):
        return processed
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'id' in obj:
                    processed.add(obj['id'])
            except json.JSONDecodeError:
                pass
    return processed


async def process_single(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                         item: dict, file_lock: asyncio.Lock, out_f) -> None:
    """处理单条数据：生成候选答案 -> 评判 -> 解析 -> 写入"""
    item_id = item.get('id')
    messages = item.get('messages', [])

    # 提取 user 消息（去掉最后一条 assistant 消息）
    user_messages = messages[:-1]

    try:
        async with sem:
            # 第一步：并发生成 N_CANDIDATES 个候选答案
            candidate_tasks = [
                text_generate_async(client, user_messages)
                for _ in range(N_CANDIDATES)
            ]
            candidates = await asyncio.gather(*candidate_tasks)
            candidates = list(candidates)

            # 第二步：用完整 messages（含 assistant 参考答案）构建评判 prompt
            selection_prompt = get_selection_prompt(
                message=messages,
                candidates=candidates
            )
            judge_messages = [{"role": "user", "content": selection_prompt}]
            judge_response = await judge_generate_async(client, judge_messages)

        # 第三步：解析结果
        selected_indices = parse_selection_result(judge_response[0])
        correct_count = len(selected_indices)

        # 构建输出对象
        out_obj = dict(item)
        out_obj['correct_count'] = correct_count

        # 写入输出文件（加锁保证顺序写入）
        async with file_lock:
            await out_f.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            await out_f.flush()

    except Exception as e:
        print(f"[ERROR] id={item_id} 处理失败: {e}", file=sys.stderr)


async def main():
    # 读取输入数据
    all_items = []
    with open(INPUTFILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] 跳过无效行: {e}", file=sys.stderr)

    # 断点续传：加载已处理 id
    processed_ids = load_processed_ids(OUTPUTFILE)
    print(f"已处理: {len(processed_ids)} 条，共 {len(all_items)} 条")

    # 过滤出未处理的数据
    remaining = [item for item in all_items if item.get('id') not in processed_ids]
    print(f"待处理: {len(remaining)} 条")

    if not remaining:
        print("所有数据已处理完毕。")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUTFILE), exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    file_lock = asyncio.Lock()

    async with aiofiles.open(OUTPUTFILE, 'a', encoding='utf-8') as out_f:
        async with httpx.AsyncClient(timeout=960.0) as client:
            tasks = [
                process_single(client, sem, item, file_lock, out_f)
                for item in remaining
            ]
            # 使用 tqdm 显示进度
            for coro in atqdm.as_completed(tasks, total=len(tasks), desc="处理进度"):
                await coro

    print("全部处理完成。")


if __name__ == "__main__":
    asyncio.run(main())
