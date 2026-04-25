#-*- coding: utf-8 -*-
import re
import random
import httpx
import json
import os
import sys
import asyncio
import aiofiles
from tqdm.asyncio import tqdm as atqdm


from prompts import get_selection_prompt, check_prompt

# ============ 模型配置 ============
REQ_URL = "http://xxxx/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}


async def text_generate_async(client: httpx.AsyncClient, messages: list, url: str = REQ_URL) -> tuple[str, str]:
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


async def check_answer(client: httpx.AsyncClient, question: str, answer: str) -> tuple[str, float]:
    """异步对答案进行质量评估，返回 (评估回复, 分数)"""
    prompt = check_prompt.strip()
    prompt = prompt.replace('{插入原始问题}', question.strip())
    prompt = prompt.replace('{插入待评估答案}', answer.strip())
    chat = [{"role": "user", "content": prompt}]
    verifier_reply, _ = await text_generate_async(client, chat)
    score = extract_evaluation_score(verifier_reply)
    return verifier_reply, score


def extract_evaluation_score(response: str) -> float:
    """
    从模型回复中提取评估分数（最后一个\\boxed{}中的0-100分数字），
    处理整数和小数格式，并将结果压缩到0-1范围
    
    参数:
        response: 包含评估分数的文本字符串
        
    返回:
        提取并处理后的分数（0.0-1.0之间的浮点数）
    """
    baseline = 50.0 #小于baseline得分直接归0
    
    # 匹配所有\boxed{}内容
    boxes = re.findall(r'\\boxed\{([^}]*)\}', response)
    
    if not boxes:
        return 0.0
    
    # 取最后一个boxed内容
    last_box = boxes[-1].strip()
    
    # 改进的数字匹配：支持整数和小数格式（如95, 96.5, 85分）
    score_match = re.search(
        r'(\d{1,3}(?:\.\d{1,2})?)\s*(?:分)?\b', 
        last_box
    )
    
    if score_match:
        try:
            score = float(score_match.group(1))
            # 处理分数范围
            if score <= baseline:  # 低分直接归零
                return 0.0
            elif score > 100.0:  # 超百分制处理
                return min(score / 100.0, 1.0)
            else:
                return score / 100.0
        except (ValueError, TypeError):
            return 0.0
    
    # 额外尝试匹配带等号的分数（如"=96.5"）
    equals_match = re.search(r'=\s*(\d{1,3}(?:\.\d{1,2})?)\b', last_box)
    if equals_match:
        try:
            score = float(equals_match.group(1))
            return max(0.0, min(score / 100.0, 1.0)) if score > baseline else 0.0
        except (ValueError, TypeError):
            pass
    
    return 0.0


INPUTFILE = "/data/questions.jsonl"
OUTPUTFILE = "/data/answer"
OUTPUTNAME = 'generate_answer.jsonl'

# 并发控制
CONCURRENCY = 16
# 评分通过阈值
SCORE_THRESHOLD = 0.9
# 每条数据生成候选答案数量
N_CANDIDATES = 4


def load_existing_ids(output_path: str) -> set:
    """读取已有输出文件中的 id 集合，用于断点续跑"""
    ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        ids.add(item["id"])
                    except Exception:
                        pass
    return ids


def extract_selected_index(response: str) -> int | None:
    """从模型回复中解析 \\boxed{[序号列表]}，返回第一个序号（1-based），失败返回None"""
    boxes = re.findall(r'\\boxed\{(\[[^\]]*\])\}', response)
    if not boxes:
        # 兼容无转义写法
        boxes = re.findall(r'\\?boxed\{(\[[^\]]*\])\}', response)
    if not boxes:
        return None
    last_box = boxes[-1].strip()
    nums = re.findall(r'\d+', last_box)
    if nums:
        return int(nums[0])
    return None


async def process_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, data: dict,
                      output_file: str):
    """处理单条数据：生成候选答案 -> 选优 -> 质量审核 -> 写入输出"""
    data_id = data['id']
    prompt = data['prompt']
    message = [{"role": "user", "content": prompt}]

    async with sem:
        # --- 步骤1：并发生成 N_CANDIDATES 条候选答案 ---
        tasks = [text_generate_async(client, message) for _ in range(N_CANDIDATES)]
        results = await asyncio.gather(*tasks)
        # results: list of (reply, reasoning_content)
        replies = [r[0] for r in results]
        reasonings = [r[1] for r in results]

        # 过滤空回复
        valid_indices = [i for i, r in enumerate(replies) if r.strip()]
        if not valid_indices:
            print(f"[SKIP] id={data_id}：所有候选答案均为空")
            return

        valid_replies = [replies[i] for i in valid_indices]
        valid_reasonings = [reasonings[i] for i in valid_indices]

        # --- 步骤2：最优答案选择 ---
        selection_prompt_text = get_selection_prompt(message, valid_replies)
        selection_chat = [{"role": "user", "content": selection_prompt_text}]
        sel_reply, _ = await text_generate_async(client, selection_chat)

        selected_idx = extract_selected_index(sel_reply)  # 1-based
        if selected_idx is None or selected_idx < 1 or selected_idx > len(valid_replies):
            # 选择失败则默认取第一个
            selected_idx = 1
        best_reply = valid_replies[selected_idx - 1]
        best_reasoning = valid_reasonings[selected_idx - 1]

        # --- 步骤3：质量审核 ---
        _, score = await check_answer(client, prompt, best_reply)
        if score < SCORE_THRESHOLD:
            print(f"[SKIP] id={data_id}：评分 {score:.2f} 低于阈值 {SCORE_THRESHOLD}")
            return

        # --- 步骤4：写入输出文件 ---
        record = {
            "id": data_id,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": best_reply}
            ],
            "reasoning_content": best_reasoning,
            "score": round(score, 4)
        }
        async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"[DONE] id={data_id}，评分={score:.2f}")


async def main():
    """主函数：读取数据，断点续传，异步并发处理"""
    output_file = os.path.join(OUTPUTFILE, OUTPUTNAME)

    # 读取断点续传进度
    processed_ids = load_existing_ids(output_file)
    print(f"已处理 {len(processed_ids)} 条，从断点继续...")

    # 读取所有输入数据
    all_data = []
    with open(INPUTFILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append(json.loads(line))

    # 过滤已处理数据
    todo_data = [d for d in all_data if d['id'] not in processed_ids]
    print(f"共 {len(all_data)} 条，待处理 {len(todo_data)} 条")

    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [process_one(client, sem, data, output_file)
                 for data in todo_data]
        # 使用 tqdm 显示整体进度
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="合成进度"):
            try:
                await coro
            except Exception as e:
                print(f"[ERROR] 处理任务时发生异常: {e}")

    print(f"全部完成，结果已保存至 {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
