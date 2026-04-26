# -*- coding: utf-8 -*-
"""
指令多样化改写核心引擎（v2 - 适配结构化版本管理格式）

设计：
- 输入：结构化版本JSON（如 v2.0.0.json），包含 task_prompt + output_formate
- task_prompt 中的 {medical_text} 是不可变占位符
- output_formate 完全不参与改写
- 输出：同格式JSON，包含改写后的变体
"""

import os
import re
import json
import copy
import random
import asyncio
from typing import Optional
from datetime import datetime

from api_config import async_text_generate_think, close_async_client
from prompts import (
    REWRITE_SYSTEM_PROMPT, REWRITE_USER_PROMPT,
    VERIFY_SYSTEM_PROMPT, VERIFY_USER_PROMPT,
    REWRITE_LEVELS
)


# ============================================================
# 常量
# ============================================================

MEDICAL_TEXT_PLACEHOLDER = "{medical_text}"


# ============================================================
# 工具函数
# ============================================================

def load_version_file(path: str) -> list[dict]:
    """
    加载结构化版本JSON文件。
    
    返回有效条目列表（过滤掉 task_prompt 为空的条目）。
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = [e for e in data if isinstance(e, dict) and e.get("task_prompt", "").strip()]
    return entries


def clean_rewrite_output(text: str) -> str:
    """清理LLM改写输出：去掉可能的代码块包裹"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def validate_placeholder(original: str, rewritten: str) -> bool:
    """
    检查 {medical_text} 占位符是否被正确保留。
    
    规则：如果原文包含占位符，改写结果也必须包含。
    """
    if MEDICAL_TEXT_PLACEHOLDER in original:
        return MEDICAL_TEXT_PLACEHOLDER in rewritten
    return True


def extract_code_blocks(text: str) -> list[str]:
    """提取文本中的所有 ``` 代码块（含围栏标记）"""
    return re.findall(r'```[^\n]*\n[\s\S]*?```', text)


def repair_code_blocks(original: str, rewritten: str) -> str:
    """
    修复改写后文本中被LLM篡改/新增的代码块。
    
    策略：
    - 原文无代码块 + 改写有代码块 → 删除发明的代码块
    - 原文有代码块 + 改写有代码块 → 用原始代码块按序替换
    - 原文有代码块 + 改写无代码块 → 追加原始代码块
    """
    orig_blocks = extract_code_blocks(original)
    rewr_blocks = extract_code_blocks(rewritten)
    
    # 两者都没有代码块，无需处理
    if not orig_blocks and not rewr_blocks:
        return rewritten
    
    # 原文无代码块，改写自行发明了 → 删除
    if not orig_blocks and rewr_blocks:
        result = rewritten
        for block in rewr_blocks:
            result = result.replace(block, "")
        print(f"    [FIX] 删除了 {len(rewr_blocks)} 个LLM自行发明的代码块")
        return re.sub(r'\n{3,}', '\n\n', result).strip()
    
    # 原文有代码块，改写丢失了 → 追加
    if orig_blocks and not rewr_blocks:
        print(f"    [FIX] 改写丢失了 {len(orig_blocks)} 个代码块，已追加恢复")
        return rewritten.rstrip() + "\n\n" + "\n".join(orig_blocks)
    
    # 都有代码块 → 按顺序用原始替换改写的
    result = rewritten
    for i, rewr_block in enumerate(rewr_blocks):
        if i < len(orig_blocks):
            result = result.replace(rewr_block, orig_blocks[i], 1)
        else:
            # 改写多出的代码块 → 删除
            result = result.replace(rewr_block, "", 1)
    
    replaced = min(len(rewr_blocks), len(orig_blocks))
    removed = max(0, len(rewr_blocks) - len(orig_blocks))
    if replaced > 0 or removed > 0:
        print(f"    [FIX] 代码块修复: 替换{replaced}个, 删除多余{removed}个")
    
    return re.sub(r'\n{3,}', '\n\n', result).strip()


# ============================================================
# 核心改写逻辑
# ============================================================

async def rewrite_task_prompt(
    task_prompt: str,
    level: str = "medium",
    output_formate: str = "",
    sem: Optional[asyncio.Semaphore] = None,
    max_retries: int = 3
) -> str:
    """
    改写单个 task_prompt（含单次调用级别重试）。
    
    参数:
        task_prompt: 原始任务指令文本
        level: 改写强度 ("light" / "medium" / "heavy")
        output_formate: 该条目的输出格式定义（传入作为上下文，防止LLM发明新格式）
        sem: 并发信号量
        max_retries: 单条改写的最大重试次数（默认 3）
    
    返回:
        改写后的文本。失败返回空字符串。
    """
    for attempt in range(1, max_retries + 1):
        level_config = REWRITE_LEVELS.get(level, REWRITE_LEVELS["medium"])
        
        # 构建 output_formate 上下文段落（有内容才展示）
        if output_formate and output_formate.strip():
            output_formate_section = (
                f"\n## 该任务的标准输出格式（由 output_formate 字段独立管理，严禁修改）\n\n"
                f"{output_formate}\n\n"
                f"⚠️ 改写时：原文有格式块则保持原样，原文无格式块则不可自行添加。\n"
            )
        else:
            output_formate_section = "\n"
        
        user_prompt = REWRITE_USER_PROMPT.format(
            rewrite_level=level_config["name"],
            rewrite_guidance=level_config["guidance"],
            task_prompt=task_prompt,
            output_formate_section=output_formate_section
        )
        
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        temperature = {"light": 0.6, "medium": 0.75, "heavy": 0.85}.get(level, 0.7)
        # 重试时微调 temperature 以增加多样性
        if attempt > 1:
            temperature = min(0.95, temperature + 0.05 * (attempt - 1))
        
        if sem:
            async with sem:
                reply = await async_text_generate_think(messages, temperature=temperature)
        else:
            reply = await async_text_generate_think(messages, temperature=temperature)
        
        if not reply:
            print(f"    [RETRY] 第{attempt}次改写API调用无返回，{'重试中...' if attempt < max_retries else '已达上限'}")
            continue
        
        result = clean_rewrite_output(reply)
        
        # 校验占位符
        if not validate_placeholder(task_prompt, result):
            print(f"    [RETRY] 第{attempt}次改写丢失了 {{medical_text}} 占位符，{'重试中...' if attempt < max_retries else '已达上限'}")
            continue
        
        # 代码块格式保护：修复被LLM篡改/新增的代码块
        result = repair_code_blocks(task_prompt, result)
        
        return result
    
    return ""


async def rewrite_with_variants(
    task_prompt: str,
    n_variants: int = 5,
    output_formate: str = "",
    sem: Optional[asyncio.Semaphore] = None,
    max_retries: int = 3
) -> list[dict]:
    """
    为一个 task_prompt 生成多个改写变体（轻度/中度/重度混合）。
    
    权重设计：轻度最低（~15%），中度为主（~55%），重度适中（~30%）
    硬性保证：每条指令至少包含 1 个中度或重度变体。
    
    返回:
        [{"level": "medium", "rewritten": "改写后文本"}, ...]
    """
    # 按权重分配改写强度：light=15%, medium=55%, heavy=30%
    levels = []
    for _ in range(n_variants):
        r = random.random()
        if r < 0.15:
            levels.append("light")
        elif r < 0.70:
            levels.append("medium")
        else:
            levels.append("heavy")
    
    # 硬性保证：至少 1 个中度或重度（轻度不是有效的多样性贡献者）
    if n_variants >= 1 and all(lvl == "light" for lvl in levels):
        # 将第一个轻度替换为中度（保留至少1个实质性改写）
        levels[0] = "medium"
        print(f"    [GUARD] 全为轻度，已将第1个替换为中度")
    
    async def _do_one(lvl):
        rewritten = await rewrite_task_prompt(task_prompt, lvl, output_formate, sem, max_retries)
        return {"level": lvl, "rewritten": rewritten}
    
    tasks = [_do_one(lvl) for lvl in levels]
    results = await asyncio.gather(*tasks)
    
    # 过滤空结果
    return [r for r in results if r["rewritten"]]


# ============================================================
# 语义一致性校验
# ============================================================

async def verify_rewrite(
    original_prompt: str,
    rewritten_prompt: str,
    sem: Optional[asyncio.Semaphore] = None
) -> dict:
    """
    校验改写后的 task_prompt 是否保留了所有硬锚点。
    
    返回:
        {"pass": True/False, "checks": {...}, "summary": "..."}
    """
    user_prompt = VERIFY_USER_PROMPT.format(
        original_prompt=original_prompt,
        rewritten_prompt=rewritten_prompt
    )
    
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    if sem:
        async with sem:
            reply = await async_text_generate_think(messages, temperature=0.3)
    else:
        reply = await async_text_generate_think(messages, temperature=0.3)
    
    if not reply:
        return {"pass": False, "checks": {}, "summary": "验证API调用失败"}
    
    # 解析JSON结果
    try:
        json_match = re.search(r'\{[\s\S]*\}', reply)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, Exception):
        pass
    
    # 降级：关键词判断
    reply_lower = reply.lower()
    passed = "不通过" not in reply and "未通过" not in reply and "false" not in reply_lower
    return {"pass": passed, "checks": {}, "summary": reply[:200]}


# ============================================================
# 单条目处理
# ============================================================

async def process_prompt_entry(
    entry: dict,
    n_variants: int,
    sem: asyncio.Semaphore,
    verify: bool,
    max_retries: int = 3
) -> list[dict]:
    """
    处理版本JSON中的单个指令条目，返回改写变体列表。
    
    每个变体保持与原始条目相同的JSON结构。
    output_formate 原封不动保留。
    
    当所有变体均生成失败时，自动重试直到至少获得一条有效结果。
    """
    task_prompt = entry["task_prompt"]
    original_id = entry.get("id", "?")
    
    print(f"\n--- 指令 [id={original_id}] 开始改写 ({n_variants} 个变体) ---")
    desc = entry.get('description', 'N/A')
    print(f"    描述: {desc}")
    prompt_preview = task_prompt[:80].replace('\n', ' ')
    print(f"    指令前80字: {prompt_preview}...")
    
    # 提取 output_formate 用于格式保护
    output_formate = entry.get("output_formate", "")
    if isinstance(output_formate, (dict, list)):
        output_formate = json.dumps(output_formate, ensure_ascii=False, indent=2)
    
    # ============================================================
    # 批次级重试：确保至少有一条改写结果
    # ============================================================
    all_variants = []
    
    for batch_attempt in range(1, max_retries + 1):
        # 生成变体
        variants = await rewrite_with_variants(
            task_prompt, n_variants, output_formate, sem, max_retries
        )
        
        if batch_attempt == 1:
            print(f"    第{batch_attempt}批生成 {len(variants)} 个变体")
        else:
            print(f"    [RETRY-BATCH] 第{batch_attempt}批生成 {len(variants)} 个变体")
        
        all_variants.extend(variants)
        
        if all_variants:
            break
        else:
            if batch_attempt < max_retries:
                print(f"    [RETRY-BATCH] 所有变体生成失败，整批重试 ({batch_attempt}/{max_retries})...")
            else:
                print(f"    [ERROR] 已重试 {max_retries} 批次，仍无有效改写结果，跳过此条目")
    
    if not all_variants:
        return []
    
    print(f"    最终有效变体: {len(all_variants)} 个")
    variants = all_variants
    
    # 可选语义校验
    if verify:
        verified = []
        verify_tasks = [
            verify_rewrite(task_prompt, v["rewritten"], sem) for v in variants
        ]
        verify_results = await asyncio.gather(*verify_tasks)
        
        for v, vr in zip(variants, verify_results):
            v["verified"] = vr.get("pass", False)
            if v["verified"]:
                verified.append(v)
            else:
                summary = vr.get('summary', '')[:80]
                print(f"    [REJECT] {v['level']}级变体未通过: {summary}")
        
        print(f"    通过校验: {len(verified)}/{len(variants)}")
        # 至少保留1个（如果全部未通过，保留第一个并标记）
        variants = verified if verified else variants[:1]
    
    # 组装输出条目（保持与输入相同的JSON结构）
    result_entries = []
    for i, v in enumerate(variants, 1):
        new_entry = copy.deepcopy(entry)
        new_entry["id"] = f"{original_id}_v{i}"
        new_entry["task_prompt"] = v["rewritten"]
        # output_formate 完全不动，由 deepcopy 保留
        new_entry["description"] = f"[改写-{v['level']}] {entry.get('description', '')}"
        new_entry["source_id"] = original_id
        new_entry["source_version"] = entry.get("version", "")
        new_entry["rewrite_level"] = v["level"]
        new_entry["rewrite_date"] = datetime.now().strftime("%Y-%m-%d")
        result_entries.append(new_entry)
    
    return result_entries


# ============================================================
# 主流程编排
# ============================================================

async def process_version_file(
    input_path: str,
    output_path: str,
    n_variants: int = 5,
    concurrency: int = 8,
    verify: bool = True,
    max_retries: int = 3
):
    """
    处理完整的结构化版本JSON文件。
    
    为每条有效指令生成多个改写变体，输出同格式JSON。
    """
    print(f"\n{'='*60}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"每条指令生成 {n_variants} 个变体, 并发={concurrency}, 校验={'开' if verify else '关'}, 最大重试={max_retries}")
    print(f"{'='*60}")
    
    # 加载数据
    entries = load_version_file(input_path)
    if not entries:
        print("[WARN] 文件为空或所有条目的 task_prompt 均为空")
        return
    
    print(f"有效指令条目数: {len(entries)}")
    
    sem = asyncio.Semaphore(concurrency)
    all_results = []
    
    # 逐条处理（每条内部并发生成变体）
    for entry in entries:
        results = await process_prompt_entry(entry, n_variants, sem, verify, max_retries)
        all_results.extend(results)
    
    # 写入输出
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*60}")
    print(f"总计生成 {len(all_results)} 条改写指令 → {output_path}")
    
    # 清理
    await close_async_client()
