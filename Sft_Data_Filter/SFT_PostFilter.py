# -*- coding: utf-8 -*-
"""
SFT 数据后处理筛选脚本（单文件版）

筛选逻辑（按 correct_count 分层处理）：
    - cc 缺失/0     → 丢弃
    - cc = 1~4      → 全部保留
    - cc = 5        → 随机保留 30%
    - cc = 6        → 随机保留 20%
    - cc >= 7       → 随机保留 5%

使用方式：
    python3 SFT_PostFilter.py --input-file INPUT.jsonl --output OUTPUT.jsonl [--seed 42]
"""

import json
import os
import random
import argparse
from collections import defaultdict


# ============ 默认配置 ============
DEFAULT_SEED = 42

# 分层抽样比例: {cc_value: 保留比例}，不在此表中且 cc>=1 的全部保留
SAMPLE_RATES = {
    5: 0.30,
    6: 0.20,
    7: 0.05,   # cc>=7 统一用此比例
}


def load_jsonl(filepath: str) -> list[dict]:
    """读取 jsonl 文件"""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def filter_items(items: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """
    对单文件数据执行分层筛选
    
    Returns:
        (filtered_items, stats_dict)
    """
    sample_start = min(SAMPLE_RATES.keys())   # 5
    max_rate_key = max(SAMPLE_RATES.keys())   # 7
    
    stats = {
        "total": len(items),
        "dropped_invalid": 0,   # cc缺失/0 丢弃
        "kept_full": 0,         # cc=1~4 全保留
        "detail": {},           # {层级: (pool_size, kept_size, rate)}
    }
    
    full_keep = []              # cc=1~4 全保留
    cc_pools = defaultdict(list)  # cc=5, 6, 7+ 分桶
    
    for item in items:
        cc = item.get("correct_count")
        if cc is None or cc <= 0:
            stats["dropped_invalid"] += 1
            continue
        if cc < sample_start:
            full_keep.append(item)
        else:
            # cc >= 7 统一归入 max_rate_key 桶
            bucket = cc if cc <= max_rate_key else max_rate_key
            cc_pools[bucket].append(item)
    
    stats["kept_full"] = len(full_keep)
    
    # 对每个桶按比例抽样
    sampled = []
    for cc_val in sorted(cc_pools.keys()):
        pool = cc_pools[cc_val]
        rate = SAMPLE_RATES.get(cc_val, SAMPLE_RATES[max_rate_key])
        n_keep = max(1, round(len(pool) * rate)) if pool else 0
        kept = rng.sample(pool, min(n_keep, len(pool)))
        sampled.extend(kept)
        label = f"cc={cc_val}" if cc_val < max_rate_key else f"cc>={cc_val}"
        stats["detail"][label] = (len(pool), len(kept), rate)
    
    filtered = full_keep + sampled
    return filtered, stats


def print_report(stats: dict, total_kept: int) -> None:
    """打印详细分层筛选报告"""
    sample_start = min(SAMPLE_RATES.keys())
    valid = stats["total"] - stats["dropped_invalid"]
    kept_pct = f"{total_kept/valid*100:.1f}%" if valid else "N/A"
    
    print(f"\n{'='*60}")
    print(f"📊 筛选报告")
    print(f"{'='*60}")
    print(f"  原始总数: {stats['total']}")
    print(f"  丢弃 (cc缺失/0): {stats['dropped_invalid']}")
    print(f"  有效数据: {valid}")
    print(f"\n  分层保留详情:")
    print(f"    • cc=1~{sample_start-1} 全保留:  {stats['kept_full']:>6} 条 (100%)")
    for label in sorted(stats["detail"].keys()):
        pool, kept, rate = stats["detail"][label]
        pct = f"{rate:.0%}"
        print(f"    • {label:8s} 抽样保留: {kept:>6} / {pool:<6} ({pct:>3})")
    print(f"\n  最终保留: {total_kept} 条 ({kept_pct} of 有效数据)")
    print(f"{'='*60}")


def main():
    input_file = "/data/final_data.jsonl"
    output = "/data/sft.jsonl"
    
    seed = 42
    rng = random.Random(seed)
    
    print(f"{'='*60}")
    print(f"SFT 数据筛选（单文件模式）")
    print(f"{'='*60}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output}")
    print(f"分层策略: cc<=0 丢弃 | cc=1~4 全保留 | cc=5 保留30% | cc=6 保留20% | cc>=7 保留5%")
    print(f"随机种子: {seed}")
    
    # 加载数据
    items = load_jsonl(input_file)
    if not items:
        print("[ERROR] 文件无有效数据")
        return
    
    # 筛选
    filtered, stats = filter_items(items, rng)
    
    # 打乱顺序
    rng.shuffle(filtered)
    
    # 计算总保留数
    total_kept = stats["kept_full"] + sum(k for _, k, _ in stats["detail"].values())
    
    # 打印报告
    print_report(stats, total_kept)
    
    # 写入输出
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for item in filtered:
            out = item
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 完成！结果已保存至: {output}")


if __name__ == "__main__":
    main()
