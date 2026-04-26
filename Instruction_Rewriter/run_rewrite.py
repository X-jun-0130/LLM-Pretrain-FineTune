# -*- coding: utf-8 -*-
"""
指令多样化合成 - 主入口（v2 - 适配结构化版本管理格式）

使用方法:

1. 改写指定版本文件（输出到同目录）:
   python run_rewrite.py --input /path/to/v2.0.0.json

2. 指定变体数量:
   python run_rewrite.py --input /path/to/v2.0.0.json --n_variants 8

3. 快速测试（1个变体，不校验，展示对比预览）:
   python run_rewrite.py --input /path/to/v2.0.0.json --test

4. 指定输出路径:
   python run_rewrite.py --input /path/to/v2.0.0.json --output /path/to/output.json

5. 跳过校验（更快，但可能引入质量问题）:
   python run_rewrite.py --input /path/to/v2.0.0.json --no_verify
"""

import os
import sys
import json
import asyncio
import argparse

from instruction_rewriter import process_version_file, load_version_file


def generate_output_filename(input_path: str, n_variants: int) -> str:
    """
    根据输入文件名自动生成输出文件名。
    输出到同目录。
    
    示例: v2.0.0.json → v2.0.0_diverse_5x.json
    """
    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"{base_name}_diverse_{n_variants}x.json"
    return os.path.join(dir_name, output_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="医疗指令多样化合成工具（结构化版本管理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="输入版本JSON文件路径（如 v2.0.0.json）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认自动生成，输出到同目录）")
    parser.add_argument("--n_variants", type=int, default=5,
                        help="每条指令生成的改写变体数量（默认: 5）")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="并发请求数（默认: 8）")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="单条指令改写失败时的最大重试次数（默认: 3）")
    parser.add_argument("--no_verify", action="store_true",
                        help="跳过语义一致性校验")
    parser.add_argument("--test", action="store_true",
                        help="测试模式：1个变体，不校验，输出对比预览")
    
    return parser.parse_args()


async def run_rewrite(args):
    """执行改写"""
    input_path = os.path.abspath(args.input)
    
    if not os.path.exists(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)
    
    n_variants = 1 if args.test else args.n_variants
    verify = False if args.test else (not args.no_verify)
    
    output_path = args.output
    if not output_path:
        output_path = generate_output_filename(input_path, n_variants)
    
    await process_version_file(
        input_path=input_path,
        output_path=output_path,
        n_variants=n_variants,
        concurrency=args.concurrency,
        verify=verify,
        max_retries=args.max_retries
    )
    
    # 测试模式：展示对比预览
    if args.test and os.path.exists(output_path):
        show_comparison(input_path, output_path)


def show_comparison(input_path: str, output_path: str):
    """展示原始指令与改写指令的对比"""
    print(f"\n{'='*60}")
    print("=== 改写结果预览 ===")
    print(f"{'='*60}")
    
    entries = load_version_file(input_path)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    for orig in entries:
        orig_id = orig.get("id", "?")
        # 找到该原始指令对应的改写结果
        matching = [r for r in results if r.get("source_id") == orig_id]
        
        if not matching:
            print(f"\n--- 指令 [id={orig_id}] 无改写结果 ---")
            continue
        
        print(f"\n{'─'*60}")
        print(f"指令 [id={orig_id}] - {orig.get('description', '')}")
        print(f"{'─'*60}")
        
        print(f"\n【原始 task_prompt（前400字）】")
        print(orig['task_prompt'][:400])
        if len(orig['task_prompt']) > 400:
            print("...(截断)")
        
        for rewr in matching:
            level = rewr.get("rewrite_level", "?")
            print(f"\n【改写后 task_prompt - {level}级（前400字）】")
            print(rewr['task_prompt'][:400])
            if len(rewr['task_prompt']) > 400:
                print("...(截断)")
        
        # 检查 output_formate 是否保持一致
        orig_fmt = orig.get('output_formate', '')
        rewr_fmt = matching[0].get('output_formate', '')
        fmt_ok = orig_fmt == rewr_fmt
        print(f"\n【output_formate 保持不变】: {'✓ 是' if fmt_ok else '✗ 否(!)' }")


def main():
    args = parse_args()
    asyncio.run(run_rewrite(args))


if __name__ == "__main__":
    main()




#python run_rewrite.py --input v1.0.0.json --max_retries 5
