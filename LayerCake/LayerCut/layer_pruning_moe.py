#!/usr/bin/env python3
"""
Qwen3.6-MoE Layer Pruning Tool
直接丢弃 MoE 模型中间块实现参数量压缩（低资源部署）

与 layer_stacking_moe.py 对偶:
    stacking: 选定块 -> 复制插入, identity 模式归零输出投影, 层数增加
    pruning:  选定块 -> 直接丢弃该块所有 4 层, 层数减少

Qwen3.6-35B-A3B 架构约束:
    - 40 层, BLOCK_SIZE=4, 共 10 个块 (0..9)
    - 每块 [linear_attn x 3, full_attn x 1]
    - 候选可删除集合: block 2..7（保留边界块 0/1 和 8/9）

用法:
    # 1. 预览减层计划
    python layer_pruning_moe.py \\
        --src_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --drop_blocks 5,6 \\
        --dry_run

    # 2. 执行减层 -> ~28B-A3B (40层 -> 32层)
    python layer_pruning_moe.py \\
        --src_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --dst_dir /workspace/.../Qwen3.6-28B-A3B-pruned \\
        --drop_blocks 5,6

    # 3. 更激进减层 -> ~21B-A3B (40层 -> 24层)
    python layer_pruning_moe.py \\
        --src_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --dst_dir /workspace/.../Qwen3.6-21B-A3B-pruned \\
        --drop_blocks 3,4,5,6
"""

import argparse
import glob
import json
import os
import re
import shutil
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from safetensors.torch import load_file, save_file

# ===== 常量（与 layer_stacking_moe.py 对齐）=====
BLOCK_SIZE = 4
BOUNDARY_PROTECT = 2
LAYER_KEY_PATTERN = re.compile(r'^(model\.language_model\.layers\.)(\d+)(\..+)$')

COPY_FILES = [
    'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
    'chat_template.jinja', 'preprocessor_config.json', 'video_preprocessor_config.json',
    'generation_config.json', 'LICENSE', 'README.md', 'configuration.json',
]


def parse_args():
    p = argparse.ArgumentParser(
        description='Qwen3.6-MoE Layer Pruning Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src_dir', type=str, required=True, help='原始 MoE 模型目录')
    p.add_argument('--dst_dir', type=str, default=None,
                   help='输出模型目录（dry_run 模式可省略）')
    p.add_argument('--drop_blocks', type=str, required=True,
                   help='要丢弃的块编号, 逗号分隔, 例如: 5,6')
    p.add_argument('--max_shard_bytes', type=int, default=5 * 1024**3,
                   help='每个 safetensors 分片的最大字节数, 默认 5GB')
    p.add_argument('--allow_boundary', action='store_true',
                   help='危险选项: 允许删除边界块 (不推荐)')
    p.add_argument('--dry_run', action='store_true', help='仅预览')
    return p.parse_args()


def validate_drop_blocks(drop_blocks, num_blocks, allow_boundary):
    if not drop_blocks:
        raise ValueError("--drop_blocks 不能为空")
    if len(drop_blocks) != len(set(drop_blocks)):
        raise ValueError(f"--drop_blocks 含重复: {drop_blocks}")
    for g in drop_blocks:
        if g < 0 or g >= num_blocks:
            raise ValueError(f"块编号 {g} 超出范围 [0, {num_blocks - 1}]")
    if len(drop_blocks) >= num_blocks - 2 * BOUNDARY_PROTECT:
        raise ValueError(
            f"--drop_blocks 数量 ({len(drop_blocks)}) 过大, 候选池最多 "
            f"{num_blocks - 2 * BOUNDARY_PROTECT} 块")

    low = BOUNDARY_PROTECT
    high = num_blocks - BOUNDARY_PROTECT
    boundary_selected = [g for g in drop_blocks if g < low or g >= high]
    if boundary_selected and not allow_boundary:
        raise ValueError(
            f"禁止删除边界块 {boundary_selected}! "
            f"候选范围应为 [{low}, {high - 1}]. "
            f"如强行删除请加 --allow_boundary (风险自负)")


def build_pruning_layer_mapping(num_original_layers, drop_blocks):
    """
    构建减层后 new -> old 映射。被删块的 4 层整体跳过, 其余层按原顺序重编号。

    Returns:
        layer_map: {new_layer_idx: old_layer_idx}
        new_total_layers
    """
    num_blocks = num_original_layers // BLOCK_SIZE
    drop_set = set(drop_blocks)
    layer_map = OrderedDict()
    new_idx = 0

    for block_idx in range(num_blocks):
        if block_idx in drop_set:
            continue
        base_old = block_idx * BLOCK_SIZE
        for offset in range(BLOCK_SIZE):
            layer_map[new_idx] = base_old + offset
            new_idx += 1

    return layer_map, new_idx


def build_new_layer_types(old_layer_types, drop_blocks):
    """根据 drop_blocks 跳过对应的 4 项。"""
    num_blocks = len(old_layer_types) // BLOCK_SIZE
    drop_set = set(drop_blocks)
    new_types = []
    for block_idx in range(num_blocks):
        if block_idx in drop_set:
            continue
        new_types.extend(
            old_layer_types[block_idx * BLOCK_SIZE: (block_idx + 1) * BLOCK_SIZE])
    return new_types


def estimate_params(total_size_bytes, old_num_layers, new_num_layers):
    """bf16: 2 bytes per param。"""
    old_params_b = total_size_bytes / 2 / 1e9
    params_per_layer = old_params_b / old_num_layers
    delta = new_num_layers - old_num_layers
    new_params_b = old_params_b + delta * params_per_layer
    return old_params_b, new_params_b


def estimate_active_params(num_layers, config):
    """每 token 激活参数量 (单位 B)。"""
    tc = config['text_config']
    hidden = tc['hidden_size']
    num_heads = tc['num_attention_heads']
    head_dim = tc['head_dim']
    kv_heads = tc['num_key_value_heads']
    moe_inter = tc['moe_intermediate_size']
    shared_inter = tc['shared_expert_intermediate_size']
    top_k = tc['num_experts_per_tok']

    attn_params = (num_heads * head_dim * hidden
                   + kv_heads * head_dim * hidden
                   + kv_heads * head_dim * hidden
                   + num_heads * head_dim * hidden)
    expert_params = top_k * (moe_inter * hidden * 3)
    shared_params = shared_inter * hidden * 3
    total_per_layer = attn_params + expert_params + shared_params
    return total_per_layer * num_layers / 1e9


def print_pruning_plan(old_num_layers, layer_map, new_num_layers, drop_blocks,
                       old_layer_types, config, total_size_bytes):
    num_old_blocks = old_num_layers // BLOCK_SIZE
    num_new_blocks = new_num_layers // BLOCK_SIZE
    removed_layers = old_num_layers - new_num_layers
    removed_blocks = removed_layers // BLOCK_SIZE

    old_params, new_params = estimate_params(
        total_size_bytes, old_num_layers, new_num_layers)
    old_active = estimate_active_params(old_num_layers, config)
    new_active = estimate_active_params(new_num_layers, config)

    tc = config['text_config']
    num_experts = tc.get('num_experts', '?')
    top_k = tc.get('num_experts_per_tok', '?')

    print("=" * 70)
    print("              Qwen3.6-MoE Layer Pruning Plan")
    print("=" * 70)
    print(f"  架构:           {config.get('architectures', ['?'])[0]}")
    print(f"  MoE 配置:       {num_experts} 专家, top-{top_k} 激活")
    print(f"  原始层数:       {old_num_layers} 层 ({num_old_blocks} 个块)")
    print(f"  新层数:         {new_num_layers} 层 ({num_new_blocks} 个块)")
    print(f"  丢弃层数:       -{removed_layers} 层 (-{removed_blocks} 个块)")
    print(f"  丢弃的块:       {sorted(drop_blocks)}")
    print(f"  总参数量:       ~{old_params:.1f}B → ~{new_params:.1f}B")
    print(f"  激活参数量:     ~{old_active:.1f}B → ~{new_active:.1f}B (每token)")
    print("-" * 70)

    sorted_drops = sorted(drop_blocks)
    adjacent_pairs = [(sorted_drops[i], sorted_drops[i+1])
                      for i in range(len(sorted_drops) - 1)
                      if sorted_drops[i+1] - sorted_drops[i] == 1]
    if adjacent_pairs:
        print(f"\n  ⚠️  警告: 检测到相邻块同时丢弃 {adjacent_pairs}")
        print(f"     残差流分布可能出现明显跳变, 强烈建议 CPT 恢复 token 量翻倍。")

    print(f"\n  块结构映射 (每块 = [linear×3, full×1]):\n")
    drop_set = set(drop_blocks)
    new_block_idx = 0
    for old_block in range(num_old_blocks):
        old_layers = f"L{old_block*4:02d}-{old_block*4+3:02d}"
        if old_block in drop_set:
            print(f"    Block {old_block} ({old_layers})  →  [DROPPED]")
        else:
            new_layers = f"L{new_block_idx*4:02d}-{new_block_idx*4+3:02d}"
            print(f"    Block {old_block} ({old_layers})  →  "
                  f"NewBlock {new_block_idx:2d} ({new_layers})")
            new_block_idx += 1

    new_layer_types = build_new_layer_types(old_layer_types, drop_blocks)
    pattern_ok = True
    expected = ["linear_attention", "linear_attention",
                "linear_attention", "full_attention"]
    for i in range(0, len(new_layer_types), BLOCK_SIZE):
        if new_layer_types[i:i+BLOCK_SIZE] != expected:
            pattern_ok = False
            break
    print(f"\n  layer_types 模式校验: "
          f"{'✅ 全部符合 [linear×3, full×1]' if pattern_ok else '❌ 模式异常!'}")
    print("=" * 70)


def load_all_safetensors(src_dir):
    """加载所有 safetensors 分片 (兼容 MoE/Dense 多种命名)。"""
    shard_files = sorted(glob.glob(os.path.join(src_dir, "model-*-of-*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(
            os.path.join(src_dir, "model.safetensors-*-of-*.safetensors")))
    if not shard_files:
        single = os.path.join(src_dir, "model.safetensors")
        if os.path.exists(single):
            shard_files = [single]
        else:
            raise FileNotFoundError(f"在 {src_dir} 中未找到 safetensors 文件")

    print(f"\n  加载 {len(shard_files)} 个权重分片...")
    all_weights = OrderedDict()
    for i, sf in enumerate(shard_files):
        print(f"    [{i+1}/{len(shard_files)}] {os.path.basename(sf)} ...",
              end=" ", flush=True)
        shard = load_file(sf, device="cpu")
        all_weights.update(shard)
        print(f"({len(shard)} tensors)")
        del shard
    print(f"  总计加载 {len(all_weights)} 个张量")
    return all_weights


def remap_weights(all_weights, layer_map, drop_blocks):
    """
    仅重命名保留层的键, 丢弃被删块的所有层权重。
    非层级权重 (embedding / lm_head / visual / mtp 等) 原样保留。
    """
    drop_layer_set = set()
    for b in drop_blocks:
        for offset in range(BLOCK_SIZE):
            drop_layer_set.add(b * BLOCK_SIZE + offset)

    old_to_new = {old: new for new, old in layer_map.items()}

    new_weights = OrderedDict()
    kept_layer_keys = 0
    dropped_layer_keys = 0
    non_layer_keys = 0

    print(f"\n  重映射权重键名 (pruning)...")
    for key, tensor in all_weights.items():
        match = LAYER_KEY_PATTERN.match(key)
        if match:
            prefix, layer_idx_str, suffix = match.groups()
            old_idx = int(layer_idx_str)
            if old_idx in drop_layer_set:
                dropped_layer_keys += 1
                continue
            if old_idx not in old_to_new:
                print(f"    ⚠️  层 {old_idx} 不在映射中也不在 drop 中, 跳过: {key}")
                continue
            new_idx = old_to_new[old_idx]
            new_key = f"{prefix}{new_idx}{suffix}"
            new_weights[new_key] = tensor
            kept_layer_keys += 1
        else:
            new_weights[key] = tensor
            non_layer_keys += 1

    print(f"  保留层级权重键: {kept_layer_keys}")
    print(f"  丢弃层级权重键: {dropped_layer_keys}")
    print(f"  非层级权重键:   {non_layer_keys} (embedding/lm_head/visual/mtp 等)")
    print(f"  新模型共 {len(new_weights)} 个张量")
    return new_weights


def save_sharded_safetensors(weights, dst_dir, max_shard_bytes):
    tensor_sizes = {k: v.numel() * v.element_size() for k, v in weights.items()}
    total_size = sum(tensor_sizes.values())

    shards = []
    current_shard = OrderedDict()
    current_size = 0
    for key in weights:
        t_size = tensor_sizes[key]
        if current_size + t_size > max_shard_bytes and len(current_shard) > 0:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0
        current_shard[key] = weights[key]
        current_size += t_size
    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}
    print(f"\n  保存 {num_shards} 个分片 (总计 {total_size / 1024**3:.2f} GB)...")
    for i, shard in enumerate(shards):
        if num_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        filepath = os.path.join(dst_dir, filename)
        shard_size = sum(v.numel() * v.element_size() for v in shard.values())
        print(f"    [{i+1}/{num_shards}] {filename} ({len(shard)} tensors, "
              f"{shard_size / 1024**3:.2f} GB)")
        save_file(shard, filepath)
        for key in shard:
            weight_map[key] = filename

    if num_shards > 1:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        index_path = os.path.join(dst_dir, "model.safetensors.index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"  已写入 {index_path}")


def update_and_save_config(src_dir, dst_dir, new_num_layers, drop_blocks):
    config_path = os.path.join(src_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    text_config = config['text_config']
    old_layer_types = text_config['layer_types']
    text_config['num_hidden_layers'] = new_num_layers
    text_config['layer_types'] = build_new_layer_types(old_layer_types, drop_blocks)

    dst_config_path = os.path.join(dst_dir, 'config.json')
    with open(dst_config_path, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\n  config.json 已更新:")
    print(f"    num_hidden_layers: {len(old_layer_types)} → {new_num_layers}")
    print(f"    layer_types: {len(old_layer_types)} 项 → "
          f"{len(text_config['layer_types'])} 项")


def copy_auxiliary_files(src_dir, dst_dir):
    copied = []
    for filename in COPY_FILES:
        src_path = os.path.join(src_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, filename))
            copied.append(filename)
    if copied:
        print(f"\n  已复制辅助文件: {', '.join(copied)}")


def main():
    args = parse_args()
    drop_blocks = [int(x.strip()) for x in args.drop_blocks.split(',') if x.strip()]

    config_path = os.path.join(args.src_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    text_config = config['text_config']
    old_num_layers = text_config['num_hidden_layers']
    old_layer_types = text_config['layer_types']
    num_blocks = old_num_layers // BLOCK_SIZE

    if 'num_experts' not in text_config:
        print("  ⚠️  警告: config 中未找到 num_experts, 该模型可能不是 MoE 架构。")
        print("  Dense 模型请使用其他工具或改写 Dense 版本。")
        return

    print(f"\n  检测到 MoE 模型: {config.get('architectures', ['?'])[0]}")
    print(f"  专家数: {text_config['num_experts']}, "
          f"每token激活: {text_config['num_experts_per_tok']}")

    validate_drop_blocks(drop_blocks, num_blocks, args.allow_boundary)

    index_path = os.path.join(args.src_dir, 'model.safetensors.index.json')
    total_size_bytes = 0
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        total_size_bytes = int(index_data.get('metadata', {}).get('total_size', 0))

    layer_map, new_num_layers = build_pruning_layer_mapping(old_num_layers, drop_blocks)

    print_pruning_plan(old_num_layers, layer_map, new_num_layers, drop_blocks,
                       old_layer_types, config, total_size_bytes)

    if args.dry_run:
        print(f"\n  [DRY RUN] 仅预览, 未实际执行。去掉 --dry_run 参数以执行。\n")
        return

    if not args.dst_dir:
        raise ValueError("非 dry_run 模式必须指定 --dst_dir")
    os.makedirs(args.dst_dir, exist_ok=True)

    all_weights = load_all_safetensors(args.src_dir)

    new_weights = remap_weights(all_weights, layer_map, drop_blocks)

    del all_weights
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    save_sharded_safetensors(new_weights, args.dst_dir, args.max_shard_bytes)

    del new_weights
    import gc; gc.collect()

    update_and_save_config(args.src_dir, args.dst_dir, new_num_layers, drop_blocks)

    copy_auxiliary_files(args.src_dir, args.dst_dir)

    old_p, new_p = estimate_params(total_size_bytes, old_num_layers, new_num_layers)
    old_a = estimate_active_params(old_num_layers, config)
    new_a = estimate_active_params(new_num_layers, config)

    print("\n" + "=" * 70)
    print("  ✅ MoE 层剪枝完成!")
    print(f"  输出目录: {args.dst_dir}")
    print(f"  新模型: {old_num_layers}层 → {new_num_layers}层")
    print(f"  总参数: ~{old_p:.1f}B → ~{new_p:.1f}B")
    print(f"  激活参数: ~{old_a:.1f}B → ~{new_a:.1f}B")
    print(f"  丢弃块: {sorted(drop_blocks)}")
    print()
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
