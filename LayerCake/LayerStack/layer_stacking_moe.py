#!/usr/bin/env python3
"""
Qwen3.6-MoE Layer Stacking Tool
直接复制 MoE 模型中间层块实现参数量扩展（无需额外训练）

Qwen3.6-35B-A3B 的层以 [linear_attn × 3, full_attn × 1] 为一个块（4层一组），
40层 = 10个块（编号0-9）。每层包含 MoE 结构（256个路由专家 + 共享专家，8个激活）。

与 Dense 版本的关键区别:
  - MLP 使用 MoE: 路由专家 experts.down_proj (packed, 无.weight后缀) + 共享专家
  - identity 模式归零: experts.down_proj + shared_expert.down_proj (而非 mlp.down_proj)
  - 模型文件命名: model-XXXXX-of-XXXXX.safetensors

用法:
    # 1. 预览堆叠计划
    python layer_stacking_moe.py \\
        --src_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --dup_groups 3,4,5,6 \\
        --dry_run

    # 2. 执行堆叠 → ~49B-A4B（40层→56层）
    python layer_stacking_moe.py \\
        --src_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --dst_dir /data1/Model-TH/Qwen3.6-49B-A4B-stacked \\
        --dup_groups 3,4,5,6

    # 3. 执行堆叠 → ~70B-A6B（40层→80层）
    python layer_stacking_moe.py \\
        --src_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --dst_dir /data1/Model-TH/Qwen3.6-70B-A6B-stacked \\
        --dup_groups 0,1,2,3,4,5,6,7,8,9
"""

import argparse
import json
import os
import re
import shutil
import glob
from collections import OrderedDict, defaultdict

import torch
from safetensors.torch import load_file, save_file

# ===== 常量 =====
BLOCK_SIZE = 4  # 每个块包含 4 层: [linear, linear, linear, full]
LAYER_KEY_PATTERN = re.compile(r'^(model\.language_model\.layers\.)(\d+)(\..+)$')

# MoE 模型中，复制层需要归零/缩放的"输出投影"权重后缀
# 归零这些权重 → 层变为 identity（纯残差直通）
#
# 与 Dense 模型的区别:
#   Dense MLP:  .mlp.down_proj.weight
#   MoE MLP:    .mlp.experts.down_proj (packed tensor, 无.weight后缀!)
#             + .mlp.shared_expert.down_proj.weight
OUTPUT_PROJ_SUFFIXES = (
    '.linear_attn.out_proj.weight',         # linear_attention 层的输出投影
    '.self_attn.o_proj.weight',             # full_attention 层的输出投影
    '.mlp.experts.down_proj',               # MoE: 全部路由专家的输出投影 (packed tensor)
    '.mlp.shared_expert.down_proj.weight',  # MoE: 共享专家的输出投影
)

# 需要原样复制的非权重文件
COPY_FILES = [
    'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
    'chat_template.jinja', 'preprocessor_config.json', 'video_preprocessor_config.json',
    'generation_config.json', 'LICENSE', 'README.md', 'configuration.json',
]


def parse_args():
    p = argparse.ArgumentParser(
        description='Qwen3.6-MoE Layer Stacking Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src_dir', type=str, required=True,
                   help='原始 MoE 模型目录路径')
    p.add_argument('--dst_dir', type=str, default=None,
                   help='输出模型目录路径（dry_run模式可省略）')
    p.add_argument('--dup_groups', type=str, default='3,4,5,6',
                   help='要复制的块编号，逗号分隔。10个块编号0-9，默认: 3,4,5,6')
    p.add_argument('--init_mode', type=str, default='identity',
                   choices=['identity', 'scaled', 'copy'],
                   help='复制层初始化模式 (默认identity):\n'
                        '  identity - 复制层归零为直通(推荐,保留原模型能力)\n'
                        '  scaled   - 复制层权重按比例缩小\n'
                        '  copy     - 完全复制(会导致乱码,仅用于后续训练)')
    p.add_argument('--scale_factor', type=float, default=0.0,
                   help='scaled模式下复制层输出投影的缩放因子 (默认0.0)')
    p.add_argument('--max_shard_bytes', type=int, default=5 * 1024**3,
                   help='每个 safetensors 分片的最大字节数，默认5GB')
    p.add_argument('--dry_run', action='store_true',
                   help='仅预览堆叠计划，不实际执行')
    return p.parse_args()


def build_layer_mapping(num_original_layers: int, dup_groups: list[int]) -> tuple[dict, int]:
    """
    构建新旧层索引映射。复制的块插在原始块之后。

    Returns:
        layer_map: {new_layer_idx: old_layer_idx}
        new_total_layers: 新模型总层数
    """
    num_blocks = num_original_layers // BLOCK_SIZE
    dup_groups = sorted(set(dup_groups))

    for g in dup_groups:
        if g < 0 or g >= num_blocks:
            raise ValueError(f"块编号 {g} 超出范围 [0, {num_blocks - 1}]")

    layer_map = OrderedDict()
    new_idx = 0

    for block_idx in range(num_blocks):
        base_old = block_idx * BLOCK_SIZE
        # 原始块
        for offset in range(BLOCK_SIZE):
            layer_map[new_idx] = base_old + offset
            new_idx += 1
        # 复制块
        if block_idx in dup_groups:
            for offset in range(BLOCK_SIZE):
                layer_map[new_idx] = base_old + offset
                new_idx += 1

    return layer_map, new_idx


def build_new_layer_types(old_layer_types: list[str], dup_groups: list[int]) -> list[str]:
    """根据复制策略构建新的 layer_types 列表。"""
    num_blocks = len(old_layer_types) // BLOCK_SIZE
    dup_groups = sorted(set(dup_groups))
    new_types = []

    for block_idx in range(num_blocks):
        block = old_layer_types[block_idx * BLOCK_SIZE: (block_idx + 1) * BLOCK_SIZE]
        new_types.extend(block)
        if block_idx in dup_groups:
            new_types.extend(block)

    return new_types


def estimate_params(total_size_bytes: int, old_num_layers: int,
                    new_num_layers: int) -> tuple[float, float]:
    """
    基于权重文件总大小估算参数量。
    Returns: (old_params_B, new_params_B)
    """
    # 从 model.safetensors.index.json 的 total_size 推算
    # bf16: 2 bytes per param
    old_params_b = total_size_bytes / 2 / 1e9
    params_per_layer = old_params_b / old_num_layers  # 近似
    added_layers = new_num_layers - old_num_layers
    new_params_b = old_params_b + added_layers * params_per_layer
    return old_params_b, new_params_b


def estimate_active_params(num_layers: int, config: dict) -> float:
    """估算每 token 激活参数量 (单位: B)。"""
    tc = config['text_config']
    hidden = tc['hidden_size']                  # 2048
    num_heads = tc['num_attention_heads']        # 16
    head_dim = tc['head_dim']                    # 256
    kv_heads = tc['num_key_value_heads']         # 2
    moe_inter = tc['moe_intermediate_size']      # 512
    shared_inter = tc['shared_expert_intermediate_size']  # 512
    top_k = tc['num_experts_per_tok']            # 8

    # 注意力: Q + K + V + O
    attn_params = (num_heads * head_dim * hidden  # Q
                   + kv_heads * head_dim * hidden  # K
                   + kv_heads * head_dim * hidden  # V (简化, linear_attn 不同但量级类似)
                   + num_heads * head_dim * hidden)  # O

    # MoE MLP: top_k 个路由专家 + 1 个共享专家
    expert_params = top_k * (moe_inter * hidden * 3)  # gate+up+down
    shared_params = shared_inter * hidden * 3

    total_per_layer = attn_params + expert_params + shared_params
    return total_per_layer * num_layers / 1e9


def print_stacking_plan(old_num_layers, layer_map, new_num_layers, dup_groups,
                        old_layer_types, config, total_size_bytes,
                        init_mode='identity', scale_factor=0.0):
    """打印可视化的堆叠计划。"""
    num_old_blocks = old_num_layers // BLOCK_SIZE
    num_new_blocks = new_num_layers // BLOCK_SIZE
    added_layers = new_num_layers - old_num_layers
    added_blocks = added_layers // BLOCK_SIZE

    old_params, new_params = estimate_params(total_size_bytes, old_num_layers, new_num_layers)
    old_active = estimate_active_params(old_num_layers, config)
    new_active = estimate_active_params(new_num_layers, config)

    tc = config['text_config']
    num_experts = tc.get('num_experts', '?')
    top_k = tc.get('num_experts_per_tok', '?')

    init_label = {
        'copy': '完全复制(⚠️会乱码)',
        'identity': '直通/Identity(推荐)',
        'scaled': f'缩放×{scale_factor}'
    }

    print("=" * 70)
    print("              Qwen3.6-MoE Layer Stacking Plan")
    print("=" * 70)
    print(f"  架构:           Qwen3_5MoeForConditionalGeneration")
    print(f"  MoE 配置:       {num_experts} 专家, top-{top_k} 激活")
    print(f"  原始层数:       {old_num_layers} 层 ({num_old_blocks} 个块)")
    print(f"  新层数:         {new_num_layers} 层 ({num_new_blocks} 个块)")
    print(f"  新增层数:       +{added_layers} 层 (+{added_blocks} 个块)")
    print(f"  复制的块:       {sorted(dup_groups)}")
    print(f"  初始化模式:     {init_label.get(init_mode, init_mode)}")
    print(f"  总参数量:       ~{old_params:.1f}B → ~{new_params:.1f}B")
    print(f"  激活参数量:     ~{old_active:.1f}B → ~{new_active:.1f}B (每token)")
    print("-" * 70)

    # MoE identity 模式归零说明
    if init_mode == 'identity':
        print("\n  identity 模式归零的 MoE 输出投影:")
        for s in OUTPUT_PROJ_SUFFIXES:
            print(f"    - {s}")

    # 可视化映射
    print(f"\n  块结构映射 (每块 = [linear×3, full×1]):\n")
    new_block_idx = 0
    for old_block in range(num_old_blocks):
        old_layers = f"L{old_block*4:02d}-{old_block*4+3:02d}"
        new_layers = f"L{new_block_idx*4:02d}-{new_block_idx*4+3:02d}"
        print(f"    Block {old_block} ({old_layers})  →  NewBlock {new_block_idx:2d} ({new_layers})")
        new_block_idx += 1

        if old_block in dup_groups:
            new_layers = f"L{new_block_idx*4:02d}-{new_block_idx*4+3:02d}"
            print(f"    Block {old_block} ({old_layers})  →  NewBlock {new_block_idx:2d} ({new_layers})  ← [COPY]")
            new_block_idx += 1

    # layer_types 验证
    new_layer_types = build_new_layer_types(old_layer_types, dup_groups)
    pattern_ok = True
    for i in range(0, len(new_layer_types), BLOCK_SIZE):
        block = new_layer_types[i:i+BLOCK_SIZE]
        expected = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        if block != expected:
            pattern_ok = False
            break

    print(f"\n  layer_types 模式校验: {'✅ 全部符合 [linear×3, full×1]' if pattern_ok else '❌ 模式异常!'}")
    print("=" * 70)


def load_all_safetensors(src_dir: str) -> OrderedDict:
    """
    加载所有 safetensors 分片到一个合并的 state_dict。
    自动检测分片命名格式:
      - model-XXXXX-of-XXXXX.safetensors  (MoE 常用)
      - model.safetensors-*-of-*.safetensors  (Dense 常用)
    """
    # 尝试 MoE 命名格式
    shard_files = sorted(glob.glob(os.path.join(src_dir, "model-*-of-*.safetensors")))
    if not shard_files:
        # 尝试 Dense 命名格式
        shard_files = sorted(glob.glob(os.path.join(src_dir, "model.safetensors-*-of-*.safetensors")))
    if not shard_files:
        # 尝试单文件
        single = os.path.join(src_dir, "model.safetensors")
        if os.path.exists(single):
            shard_files = [single]
        else:
            raise FileNotFoundError(f"在 {src_dir} 中未找到 safetensors 文件")

    print(f"\n  加载 {len(shard_files)} 个权重分片...")
    all_weights = OrderedDict()
    for i, sf in enumerate(shard_files):
        print(f"    [{i+1}/{len(shard_files)}] {os.path.basename(sf)} ...", end=" ", flush=True)
        shard = load_file(sf, device="cpu")
        all_weights.update(shard)
        print(f"({len(shard)} tensors)")
        del shard

    print(f"  总计加载 {len(all_weights)} 个张量")
    return all_weights


def remap_weights(all_weights: OrderedDict, layer_map: dict,
                  init_mode: str = 'identity', scale_factor: float = 0.0) -> OrderedDict:
    """
    根据层映射重新映射权重键名。

    init_mode:
        'copy'     - 完全复制权重（双重变换，会导致乱码）
        'identity' - 复制层的输出投影归零（纯残差直通，保留原模型能力）
        'scaled'   - 复制层的输出投影按 scale_factor 缩放

    MoE 特殊处理:
        experts.down_proj 是 packed tensor (shape: [num_experts, inter_size, hidden_size]),
        归零时整个 tensor 置零，相当于所有专家输出为 0。
    """
    reverse_map = defaultdict(list)
    for new_idx, old_idx in layer_map.items():
        reverse_map[old_idx].append(new_idx)

    duplicate_new_indices = set()
    for old_idx, new_indices in reverse_map.items():
        if len(new_indices) > 1:
            for ni in new_indices[1:]:
                duplicate_new_indices.add(ni)

    new_weights = OrderedDict()
    layer_keys_processed = 0
    zeroed_keys = 0
    scaled_keys = 0

    init_mode_label = {
        'copy': '完全复制', 'identity': '直通(identity)', 'scaled': f'缩放(×{scale_factor})'
    }
    print(f"\n  重映射权重键名... (复制层初始化: {init_mode_label.get(init_mode, init_mode)})")

    for key, tensor in all_weights.items():
        match = LAYER_KEY_PATTERN.match(key)
        if match:
            prefix, layer_idx_str, suffix = match.groups()
            old_idx = int(layer_idx_str)

            if old_idx not in reverse_map:
                print(f"    ⚠️  层 {old_idx} 不在映射中，跳过: {key}")
                continue

            for new_idx in reverse_map[old_idx]:
                new_key = f"{prefix}{new_idx}{suffix}"
                is_dup = new_idx in duplicate_new_indices
                is_output_proj = key.endswith(OUTPUT_PROJ_SUFFIXES)

                if is_dup and init_mode == 'identity' and is_output_proj:
                    # 归零输出投影 → 层变为纯残差直通
                    # experts.down_proj: [256, 512, 2048] → 全部归零
                    # shared_expert.down_proj: [hidden, inter] → 归零
                    new_weights[new_key] = torch.zeros_like(tensor)
                    zeroed_keys += 1
                elif is_dup and init_mode == 'scaled' and is_output_proj:
                    new_weights[new_key] = tensor.clone() * scale_factor
                    scaled_keys += 1
                elif is_dup:
                    new_weights[new_key] = tensor.clone()
                else:
                    new_weights[new_key] = tensor

            layer_keys_processed += 1
        else:
            # 非层权重直接保留 (embedding, lm_head, visual, mtp 等)
            new_weights[key] = tensor

    print(f"  处理了 {layer_keys_processed} 个层级权重键")
    print(f"  新模型共 {len(new_weights)} 个张量")
    if init_mode == 'identity':
        print(f"  ✅ 已归零 {zeroed_keys} 个复制层输出投影 (identity模式)")
        print(f"     包含: MoE experts.down_proj (全部路由专家) + shared_expert.down_proj + attn output")
    elif init_mode == 'scaled':
        print(f"  ✅ 已缩放 {scaled_keys} 个复制层输出投影 (×{scale_factor})")
    return new_weights


def save_sharded_safetensors(weights: OrderedDict, dst_dir: str, max_shard_bytes: int):
    """将权重保存为分片 safetensors 文件，并生成 index.json。"""
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

    # 写 index.json
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        index_path = os.path.join(dst_dir, "model.safetensors.index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"  已写入 {index_path}")


def update_and_save_config(src_dir: str, dst_dir: str, new_num_layers: int,
                           dup_groups: list[int]):
    """更新 config.json 并保存到目标目录。"""
    config_path = os.path.join(src_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    text_config = config['text_config']
    old_layer_types = text_config['layer_types']

    text_config['num_hidden_layers'] = new_num_layers
    text_config['layer_types'] = build_new_layer_types(old_layer_types, dup_groups)

    dst_config_path = os.path.join(dst_dir, 'config.json')
    with open(dst_config_path, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\n  config.json 已更新:")
    print(f"    num_hidden_layers: {len(old_layer_types)} → {new_num_layers}")
    print(f"    layer_types: {len(old_layer_types)} 项 → {len(text_config['layer_types'])} 项")


def copy_auxiliary_files(src_dir: str, dst_dir: str):
    """复制 tokenizer、模板等辅助文件。"""
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
    dup_groups = [int(x.strip()) for x in args.dup_groups.split(',')]

    # 读取原始 config
    config_path = os.path.join(args.src_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    text_config = config['text_config']
    old_num_layers = text_config['num_hidden_layers']
    old_layer_types = text_config['layer_types']
    num_blocks = old_num_layers // BLOCK_SIZE

    # 校验是否为 MoE 模型
    if 'num_experts' not in text_config:
        print("  ⚠️  警告: config 中未找到 num_experts，该模型可能不是 MoE 架构。")
        print("  如果是 Dense 模型，请使用 layer_stacking.py 代替。")
        return

    print(f"\n  检测到 MoE 模型: {config.get('architectures', ['?'])[0]}")
    print(f"  专家数: {text_config['num_experts']}, "
          f"每token激活: {text_config['num_experts_per_tok']}")

    # 边界块警告
    boundary_blocks = {0, 1, num_blocks - 2, num_blocks - 1}
    boundary_selected = set(dup_groups) & boundary_blocks
    if boundary_selected and args.init_mode == 'copy':
        print(f"\n  ⚠️  警告: 你选择了边界块 {sorted(boundary_selected)} 且使用 copy 模式!")
        print("  边界块(最前/最后)用 copy 模式极易导致乱码。")
        print("  强烈建议:")
        print("    1. 改用 --init_mode identity (推荐) 或 --init_mode scaled --scale_factor 0.1")
        print(f"    2. 或改为中间块, 如 --dup_groups 3,4,5,6")
        print()

    # 读取 total_size 用于参数量估算
    index_path = os.path.join(args.src_dir, 'model.safetensors.index.json')
    total_size_bytes = 0
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        total_size_bytes = int(index_data.get('metadata', {}).get('total_size', 0))

    # 构建映射
    layer_map, new_num_layers = build_layer_mapping(old_num_layers, dup_groups)

    # 打印堆叠计划
    print_stacking_plan(old_num_layers, layer_map, new_num_layers, dup_groups,
                        old_layer_types, config, total_size_bytes,
                        args.init_mode, args.scale_factor)

    if args.dry_run:
        print(f"\n  [DRY RUN] 仅预览，未实际执行。去掉 --dry_run 参数以执行。\n")
        return

    # 校验输出目录
    if not args.dst_dir:
        raise ValueError("非 dry_run 模式必须指定 --dst_dir")

    os.makedirs(args.dst_dir, exist_ok=True)

    # 1. 加载权重
    all_weights = load_all_safetensors(args.src_dir)

    # 2. 重映射 (应用 init_mode)
    new_weights = remap_weights(all_weights, layer_map,
                                init_mode=args.init_mode,
                                scale_factor=args.scale_factor)

    # 释放原始权重内存
    del all_weights
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # 3. 保存分片
    save_sharded_safetensors(new_weights, args.dst_dir, args.max_shard_bytes)

    # 释放
    del new_weights
    import gc; gc.collect()

    # 4. 更新 config
    update_and_save_config(args.src_dir, args.dst_dir, new_num_layers, dup_groups)

    # 5. 复制辅助文件
    copy_auxiliary_files(args.src_dir, args.dst_dir)

    # 完成提示
    old_p, new_p = estimate_params(total_size_bytes, old_num_layers, new_num_layers)
    old_a = estimate_active_params(old_num_layers, config)
    new_a = estimate_active_params(new_num_layers, config)

    print("\n" + "=" * 70)
    print("  ✅ MoE 层堆叠完成!")
    print(f"  输出目录: {args.dst_dir}")
    print(f"  新模型: {old_num_layers}层 → {new_num_layers}层")
    print(f"  总参数: ~{old_p:.1f}B → ~{new_p:.1f}B")
    print(f"  激活参数: ~{old_a:.1f}B → ~{new_a:.1f}B")
    print(f"  初始化模式: {args.init_mode}")
    print()
    print("  后续步骤:")
    print(f"    1. 验证: python verify_model.py --model_dir {args.dst_dir}")
    print(f"    2. vLLM部署: vllm serve {args.dst_dir} --trust-remote-code")
    if args.init_mode == 'identity':
        print()
        print("  💡 identity 模式说明:")
        print("     复制层的所有路由专家和共享专家的 down_proj 已归零。")
        print("     这些层是纯残差直通，模型行为等同原始模型。")
        print("     后续可对新层做 CPT/SFT 微调，让专家学到新知识。")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
