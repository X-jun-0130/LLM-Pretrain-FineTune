#!/usr/bin/env python3
"""
Qwen3.6-MoE Layer Importance Analysis
基于 Angular Distance 的块级冗余度分析工具

核心思想（Gromov et al., 2024 "Unreasonable Ineffectiveness of the Deeper Layers"）:
    对于一个 Transformer 块 B_k，其 residual 贡献越小，角距离 d(h_in, h_out) 越小，
    该块越接近 identity（即删除后对后续层表示影响越小），越适合被剪枝。

    angular_distance(x, y) = arccos(cos_sim(x, y)) / pi
    值域 [0, 1], 越小越冗余。

Qwen3.6-35B-A3B 架构约束:
    - 40 层, BLOCK_SIZE=4, 共 10 个块 (0..9)
    - 每块模式 [linear_attn x 3, full_attn x 1]
    - 候选可删除集合: block 2..7（保留边界块 0/1 和 8/9）

用法:
    # 1. 使用内置医疗校准文本（快速验证）
    python layer_importance_moe.py \\
        --model_dir /data1/Model-TH/Qwen3.6-35B-A3B

    # 2. 使用自定义校准数据（推荐, jsonl, 每行 {"text": "..."} 或 {"messages":[...]}）
    python layer_importance_moe.py \\
        --model_dir /data1/Model-TH/Qwen3.6-35B-A3B \\
        --calib_file /path/to/medical_calib.jsonl \\
        --num_samples 300 \\
        --max_length 1024 \\
        --top_k 2 \\
        --output_json ./block_importance.json
"""

import argparse
import json
import math
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import torch

# ===== 架构常量（与 layer_stacking_moe.py 保持一致）=====
BLOCK_SIZE = 4
BOUNDARY_PROTECT = 2  # 前后各保护的块数: {0,1} 与 {N-2,N-1} 不参与候选
DEFAULT_TOP_K = 2     # 目标减 2 块 (40 -> 32)

# ===== 内置医疗校准文本（数据不可用时的兜底）=====
BUILTIN_CALIB_TEXTS = [
    "急性心肌梗死的典型表现为胸骨后压榨性疼痛，持续时间超过30分钟，含服硝酸甘油不缓解，心电图可见ST段弓背向上抬高。治疗上应尽早行冠状动脉造影和介入治疗。",
    "2型糖尿病一线治疗药物是二甲双胍。当HbA1c大于7.0%时，可加用SGLT2抑制剂或GLP-1受体激动剂，后者在合并心血管疾病的患者中优先考虑。",
    "社区获得性肺炎最常见致病菌为肺炎链球菌，经验性治疗首选阿莫西林或呼吸喹诺酮类抗生素。重症患者需覆盖非典型病原体。",
    "慢性肾脏病患者血压控制目标低于130/80 mmHg，首选ACEI或ARB，可减少蛋白尿并延缓肾功能恶化。禁用于双侧肾动脉狭窄患者。",
    "缺血性脑卒中发病4.5小时内可行静脉溶栓，阿替普酶0.9 mg/kg，最大剂量90 mg，10%团注后余量60分钟内静滴。",
    "乳腺癌根据分子分型分为Luminal A、Luminal B、HER2阳性、三阴性四类。三阴性乳腺癌预后较差，化疗敏感，靶向治疗选择有限。",
    "非小细胞肺癌EGFR突变阳性患者首选奥希替尼，疾病进展后需重新活检明确耐药机制，常见为T790M、C797S或MET扩增。",
    "胃食管反流病典型症状为反酸烧心，PPI治疗疗程8周，难治性患者需行24小时食管pH监测和高分辨率食管测压。",
    "结直肠癌TNM分期中，T1指肿瘤侵犯黏膜下层，T2侵及固有肌层，T3穿透固有肌层至浆膜下，T4a穿透脏层腹膜，T4b直接侵犯其他器官。",
    "心力衰竭NYHA分级IV级患者，射血分数低于40%时应使用ARNI、β受体阻滞剂、MRA和SGLT2抑制剂四联治疗。",
    "胰腺癌CA19-9是最常用的肿瘤标志物，但特异性不高，需结合CT、MRI和EUS-FNA明确诊断。可切除者首选根治性手术，辅助化疗方案为FOLFIRINOX或吉西他滨联合白蛋白紫杉醇。",
    "高血压3级合并靶器官损害属于极高危分层，需在生活方式干预基础上立即启动降压药物治疗，常用联合方案为ACEI/ARB联合CCB或噻嗪类利尿剂。",
    "血小板减少性紫癜的一线治疗为糖皮质激素，无效者可选用IVIG或TPO受体激动剂，严重出血或激素抵抗者考虑脾切除。",
    "病毒性肝炎中，乙肝核心抗体IgM阳性提示急性感染，HBsAg持续阳性超过6个月定义为慢性乙肝，核苷类似物或PEG-IFN可用于抗病毒治疗。",
    "肾病综合征四联征为大量蛋白尿、低蛋白血症、高度水肿和高脂血症，微小病变型对糖皮质激素敏感，膜性肾病则多需加用环磷酰胺或利妥昔单抗。",
]


def parse_args():
    p = argparse.ArgumentParser(
        description='Qwen3.6-MoE Block Importance (Angular Distance) Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_dir', type=str, required=True, help='MoE 模型目录')
    p.add_argument('--calib_file', type=str, default=None,
                   help='校准数据 jsonl, 每行 {"text": "..."} 或 {"messages":[...]}')
    p.add_argument('--num_samples', type=int, default=200,
                   help='使用的校准样本数, 默认 200')
    p.add_argument('--max_length', type=int, default=1024,
                   help='每条样本最大 token 数, 默认 1024')
    p.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                   help='推荐的待剪枝块数, 默认 2')
    p.add_argument('--device', type=str, default='auto', help='cuda:0 / cpu / auto')
    p.add_argument('--output_json', type=str, default=None,
                   help='分析结果保存路径 (可选)')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    return p.parse_args()


def load_calib_texts(calib_file, num_samples, seed):
    """加载校准文本。支持 jsonl 格式: {"text": "..."} 或 {"messages": [...]}。"""
    if calib_file is None:
        print(f"  [校准数据] 未指定 --calib_file, 使用内置医疗文本 ({len(BUILTIN_CALIB_TEXTS)} 条)")
        return BUILTIN_CALIB_TEXTS

    texts = []
    with open(calib_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                if 'text' in item:
                    texts.append(item['text'])
                elif 'messages' in item:
                    parts = []
                    for m in item['messages']:
                        role = m.get('role', '')
                        content = m.get('content', '')
                        if isinstance(content, str) and content:
                            parts.append(f"{role}: {content}")
                    if parts:
                        texts.append('\n'.join(parts))
                elif 'prompt' in item and 'response' in item:
                    texts.append(f"{item['prompt']}\n{item['response']}")
            elif isinstance(item, str):
                texts.append(item)

    if not texts:
        raise ValueError(f"未能从 {calib_file} 读取任何文本")

    rng = random.Random(seed)
    if len(texts) > num_samples:
        texts = rng.sample(texts, num_samples)
    print(f"  [校准数据] 从 {calib_file} 加载 {len(texts)} 条样本")
    return texts


def detect_device(pref):
    if pref == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return pref


def load_model(model_dir, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\n  加载模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if device.startswith('cuda'):
        model_kwargs['device_map'] = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if device == 'cpu':
        model = model.to('cpu')
    model.eval()
    return model, tokenizer


def get_num_blocks_from_config(model_dir):
    """从 config.json 推断 (num_layers, num_blocks)。"""
    cfg_path = os.path.join(model_dir, 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    tc = cfg.get('text_config', cfg)
    num_layers = tc['num_hidden_layers']
    if num_layers % BLOCK_SIZE != 0:
        raise ValueError(f"num_hidden_layers={num_layers} 不是 {BLOCK_SIZE} 的倍数, 架构异常")
    return num_layers, num_layers // BLOCK_SIZE


def angular_distance(h_in, h_out):
    """
    对形状 [seq_len, hidden] 的两个张量, 按 token 计算 cos_sim 再取平均,
    最后转角距离 arccos(cos) / pi.
    """
    h_in = h_in.float()
    h_out = h_out.float()
    cos = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1)
    cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    d = torch.arccos(cos) / math.pi
    return float(d.mean().item())


@torch.no_grad()
def analyze_importance(model, tokenizer, texts, num_blocks, max_length):
    """
    对每条样本前向一次, 收集 hidden_states, 计算每个块的 angular distance,
    最后对所有样本求均值+方差。
    hidden_states 是 embeddings 后 + 每层输出, 共 num_layers+1 个张量。
    块 k 的入/出: hs[k*BLOCK_SIZE] 与 hs[(k+1)*BLOCK_SIZE]。
    """
    device = next(model.parameters()).device
    per_block_dists = [[] for _ in range(num_blocks)]

    print(f"\n  运行前向, 共 {len(texts)} 条样本...")
    for i, text in enumerate(texts):
        if not text:
            continue
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = outputs.hidden_states

        if attention_mask is not None:
            mask = attention_mask[0].bool()
        else:
            mask = torch.ones(input_ids.size(1), dtype=torch.bool, device=device)

        for k in range(num_blocks):
            idx_in = k * BLOCK_SIZE
            idx_out = (k + 1) * BLOCK_SIZE
            h_in = hs[idx_in][0][mask]
            h_out = hs[idx_out][0][mask]
            per_block_dists[k].append(angular_distance(h_in, h_out))

        del outputs, hs
        if (i + 1) % 20 == 0 or i == len(texts) - 1:
            print(f"    [{i+1}/{len(texts)}] 完成")

    results = []
    for k in range(num_blocks):
        vals = per_block_dists[k]
        if not vals:
            results.append({'block': k, 'mean_dist': float('nan'),
                            'std_dist': float('nan'), 'n_samples': 0})
            continue
        t = torch.tensor(vals)
        results.append({
            'block': k,
            'mean_dist': float(t.mean().item()),
            'std_dist': float(t.std().item()) if len(vals) > 1 else 0.0,
            'n_samples': len(vals),
        })
    return results


def recommend_drop_blocks(results, num_blocks, top_k):
    """只从候选集 [BOUNDARY_PROTECT, num_blocks - BOUNDARY_PROTECT) 中挑 top_k 最小角距离块。"""
    low = BOUNDARY_PROTECT
    high = num_blocks - BOUNDARY_PROTECT
    cands = [r for r in results if low <= r['block'] < high]
    cands_sorted = sorted(cands, key=lambda r: r['mean_dist'])
    return [r['block'] for r in cands_sorted[:top_k]]


def print_report(results, num_blocks, recommended):
    low = BOUNDARY_PROTECT
    high = num_blocks - BOUNDARY_PROTECT
    print(f"\n{'=' * 70}")
    print(f"         MoE Block Importance (Angular Distance) Report")
    print(f"{'=' * 70}")
    print(f"  块总数: {num_blocks} (每块 {BLOCK_SIZE} 层)")
    print(f"  候选集: block {low}..{high - 1} (保护边界块 0..{low-1} 与 {high}..{num_blocks-1})")
    print(f"{'-' * 70}")
    print(f"  {'Block':<7}{'Layers':<12}{'MeanDist':<12}{'StdDist':<12}{'Note':<20}")
    print(f"{'-' * 70}")
    rec_set = set(recommended)
    min_in_cand = min((r['mean_dist'] for r in results if low <= r['block'] < high),
                      default=float('inf'))
    for r in results:
        k = r['block']
        layers = f"L{k*BLOCK_SIZE:02d}-{(k+1)*BLOCK_SIZE-1:02d}"
        is_boundary = k < low or k >= high
        note = ""
        if is_boundary:
            note = "[边界保护]"
        elif k in rec_set:
            note = "<< 推荐剪枝 >>"
        elif r['mean_dist'] == min_in_cand:
            note = "(候选集最小)"
        print(f"  {k:<7}{layers:<12}{r['mean_dist']:<12.4f}"
              f"{r['std_dist']:<12.4f}{note:<20}")
    print(f"{'-' * 70}")
    print(f"  推荐 drop_blocks = {sorted(recommended)}")
    print(f"  下一步: python layer_pruning_moe.py --src_dir <src> --dst_dir <dst> "
          f"--drop_blocks {','.join(str(x) for x in sorted(recommended))} --dry_run")
    print(f"{'=' * 70}\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_layers, num_blocks = get_num_blocks_from_config(args.model_dir)
    print(f"  [Config] num_hidden_layers = {num_layers}, num_blocks = {num_blocks}")

    texts = load_calib_texts(args.calib_file, args.num_samples, args.seed)

    device = detect_device(args.device)
    model, tokenizer = load_model(args.model_dir, device)

    results = analyze_importance(model, tokenizer, texts, num_blocks, args.max_length)

    recommended = recommend_drop_blocks(results, num_blocks, args.top_k)
    print_report(results, num_blocks, recommended)

    if args.output_json:
        payload = {
            'model_dir': args.model_dir,
            'num_layers': num_layers,
            'num_blocks': num_blocks,
            'num_samples': len(texts),
            'block_size': BLOCK_SIZE,
            'boundary_protect': BOUNDARY_PROTECT,
            'top_k': args.top_k,
            'results': results,
            'recommended_drop_blocks': sorted(recommended),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {args.output_json}\n")


if __name__ == '__main__':
    main()
