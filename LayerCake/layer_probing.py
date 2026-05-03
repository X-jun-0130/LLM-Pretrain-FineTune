#!/usr/bin/env python3
"""
Qwen3.5 Medical Knowledge Layer Probing Tool
检测模型各层对医疗知识的掌握程度，输出层/块重要性排名，
生成推荐的 --dup_groups 参数（用于 layer_stacking.py）

分析方法:
  1. Logit Lens: 每层输出投射到词表空间，检测已掌握的医疗token正确率
  2. Layer Knockout: 逐块旁路，检测医疗文本loss增幅（越大=越重要）
  3. Medical Specificity: 对比医疗vs通用文本的层级差异

用法:
    # 使用内置医疗评估数据（推荐先试）
    python layer_probing.py --model_dir /data1/Model-TH/Qwen3.5-9B --device cuda:0

    # 使用自定义医疗数据 (jsonl, 每行 {"text": "..."})
    python layer_probing.py --model_dir /data1/Model-TH/Qwen3.5-9B --eval_file medical_eval.jsonl

    # 仅做 Logit Lens 分析（更快）
    python layer_probing.py --model_dir /data1/Model-TH/Qwen3.5-9B --method logit_lens

    # 仅做 Layer Knockout 分析（更精确但更慢）
    python layer_probing.py --model_dir /data1/Model-TH/Qwen3.5-9B --method knockout
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

# ===== 常量 =====
BLOCK_SIZE = 4  # 每块 4 层: [linear×3, full×1]

# ===== 内置医疗评估数据 =====
# 涵盖多个医学专科，每条文本包含医学知识的关键completion
MEDICAL_EVAL_TEXTS = [
    # 心内科
    "患者男性65岁，突发胸骨后压榨性疼痛2小时，伴大汗淋漓、濒死感。心电图示V1-V4导联ST段弓背向上抬高，肌钙蛋白I明显升高。诊断为急性前壁ST段抬高型心肌梗死，应立即行经皮冠状动脉介入治疗，术前给予阿司匹林和氯吡格雷双联抗血小板治疗。",
    "心力衰竭患者出现夜间阵发性呼吸困难，端坐呼吸，双下肢水肿。BNP明显升高。治疗方案包括利尿剂呋塞米减轻容量负荷，ACEI类药物如依那普利改善心室重构，β受体阻滞剂如美托洛尔降低心率。",
    # 内分泌
    "2型糖尿病患者空腹血糖8.5mmol/L，餐后2小时血糖14.2mmol/L，糖化血红蛋白HbA1c为8.3%。首选二甲双胍治疗，若血糖控制不佳可联合使用DPP-4抑制剂西格列汀或SGLT2抑制剂达格列净。需定期监测肾功能和糖化血红蛋白。",
    "甲状腺功能亢进患者表现为心悸、多汗、消瘦、手抖，突眼征阳性。实验室检查TSH降低，FT3和FT4升高，TRAb阳性。诊断为Graves病，治疗首选抗甲状腺药物甲巯咪唑，需定期复查甲状腺功能和血常规。",
    # 呼吸内科
    "社区获得性肺炎患者，发热、咳嗽、咳铁锈色痰，胸部X线示右下肺实变影。痰培养示肺炎链球菌。经验性治疗首选阿莫西林克拉维酸钾或呼吸喹诺酮类莫西沙星。重症患者需入住ICU，给予广谱抗生素联合治疗。",
    "慢性阻塞性肺疾病急性加重期患者，表现为咳嗽加重、痰量增多、呼吸困难加重。治疗包括短效支气管扩张剂沙丁胺醇雾化吸入，全身糖皮质激素甲泼尼龙静脉使用，必要时给予无创正压通气辅助呼吸。",
    # 神经内科
    "缺血性脑卒中患者在发病4.5小时内，应考虑静脉溶栓治疗，首选药物为重组组织型纤溶酶原激活剂阿替普酶。溶栓后24小时内禁用抗凝和抗血小板药物，需密切监测血压和神经功能变化。",
    "帕金森病患者出现静止性震颤、肌强直、运动迟缓和姿势步态异常。早期治疗可选用多巴胺受体激动剂普拉克索，中晚期需联合使用左旋多巴复方制剂美多芭。需注意运动并发症如剂末现象和异动症。",
    # 消化内科
    "上消化道出血患者，呕血伴黑便，血红蛋白下降至70g/L。急诊胃镜检查发现胃溃疡活动性出血，给予内镜下止血治疗。同时静脉使用质子泵抑制剂奥美拉唑持续泵入，联合生长抑素抑制胃酸分泌和内脏血管收缩。",
    "肝硬化失代偿期患者出现大量腹水、脾大、食管胃底静脉曲张。治疗包括限钠饮食、螺内酯联合呋塞米利尿，白蛋白输注提高胶体渗透压。预防食管静脉曲张破裂出血可使用非选择性β受体阻滞剂普萘洛尔。",
    # 肿瘤
    "非小细胞肺癌患者EGFR基因检测示19号外显子缺失突变。一线治疗推荐第三代EGFR-TKI奥希替尼，相比一二代TKI具有更好的无进展生存期和中枢神经系统活性。治疗期间需定期监测肝功能和心电图QTc间期。",
    "HER2阳性乳腺癌术后辅助治疗方案包括蒽环类联合紫杉类化疗，同时给予曲妥珠单抗靶向治疗，疗程为1年。治疗期间需每3个月监测心脏射血分数，若LVEF下降超过16%或低于50%需暂停靶向治疗。",
    # 肾内科
    "慢性肾脏病3期患者eGFR为45mL/min/1.73m2，伴蛋白尿。治疗目标为延缓肾功能恶化，首选ACEI或ARB类药物如缬沙坦控制血压和减少蛋白尿。新型药物SGLT2抑制剂达格列净也被证实具有肾脏保护作用。需限制蛋白质摄入并监测血钾水平。",
    # 血液科
    "急性早幼粒细胞白血病APL患者PML-RARα融合基因阳性。诱导治疗方案为全反式维甲酸ATRA联合三氧化二砷ATO。治疗期间需警惕分化综合征，表现为发热、呼吸困难、体重增加，需及时给予地塞米松处理。",
    # 骨科
    "股骨颈骨折Garden IV型，患者72岁。移位型股骨颈骨折老年患者首选人工关节置换术。术后需预防深静脉血栓形成，使用低分子肝素抗凝，早期进行康复训练包括踝泵运动和股四头肌等长收缩练习。",
    # 儿科
    "小儿川崎病急性期表现为持续发热5天以上、双侧非化脓性结膜充血、口唇皲裂、颈部淋巴结肿大。治疗首选大剂量静脉用丙种球蛋白IVIG联合阿司匹林。需超声心动图监测冠状动脉是否扩张或形成动脉瘤。",
    # 影像诊断
    "CT平扫示肝脏S6段低密度占位，增强扫描动脉期明显强化，门脉期和延迟期强化减退呈低密度，即典型的快进快出强化模式，结合AFP升高，考虑肝细胞癌。建议行MRI肝脏特异性对比剂增强扫描进一步评估。",
    # 检验
    "血常规示白细胞15.6×10^9/L，中性粒细胞比例85%，C反应蛋白156mg/L，降钙素原3.2ng/mL。以上指标提示细菌感染可能性大，降钙素原大于0.5ng/mL时高度提示细菌性脓毒症。需完善血培养后尽早启动经验性抗感染治疗。",
    # 药理学
    "华法林是维生素K拮抗剂类口服抗凝药，通过抑制维生素K依赖性凝血因子II、VII、IX、X的合成发挥抗凝作用。治疗窗窄，需定期监测国际标准化比值INR，目标范围一般为2.0-3.0。与多种药物和食物存在相互作用，需注意避免与含维生素K丰富的绿叶蔬菜大量同时摄入。",
    # 急诊
    "过敏性休克患者出现皮疹、呼吸困难、血压下降。首选治疗药物为肾上腺素，成人剂量0.3-0.5mg肌肉注射，必要时每5-15分钟重复。同时开放静脉通路快速补液，给予糖皮质激素甲泼尼龙和抗组胺药异丙嗪。",
]

# 通用对比文本（用于计算 medical specificity）
GENERAL_EVAL_TEXTS = [
    "春天到了，花园里的花朵竞相开放，蝴蝶在花丛中翩翩起舞。孩子们在草地上奔跑嬉戏，享受着温暖的阳光。远处的山峦被薄雾笼罩，呈现出一幅美丽的画卷。",
    "人工智能技术近年来取得了巨大进展，特别是在自然语言处理和计算机视觉领域。大语言模型的出现改变了人们与计算机交互的方式，使得机器能够理解和生成更加自然流畅的文本。",
    "中国的高速铁路网络已经成为世界上最大的高铁系统。从北京到上海只需要四个多小时，极大地方便了人们的出行。高铁站的设计也越来越现代化，成为城市的新地标。",
    "今天的股市表现强劲，上证综指上涨了百分之二点三。科技板块领涨，多只芯片概念股涨停。分析师认为，政策利好和经济数据改善是推动市场上涨的主要因素。",
    "这道红烧肉的做法是先将五花肉切块焯水，然后加入冰糖炒出焦糖色，再放入生抽老抽料酒和八角桂皮等香料，最后小火慢炖一个半小时，直到肉质软烂入味。",
    "足球世界杯是全球最受瞩目的体育赛事之一。每四年举办一次，来自世界各地的球队争夺冠军荣誉。比赛期间，数十亿观众通过电视和网络观看直播。",
    "量子计算利用量子力学的叠加态和纠缠态原理进行计算，理论上能够在某些问题上远超经典计算机的性能。目前各大科技公司都在积极研发量子处理器。",
    "近年来新能源汽车市场快速发展，电池技术不断突破，续航里程持续提升。充电基础设施建设也在加速推进，越来越多的消费者开始选择纯电动或插电式混合动力汽车。",
]


def parse_args():
    p = argparse.ArgumentParser(description='Qwen3.5 Medical Knowledge Layer Probing')
    p.add_argument('--model_dir', type=str, required=True,
                   help='模型目录路径')
    p.add_argument('--eval_file', type=str, default=None,
                   help='自定义医疗评估数据 (jsonl, 每行 {"text": "..."})')
    p.add_argument('--method', type=str, default='both',
                   choices=['logit_lens', 'knockout', 'both'],
                   help='分析方法: logit_lens / knockout / both (默认both)')
    p.add_argument('--device', type=str, default='auto',
                   help='设备: cuda:0 / cpu / auto')
    p.add_argument('--max_length', type=int, default=256,
                   help='每条文本最大token长度 (默认256)')
    p.add_argument('--top_k', type=int, default=5,
                   help='Logit Lens 的 top-k 正确率 (默认5)')
    p.add_argument('--target_blocks', type=int, default=4,
                   help='推荐复制的块数量 (默认4)')
    p.add_argument('--output_json', type=str, default=None,
                   help='输出分析结果到 JSON 文件')
    return p.parse_args()


def detect_device(preference: str) -> torch.device:
    if preference == 'auto':
        if torch.cuda.is_available():
            # 选择显存最大的 GPU
            max_mem = 0
            best_gpu = 0
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory
                if mem > max_mem:
                    max_mem = mem
                    best_gpu = i
            device = torch.device(f'cuda:{best_gpu}')
            print(f"  自动选择设备: {device} ({torch.cuda.get_device_name(best_gpu)}, "
                  f"{max_mem / 1024**3:.1f} GB)")
        else:
            device = torch.device('cpu')
            print("  自动选择设备: CPU (未检测到 GPU)")
    else:
        device = torch.device(preference)
    return device


def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """加载模型和 tokenizer，返回模型组件引用。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    print(f"\n  加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"  加载模型到 {device} (bf16)...")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # 尝试多种加载方式
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            device_map=device if device.type == 'cuda' else None,
            trust_remote_code=True
        )
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            device_map=device if device.type == 'cuda' else None,
            trust_remote_code=True
        )

    if device.type == 'cpu':
        model = model.to(device)

    model.eval()

    # 自动检测模型内部结构
    components = detect_model_components(model)
    num_layers = len(components['layers'])
    print(f"  模型加载完成: {num_layers} 层, {num_layers // BLOCK_SIZE} 个块")

    return model, tokenizer, components


def detect_model_components(model) -> dict:
    """
    自动检测模型的 layers / final_norm / lm_head 引用。
    兼容多种模型结构。
    """
    components = {'layers': None, 'final_norm': None, 'lm_head': None}

    # 搜索 lm_head
    for name, module in model.named_modules():
        if name.endswith('lm_head') and hasattr(module, 'weight'):
            components['lm_head'] = module
            break

    # 搜索 language_model.layers 或 model.layers
    search_paths = [
        'model.language_model.layers',
        'language_model.layers',
        'model.layers',
        'transformer.h',
    ]
    for path in search_paths:
        try:
            obj = model
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and len(obj) > 0:
                components['layers'] = obj
                # 找对应的 norm
                parent_path = '.'.join(path.split('.')[:-1])
                parent = model
                for attr in parent_path.split('.'):
                    parent = getattr(parent, attr)
                if hasattr(parent, 'norm'):
                    components['final_norm'] = parent.norm
                break
        except AttributeError:
            continue

    # 验证
    missing = [k for k, v in components.items() if v is None]
    if missing:
        print(f"\n  ⚠️  未找到以下组件: {missing}")
        print("  模型结构:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                print(f"    {name}: {type(module).__name__}")
        raise RuntimeError("无法自动检测模型结构，请检查模型类型")

    return components


def prepare_eval_data(tokenizer, eval_file, max_length, device):
    """准备评估数据。"""
    # 加载医疗文本
    if eval_file and os.path.exists(eval_file):
        print(f"\n  加载自定义评估数据: {eval_file}")
        medical_texts = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    medical_texts.append(obj.get('text', obj.get('content', '')))
        print(f"  加载了 {len(medical_texts)} 条医疗文本")
    else:
        medical_texts = MEDICAL_EVAL_TEXTS
        print(f"\n  使用内置医疗评估数据: {len(medical_texts)} 条")

    general_texts = GENERAL_EVAL_TEXTS
    print(f"  通用对比数据: {len(general_texts)} 条")

    # Tokenize
    def tokenize_texts(texts):
        results = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True,
                                   max_length=max_length, truncation=True)
            results.append(torch.tensor([ids], device=device))
        return results

    return tokenize_texts(medical_texts), tokenize_texts(general_texts)


# ===================================================================
#                          Logit Lens 分析
# ===================================================================

class LogitLensAnalyzer:
    """在每一层的输出上做 Logit Lens，检测各层对正确 token 的预测能力。"""

    def __init__(self, layers, final_norm, lm_head, top_k=5):
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.top_k = top_k
        self.num_layers = len(layers)

        # 累积结果: layer_idx -> {correct_top1, correct_topk, total, loss_sum, n_samples}
        self.results = defaultdict(lambda: {
            'correct_top1': 0, 'correct_topk': 0,
            'total': 0, 'loss_sum': 0.0, 'n_samples': 0
        })
        self.target_ids = None
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if self.target_ids is None:
                return
            hs = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                normed = self.final_norm(hs.float() if hs.dtype != self.final_norm.weight.dtype
                                         else hs)
                logits = self.lm_head(normed)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = self.target_ids[:, 1:].contiguous()

                # Top-1 accuracy
                preds_top1 = shift_logits.argmax(dim=-1)
                self.results[layer_idx]['correct_top1'] += (
                    (preds_top1 == shift_labels).sum().item()
                )

                # Top-k accuracy
                _, preds_topk = shift_logits.topk(self.top_k, dim=-1)
                match_topk = (preds_topk == shift_labels.unsqueeze(-1)).any(dim=-1)
                self.results[layer_idx]['correct_topk'] += match_topk.sum().item()

                # Cross-entropy loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), reduction='mean'
                )
                self.results[layer_idx]['loss_sum'] += loss.item()
                self.results[layer_idx]['total'] += shift_labels.numel()
                self.results[layer_idx]['n_samples'] += 1

        return hook_fn

    def register_hooks(self):
        for i in range(self.num_layers):
            h = self.layers[i].register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(self, model, input_ids_list, label=''):
        self.register_hooks()
        try:
            for idx, input_ids in enumerate(input_ids_list):
                self.target_ids = input_ids
                with torch.no_grad():
                    model(input_ids=input_ids)
                if (idx + 1) % 5 == 0 or idx == len(input_ids_list) - 1:
                    print(f"\r    Logit Lens [{label}]: {idx+1}/{len(input_ids_list)}", end="")
            print()
        finally:
            self.remove_hooks()
            self.target_ids = None

    def get_layer_scores(self) -> list[dict]:
        """返回每层的分数汇总。"""
        scores = []
        for i in range(self.num_layers):
            r = self.results[i]
            total = max(r['total'], 1)
            n = max(r['n_samples'], 1)
            scores.append({
                'layer': i,
                'top1_acc': r['correct_top1'] / total,
                'topk_acc': r['correct_topk'] / total,
                'avg_loss': r['loss_sum'] / n,
            })
        return scores


# ===================================================================
#                        Layer Knockout 分析
# ===================================================================

def run_knockout_analysis(model, components, medical_ids_list):
    """
    逐块旁路（bypass），测量医疗文本 loss 的增幅。
    loss 增幅越大 = 该块对医疗知识越关键。
    """
    layers = components['layers']
    num_layers = len(layers)
    num_blocks = num_layers // BLOCK_SIZE

    print(f"\n    Layer Knockout: {num_blocks} 个块, {len(medical_ids_list)} 条文本")

    # 1. 基线 loss (无旁路)
    print(f"    计算基线 loss...", end=" ", flush=True)
    baseline_loss = _compute_avg_loss(model, medical_ids_list)
    print(f"{baseline_loss:.4f}")

    # 2. 逐块旁路
    block_knockout_results = {}
    for block_idx in range(num_blocks):
        layer_start = block_idx * BLOCK_SIZE
        layer_end = layer_start + BLOCK_SIZE

        # 注册旁路 hooks
        bypass_hooks = []
        for li in range(layer_start, layer_end):
            def make_bypass_hook():
                def hook_fn(module, input, output):
                    # 将输出替换为输入（跳过该层）
                    inp_hs = input[0] if isinstance(input, tuple) else input
                    if isinstance(output, tuple):
                        return (inp_hs,) + output[1:]
                    return inp_hs
                return hook_fn

            h = layers[li].register_forward_hook(make_bypass_hook())
            bypass_hooks.append(h)

        # 计算旁路后 loss
        knockout_loss = _compute_avg_loss(model, medical_ids_list)
        loss_increase = knockout_loss - baseline_loss

        block_knockout_results[block_idx] = {
            'baseline_loss': baseline_loss,
            'knockout_loss': knockout_loss,
            'loss_increase': loss_increase,
            'relative_increase': loss_increase / max(baseline_loss, 1e-8),
        }

        # 移除旁路 hooks
        for h in bypass_hooks:
            h.remove()

        print(f"    Block {block_idx} (L{layer_start}-{layer_end-1}): "
              f"loss {knockout_loss:.4f} (Δ={loss_increase:+.4f}, "
              f"{loss_increase / max(baseline_loss, 1e-8) * 100:+.1f}%)")

    return block_knockout_results


def _compute_avg_loss(model, input_ids_list) -> float:
    """计算模型在给定输入上的平均 loss。"""
    total_loss = 0.0
    for input_ids in input_ids_list:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
    return total_loss / max(len(input_ids_list), 1)


# ===================================================================
#                         结果分析与输出
# ===================================================================

def aggregate_to_blocks(layer_scores: list[dict], key: str) -> list[float]:
    """将层级分数聚合为块级分数（取块内平均）。"""
    num_layers = len(layer_scores)
    num_blocks = num_layers // BLOCK_SIZE
    block_scores = []
    for b in range(num_blocks):
        vals = [layer_scores[b * BLOCK_SIZE + i][key] for i in range(BLOCK_SIZE)]
        block_scores.append(sum(vals) / len(vals))
    return block_scores


def compute_medical_specificity(medical_scores, general_scores, key='avg_loss'):
    """
    计算医疗特异性得分: 医疗文本在该层的loss下降幅度 vs 通用文本。
    specificity > 0 表示该层对医疗文本更重要。
    """
    num_layers = len(medical_scores)
    specificity = []
    for i in range(num_layers):
        # loss 越低越好, 所以用 general_loss - medical_loss
        # 如果该层让医疗 loss 下降得更多, specificity 为正
        med_loss = medical_scores[i][key]
        gen_loss = general_scores[i][key]
        specificity.append(gen_loss - med_loss)
    return specificity


def compute_layer_contribution(layer_scores, key='avg_loss'):
    """
    计算每层对 loss 下降的边际贡献。
    contribution[i] = loss[i-1] - loss[i] （越大 = 该层贡献越大）
    """
    contributions = []
    for i in range(len(layer_scores)):
        if i == 0:
            contributions.append(0.0)  # 第0层无参照
        else:
            prev_loss = layer_scores[i - 1][key]
            curr_loss = layer_scores[i][key]
            contributions.append(prev_loss - curr_loss)
    return contributions


def rank_blocks(block_scores: list[float], higher_is_better=True) -> list[tuple[int, float]]:
    """按得分排序块，返回 [(block_idx, score), ...] 降序。"""
    ranked = [(i, s) for i, s in enumerate(block_scores)]
    ranked.sort(key=lambda x: x[1], reverse=higher_is_better)
    return ranked


def print_bar(value, max_value, width=30, char='█'):
    """生成文本条形图。"""
    if max_value == 0:
        return ''
    fill = int(width * abs(value) / max_value)
    return char * fill


def print_report(logit_lens_medical, logit_lens_general,
                 knockout_results, num_layers, top_k, target_blocks):
    """打印综合分析报告。"""
    num_blocks = num_layers // BLOCK_SIZE

    print("\n" + "=" * 75)
    print("             Qwen3.5 Medical Knowledge Layer Analysis Report")
    print("=" * 75)

    # ===== Logit Lens Report =====
    if logit_lens_medical:
        med_scores = logit_lens_medical.get_layer_scores()
        gen_scores = logit_lens_general.get_layer_scores() if logit_lens_general else None

        # 块级 top-k accuracy (医疗)
        block_topk = aggregate_to_blocks(med_scores, 'topk_acc')
        block_loss = aggregate_to_blocks(med_scores, 'avg_loss')

        print(f"\n{'─' * 75}")
        print(f"  [Logit Lens] 各块医疗知识掌握度 (Top-{top_k} Accuracy)")
        print(f"{'─' * 75}")

        max_acc = max(block_topk) if block_topk else 1
        for b in range(num_blocks):
            layer_range = f"L{b*4:2d}-{b*4+3:2d}"
            # 层类型标注
            types = "lin×3+full"
            acc = block_topk[b]
            loss = block_loss[b]
            bar = print_bar(acc, max_acc, width=35)
            print(f"  Block {b} ({layer_range}) [{types}]  "
                  f"top{top_k}={acc:.3f}  loss={loss:.2f}  {bar}")

        # 层级贡献度
        contributions = compute_layer_contribution(med_scores, 'avg_loss')
        block_contributions = []
        for b in range(num_blocks):
            bc = sum(contributions[b * BLOCK_SIZE + i] for i in range(BLOCK_SIZE))
            block_contributions.append(bc)

        print(f"\n{'─' * 75}")
        print(f"  [Logit Lens] 各块 Loss 下降贡献度 (越大=该块减少loss越多)")
        print(f"{'─' * 75}")

        max_contrib = max(abs(c) for c in block_contributions) if block_contributions else 1
        for b in range(num_blocks):
            layer_range = f"L{b*4:2d}-{b*4+3:2d}"
            c = block_contributions[b]
            bar = print_bar(c, max_contrib, width=35, char='▓' if c > 0 else '░')
            print(f"  Block {b} ({layer_range})  Δloss={c:+.4f}  {bar}")

        # 医疗特异性
        if gen_scores:
            specificity = compute_medical_specificity(med_scores, gen_scores)
            block_spec = []
            for b in range(num_blocks):
                bs = sum(specificity[b * BLOCK_SIZE + i] for i in range(BLOCK_SIZE)) / BLOCK_SIZE
                block_spec.append(bs)

            print(f"\n{'─' * 75}")
            print(f"  [Medical Specificity] 各块医疗特异性 (越大=越'专注'医疗)")
            print(f"{'─' * 75}")

            max_spec = max(abs(s) for s in block_spec) if block_spec else 1
            for b in range(num_blocks):
                layer_range = f"L{b*4:2d}-{b*4+3:2d}"
                s = block_spec[b]
                bar = print_bar(s, max_spec, width=35, char='●' if s > 0 else '○')
                print(f"  Block {b} ({layer_range})  spec={s:+.4f}  {bar}")

    # ===== Knockout Report =====
    if knockout_results:
        print(f"\n{'─' * 75}")
        print(f"  [Layer Knockout] 各块被移除后的 Loss 增幅 (越大=越关键)")
        print(f"{'─' * 75}")

        max_increase = max(r['loss_increase'] for r in knockout_results.values()) if knockout_results else 1
        for b in range(num_blocks):
            if b not in knockout_results:
                continue
            r = knockout_results[b]
            layer_range = f"L{b*4:2d}-{b*4+3:2d}"
            bar = print_bar(r['loss_increase'], max_increase, width=35, char='▰')
            print(f"  Block {b} ({layer_range})  Δloss={r['loss_increase']:+.4f} "
                  f"({r['relative_increase']*100:+.1f}%)  {bar}")

    # ===== 综合排名与推荐 =====
    print(f"\n{'=' * 75}")
    print(f"  综合排名与推荐 (目标: 复制 {target_blocks} 个块)")
    print(f"{'=' * 75}")

    # 计算综合得分 (多指标加权)
    block_composite_scores = [0.0] * num_blocks

    if logit_lens_medical:
        med_scores = logit_lens_medical.get_layer_scores()

        # 指标1: loss贡献 (权重 0.3)
        contribs = compute_layer_contribution(med_scores, 'avg_loss')
        block_contribs = [sum(contribs[b*4+i] for i in range(BLOCK_SIZE)) for b in range(num_blocks)]
        max_c = max(abs(c) for c in block_contribs) if any(block_contribs) else 1
        for b in range(num_blocks):
            block_composite_scores[b] += 0.3 * (block_contribs[b] / max_c)

        # 指标2: top-k accuracy (权重 0.2)
        block_accs = aggregate_to_blocks(med_scores, 'topk_acc')
        max_a = max(block_accs) if block_accs else 1
        for b in range(num_blocks):
            block_composite_scores[b] += 0.2 * (block_accs[b] / max_a)

        # 指标3: 医疗特异性 (权重 0.2)
        if logit_lens_general:
            gen_scores = logit_lens_general.get_layer_scores()
            spec = compute_medical_specificity(med_scores, gen_scores)
            block_spec = [sum(spec[b*4+i] for i in range(BLOCK_SIZE)) / BLOCK_SIZE
                          for b in range(num_blocks)]
            max_s = max(abs(s) for s in block_spec) if any(block_spec) else 1
            for b in range(num_blocks):
                block_composite_scores[b] += 0.2 * (block_spec[b] / max_s)

    if knockout_results:
        # 指标4: knockout loss增幅 (权重 0.3)
        knockout_increases = [knockout_results.get(b, {}).get('loss_increase', 0)
                              for b in range(num_blocks)]
        max_ki = max(abs(ki) for ki in knockout_increases) if any(knockout_increases) else 1
        for b in range(num_blocks):
            block_composite_scores[b] += 0.3 * (knockout_increases[b] / max_ki)

    # 排序
    ranked = rank_blocks(block_composite_scores, higher_is_better=True)

    print(f"\n  块综合得分排名:")
    for rank, (b, score) in enumerate(ranked):
        layer_range = f"L{b*4:2d}-{b*4+3:2d}"
        marker = " ← 推荐复制" if rank < target_blocks else ""
        bar = print_bar(score, max(s for _, s in ranked), width=30)
        print(f"    #{rank+1}  Block {b} ({layer_range})  score={score:.4f}  {bar}{marker}")

    # 推荐的 dup_groups
    recommended = sorted([b for b, _ in ranked[:target_blocks]])
    recommended_str = ','.join(str(b) for b in recommended)

    new_layers = num_layers + target_blocks * BLOCK_SIZE
    est_params = 9.0 + (target_blocks * BLOCK_SIZE * 0.2)

    print(f"\n  {'─' * 50}")
    print(f"  推荐 dup_groups: {recommended_str}")
    print(f"  效果: {num_layers}层 → {new_layers}层, 估算 ~{est_params:.1f}B")
    print(f"\n  执行命令:")
    print(f"    python layer_stacking.py \\")
    print(f"        --src_dir <MODEL_DIR> \\")
    print(f"        --dst_dir <OUTPUT_DIR> \\")
    print(f"        --dup_groups {recommended_str}")
    print(f"{'=' * 75}\n")

    return recommended, block_composite_scores


def main():
    args = parse_args()
    device = detect_device(args.device)

    # 1. 加载模型
    model, tokenizer, components = load_model_and_tokenizer(args.model_dir, device)
    num_layers = len(components['layers'])

    # 2. 准备数据
    medical_ids, general_ids = prepare_eval_data(
        tokenizer, args.eval_file, args.max_length, device
    )

    # 3. Logit Lens 分析
    logit_lens_medical = None
    logit_lens_general = None

    if args.method in ('logit_lens', 'both'):
        print(f"\n  ===== Logit Lens Analysis =====")

        logit_lens_medical = LogitLensAnalyzer(
            components['layers'], components['final_norm'],
            components['lm_head'], top_k=args.top_k
        )
        logit_lens_medical.run(model, medical_ids, label='Medical')

        logit_lens_general = LogitLensAnalyzer(
            components['layers'], components['final_norm'],
            components['lm_head'], top_k=args.top_k
        )
        logit_lens_general.run(model, general_ids, label='General')

    # 4. Layer Knockout 分析
    knockout_results = None

    if args.method in ('knockout', 'both'):
        print(f"\n  ===== Layer Knockout Analysis =====")
        knockout_results = run_knockout_analysis(model, components, medical_ids)

    # 5. 生成报告
    recommended, scores = print_report(
        logit_lens_medical, logit_lens_general,
        knockout_results, num_layers,
        args.top_k, args.target_blocks
    )

    # 6. 保存结果
    if args.output_json:
        output = {
            'num_layers': num_layers,
            'num_blocks': num_layers // BLOCK_SIZE,
            'recommended_dup_groups': recommended,
            'block_composite_scores': {i: scores[i] for i in range(len(scores))},
        }
        if logit_lens_medical:
            output['logit_lens_medical'] = logit_lens_medical.get_layer_scores()
        if logit_lens_general:
            output['logit_lens_general'] = logit_lens_general.get_layer_scores()
        if knockout_results:
            output['knockout_results'] = knockout_results

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  分析结果已保存到: {args.output_json}")


if __name__ == '__main__':
    main()
