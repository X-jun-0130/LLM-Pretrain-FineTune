# 📊 LLM-Pretrain-FineTune

> **Deepspeed · LLM · Medical_Dialogue · 医疗大模型 · 增量预训练 · 微调 · 指令多样化**

## 1. 项目概述

这是一个**中文医疗大语言模型（LLM）增量预训练和微调**的完整工作流项目，覆盖从数据合成、指令多样化改写、数据过滤、模型预训练、指令微调到模型保存/转换的端到端解决方案。

**核心亮点：**
- 🧬 **GenRM 数据合成与过滤**：基于生成式奖励模型的高质量医疗答案合成与质量筛选
- 🔄 **指令多样化改写**：三级改写强度（轻度/中度/重度）驱动指令表达形式多样性，提升模型泛化能力
- 🧠 **混合思维训练**：非Think + 独立Think + 混合Think 三种训练模式
- ⚡ **高性能训练**：DeepSpeed ZeRO-3 + CPU Offload + Flash Attention 2 + Liger Kernel

---

## 2. 顶层目录结构

```
LLM-Pretrain-FineTune/
├── Data_Synthesis/              # 医疗数据合成（答案生成 + GenRM优选）
│   ├── prompts.py               #   合成提示模板（GenRM选择模式 + 质量评估）
│   └── synth_data.py            #   异步数据合成工作流（含断点续传）
├── Instruction_Rewrite/         # 指令多样化改写（提升模型泛化性）
│   ├── api_config.py            #   LLM API 配置
│   ├── prompts.py               #   改写与校验提示词模板
│   ├── instruction_rewriter.py  #   核心改写引擎（三级强度 + 语义校验）
│   ├── run_rewrite.py           #   命令行入口
│   └── merge_data.py            #   原始数据与改写数据合并工具
├── Sft_Data_Filter/             # SFT 数据质量过滤
│   ├── data_filter.py           #   基于 Judge 模型的质量验证
│   ├── Verifier_Prompts.py      #   验证器提示模板（有/无参考答案）
│   └── SFT_PostFilter.py        #   分层后处理筛选（按 correct_count 采样）
├── examples/                    # 示例截图
│   ├── report.png               #   报告解读示例
│   ├── dialogue.png             #   多轮对话示例
│   ├── medical_consult.png      #   医疗问答示例
│   ├── harmlessness.png         #   无害性回复示例
│   ├── genechat.png             #   单轮转多轮对话示例
│   └── kb_instruction.png       #   知识库 Prompt 示例
├── Model_MIX.py                 # 核心训练脚本（混合思维 + 混合预训练/微调）
├── model_convert16save.py       # 训练后模型保存/转换（支持 Qwen3.5 / VL 模型）
├── ds_config.json               # DeepSpeed ZeRO-3 配置
└── README.md                    # 项目文档
```

---

## 3. 主要模块详解

### 3.1 Data_Synthesis（数据合成）

从原始医疗问题出发，调用大模型生成高质量候选答案，并通过 GenRM 机制筛选最优结果。

**`synth_data.py`** — 异步数据合成工作流，核心流程：

1. 读取原始问题数据
2. 为每个问题调用大模型生成 **N 个候选答案**（默认 4 个）
3. 使用 **GenRM 同行评议选择模式**从候选中挑出最优答案
4. 通过**质量评估**（满分 100 分，阈值 0.9 筛选）
5. 输出带 `reasoning_content` 和 `score` 的结构化数据
6. 支持**断点续传**（`load_existing_ids`）

**`prompts.py`** — 包含两个核心 Prompt 模板：

| 模板 | 用途 |
|------|------|
| `SELECTION_PROMPT_TEMPLATE` | GenRM 同行评议：从多个候选中选出最优答案 |
| `check_prompt` | 答案质量评估：医学准确性、语言表达等多维度评分 |

---

### 3.2 Instruction_Rewrite（指令多样化改写）🆕

> **目的**：通过对 SFT 指令（task_prompt）进行表达形式多样化改写，增加微调数据的指令多样性，从而**提升模型的泛化能力和鲁棒性**，避免模型对特定指令措辞过拟合。

#### 设计理念

```
原始指令（单一表达）──→ [三级改写引擎] ──→ 多样化指令变体（5x~8x 扩充）
       ↑                                            │
       └──── [语义一致性校验] ← 9 维检查清单 ←──────┘
```

**核心原则：**
- **形式可变，语义不变**：医学规则、诊断逻辑、质控原则等业务核心保持 100% 一致
- **占位符保护**：`{medical_text}` 等输入占位符绝对不可变
- **格式独立管理**：`output_formate` 字段完全独立，不参与改写，防止格式漂移
- **代码块修复**：自动检测并修复 LLM 在改写过程中篡改/丢失/新增的代码块

#### 三级改写强度

| 强度 | 名称 | 温度 | 行为 |
|------|------|------|------|
| 🌿 `light` | 轻度改写 | 0.60 | 保持结构不变，替换同义词、调整句式（主动 ↔ 被动）、微调标题编号 |
| 🌳 `medium` | 中度改写 | 0.75 | 重组段落结构、列表 ↔ 叙述互转、调整 Markdown 格式、改变角色设定措辞 |
| 🔥 `heavy` | 重度改写 | 0.85 | 大幅重构（平铺 ↔ 层级、文档式 ↔ 对话式 ↔ 步骤式）、可去 Markdown 改纯文本 |

**权重分配**：轻度 ~15% | 中度 ~55% | 重度 ~30%（硬性保证每条指令至少 1 个中度/重度变体）

#### 两级重试机制

确保每条指令都能产出有效改写结果：

1. **调用级重试**（默认 3 次）：单次 LLM 调用失败时自动重试，每次微调 temperature 增加多样性
2. **批次级重试**（默认 3 批）：整批变体全部失败时，重新生成全新一批

#### 九维语义校验

对每个改写结果进行自动化质量审核，覆盖以下维度：

| # | 检查维度 | 说明 |
|---|----------|------|
| 1 | 占位符完整性 | `{medical_text}` 是否原封不动保留 |
| 2 | 医学术语完整性 | 所有医学专业术语是否完整 |
| 3 | 错误类型/评判维度完整性 | 分类项是否一项不少 |
| 4 | 质控原则/限制条件完整性 | 规则语义是否未被扭曲 |
| 5 | 医学示例完整性 | 示例性文字是否保留 |
| 6 | 分析步骤完整性 | 步骤间逻辑关系是否保留 |
| 7 | 无新增约束 | 是否引入了原文不存在的新规则 |
| 8 | 无语义偏移 | 规则含义是否被改变 |
| 9 | 无格式冲突 | 是否自行发明了原文不存在的输出格式 |

#### 使用方式

```bash
# 1. 改写指定版本文件（每条指令生成 5 个变体，默认行为）
python Instruction_Rewrite/run_rewrite.py \
    --input /path/to/prompts/v2.0.0.json

# 2. 指定变体数量
python Instruction_Rewrite/run_rewrite.py \
    --input /path/to/prompts/v2.0.0.json \
    --n_variants 8

# 3. 快速测试（1 个变体，不校验，展示对比预览）
python Instruction_Rewrite/run_rewrite.py \
    --input /path/to/prompts/v2.0.0.json \
    --test

# 4. 合并原始数据与改写数据为最终训练集
python Instruction_Rewrite/merge_data.py \
    --original_dir /path/to/Minimalism-SFT/Medical \
    --rewritten_dir /path/to/output \
    --output /path/to/merged.jsonl \
    --clean
```

**`run_rewrite.py`** 参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | （必填） | 输入的结构化版本 JSON 文件路径 |
| `--output` | 自动生成 | 输出路径，默认同目录下生成 `{name}_diverse_{N}x.json` |
| `--n_variants` | 5 | 每条指令生成的改写变体数量 |
| `--concurrency` | 8 | 并发请求数 |
| `--max_retries` | 3 | 单条指令改写失败最大重试次数 |
| `--no_verify` | False | 跳过语义一致性校验 |
| `--test` | False | 测试模式：1 个变体 + 不校验 + 对比预览 |

**`instruction_rewriter.py`** — 核心改写引擎：

- `rewrite_task_prompt()`：单次改写（含调用级重试 + 占位符校验 + 代码块修复）
- `rewrite_with_variants()`：按权重分配生成多强度变体
- `verify_rewrite()`：九维语义一致性自动化校验
- `process_prompt_entry()`：单条目全流程（改写 + 校验 + 批次级重试）
- `process_version_file()`：完整版本文件批量处理编排

---

### 3.3 Sft_Data_Filter（SFT 数据质量过滤）

对合成数据进行基于 Judge 模型的质量验证与分层筛选。

**`data_filter.py`** — 基于模型的数据过滤流水线：

1. 读取合成答案，调用 `Qwen3.5` 模型生成 **N 个候选答案**（默认 7 个）
2. 使用 `deepseek32` 作为 Judge 模型，以参考答案为基准选出正确候选
3. 输出 `correct_count`（匹配正确的候选数量）
4. 支持断点续传

**`Verifier_Prompts.py`** — 两种 GenRM 验证模式：

| 模式 | 适用场景 |
|------|----------|
| `SELECTION_PROMPT_TEMPLATE_NO_SOLUTION` | 无参考答案的同行评议 |
| `SELECTION_PROMPT_TEMPLATE_WITH_SOLUTION` | 有参考答案的正确答案遴选 |

**`SFT_PostFilter.py`** — 分层后处理筛选策略：

| correct_count | 处理策略 |
|---------------|----------|
| cc = 0 / 缺失 | ❌ 丢弃 |
| cc = 1 ~ 4 | ✅ 100% 保留 |
| cc = 5 | 🎲 30% 随机保留 |
| cc = 6 | 🎲 20% 随机保留 |
| cc ≥ 7 | 🎲 5% 随机保留 |

---

### 3.4 Model_MIX.py（核心训练脚本）

项目的核心训练文件，支持**四种训练模式**的灵活切换：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **预训练模式** | 直接 tokenize，`input_ids = labels`（全序列训练） | 增量预训练 / CPT |
| **非 Think 模式** | 仅对 `assistant\n` 之后的回复计算 loss（输入 mask 为 -100） | 标准 SFT 指令微调 |
| **独立 Think 模式** | 匹配 `assistant\n thinking\n` 到结束符的内容进行训练 | 思维链专项微调 |
| **混合 Think 模式** | 支持空 think 块和带 think 块的混合数据，处理单轮/多轮对话 | 真实业务场景混合训练 |

**关键特性：**

- 🚀 使用 `liger_kernel` 优化的 `AutoLigerKernelForCausalLM`（内存与速度双重优化）
- ⚡ Flash Attention 2（高效注意力计算）
- 🔢 BF16 混合精度训练
- 📦 DeepSpeed ZeRO-3 + CPU Offload（参数 + 优化器卸载）
- 🔄 Gradient Checkpointing（显存优化）
- 📉 Cosine 学习率调度（带 `min_lr`，平滑退火）

---

### 3.5 model_convert16save.py（模型保存/转换）

训练完成后从 DeepSpeed checkpoint 提取并保存最终模型：

- 从 DeepSpeed ZeRO-1 / ZeRO-2 / ZeRO-3 checkpoint 提取并合并分片权重
- 支持转换为 **FP32 / BF16 / FP16** 多种精度
- 支持 **VL（视觉语言）模型**的转换（如 Qwen3.5-VL），保留视觉模块权重完整
- 输出 **safetensors** 格式，便于分发与部署

---

### 3.6 ds_config.json（DeepSpeed 配置）

面向大规模训练的 DeepSpeed 优化配置：

- **ZeRO Stage 3**：参数 + 梯度 + 优化器状态全分片
- **CPU Offload**：参数和优化器状态卸载至 CPU
- **BF16** 混合精度
- **AdamW** 优化器
- 通信优化：`overlap_comm`、`reduce_scatter`、`allgather_bucket_size` 等
- 子分组通信（`sub_group_size`）优化大集群训练效率

---

## 4. 完整训练流水线

```
                            ┌─────────────────────────────┐
                            │    Instruction_Rewrite 🆕    │
                            │   指令多样化改写（3级强度）     │
                            │   9维语义校验 + 两级重试      │
                            └──────────────┬──────────────┘
                                           │ 5x~8x 扩充
                                           ▼
原始医疗数据 ──→ [Data_Synthesis] ──→ 候选答案生成
                     │                       │
                     │              [GenRM 选择 + 质量评估]
                     │                       │
                     ▼                       ▼
               原始指令 + 答案        结构化 SFT 数据
                     │                       │
                     └───────────┬───────────┘
                                 │
                                 ▼
                        [Sft_Data_Filter]
                         Judge 模型验证
                        correct_count 计算
                                 │
                                 ▼
                        [SFT_PostFilter]
                         分层抽样筛选
                                 │
                                 ▼
                          最终 SFT 数据集
                                 │
                        [merge_data.py]
                        原始 + 改写混合
                                 │
                                 ▼
                        [Model_MIX.py]
                    混合思维多模式训练
                   DeepSpeed ZeRO-3 + BF16
                                 │
                                 ▼
                     [model_convert16save.py]
                     Checkpoint → SafeTensors
                                 │
                                 ▼
                            🎯 SFT 模型
```

---

## 5. 快速开始

### 5.1 典型工作流

```bash
# Step 1: 数据合成 —— 从原始问题生成高质量答案
python Data_Synthesis/synth_data.py --input questions.jsonl --output answers.jsonl

# Step 2: 指令多样化改写 —— 扩充指令表达多样性
python Instruction_Rewrite/run_rewrite.py \
    --input prompts/v2.0.0.json \
    --n_variants 5

# Step 3: 数据过滤 —— Judge 模型质量验证
python Sft_Data_Filter/data_filter.py --input answers.jsonl --output filtered.jsonl

# Step 4: 后处理筛选 —— 分层抽样
python Sft_Data_Filter/SFT_PostFilter.py --input filtered.jsonl --output final.jsonl

# Step 5: 合并数据 —— 原始数据 + 改写数据混合
python Instruction_Rewrite/merge_data.py \
    --original_dir ./original/ \
    --rewritten_dir ./rewritten/ \
    --output ./train_data.jsonl \
    --clean

# Step 6: 模型训练
deepspeed --num_gpus 8 Model_MIX.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --train_data_path ./train_data.jsonl \
    --output_dir ./checkpoints \
    --deepspeed ds_config.json

# Step 7: 模型转换
python model_convert16save.py \
    --checkpoint_path ./checkpoints/global_step_xxx \
    --output_path ./final_model \
    --dtype bf16
```

---

## 6. 项目特点总结

| 维度 | 实现方式 |
|------|----------|
| **数据质量** | GenRM 同行评议选择 + 多维度质量评估（100 分制） |
| **指令多样性** | 三级改写强度（轻度/中度/重度）+ 权重化分配 + 九维语义校验 |
| **鲁棒性保障** | 两级重试机制（调用级 + 批次级）+ 代码块自动修复 + 占位符硬校验 |
| **训练效率** | DeepSpeed ZeRO-3 + CPU Offload + Flash Attention 2 + Liger Kernel |
| **训练灵活性** | 增量预训练 / 非Think / 独立Think / 混合Think 四种模式自由切换 |
| **模型输出** | SafeTensors 格式 + 多精度支持 + VL 模型兼容 |
| **容错能力** | 全链路断点续传 + 自动重试 + 降级兜底 |

---
