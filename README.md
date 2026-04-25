## 📊 LLM-Pretrain-FineTune 仓库全面分析

### 1. 项目概述

**描述**: Deepspeed、LLM、Medical_Dialogue、医疗大模型、预训练、微调

这是一个**中文医疗大语言模型（LLM）预训练和微调**的完整工作流项目。该项目从数据合成、数据过滤、模型预训练、指令微调到模型保存/转换提供了端到端的解决方案。


---

### 2. 顶层目录结构

```
LLM-Pretrain-FineTune/
├── Data_Synthesis/          # 医疗数据合成（指令生成）
│   ├── prompts.py           #   合成提示模板（GenRM选择模式+质量评估）
│   └── synth_data.py        #   异步数据合成工作流
├── Sft_Data_Filter/         # SFT数据过滤（质量筛选）
│   ├── data_filter.py       #   基于模型的数据过滤（GenRM方法）
│   ├── Verifier_Prompts.py  #   验证器提示模板（有/无参考答案两种模式）
│   └── SFT_PostFilter.py    #   分层后处理筛选（按correct_count抽样）
├── examples/                # 示例截图
│   ├── report.png           #   报告解读示例
│   ├── dialogue.png         #   对话示例
│   ├── medical_consult.png  #   医疗问答示例
│   ├── harmlessness.png     #   无害性回复示例
│   ├── genechat.png         #   单轮转多轮对话示例
│   └── kb_instruction.png   #   知识库Prompt示例
├── Model_MIX.py             # 核心训练脚本（混合思维+混合预训练/微调）
├── model_convert16save.py   # 训练后模型保存/转换（支持Qwen3.5）
├── ds_config.json           # DeepSpeed ZeRO-3配置
└── README.md                # 项目文档
```

---

### 3. 主要组件/模块详解

#### 3.1 **Data_Synthesis（数据合成）**
- **`synth_data.py`**: 异步数据合成工作流，核心流程：
  1. 读取原始问题数据
  2. 为每个问题调用 `deepseek32` 模型生成 **N个候选答案**（默认4个）
  3. 使用 GenRM 选择模式从候选中挑出最优答案
  4. 通过质量评估（满分100分，阈值0.9筛选）
  5. 输出带 reasoning_content 和 score 的结构化数据
  6. 支持断点续传（`load_existing_ids`）
- **`prompts.py`**: 包含两个核心 Prompt 模板：
  - `SELECTION_PROMPT_TEMPLATE`: GenRM同行评议选择最优答案
  - `check_prompt`: 答案质量评估（医学准确性、语言表达等维度）

#### 3.2 **Sft_Data_Filter（数据过滤）**
- **`data_filter.py`**: 基于模型的数据过滤流水线：
  1. 读取合成答案，调用 `Qwen3.5` 模型生成 N个候选答案（默认7个）
  2. 使用 `deepseek32` 作为评判模型（Judge），以参考答案为基准选出正确候选
  3. 输出 `correct_count`（匹配正确的候选数量）
  4. 断点续传支持
- **`Verifier_Prompts.py`**: 两种 GenRM 验证模式：
  - `SELECTION_PROMPT_TEMPLATE_NO_SOLUTION`: 无参考答案的同行评议
  - `SELECTION_PROMPT_TEMPLATE_WITH_SOLUTION`: 有参考答案的正确答案遴选
- **`SFT_PostFilter.py`**: 分层后处理筛选：
  - cc=0/缺失 → 丢弃
  - cc=1~4 → 100%保留
  - cc=5 → 30%随机保留
  - cc=6 → 20%随机保留
  - cc≥7 → 5%随机保留

#### 3.3 **Model_MIX.py（核心训练脚本）**
这是项目的核心训练文件，支持三种模式：

| 模式 | 说明 |
|------|------|
| **预训练模式** | 直接tokenize文本，input_ids=labels（全序列训练） |
| **非think模式** | 仅对 `assistant\n` 之后的回复部分计算loss（输入部分mask为-100） |
| **独立think模式** | 仅匹配 `assistant\n<think>\n` 并训练从该标记到 `<|im_end|>` 的内容 |
| **混合think模式** | 支持空think块和带think块的混合数据，处理单轮/多轮对话 |

关键特性：
- 使用 `liger_kernel` 优化的 `AutoLigerKernelForCausalLM`
- Flash Attention 2
- BF16 训练
- DeepSpeed ZeRO-3 + CPU Offload
- Gradient Checkpointing
- Cosine学习率调度（带min_lr）

#### 3.4 **model_convert16save.py（模型保存/转换）**
- 从 DeepSpeed ZeRO-1/2/3 checkpoint 提取并合并权重
- 支持转换为 FP32/BF16/FP16
- 支持 **VL（视觉语言）模型** 的转换（Qwen3.5-VL），保留视觉模块权重
- 输出 safetensors 格式

#### 3.5 **ds_config.json（DeepSpeed配置）**
- ZeRO Stage 3
- CPU Offload（参数+优化器）
- BF16 混合精度
- AdamW 优化器
- 通信优化（overlap_comm、reduce_scatter等）

---

### 4. 整体目的和功能

该项目是一个**完整的医疗大模型训练流水线**，涵盖以下阶段：

```
原始医疗数据 → [数据合成] → 候选答案生成 → [GenRM过滤] → 质量筛选
    → [分层抽样] → 最终SFT数据 → [Model_MIX训练] → [模型转换] → WiNGPT模型
```

