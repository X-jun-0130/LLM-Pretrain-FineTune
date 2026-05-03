# deepspeed --include localhost:0,1,2,3,4,5,6,7 /workspace/Xuxiangjun/LLM-Train/LLM_SFT/Model_Sft_Stack.py
import os
import re
import torch
from datasets import load_dataset, concatenate_datasets
from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


import warnings
warnings.filterwarnings("ignore")



'''📊 Layer-wise Learning Rate Grouping
----------------------------------------------------------------------
复制层 (lr=5.00e-05): [12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39]
原始层 (lr=1.00e-05): 其余所有层 + embedding/lm_head/norm
参数量: 复制层 ~3.0B | 其余 ~10.0B    ← ⚠️ 两个数字必须都非零！
'''

# ================================================================
# ===== 堆叠模型预设（新增模型时在这里加一条配置即可）==============
# ================================================================
STACK_PRESETS = {
    # Dense 12B：原 32 层 → 堆叠 48 层
    'dense_12b': {
        'model_path':         '/Model_Stack/Qwen3.5-12B/',
        'original_num_layers': 32,
        'block_size':          4,
        'dup_groups':          [2, 3, 4, 5],
        'base_lr':             3e-6,     # 原始层学习率
        'copy_lr_multiplier':  10.0,      # 复制层 = base × 倍数
    },
    # MoE 50B：原 40 层 → 堆叠 56 层（35B-A3B → ~50B-A4B）
    'moe_49b': {
        'model_path':         '/Model_Stack/Qwen3.6-49B-A4B/',
        'original_num_layers': 40,
        'block_size':          4,
        'dup_groups':          [3, 4, 5, 6],
        'base_lr':             3e-6,     # MoE 建议更小的 base_lr（专家数量多，更新要保守）
        'copy_lr_multiplier':  10.0,      # copy_lr = 3e-5
    },
}


# ⬇️⬇️⬇️ 切换模型只改这一行 ⬇️⬇️⬇️
ACTIVE_PRESET = 'dense_12b'     # 可选: 'dense_12b' / 'moe_49b'
# ⬆️⬆️⬆️ 切换模型只改这一行 ⬆️⬆️⬆️


_cfg = STACK_PRESETS[ACTIVE_PRESET]


def _compute_copy_layer_indices(num_original_layers: int, block_size: int,
                                dup_groups: list) -> set:
    """
    根据堆叠参数自动推算复制层在新模型中的索引。
    算法与 layer_stacking[_moe].py 中的 build_layer_mapping 完全一致。
    """
    num_blocks = num_original_layers // block_size
    dup_groups_sorted = sorted(set(dup_groups))
    copy_indices = set()
    new_idx = 0
    for block_idx in range(num_blocks):
        # 原始块
        new_idx += block_size
        # 复制块（紧跟在原始块之后）
        if block_idx in dup_groups_sorted:
            for offset in range(block_size):
                copy_indices.add(new_idx + offset)
            new_idx += block_size
    return copy_indices


# ===== 自动派生的运行时常量 =====
COPY_LAYER_INDICES = _compute_copy_layer_indices(
    _cfg['original_num_layers'], _cfg['block_size'], _cfg['dup_groups']
)
COPY_LR_MULTIPLIER = _cfg['copy_lr_multiplier']
LAYER_INDEX_PATTERN = re.compile(r'\.layers\.(\d+)\.')   # 通用正则，兼容 Liger 简化命名

# 打印激活配置（便于日志追溯）
print(f"\n[StackPreset] ACTIVE={ACTIVE_PRESET}")
print(f"[StackPreset] model_path      = {_cfg['model_path']}")
print(f"[StackPreset] original_layers = {_cfg['original_num_layers']}")
print(f"[StackPreset] dup_groups      = {_cfg['dup_groups']}")
print(f"[StackPreset] new_total_layer = {_cfg['original_num_layers'] + len(_cfg['dup_groups']) * _cfg['block_size']}")
print(f"[StackPreset] copy_indices    = {sorted(COPY_LAYER_INDICES)}")
print(f"[StackPreset] base_lr         = {_cfg['base_lr']:.2e}")
print(f"[StackPreset] copy_lr         = {_cfg['base_lr'] * COPY_LR_MULTIPLIER:.2e}\n")


LAYER_INDEX_PATTERN = re.compile(r'\.layers\.(\d+)\.')
learning_rate = _cfg['base_lr']

IGNORE_INDEX = -100

model_name = _cfg['model_path']
tokenizer = AutoTokenizer.from_pretrained(model_name)


padding_id = tokenizer.pad_token_id
print(tokenizer.decode(padding_id))
END_IDS = tokenizer.convert_tokens_to_ids('<|im_end|>')

############################################
'''
预训练模式使用
'''
###########################################
def tokenize_function(examples):
    inputs_with_mask = tokenizer(examples['text'])
    inputs = inputs_with_mask['input_ids']

    return dict(
        input_ids=inputs,
        attention_mask=inputs_with_mask['attention_mask'],
        labels=inputs.copy(),
    )

# ############################################
# '''
# 非<think>模式使用
# '''
# ###########################################
# START_IDS = tokenizer.encode("<|im_start|>assistant\n")
# END_IDS = tokenizer.convert_tokens_to_ids('<|im_end|>')
# print([tokenizer.decode(START_IDS)])
# print(tokenizer.decode(END_IDS))
# print(START_IDS)
# print(END_IDS)

# def tokenize(text: dict) -> dict:
#     """
#     Tokenize input text and prepare it for model training with appropriate labels.
#     Args:
#         text (dict): Input dictionary containing 'text' key with the input string
#     Returns:
#         dict: Dictionary containing:
#             - input_ids: Tokenized input sequence
#             - attention_mask: Attention mask for the input sequence
#             - labels: Training labels (-100 for non-target tokens)
#     """
#     # Tokenize the input text and get input IDs and attention mask
#     inputs_with_mask = tokenizer(text['text'])
#     inputs = inputs_with_mask['input_ids']
#     attention_mask = inputs_with_mask['attention_mask']
    
#     # Initialize labels with -100 (ignored in loss calculation)
#     labels = [-100] * len(inputs)
    
#     for i in range(len(labels)):
#         if inputs[i - len(START_IDS): i]  == START_IDS:
#             j = inputs.index(END_IDS, i)
#             for k in range(i,j+1):
#                 labels[k] = inputs[k]

#     return dict(
#         input_ids=inputs,
#         attention_mask=attention_mask,
#         labels=labels,
#     )





# ############################################
# '''
# 独立<think>模式使用
# 仅有 assistant\n<think>\n 这一种形式，都是带有思考的
# label仅计算从<think>\n之后开始训练
# '''
# ############################################
# START_IDS = tokenizer.encode("assistant\n<think>\n")  # 匹配 assistant\n<think>\n
# print("START_IDS:", START_IDS)
# print("START_IDS decoded:", [tokenizer.decode(START_IDS)])
# print("END_IDS:", END_IDS)

# def tokenize_think(text: dict) -> dict:
#     """
#     Tokenize input text and prepare it for model training with appropriate labels.
#     独立<think>模式：从 <think>\n 之后开始训练到 <|im_end|>
    
#     Args:
#         text (dict): Input dictionary containing 'text' key with the input string
#     Returns:
#         dict: Dictionary containing:
#             - input_ids: Tokenized input sequence
#             - attention_mask: Attention mask for the input sequence
#             - labels: Training labels (-100 for non-target tokens)
#     """
#     # Tokenize the input text and get input IDs and attention mask
#     inputs_with_mask = tokenizer(text['text'])
#     inputs = inputs_with_mask['input_ids']
#     attention_mask = inputs_with_mask['attention_mask']
    
#     # Initialize labels with -100 (ignored in loss calculation)
#     labels = [-100] * len(inputs)
    
#     len_start = len(START_IDS)
    
#     # 遍历找到所有 assistant\n<think>\n 块，从该模式之后开始设置labels
#     i = 0
#     while i < len(inputs):
#         # 匹配 assistant\n<think>\n
#         if i >= len_start and inputs[i - len_start: i] == START_IDS:
#             # 从 <think>\n 之后开始训练到 <|im_end|>
#             try:
#                 j = inputs.index(END_IDS, i)
#                 for k in range(i, j + 1):
#                     labels[k] = inputs[k]
#                 i = j + 1
#             except ValueError:
#                 # 如果找不到 <|im_end|>，则将剩余部分都作为 labels
#                 for k in range(i, len(inputs)):
#                     labels[k] = inputs[k]
#                 break
#         else:
#             i += 1

#     return dict(
#         input_ids=inputs,
#         attention_mask=attention_mask,
#         labels=labels,
#     )


###########################################
'''
混合<think>模式使用,需要将<think>设置为单独的token
支持单轮和多轮对话，每轮对话可能是:
1. 空思考块: assistant
<think>

</think>

回复内容 -> 只训练回复内容
2. 有思考块: assistant
<think>
思考内容</think>

回复内容 -> 训练思考+回复
多轮对话中可能混合存在空思考和有思考的情况
'''
############################################
print(tokenizer.decode(END_IDS))
print(END_IDS)

# 预定义不同模式的START_IDS
START_IDS_EMPTY_THINK = tokenizer.encode("assistant\n<think>\n\n</think>\n\n")  # 空思考块，从这之后开始训练
START_IDS_WITH_THINK = tokenizer.encode("assistant\n<think>\n")  # 有思考块，从<think>\n之后开始训练

print("Empty think START_IDS:", START_IDS_EMPTY_THINK)
print("With think START_IDS:", START_IDS_WITH_THINK)

def tokenize_think(text: dict) -> dict:
    """
    Tokenize input text and prepare it for model training with appropriate labels.
    支持混合<think>模式，处理单轮和多轮对话：
    - 空思考块: 只训练</think>\n\n后的回复内容
    - 有思考块: 训练<think>\n后的思考内容和回复内容 
    - 混合模式: 多轮对话中每轮独立判断
    
    Args:
        text (dict): Input dictionary containing 'text' key with the input string
    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequence
            - attention_mask: Attention mask for the input sequence
            - labels: Training labels (-100 for non-target tokens)
    """
    # Tokenize the input text and get input IDs and attention mask
    inputs_with_mask = tokenizer(text['text'])
    inputs = inputs_with_mask['input_ids']
    attention_mask = inputs_with_mask['attention_mask']
    
    # Initialize labels with -100 (ignored in loss calculation)
    labels = [-100] * len(inputs)
    
    len_empty = len(START_IDS_EMPTY_THINK)
    len_with = len(START_IDS_WITH_THINK)
    # 空思考块的后缀部分（用于检查是否是空思考）
    empty_suffix = START_IDS_EMPTY_THINK[len_with:]
    
    # 遍历找到所有assistant回复块，根据每轮的情况设置对应的labels
    i = 0
    while i < len(inputs):
        # 优先匹配空思考块（更长的模式优先）
        if i >= len_empty and inputs[i - len_empty: i] == START_IDS_EMPTY_THINK:
            # 空思考块：从</think>\n\n之后开始训练到<|im_end|>
            try:
                j = inputs.index(END_IDS, i)
                for k in range(i, j + 1):
                    labels[k] = inputs[k]
                i = j + 1
            except ValueError:
                for k in range(i, len(inputs)):
                    labels[k] = inputs[k]
                break
        
        # 匹配有思考块（需要排除空思考块的情况）
        elif i >= len_with and inputs[i - len_with: i] == START_IDS_WITH_THINK:
            # 检查是否是空思考块的一部分（向后看是否紧跟空思考的后缀）
            is_empty_think = False
            suffix_len = len(empty_suffix)
            if i + suffix_len <= len(inputs):
                if inputs[i: i + suffix_len] == empty_suffix:
                    is_empty_think = True
            
            if not is_empty_think:
                # 有思考块：从<think>\n之后开始训练到<|im_end|>
                try:
                    j = inputs.index(END_IDS, i)
                    for k in range(i, j + 1):
                        labels[k] = inputs[k]
                    i = j + 1
                except ValueError:
                    for k in range(i, len(inputs)):
                        labels[k] = inputs[k]
                    break
            else:
                i += 1
        else:
            i += 1

    return dict(
        input_ids=inputs,
        attention_mask=attention_mask,
        labels=labels,
    )



### 预训练数据
pt_dataset = load_dataset("json", data_files="/AAA-Training-CPT_DATA/data/20250804-no_packing-slice-pt_data.json", split="train",cache_dir="/workspace/cache_dir/")
pt_tokenized_dataset = pt_dataset.map(tokenize_function, remove_columns=pt_dataset.column_names, num_proc=32, keep_in_memory=False)
print(len(pt_tokenized_dataset))


### 微调数据
instruction_dataset = load_dataset("json", data_files="/LLM_SFT/make_sft_data/final_data/20260502-sft-qwen36.jsonl", split="train",cache_dir="/workspace/cache_dir/")
tokenized_dataset = instruction_dataset.map(tokenize_think, remove_columns=instruction_dataset.column_names, num_proc=32, keep_in_memory=False)
print(len(tokenized_dataset))

# Shuffle 数据集
merged_dataset = concatenate_datasets([pt_tokenized_dataset, tokenized_dataset])
shuffled_tokenized_dataset = merged_dataset.shuffle(seed=42)
#3241 + 38155
print(len(shuffled_tokenized_dataset))

''' 23342
model training
'''
training_args = TrainingArguments(
    output_dir='./Train_Result/wingpt-12B',        # output directory
    max_steps = 1294,
    per_device_train_batch_size= 4,              # batch size per device during training
    warmup_steps=10,                             # number of warmup steps for learning rate scheduler
    lr_scheduler_type ='cosine_with_min_lr',
    lr_scheduler_kwargs = {'min_lr_rate': 0.1},
    weight_decay=0.01,                            # strength of weight decay
    logging_steps=20,
    save_strategy='steps',
    gradient_accumulation_steps = 2,
    save_steps = 1294,
    learning_rate=learning_rate,
    bf16=True,
    gradient_checkpointing=True,
    report_to='tensorboard',
    deepspeed='./ds_config_stack.json'
    )

model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2",  dtype=torch.bfloat16)
model.config.use_cache = False
model.config.tie_word_embeddings=False
print('model_loaded')


def the_collate_fn(batch):  
    maxlength = max([len(f['input_ids']) for f in batch])
    input_ids = torch.stack([torch.tensor(f['input_ids'] + [padding_id]*(maxlength-len(f['input_ids']))) for f in batch])
    attention_mask = torch.stack([torch.tensor(f['attention_mask'] + [0]*(maxlength-len(f['attention_mask']))) for f in batch])
    labels = torch.stack([torch.tensor(f['labels']+[-100]*(maxlength-len(f['labels'])))  for f in batch])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}


class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"], num_items_in_batch=num_items_in_batch)
        return outputs.loss

    def log(self, logs, start_time=None):
        if self.optimizer is not None and len(self.optimizer.param_groups) >= 3:
            logs["lr_copy"] = self.optimizer.param_groups[0]["lr"]
            logs["lr_base"] = self.optimizer.param_groups[2]["lr"]
        super().log(logs, start_time)

    def create_optimizer(self):
        """
        分层学习率：
          - 复制层 (COPY_LAYER_INDICES)        : lr = base_lr * COPY_LR_MULTIPLIER
          - 原始层 + embedding/lm_head/norm 等 : lr = base_lr
        weight_decay 同时按 LayerNorm/bias 区分。
        """
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        # 调试打印 + 防御性类型转换，兼容 DeepSpeed auto / str / tuple 等异常类型
        raw_lr = self.args.learning_rate
        print(f"[DEBUG] self.args.learning_rate type={type(raw_lr).__name__}, value={raw_lr!r}")
        if isinstance(raw_lr, (list, tuple)):
            base_lr = float(raw_lr[0])
        else:
            base_lr = float(raw_lr)

        raw_wd = self.args.weight_decay
        weight_decay = float(raw_wd) if not isinstance(raw_wd, (list, tuple)) else float(raw_wd[0])

        copy_lr = base_lr * COPY_LR_MULTIPLIER

        groups = {
            "copy_decay":   {"params": [], "lr": copy_lr, "weight_decay": self.args.weight_decay},
            "copy_nodecay": {"params": [], "lr": copy_lr, "weight_decay": 0.0},
            "base_decay":   {"params": [], "lr": base_lr, "weight_decay": self.args.weight_decay},
            "base_nodecay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        }

        # 在 DeepSpeed ZeRO-3 下，param.numel() 返回的是本地 shard 的大小（往往为 0），
        # 全局参数量保存在 param.ds_numel 属性里。此处优先取 ds_numel。
        def _pnumel(p):
            return getattr(p, 'ds_numel', None) or p.numel()

        n_copy, n_base = 0, 0
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            m = LAYER_INDEX_PATTERN.search(name)
            is_copy = bool(m and int(m.group(1)) in COPY_LAYER_INDICES)
            use_decay = name in decay_parameters

            if is_copy:
                groups["copy_decay" if use_decay else "copy_nodecay"]["params"].append(param)
                n_copy += _pnumel(param)
            else:
                groups["base_decay" if use_decay else "base_nodecay"]["params"].append(param)
                n_base += _pnumel(param)

        optimizer_grouped_parameters = [g for g in groups.values() if len(g["params"]) > 0]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.args.local_rank in (-1, 0):
            print("=" * 70)
            print("  📊 Layer-wise Learning Rate Grouping")
            print("-" * 70)
            print(f"  复制层 (lr={copy_lr:.2e}): {sorted(COPY_LAYER_INDICES)}")
            print(f"  原始层 (lr={base_lr:.2e}): 其余所有层 + embedding/lm_head/norm")
            print(f"  参数量: 复制层 {n_copy/1e9:.2f}B | 其余 {n_base/1e9:.2f}B")
            for gname, g in groups.items():
                if g["params"]:
                    total = sum(_pnumel(p) for p in g["params"])
                    print(f"    [{gname:14s}] tensors={len(g['params']):4d}  "
                          f"params={total/1e6:10.4f}M  lr={g['lr']:.2e}  wd={g['weight_decay']}")
            print("=" * 70)

        return self.optimizer
    
    
trainer = Mytrainer(model=model, 
                    args=training_args, 
                    train_dataset=shuffled_tokenized_dataset,
                    data_collator=the_collate_fn
                    )
trainer.train()

