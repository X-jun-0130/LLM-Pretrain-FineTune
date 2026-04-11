# deepspeed --include localhost:0,1,2,3,4,5,6,7  /Model_Sft.py

import os
import torch
from datasets import load_dataset,concatenate_datasets
from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM
#from transformers import  AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

IGNORE_INDEX = -100

model_name ="/data1/Model-TH/Qwen3.5-35B-A3B/"
tokenizer = AutoTokenizer.from_pretrained(model_name)


padding_id = tokenizer.pad_token_id
print(tokenizer.decode(padding_id))
END_IDS = tokenizer.convert_tokens_to_ids('<|im_end|>')

# ############################################
# '''
# 预训练模式使用
# '''
# ###########################################
# def tokenize_function(examples):
#     inputs_with_mask = tokenizer(examples['text'])
#     inputs = inputs_with_mask['input_ids']

#     return dict(
#         input_ids=inputs,
#         attention_mask=inputs_with_mask['attention_mask'],
#         labels=inputs.copy(),
#     )

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


############################################
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
# pt_dataset = load_dataset("json", data_files="/data/private/WAIR/dataset/已整理数据/wndata_sft/AAA-Training-CPT_DATA/data/20250804-no_packing-slice-pt_data.json", split="train",cache_dir="/workspace/cache_dir/")
# pt_tokenized_dataset = pt_dataset.map(tokenize_function, remove_columns=pt_dataset.column_names, num_proc=32, keep_in_memory=False)
# print(len(pt_tokenized_dataset))


### 微调数据
instruction_dataset = load_dataset("json", data_files="/LLM-Train/LLM_SFT/make_sft_data/final_data/20260411-sft-qwen35.jsonl", split="train",cache_dir="/workspace/cache_dir/")
tokenized_dataset = instruction_dataset.map(tokenize_think, remove_columns=instruction_dataset.column_names, num_proc=32, keep_in_memory=False)
print(len(tokenized_dataset))

# Shuffle 数据集
# merged_dataset = concatenate_datasets([pt_tokenized_dataset, tokenized_dataset])
shuffled_tokenized_dataset = tokenized_dataset.shuffle(seed=42)
#3241 + 47924
print(len(shuffled_tokenized_dataset))

''' 23338
model training
'''
training_args = TrainingArguments(
    output_dir='./Train_Result/wingpt-35B',        # output directory
    max_steps = 365,
    per_device_train_batch_size= 16,              # batch size per device during training
    warmup_steps=10,                             # number of warmup steps for learning rate scheduler
    lr_scheduler_type ='cosine_with_min_lr',
    lr_scheduler_kwargs = {'min_lr_rate': 0.1},
    weight_decay=0.01,                            # strength of weight decay
    logging_steps=20,
    save_strategy='steps',
    gradient_accumulation_steps = 1,
    save_steps = 365,
    learning_rate=3e-6,
    bf16=True,
    gradient_checkpointing=True,
    report_to='tensorboard',
    deepspeed='./config_file/ds_config.json'
    )

model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2",  dtype=torch.bfloat16)
model.config.use_cache = False
model.config.tie_word_embeddings=False
print('model_loaded')


# from types import MethodType
# neft_alpha = 5
# input_embed = model.get_input_embeddings()
# if isinstance(input_embed, torch.nn.Embedding):
#     def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
#         embeddings = input_embed.__class__.forward(self, x)
#         dims = self.num_embeddings * self.embedding_dim
#         mag_norm = neft_alpha / (dims ** 0.5)
#         return embeddings + torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)

#     input_embed.forward = MethodType(noisy_forward, input_embed)


def the_collate_fn(batch):  
    maxlength = max([len(f['input_ids']) for f in batch])
    input_ids = torch.stack([torch.tensor(f['input_ids'] + [padding_id]*(maxlength-len(f['input_ids']))) for f in batch])
    attention_mask = torch.stack([torch.tensor(f['attention_mask'] + [0]*(maxlength-len(f['attention_mask']))) for f in batch])
    labels = torch.stack([torch.tensor(f['labels']+[-100]*(maxlength-len(f['labels'])))  for f in batch])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}


class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"], num_items_in_batch=num_items_in_batch)
        return outputs.loss


trainer = Mytrainer(model=model, 
                    args=training_args, 
                    train_dataset=shuffled_tokenized_dataset,
                    data_collator=the_collate_fn
                    )
trainer.train()
