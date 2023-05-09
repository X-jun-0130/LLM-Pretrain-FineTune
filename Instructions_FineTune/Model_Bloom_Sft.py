# deepspeed --master_addr 172.16.0.126 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  ./Model_Bloom_Sft.py
import os
import re
import copy
import torch
import json
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

IGNORE_INDEX = -100

model_name = "./Bloom_6b4_pretrain/"
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
training_args = TrainingArguments(output_dir='./results', 
                                 num_train_epochs=3, 
                                 learning_rate=2e-6,
                                 logging_steps=10, 
                                 save_strategy='steps',
                                 save_steps = 337,
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=32, 
                                 per_device_eval_batch_size=32, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps=30,
                                 weight_decay=0.01,
                                 fp16=True,
                                 gradient_checkpointing=True,
                                 deepspeed='./config_file/ds_config_sft.json')
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache =False).cuda()

def tokenize(text):
    inputs_with_offsets = tokenizer(text, return_offsets_mapping=True)
    labels = copy.deepcopy(inputs_with_offsets['input_ids'])
    offsets = inputs_with_offsets["offset_mapping"]
    matches = re.finditer(r'(\n Assistant:)(.*?)</s>', text, re.DOTALL)

    idx = []
    for match in matches:
        start_pos, end_pos = match.span()
        start_idx = None
        end_idx = None

        for i, (start, end) in enumerate(offsets):
            if start <= start_pos < end+1:
                start_idx = i
            if start <= end_pos < end+1:
                end_idx = i

            if start_idx is not None and end_idx is not None:
                idx.extend([j for j in range(start_idx, int(end_idx))])

    if len(idx) > 0:
        for k in range(len(labels)):
            if k not in idx:
                labels[k] = IGNORE_INDEX
    labels[-1] = 2

    return dict(
        input_ids=inputs_with_offsets['input_ids'],
        attention_mask=inputs_with_offsets['attention_mask'],
        labels=labels,
    )

'''
1.缺少指标异常、正常的指令  qa   2000
2.缺少多轮知识问答的数据（英翻中） dia  500

3.医患对话生成报告   300
4.书籍问答知识题   600（各类)
5.诊断结论  400(各类)
6.实体识别  300 
7.报告摘要  600(各类)
8.报告解读 500（各类）
9.检验指标抽取分析 500（各类）
10.阴阳性 (200)
'''
class data_sets(Dataset):
    def __init__(self, txt_list,  max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for i, txt in enumerate(txt_list):
            encodings_dict = tokenize(txt)
            padding_len = max(0, max_length - len(encodings_dict['input_ids']))
            self.input_ids.append(torch.tensor([3]*padding_len + encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor([0]*padding_len +  encodings_dict['attention_mask']))
            self.labels.append(torch.tensor([-100] * padding_len +  encodings_dict['labels']))
            print(i)
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


# class data_sets(Dataset):
#     def __init__(self, txt_list,  max_length):
#         self.input_ids = []
#         self.attn_masks = []
#         for txt in txt_list:
#             encodings_dict = tokenizer(txt,  max_length=max_length, padding="max_length")
#             self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
#             self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.attn_masks[idx]
  
'''
data_process
'''
instructions = json.load(open('./Nlp_2023/Medical_data/instruction/all_instruction.json', 'r', encoding='utf-8'))
print(len(instructions))
print(instructions[100])

instruction_list = []
for line in instructions:
    k = tokenizer.encode(line)
    if len(k) <= 1024:
        instruction_list.append(line)

print(len(instruction_list))

dataset = data_sets(instruction_list, 1024)
train_size = int(0.88 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
print(len(instruction_list), len(train_dataset))
#49371

'''
model training
'''
# def the_collate_fn(batch):  
#     input_ids = torch.stack([f[0] for f in batch])
#     attention_mask = torch.stack([f[1] for f in batch])
#     return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}


def the_collate_fn(batch):  
    input_ids = torch.stack([f[0] for f in batch])
    attention_mask = torch.stack([f[1] for f in batch])
    labels = torch.stack([f[2] for f in batch])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}


class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss


trainer = Mytrainer(model=model, 
                    args=training_args, 
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset, 
                    data_collator=the_collate_fn
                    )
trainer.train()
