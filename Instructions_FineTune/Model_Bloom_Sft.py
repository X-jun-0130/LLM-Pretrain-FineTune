# deepspeed --master_addr 58.210.77.164 --master_port 60147 --include localhost:0,1,2,3,4,5,6,7  ./sft_model.py
import os
import re
import numpy as np
import torch
import json
import random
import copy
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

IGNORE_INDEX = -100

model_name = '/workspace/gptpretarin/'
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)

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
                
    idx = list(set(idx))
    if len(idx) > 0:
        not_idx = [k for k in range(len(labels)) if k not in idx]
        for t in not_idx:
            labels[t] = IGNORE_INDEX
    labels[-1] = 2

    return dict(
        input_ids=inputs_with_offsets['input_ids'],
        attention_mask=inputs_with_offsets['attention_mask'],
        labels=labels,
    )

'''
data_process
'''
instruction_dataset = load_dataset("json", data_files='/workspace/model_sft/instruction_2048.json', split="train")
tokenized_dataset = instruction_dataset.shuffle().map(lambda x: tokenize(x["text"]), num_proc=32)
print(len(tokenized_dataset))

train_size = int(0.95 * len(tokenized_dataset))
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, len(tokenized_dataset) - train_size])
print(len(train_dataset), len(val_dataset))
'''
model training
'''
training_args = TrainingArguments(output_dir='./results', 
                                 num_train_epochs=2, 
                                 learning_rate=3e-6,
                                 logging_steps=100, 
                                 save_strategy='epoch',
                                #  save_steps = 150,
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=20, 
                                 per_device_eval_batch_size=20, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps= 3000,
                                 weight_decay=0.01,
                                 fp16=True,
                                 gradient_checkpointing=True,
                                 deepspeed='./config_file/config_sft.json')
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache =False).cuda()
print('model_loaded')

def the_collate_fn(batch):  
    maxlength = max([len(f['input_ids']) for f in batch])
    input_ids = torch.stack([torch.tensor([3]*(maxlength-len(f['input_ids'])) + f['input_ids']) for f in batch])
    attention_mask = torch.stack([torch.tensor([0]*(maxlength-len(f['attention_mask'])) + f['attention_mask']) for f in batch])
    labels = torch.stack([torch.tensor([-100]*(maxlength-len(f['labels'])) + f['labels']) for f in batch])
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
