# deepspeed --master_addr 172.16.0.126 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  ./Model_Bloom.py
import os
os.chdir('/Nlp_2023/Dialogue_Bloom/')

import torch
import json
import numpy as np
from torch.utils.data import Dataset,random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from data_chunk import chunk_1024

model_name = "/Bloom_6B4_zh/"
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
training_args = TrainingArguments(output_dir='./results', 
                                 overwrite_output_dir=True,
                                 num_train_epochs=2, 
                                 logging_steps=100, 
                                 save_strategy='steps',
                                 save_steps = 1300,
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=32, 
                                 per_device_eval_batch_size=32, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps=100,
                                 weight_decay=0.01,
                                 fp16=True,
                                 gradient_checkpointing=True,
                                 deepspeed='./config_file/ds_config.json')
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache =False).cuda()
# model.resize_token_embeddings(len(tokenizer))

class data_sets(Dataset):
    def __init__(self, txt_list,  max_length):
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt,  max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def sliding_window_template_with_examples(text, length, step):
    left = 0
    right = length
    _list = []
    while right < len(text) +step:
        line = text[left:right]
        _list.append(line)
        left += step
        right += step
    
    return _list

'''
docx_book 51 1600w
docx_zhinan[~800]
drug_xls[~9700]
disease_xls[~6700]

bingli[:50000]

qa[:50000]
op[:50000]
conversion[:50000]
ner[:30000]
classify[:20000]
similarity[:10000]
yin/yang [:5000]
norli[]
'''
kg_list = json.load(open('./data/bookdoc.json', 'r', encoding='utf-8')) + json.load(open('./data/drug_disease_json.json', 'r', encoding='utf-8'))
chunk_list,_ = chunk_1024([['data/bisai_dia.json',100000], ['data/dia_data.json',100000], ['data/kuake_data.json',100000], ['data/ner_data.json',100000], ['data/opqa_list1.json',100000], ['data/qa_data.json',20000]])

doc_all = []
for k in (kg_list+chunk_list):
    if len(k) <= 1100:
        doc_all.append(k)
    else:
        line_list = sliding_window_template_with_examples(k, 1100, 950)
        doc_all.extend(line_list)
        

kg_dataset = ['<s>' + key +'</s>'  for key in doc_all]

cut_dataset = []
for line in kg_dataset:
    k = tokenizer.encode(line)
    if len(k) <= 1024:
        cut_dataset.append(line)

print(cut_dataset[100:102])
dataset = data_sets(cut_dataset, 1024)
train_size = int(0.995 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
print(len(cut_dataset), len(train_dataset))

'''
model training
'''
def the_collate_fn(batch):  
    input_ids = torch.stack([f[0] for f in batch])
    attention_mask = torch.stack([f[1] for f in batch])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}



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
