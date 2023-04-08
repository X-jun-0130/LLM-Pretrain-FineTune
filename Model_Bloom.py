# deepspeed --master_addr 0.0.0.0 --master_port 6006 --include localhost:0,1,2,3 ./Model_Bloom.py
import torch
import time
import json
import numpy as np
from torch.utils.data import Dataset, random_split
import pandas as pd
# from transformers import BloomModel
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy



model_name = "./Bloom_7B1/"


torch.manual_seed(42)



tokenizer = AutoTokenizer.from_pretrained(model_name)
training_args = TrainingArguments(output_dir='./results', 
                                 overwrite_output_dir=True,
                                 num_train_epochs=2, 
                                 learning_rate=5e-5,
                                 logging_steps=50, 
                                 save_strategy='steps',
                                 save_steps = 450,
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=32, 
                                 per_device_eval_batch_size=32, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps=200,
                                 weight_decay=0.01,
                                 fp16=True,
                                 gradient_checkpointing=True,
                                 deepspeed='./config_file/ds_config_sft.json')
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


'''
data_process
'''
kg_list = json.load(open('./data_sft/medical_sft.json', 'r', encoding='utf-8')) + json.load(open('./data_sft/ner_sft.json', 'r', encoding='utf-8'))[:1000] + json.load(open('./data_sft/opqa_test.json', 'r', encoding='utf-8'))[:1000] + json.load(open('./data_sft/kuake_data.json', 'r', encoding='utf-8'))+json.load(open('./data_sft/qa_sft.json', 'r', encoding='utf-8'))
# kg_dataset = [['<s>'+key['text'] +'</s>' + key['answer'] +'</s>', j] for j,key in enumerate(kg_list)]
kg_dataset = ['<s>'+ 'User:'+key['text'] +'</s>' + 'System:'+key['answer'] +'</s>'  for key in kg_list]

#4000
dia_list = json.load(open('./data_sft/dia_sft.json', 'r', encoding='utf-8')) 
dia_2 = [['<s>'+  '</s>'.join(key) +'</s>', j] for j,key in enumerate(json.load(open('./data_sft/dia_data.json', 'r', encoding='utf-8')) )]

dia_dataset = ['<s>'+ key +'</s>' for key in  (dia_2 + dia_list)]
print(dia_dataset[0])


cut_dataset = []
for line in (kg_dataset + dia_dataset):
    k = tokenizer.encode(line)
    if len(k) <= 1024:
        cut_dataset.append(line)
print(cut_dataset[0:2])

dataset = data_sets(cut_dataset, 1024)
train_size = int(0.98 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
print(len(cut_dataset), len(train_dataset))
#11425

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
