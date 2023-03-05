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
                                 num_train_epochs=1, 
                                 logging_steps=200, 
                                 save_total_limit = 90,
                                 save_strategy='epoch',
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=8, 
                                 per_device_eval_batch_size=8, 
                                 warmup_steps=1000,
                                 weight_decay=0.01,
                                 fp16=True,
                                 logging_dir='./logs', 
                                 deepspeed='./ds_config.json')
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
# model.resize_token_embeddings(len(tokenizer))



'''
data_process
'''
qa_list = json.load(open('data/qa_train_data.json', 'r', encoding='utf-8'))
# dialogue_list = json.load(open('data/small_dialogue_train_data.json', 'r', encoding='utf-8'))
kg_list = json.load(open('data/kg_drug_train_data.json', 'r', encoding='utf-8')) + json.load(open('data/kg_dis_train_data.json', 'r', encoding='utf-8')) + json.load(open('data/cmekg_dis_train_data.json', 'r', encoding='utf-8'))+json.load(open('data/cmekg_med_train_data.json', 'r', encoding='utf-8'))

qa_dataset = [['<s>' + k['text'] + '</s>' + k['answer'][:400] + '</s>', j] for j, k in enumerate(qa_list[:100000])]
# dia_dataset = [['<s>' + '</s>'.join(d) + '</s>', i] for i, d in enumerate(dialogue_list[:100])]
kg_dataset = [['<s>' + k['text'] + '</s>' + k['answer'][:400] + '</s>', j] for j, k in enumerate(kg_list)]

dataset = qa_dataset + kg_dataset

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

print(len(train_dataset))


'''
model training
'''
def the_collate_fn(batch):  
    r = tokenizer([b[0] for b in batch], padding=True )
    input_ids = torch.LongTensor(r['input_ids'])
    attention_mask = torch.LongTensor(r['attention_mask'])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}

class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

trainer = Mytrainer(model=model, args=training_args, train_dataset=train_dataset,eval_dataset=val_dataset, data_collator=the_collate_fn)
trainer.train()

