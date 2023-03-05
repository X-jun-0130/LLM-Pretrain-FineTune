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
# data_2 = ['<s>' + k['text'] + '</s>' + k['answer'] + '</s>' for j, k in enumerate(payload_list[1600:2000])]
# max_length = max([len(tokenizer.encode(k))  for k in data_1])


# input_data = data_1 
# print("Max length: {}".format(max_length))

# class ModelDataset(Dataset):
#     def __init__(self, txt_list, tokenizer, max_length):
#         self.input_ids = []
#         self.attn_masks = []
#         self.labels = []
#         for txt in txt_list:
#             encodings_dict = tokenizer('<s>' + txt + '</s>', truncation=True,
#                                        max_length=max_length, padding="max_length")
#             self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
#             self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.attn_masks[idx]


# dataset = ModelDataset(input_data, tokenizer, max_length=max_length)
# train_size = int(0.95 * len(dataset))
# train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# Trainer(model=model, args=training_args, train_dataset=train_dataset,
#         eval_dataset=val_dataset, 
#         data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
#                                     'attention_mask': torch.stack([f[1] for f in data]),
#                                     'labels': torch.stack([f[0] for f in data])}).train()

# class ModelDataset(Dataset):
#     def __init__(self, txt_list, tokenizer, max_length):
#         self.input_ids = []
#         self.attn_masks = []
#         self.labels = []
#         for txt in txt_list:
#             encodings_dict = tokenizer('<s>' + txt + '</s>', truncation=True, max_length=max_length)
#             self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
#             self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.attn_masks[idx]

# dataset = ModelDataset(input_data, tokenizer, max_length=max_length)
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


# def evaluate_bleu(predictions, references)


#     return {'bleu':score}

# def compute_metrics(p):
#     loss, logits = p



class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

trainer = Mytrainer(model=model, args=training_args, train_dataset=train_dataset,eval_dataset=val_dataset, data_collator=the_collate_fn)
trainer.train()





'''
inference
'''
def infer(model, payload):
    input_ids = tokenizer('<s>' + payload+ '</s>', return_tensors="pt").input_ids.cuda()
    logits = model.generate(input_ids, num_beams=1, top_k=10, max_length=len(payload)+100)
    out = tokenizer.decode(logits[0].tolist())
    return out


infer_list = ['男孩早泄究竟是什么因素引发的', '糖脉康颗粒适应症有哪些']

s = time.time()
for payload in infer_list:
    model.eval()
    out = infer(model, payload)
    out = out.replace('<s>' + payload + '</s>', '')
    print("="*70+" 模型输入输出 "+"="*70)
    print(f"模型输入: {payload}")
    print(f"模型输出: {out}")
e = time.time()
print('推理耗时：' , str(e-s)+ 's' )

# generated = tokenizer("<s>", return_tensors="pt").input_ids.cuda()
# sample_outputs = model.generate(generated, do_sample=True, top_k=50,
#                                 bos_token='<s>',
#                                 eos_token='</s>',
#                                 max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))