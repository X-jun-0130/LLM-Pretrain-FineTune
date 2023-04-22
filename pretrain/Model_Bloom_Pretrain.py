# deepspeed --master_addr 172.16.0.126 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  ./Model_Bloom_Pretrain.py
import os
os.chdir('./Nlp_2023/Dialogue_Bloom/')

import torch
import json
from torch.utils.data import Dataset,random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM


model_name = "./Model_TH/Bloom_6B4_zh/"
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
training_args = TrainingArguments(output_dir='./results', 
                                 num_train_epochs=2.2, 
                                 logging_steps=100, 
                                 learning_rate=1e-5,
                                 save_strategy='steps',
                                 save_steps = 6200,
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=32, 
                                 per_device_eval_batch_size=32, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps=500,
                                 weight_decay=0.01,
                                 fp16=True,
                                 gradient_checkpointing=True,
                                 deepspeed='./ds_config.json')
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



diaqa_dataset = json.load(open('./Nlp_2023/Medical_data/pretrain_data/diaqa_data_1024.json', 'r', encoding='utf-8'))
doc_data = json.load(open('./Nlp_2023/Medical_data/pretrain_data/doc_1024.json', 'r', encoding='utf-8'))
kg_data = json.load(open('./Nlp_2023/Medical_data/pretrain_data/pretrain_kg_1024.json', 'r', encoding='utf-8'))
pretrain_data = diaqa_dataset + doc_data + kg_data

print(len(pretrain_data))
print(pretrain_data[100000:100002])

#362570
dataset = data_sets(pretrain_data, 1024)
train_size = int(0.9995 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
print(len(pretrain_data), len(train_dataset))

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
