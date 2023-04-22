import os
os.chdir('./Nlp_2023/Dialogue_Bloom/')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import json

from torch.utils.data import Dataset, random_split
from data_chunk import chunk_list
import transformers
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["query_key_value"]
OUTPUT_DIR = './data_sft/'
model_name = "./Bloom_6b4_pretrain/"
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
training_args = TrainingArguments(
                                 output_dir=OUTPUT_DIR,
                                 num_train_epochs=3, 
                                 learning_rate=5e-6,
                                 logging_steps=50, 
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=24, 
                                 per_device_eval_batch_size=24, 
                                 lr_scheduler_type="cosine",
                                 warmup_steps=50,
                                 weight_decay=0.01,
                                 fp16=True,
                                 gradient_checkpointing=True
                                 )
model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit =True, device_map='auto', use_cache =False)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    )
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)
model.print_trainable_parameters()
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
kg_list = json.load(open('./data_sft/medical_sft.json', 'r', encoding='utf-8')) 
kg_dataset = ['#User:'+key['text'].strip('\n') +'</s>' + '#System:'+key['answer'].strip('\n')  for key in kg_list]

kg_qa  =[k +'</s>' for k in  chunk_list([kg_dataset])]
print(kg_qa[-5])
#4000

cut_dataset = []
for line in kg_qa :
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

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

if torch.__version__ >= "2" :
    model = torch.compile(model)

trainer.train()
print("\n If there's a warning about missing keys above, please disregard :)")
model.save_pretrained(OUTPUT_DIR)



'''
inference
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16,device_map="auto")
model = PeftModel.from_pretrained(model,args.lora_weights,torch_dtype=torch.float16)

'''
