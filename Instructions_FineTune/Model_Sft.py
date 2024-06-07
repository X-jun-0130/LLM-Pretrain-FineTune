# deepspeed --master_addr 172.xxx.94 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  /Model_Sft.py
import os
import torch
from datasets import load_dataset
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

IGNORE_INDEX = -100

model_name = '/Model_WinGPT_pretrain/Meta-Llama-3-8B/'
torch.manual_seed(2024)

tokenizer = AutoTokenizer.from_pretrained(model_name)

#Yi-34B
START_IDS =[198, 22103, 5232]
# START_IDS = [144, 20921, 106]
#START_IDS = [144, 20921, 59601]
END_IDS = tokenizer.eos_token_id
padding_id = tokenizer.eos_token_id

#qwen
# START_IDS = [198, 21388, 25]
# END_IDS = 151643
# padding_id = 151643

print([tokenizer.decode(START_IDS)])
print(tokenizer.decode(END_IDS))
print(tokenizer.decode(padding_id))

def tokenize(text):
    inputs_with_mask = tokenizer(text['text'])
    inputs = inputs_with_mask['input_ids']
    labels = [-100] * len(inputs)
    
    for i in range(len(labels)):
        if inputs[i - len(START_IDS): i]  == START_IDS:
            j = inputs.index(END_IDS, i)
            for k in range(i,j+1):
                labels[k] = inputs[k]

    return dict(
        input_ids=inputs,
        attention_mask=inputs_with_mask['attention_mask'],
        labels=labels,
    )

'''
{'text':'data_text'}
'''
instruction_dataset = load_dataset("json", data_files="/Final_Files/llama_8b_medical.json", split="train",cache_dir="/workspace/cache_dir/")
tokenized_dataset = instruction_dataset.map(tokenize, remove_columns=instruction_dataset.column_names, num_proc=32, keep_in_memory=False)
print(len(tokenized_dataset))

train_size = int(0.995 * len(tokenized_dataset))
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, len(tokenized_dataset) - train_size])
print(len(train_dataset), len(val_dataset))
#81295 409
'''
model training
'''
training_args = TrainingArguments(
    output_dir='./WiNGPT8B_medical',      # output directory
    max_steps = 4235,
    # num_train_epochs=3,                   # total number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    per_device_eval_batch_size=2,           # batch size for evaluation
    warmup_steps=100,                       # number of warmup steps for learning rate scheduler
    weight_decay=0.05,                      # strength of weight decay
    evaluation_strategy="steps",            # Evaluation is done at the end of each epoch
    eval_steps=4886,
    logging_steps=100,
    save_strategy='steps',
    save_steps = 4886,
    learning_rate=5e-5,
    bf16=True,
    gradient_checkpointing=True,
    deepspeed='/config_file/ds_config_sft.json'
)

model = AutoModelForCausalLM.from_pretrained(model_name, use_cache =False, attn_implementation="flash_attention_2").cuda()
print('model_loaded')

from types import MethodType
neft_alpha = 5
input_embed = model.get_input_embeddings()
if isinstance(input_embed, torch.nn.Embedding):
    def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
        embeddings = input_embed.__class__.forward(self, x)
        dims = self.num_embeddings * self.embedding_dim
        mag_norm = neft_alpha / (dims ** 0.5)
        return embeddings + torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)

    input_embed.forward = MethodType(noisy_forward, input_embed)

def the_collate_fn(batch):  
    maxlength = max([len(f['input_ids']) for f in batch])
    input_ids = torch.stack([torch.tensor(f['input_ids'] + [padding_id]*(maxlength-len(f['input_ids']))) for f in batch])
    attention_mask = torch.stack([torch.tensor(f['attention_mask'] + [0]*(maxlength-len(f['attention_mask']))) for f in batch])
    labels = torch.stack([torch.tensor(f['labels']+[-100]*(maxlength-len(f['labels'])))  for f in batch])
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
