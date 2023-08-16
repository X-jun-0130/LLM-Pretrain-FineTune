# deepspeed --master_addr 172.xx.xx --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  ./Model_Bloom_Sft.py
import torch
import copy
from datasets import load_dataset
from torch.utils.data import random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM,LlamaTokenizer,LlamaForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

IGNORE_INDEX = -100
flash_attn = True

model_name = '/workspace/Chinese_llama2_7B/epoch3/'
torch.manual_seed(42)

if flash_attn:
    from flash_attn_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()

tokenizer = LlamaTokenizer.from_pretrained(model_name)

START_IDS = [13, 4007, 22137, 29901]
print([tokenizer.decode(START_IDS)])
END_IDS = 2

def tokenize(text):
    inputs_with_mask = tokenizer(text['text'])
    inputs = inputs_with_mask['input_ids']
    labels = [-100] * len(inputs)
    
    for i in range(len(labels)):
        if inputs[i - len(START_IDS): i]  == START_IDS:
            j = inputs.index(2, i)
            for k in range(i,j+1):
                labels[k] = inputs[k]

    return dict(
        input_ids=inputs,
        attention_mask=inputs_with_mask['attention_mask'],
        labels=labels,
    )

'''
data_process
'''
instruction_dataset = load_dataset("json", data_files='/workspace/Nlp_2023/Medical_data/mix_instruction/all_instruction_Chinesellama.json', split="train")
tokenized_dataset = instruction_dataset.map(tokenize, remove_columns=instruction_dataset.column_names, num_proc=64, keep_in_memory=False)
print(len(tokenized_dataset))

train_size = int(0.99 * len(tokenized_dataset))
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, len(tokenized_dataset) - train_size])
print(len(train_dataset), len(val_dataset))

'''
model training
'''
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=150,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch
    logging_steps=10,
    save_strategy='epoch',
    learning_rate=5e-6,
    fp16=True,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    deepspeed='./config_file/ds_config_sft.json'
)

def my_scheduler(optimizer):
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=150, num_training_steps=1600)

training_args.lr_scheduler_fn = my_scheduler

model = LlamaForCausalLM.from_pretrained(model_name, use_cache =False).cuda()
print('model_loaded')


def the_collate_fn(batch):  
    maxlength = max([len(f['input_ids']) for f in batch])
    input_ids = torch.stack([torch.tensor([0]*(maxlength-len(f['input_ids'])) + f['input_ids']) for f in batch])
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
