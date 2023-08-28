# deepspeed --master_addr 172.xx.xx.94 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  ./Model_Pretrain.py
import os
os.chdir('/workspace/Nlp_2023/Dialogue_Bloom/')
import torch
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer, LlamaTokenizer,LlamaForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup
from itertools import chain
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

model_name = '/workspace/Model_TH/Tiger-llama2-13B/'
torch.manual_seed(42)

model_max_length = 4096
flash_attn = True

# Initialize a  tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# TrainingArguments include various training parameters

training_args = TrainingArguments(
    output_dir='./pretrain13B_results',  # output directory
    num_train_epochs=3,                  # total number of training epochs
    per_device_train_batch_size=4,       # batch size per device during training
    per_device_eval_batch_size=4,        # batch size for evaluation
    warmup_steps=500,                   # number of warmup steps for learning rate scheduler
    weight_decay=0.05,                   # strength of weight decay
    evaluation_strategy="epoch",         # Evaluation is done at the end of each epoch
    logging_steps=100,
    save_strategy='epoch',
    learning_rate=5e-5,
    fp16=True,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    deepspeed='./config_file/ds_config.json'
)

def my_scheduler(optimizer):
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=12000)

training_args.lr_scheduler_fn = my_scheduler

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // model_max_length) * model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + model_max_length] for i in range(0, total_length, model_max_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'])

# Load your data
texts = load_dataset("json", data_files='/workspace/Nlp_2023/Medical_data/pretrain_data_llama/pretrain_data.json', split="train")
texts_token = texts.map(tokenize_function, batched=True,remove_columns=texts.column_names,num_proc=32, keep_in_memory=False, fn_kwargs={"tokenizer":tokenizer})
dataset = texts_token.map(group_texts, batched=True, num_proc=32, keep_in_memory=False)

eval_size = int(0.0002 * len(dataset))

print('loaded data')
# Split the dataset into a train and eval dataset
train_size = len(dataset) - eval_size
train_dataset, eval_dataset = random_split(dataset , [train_size, eval_size])

print(len(dataset), len(eval_dataset))

# Initialize a  GPT model
model = LlamaForCausalLM.from_pretrained(model_name, use_cache =False).cuda()
print('model_loaded')


def the_collate_fn(batch): 
    input_ids = torch.stack([torch.tensor(f['input_ids']) for f in batch])
    attention_mask = torch.stack([torch.tensor(f['attention_mask']) for f in batch])
    labels = torch.stack([torch.tensor(f['labels']) for f in batch])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}

class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

# Define the Trainer
trainer = Mytrainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,           # evaluation dataset
    data_collator=the_collate_fn
)

# Train the model
trainer.train()
