import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from transformers import AutoTokenizer,AutoModelForCausalLM
import time

# # do the training and checkpoint saving
# # state_dict = get_fp32_state_dict_from_zero_checkpoint('./results/checkpoint-5/') # already on cpu
# # model = model.cpu() # move to cpu
# # model.load_state_dict(state_dict)
# # submit to model hub or save the model to share with others
# # In this example the ``model`` will no longer be usable in the deepspeed context of the same
# # application. i.e. you will need to re-initialize the deepspeed engine, since
# # ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.
# # If you want it all done for you, use ``load_state_dict_from_zero_checkpoint`` instead.
# model = load_state_dict_from_zero_checkpoint(model, './results/checkpoint-5/')
model_name = './Bloom_Save/'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')



# '''
# inference
# '''
def infer(model, payload):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.cuda()
    logits = model.generate(input_ids, num_beams=1, top_k=3, max_length=len(payload)+100)
    out = tokenizer.decode(logits[0].tolist())
    return out


infer_list = ['你好', '糖脉康颗粒适应症有哪些']

s = time.time()
for payload in infer_list:
    model.eval()
    if '</s>' not in payload:
        input_text = '<s>' + payload + '</s>'
        out = infer(model, input_text)
        out = out.replace(input_text, '')
    else:
        input_text = '<s>' + payload
        out = infer(model, input_text)
        out = out.replace(input_text, '')
    print("="*70+" 模型输入输出 "+"="*70)
    print(f"模型输入: {payload}")
    print(f"模型输出: {out}")
e = time.time()
print('推理耗时：' , str(e-s)+ 's' )
