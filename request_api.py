import json
import numpy as np
import random
import requests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./Model_TH/Chatglm_6B/", trust_remote_code=True)
model = AutoModel.from_pretrained("./Model_TH/Chatglm_6B/", trust_remote_code=True, device_map='auto').half().cuda()
model = model.eval()

def get_prompt_answer(text, ues_model=True):
    if ues_model:
        response, history = model.chat(tokenizer, text, history=[])
    else:
        Post_url = "http://{}:8001/ask/"
        Post_data = json.dumps({"user_request": text, history:[]})
        r_json = requests.post(Post_url, Post_data)
        text = json.loads(r_json.text)
        response, history = text['response'] , text['history'] 
    
    return response, history

sft = [k.strip('\n') for k in open('./harmlessness.txt', 'r', encoding='utf-8')]
for k in sft[10000:15000]:
    text = '请扮演医疗人工智能助手，并委婉的拒绝回答下面的问题：\n'
    res, _ = get_prompt_answer(text+k)
    if len(res) > 0:
        print(k + '||' + res.replace('\n', ''))
        print('-------------------')
        json_str = json.dumps({'instruction':k, 'input':'', 'output':res}, ensure_ascii=False)
        with open('./instruction/harmlessness.json', 'a+', encoding='utf-8') as qa_list:
            qa_list.write(json_str + ',' + '\n')

