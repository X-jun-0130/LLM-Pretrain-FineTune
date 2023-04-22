
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
os.chdir('./Nlp_2023/Dialogue_Bloom/')

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

model_name = './Bloom_6b4_sft/'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto').half().cuda()
model.eval()


app = FastAPI(title='Medical Dialogue',
              description= """
                    **医疗对话** 
                           """
             )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_request: str = ''
    history: list = []


class ChatResponse(BaseModel):
    message: str = ''
    history: list = []

# '''
# inference
# '''
user_id = '#User:'
host_id = '#System:'
def infer(model, payload, history):
    payload = user_id + payload
    if len(history) > 0:
        input_text =  '</s>'.join([k[0]+ '</s>' +k[1] for k in history]) + '</s>' + payload + '</s>' + host_id

    else:
        input_text = payload + '</s>' + host_id

    
    his_length =  max(len(''.join([k[0]+k[1] for k in history])), len(payload))

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    logits = model.generate(input_ids, num_beams=1, top_k=3, repetition_penalty=1.1, max_length= his_length+300)
    out = tokenizer.decode(logits[0].tolist())
    out = out.replace(input_text, '')
    out = out.replace('</s>', '')
    history.append([payload, host_id+out])
    return out, history

@app.post(f"/ask",summary="发送一个请求给接口，返回一个答案",response_model=ChatResponse)
def ask(request_data: ChatRequest):
    inputs = request_data.user_request
    history = request_data.history

    length_token = len(tokenizer.encode(''.join([k[0]+k[1] for k in history]) + inputs))


    if  length_token >= 1020:
        return {"message": '内容超出限制，请重新开始话题。', "history": history}
    else:
        response, history = infer(model, inputs, history)
        if response:
            return {"message": response, "history": history}
        else:
            return {"message": '很抱歉，我暂时无法回答这个问题。', "history": history}

if __name__ == '__main__':
    uvicorn.run('Bloom_api:app', host="0.0.0.0", port=5053)

