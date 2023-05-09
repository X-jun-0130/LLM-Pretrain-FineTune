"""
A model worker executes the model.
"""
import argparse
import asyncio
import logging
import json
import time
import threading
import uuid
import torch

from typing import List, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.encoders import jsonable_encoder

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import warnings

warnings.simplefilter("ignore", UserWarning)


class Message(BaseModel):
    role: str = Field(regex='^(User|Assistant)$')
    content: str = ''


class ChatResponse(BaseModel):
    message: str = ''
    finish_reason: str = None
    error_code: str = None


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False


def setup_logger(name, filename, level=logging.DEBUG) -> logging.Logger:
    FORMAT = "[%(levelname)s    %(name)s %(module)s:%(lineno)s - %(funcName)s() - %(asctime)s]\n\t %(message)s \n"
    TIME_FORMAT = "%Y.%m.%d %I:%M:%S %p"
    logging.basicConfig(
        format=FORMAT, datefmt=TIME_FORMAT, level=level, filename=filename
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(FORMAT, TIME_FORMAT))
    logging.getLogger('').addHandler(console_handler)
    logger = logging.getLogger(name)
    return logger


def extract_history_by_length(lst, max_length):
    result = []
    current_length = 0
    for item in reversed(lst):
        new_length = len(''.join(item))
        if current_length + new_length > max_length:
            break
        result.append(item)
        current_length += new_length
    return list(reversed(result))


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model(model_path, device, num_gpus, max_gpu_memory=None):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        logger.info(f"init_kwargs: {kwargs}")
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    return model, tokenizer


def generate_stream(model, tokenizer, inputs, device, context_len=1024):
    inputs = tokenizer(inputs, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        num_beams=1,
        top_k=3,
        repetition_penalty=1.1,
        max_new_tokens=context_len,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for i, output in enumerate(streamer):
        if i > 1:
            yield output.strip('</s>').replace('Assistant:', '')


def generate(model, tokenizer, inputs, device, context_len=1024):
    inputs = tokenizer.encode(inputs, return_tensors="pt").to(device)
    outputs = model.generate(inputs, num_beams=1, top_k=3, repetition_penalty=1.1, max_new_tokens=context_len)
    output = tokenizer.decode(outputs[0])
    return output.strip('</s>')


worker_id = str(uuid.uuid4())[:6]
logger = setup_logger(__name__, f"worker_{worker_id}.log")
global_counter = 0
model_semaphore = None


class ModelWorker:
    def __init__(
        self,
        worker_id,
        model_path,
        model_name,
        device,
        num_gpus,
        max_gpu_memory,
    ):
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]
        self.device = device

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.model, self.tokenizer = load_model(
            model_path, device, num_gpus, max_gpu_memory
        )

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        else:
            self.context_len = 2048

        self.generate_stream_func = generate_stream
        self.generate_func = generate

    def prepare_inputs(self, request_json):
        messages = request_json['messages']
        print(messages)
        inputs = extract_history_by_length(messages, self.context_len - len(messages))
        inputs = [
            i['role'] + ':' + i['content'].strip('\n')  for i in inputs
        ]
        return '</s>\n '.join(inputs) + '</s>\n Assistant:'

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, params):
        try:
            start_time = time.perf_counter()
            inputs = self.prepare_inputs(params)
            print(inputs)
            logger.info('\n' + inputs)
            response = ''
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                inputs,
                self.device,
                self.context_len,
            ):
                if output:
                    response += output
                    ret = {
                        "message": output,
                        "finish_reason": None,
                        "error_code": None,
                    }
                    yield json.dumps(ret, ensure_ascii=False) + '\n'
            ret = {
                "message": "",
                "finish_reason": "stop",
                "error_code": None,
            }
            yield json.dumps(ret, ensure_ascii=False) + '\n'
            execution_time = time.perf_counter() - start_time
            logger.info(
                "\n" + response + f"API execution time: {execution_time:.6f}s"
            )
        except Exception as e:
            logger.info(e)
            ret = {
                "message": e,
                "finish_reason": "error",
                "error_code": 500,
            }
            yield json.dumps(ret, ensure_ascii=False) + '\n'


    def generate_gate(self, params):
        try:
            start_time = time.perf_counter()
            inputs = self.prepare_inputs(params)
            logger.info('\n' + inputs)
            response = self.generate_func(
                self.model,
                self.tokenizer,
                inputs,
                self.device,
                self.context_len,
            ).replace(inputs, '')
            ret = {
                "message": response,
                "finish_reason": None,
                "error_code": None,
            }
            execution_time = time.perf_counter() - start_time
            logger.info(
                "\n" + response + f"API execution time: {execution_time:.6f}s"
            )
            return json.dumps(ret, ensure_ascii=False)
        except Exception as e:
            logger.info(e)
            ret = {
                "message": e,
                "finish_reason": "error",
                "error_code": 500,
            }
            return json.dumps(ret, ensure_ascii=False)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def release_model_semaphore():
    model_semaphore.release()


@app.post(
    "/worker_generate",
    summary="发送一个请求给接口，返回一个流式答案",
    tags=["winGPT"],
    response_model=ChatResponse,
)
async def worker_generate(request: ChatRequest):
    global model_semaphore, global_counter
    global_counter += 1
    params = jsonable_encoder(request)
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    if params['stream']:
        generator = worker.generate_stream_gate(params)
        return StreamingResponse(generator, background=background_tasks)
    else:
        generator = worker.generate_gate(params)
        return Response(generator, background=background_tasks)


@app.get("/worker_get_status")
async def api_get_status():
    return worker.get_status()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5053)
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Nlp_2023/Dialogue_Bloom/Bloom_6b4_sft/",
        help="The path to the weights",
    )
    parser.add_argument("--model-name", type=str, help="Optional name")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(
        worker_id,
        args.model_path,
        args.model_name,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
    )
    uvicorn.run(app, host=args.host, port=args.port)
