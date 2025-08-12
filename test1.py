import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch

import json
import time
import re
import itertools
import numpy as np
import pandas as pd
import datetime as DT
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import curve_fit
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# Load pretrained LLM
QWEN2d5_7B_INST = "/data/liangyunfei/Qwen2.5-7B-Instruct"
QWEN2d5_32B_INST_AWQ = "/home/peizhengqi/Qwen/Qwen2.5-32B-Instruct-AWQ"
QWEN2d5_72B_INST_AWQ = "/data/liangyunfei/Qwen2.5-72B-Instruct-AWQ"

#model_7Bn = AutoModelForCausalLM.from_pretrained(QWEN2d5_7B_INST, device_map="auto")
#tokenizer_7Bn = AutoTokenizer.from_pretrained(QWEN2d5_7B_INST, device_map="auto")

model_32Bn = AutoModelForCausalLM.from_pretrained(QWEN2d5_72B_INST_AWQ, device_map="auto")
tokenizer_32Bn = AutoTokenizer.from_pretrained(QWEN2d5_72B_INST_AWQ, device_map="auto")


from transformers import TextStreamer

streamer = TextStreamer(tokenizer_32Bn, skip_prompt=True, skip_special_tokens=True)



sys_prompt1 = """
    You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature.
    If you don't know the answer to a question, please don't share false information.
"""
sys_prompt1 = '你是一个非常专业的糖尿病患者照护师，请专注于给对方提供情绪价值，输出用简短的话语'

def message_to_response_via_LLM(model, tokenizer, messages, max_new_tokens=256):

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.01, streamer=streamer)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
def message_to_response_no_stream(model, tokenizer, messages, max_new_tokens=256):

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.01)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, datetime
app = FastAPI()

class Query(BaseModel):
    text: str



@app.post("/chat/")
async def chat(query:Query):
    print(123)
    answer = message_to_response_via_LLM(model_32Bn, tokenizer_32Bn, query.text, max_new_tokens=1000)
    #json.loads(postdata)
    return {"result": query.text}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80055)    #openai_api_base = "http://127.0.0.1:50055/v1"  uvicorn.run(app, host="0.0.0.0", port=6667)