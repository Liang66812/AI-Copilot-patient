
import re
import json
import time
import random
import torch
import transformers
from datetime import datetime
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore') 
#from datasets import load_dataset


model_id = '/data/zhanganran/meta-llama/Meta-Llama-3-8B-Instruct'
model_id = '/data/zhanganran/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/2b724926966c141d5a60b14e75a5ef5c0ab7a6f0'
QWEN2d5_7B_INST = "/data/liangyunfei/Qwen2.5-7B-Instruct"
QWEN2d5_7B_Code = "/data/liangyunfei/google_download/Qwen2.5-Coder-7B-Instruct"
QWEN2d5_7B_Math = "/data/liangyunfei/google_download/Qwen2.5-Math-7B-Instruct"
QWEN2d5_72B_INST_AWQ = "/data/liangyunfei/Qwen2.5-72B-Instruct-AWQ"
#model_id = 'shenzhi-wang/Llama3-8B-Chinese-Chat'
#model_id = 'casperhansen/llama-3-70b-instruct-awq'
phi4_path = '/data/liangyunfei/google_download/phi-4'
# begin initializing HF items, need auth token for these
#hf_auth = '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'

model_7B = AutoModelForCausalLM.from_pretrained(QWEN2d5_7B_INST, device_map="auto")
tokenizer_7B = AutoTokenizer.from_pretrained(QWEN2d5_7B_INST, device_map="auto")

model_phi4 = AutoModelForCausalLM.from_pretrained(QWEN2d5_7B_INST, device_map="auto")
tokenizer_phi4 = AutoTokenizer.from_pretrained(QWEN2d5_7B_INST, device_map="auto")

#model_72B = AutoModelForCausalLM.from_pretrained(QWEN2d5_72B_INST_AWQ, device_map="auto")
#tokenizer_72B = AutoTokenizer.from_pretrained(QWEN2d5_72B_INST_AWQ, device_map="auto")
#model_Code = AutoModelForCausalLM.from_pretrained(QWEN2d5_7B_Code, device_map="auto")
#tokenizer_COde = AutoTokenizer.from_pretrained(QWEN2d5_7B_Code, device_map="auto")


from transformers import TextStreamer

streamer = TextStreamer(tokenizer_7B, skip_prompt=True, skip_special_tokens=True)
sys_prompt1 = """
    You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature.
    If you don't know the answer to a question, please don't share false information.
"""
sys_prompt = """
你是一个聊天agent，判断对方是不是已经聊天完了。
""" 

sys_prompt1 = '你是一个非常专业的糖尿病患者照护师，你需要了解患者的就餐情况（早餐前、早餐后、午餐前、午餐后、晚餐前、晚餐后、餐间时刻）以及血糖情况（非常偏低、偏低、适宜、偏高、非常偏高），如果不知道的话要进行提问'
sys_prompt1 = '你不是一个好人，你只会吐槽别人，不说好话'
def message_to_response_via_LLM(model, tokenizer, messages, max_new_tokens=256):

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.005)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

#判别是否含有专业问题agent
def yes_no_need(_query):
    _formal = '请判断上一句话包含结束语（如谢谢，再见，明天见，那我先去忙了哈），如果包含仅仅输出是，否则仅仅输出否'
    res_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": _query + _formal},  # 37 62
        ]
    yes_no = message_to_response_via_LLM(model_phi4, tokenizer_phi4, res_messages, max_new_tokens=1000)
    return yes_no

def yes_no_supervise(_query):
    sys_prompt = '你是一个血糖检测的判断师。'
    _formal = '请判断以上一段话是否含有血糖测量的信息，如果有仅仅输出是，否则输出否'
    res_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": _query + _formal},  # 37 62
        ]
    yes_no = message_to_response_via_LLM(model_phi4, tokenizer_phi4, res_messages, max_new_tokens=1000)
    return yes_no

def generate(query):
    sys_prompt = """
    你是一个聊天agent，判断对方是不是已经聊天完了,对方的话进行简短的回应并给予很高的情绪价值
    """ 
    res_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query},  # 37 62
        ]
    yes_no = message_to_response_via_LLM(model_7B, tokenizer_7B, res_messages, max_new_tokens=1000)
    return yes_no



import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, datetime
app = FastAPI()

class Query(BaseModel):
    text: str

list_generate = ['顺便说一下，今天的血糖检测有助于我们更全面地掌握您的健康情况，让我们共同努力，确保您处于最佳状态。', '顺口提一下，通过今天的血糖测试，我们可以更好地理解您的身体状况，让我们携手合作，维持您的最佳健康水平。', '差点忘了，您最近血糖监测不多，尽量每周抽取一餐饭的餐前餐后监测看看哈～']

@app.post("/chat/")
async def chat(query: Query):
    query = query.text
    
    return {"result": generate(query)}
    if '血糖测量' in query or yes_no_supervise(query) == '是':
        return {"result": ''}
    if '再见' in query or yes_no_need(query) == '是':
        return {"result": generate(query) + '\n' + list_generate[random.randint(0, 2)]}
    else:
        return {"result": ''}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50055)  #127.0.0.1    47.98.130.98 0.0.0.0


    

