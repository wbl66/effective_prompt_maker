# This file contains necessary helper functions
# e.g. GPT request, create_dataloader
import openai
import random
import sys
import numpy as np
import torch
import json
import re
from collections import Counter
import time
from openai import OpenAI
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最新生成的 token 是否在停用词列表中
        return input_ids[0][-1].item() in self.stop_token_ids
    
# put your API key here
API_KEY = "sk-eeaae589ea7549f398b19cc27a3c2d09"

NO_SOLUTION = '-10086' # use this when calculating numerical results

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def Qwen_request(model, tokenizer, input_prompt, max_tokens, time_interval, temperature=0.7, stop=None, device_num=1):
    resp = None
    done = False
    while not done:
        try:
            # 对输入提示进行分词，并获取注意力掩码
            inputs = tokenizer(input_prompt[0], return_tensors='pt', padding=True)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            stop_token_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in stop]

            # 创建停止条件
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

            if torch.cuda.is_available():
                input_ids = input_ids.cuda(device_num)
                attention_mask = attention_mask.cuda(device_num)

            # 生成文本
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 传递注意力掩码
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                stopping_criteria=stopping_criteria  # 使用 stopping_criteria
                # 移除不支持的参数（如果模型不支持）
                # frequency_penalty=0,
                # presence_penalty=0,
                # stop=stop
            )

            # 解码生成的文本
            resp = tokenizer.decode(output[0], skip_special_tokens=True)

            done = True
        except Exception as e:
            print(f"Error: {type(e).__name__}\n")
            print(f"Reason: {str(e)}\n")
            # pause between each request to avoid rate limit
            time.sleep(time_interval)
    return resp

def deepseek_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

            resp = client.chat.completions.create(
                model=model,  # 如 "deepseek-chat"
                messages=[{"role": "user", "content": input_prompt[0]}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
            )
            done = True
        except Exception as e:
            err_type, err_msg = sys.exc_info()[:2]
            if "InvalidRequestError" in str(err_type):  # 假设 DeepSeek 有类似的错误类型
                print(f"Invalid Request\nPrompt: {input_prompt}\nReason: {err_msg}")
                raise  # 抛出异常，终止程序
            else:
                print(f"Error: {err_type}\nReason: {err_msg}")
            time.sleep(time_interval)  # 避免速率限制
    return resp

def load_data(args):
    with open(args.dataset_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    question_idxes = [item['question_idx'] for item in data]

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers, question_idxes, data


# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    questions, answers, question_idxes, complete_data = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":question_idxes[idx], "complete_data":complete_data[idx]}) # 每个样本是一个字典

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_input_prompt(args, cot_flag:bool)->str:
    x, z, y = [], [], []
    
    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag: # 是否使用中间推理
            if args.dataset in ("strategyqa", "pubmedqa", "strategyqa_tiny", "pubmedqa_tiny"):
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
            else:
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text


def answer_extraction(args, responses):
    pred_ans = ""
    temp = ""
    if args.model == 'gpt-3.5-turbo' or args.model == 'deepseek-chat':
        try:
            temp = responses['choices'][0]['message']['content']
        except:
            temp = responses.model_dump()['choices'][0]['message']['content'] # 先转为dict再提取
    elif args.model == 'Qwen2.5-7B':
        temp = responses
    else:
        temp = responses['choices'][0].text
    if args.dataset in ("gsm8k", "svamp", "asdiv"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)] # 筛选temp中的所有数字，以列表形式返回
    elif args.dataset in ("strategyqa", "pubmedqa"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("csqa", "qasc"):
        temp = re.findall(r'A|B|C|D|E', temp)

    if len(temp) != 0:
        answer = temp[-1] # 最后出现的是答案
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp", "asdiv"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        pred_ans = answer
    else:
        pred_ans = ""
    return pred_ans


def find_most_frequent(arr, n):
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item