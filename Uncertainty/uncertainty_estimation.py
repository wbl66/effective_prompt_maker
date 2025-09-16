# This file used to generate uncertainty score for each question
from utils import *
import time
import argparse
import numpy as np
import json
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_clusters", type=int, default=11, help="cluster number, default is 11")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "asdiv", "strategyqa", "csqa", "qasc", "pubmedqa"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="./basic_sot_prompts/math_word_problems", help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen2.5-7B", choices=["Qwen2.5-7B", "deepseek-chat"], help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot", "few_shot_sot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./add_uncertainty_scores", help="output directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--num_trails", type=int, default=10, help="number of trails to run for each question"
    )
    parser.add_argument(
        "--sort_by", type=str, default='disagreement', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    
    args = parser.parse_args()

    args.dataset_path = f"./add_typicality_scores_and_labels_dataset/{args.dataset}/dataset_with_scores_and_labels_{args.num_clusters}.json"
    
    # Fill in the dataset path
    if args.dataset in ("gsm8k", "svamp", "asdiv"):
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset in ("strategyqa", "pubmedqa"):
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset in ("csqa", "qasc"):
        args.direct_answer_trigger = "\nSo the optimal answer choice is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.dataset in ("csqa", "qasc"):
        args.direct_answer_trigger_for_fewshot = "So the optimal answer choice is"
    
    return args

def generate_uncertainty_qes(args, question, Qwen_model, Qwen_tokenizer):
    if args.method in ("few_shot_cot", "few_shot_sot"):
        given_prompt = create_input_prompt(args, True) # demonstrations
        # print(f"Given prompt: {given_prompt}")

    if args.dataset in ("gsm8k", "svamp", "asdiv"):
        # the float is reserved for variance calculation result
        uncertainty_record = {'dataset_idx':question['question_idx'], 'variance':float, 'entropy':float, 'occurrence':{}}
    elif args.dataset in ("strategyqa", "pubmedqa"):
        uncertainty_record = {'dataset_idx':question['question_idx'], 'entropy':float, 'occurrence':{"yes":0, "no":0}}
    else:
        uncertainty_record = {'dataset_idx':question['question_idx'], 'entropy':float, 'occurrence':{}}

    for trail in range(args.num_trails): # 为测不确定性，多次运行模型
        # if zero-shot to generate uncertainty, construct first stage zero-shot prompt (step by step)
        if args.method == "few_shot_cot":
            prompt = given_prompt + "Q: " + question['question'] + "\nA: Let's think step by step."
        elif args.method == "few_shot_sot":
            prompt = given_prompt + "Q: " + question['question'] + "\nA:" 
        elif args.method == "zero_shot_cot":
            prompt = "Q: " + question['question'] + "\nA: Let's think step by step."
        prompt_list = [prompt]
        # print(prompt_list)

        if args.model == "Qwen2.5-7B": # 调用本地部署的Qwen2.5-7B模型
            responses = Qwen_request(model=Qwen_model, tokenizer=Qwen_tokenizer, input_prompt=prompt_list, max_tokens=args.max_length, time_interval=args.api_time_interval
                                    , temperature=args.temperature , stop=['Question:', "Q:"], device_num=int(args.device[-1]))
                
            if args.method == "zero_shot_cot":
                prompt_list[0] += responses + args.direct_answer_trigger # 总结最终答案

                responses = Qwen_request(model=args.model, tokenizer=Qwen_tokenizer, input_prompt=prompt_list, max_tokens=args.max_length, time_interval=args.api_time_interval,
                                        temperature=args.temperature, stop='.', device_num=int(args.device[-1]))
        else:
            # if use zero-shot, here we get the first stage zero-shot result
            # if not use zero-shot, here we get the final output
            responses = deepseek_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length, time_interval=args.api_time_interval
                                    , temperature=args.temperature , stop=['Question:', "Q:"])
                
            # construct second stage prompt, to generate a single arabic num answer
            if args.method == "zero_shot_cot":
                prompt_list[0] += responses["choices"][0]["message"]["content"] + args.direct_answer_trigger # 总结最终答案

                # get the second stage zero-shot rationale result -> arabic num answer
                responses = deepseek_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length, time_interval=args.api_time_interval,
                                        temperature=args.temperature, stop='.')

        # extract the pred answer
        # print(responses)
        pred_ans = answer_extraction(args, responses)
        # print(pred_ans)

        # check uncertainty
        if pred_ans != "":
            if pred_ans in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][pred_ans] += 1 # increment answer occurrence
            else:
                uncertainty_record['occurrence'][pred_ans] = 1 # first occurence
        else:
            # Handle no solution case
            if NO_SOLUTION in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][NO_SOLUTION] += 1
            else:
                uncertainty_record['occurrence'][NO_SOLUTION] = 1

    # calculate the variance for the question (only applied to datasets with numerical answer)
    if args.dataset in ("gsm8k", "svamp", "asdiv"):
        ans_list = [] # 记录num_trails次运行的答案
        for ans, occurs in uncertainty_record['occurrence'].items():
            for i in range(int(occurs)):
                ans_list.append(float(ans))
        uncertainty_record['variance'] = np.var(ans_list)
        
    # calculate the entropy for all dataset
    frequency_list = list(uncertainty_record['occurrence'].values())
    uncertainty_record['entropy'] = entropy(frequency_list)

    # calculate the disagreement for all dataset
    uncertainty_record['disagreement'] = len(uncertainty_record['occurrence'])
    
    return uncertainty_record


# return a sorted list by uncertainty from high to low
def create_uncertainty(args, questions, Qwen_model, Qwen_tokenizer):
    result = []
    count = 0

    for qes in tqdm(questions): # 每个qes样本是一个dict：{question, answer, question_idx, complete_data}
        if count == args.qes_limit:
            break
        uncertainty_record = generate_uncertainty_qes(args, qes, Qwen_model, Qwen_tokenizer) # 为每个qes样本生成不确定性分数，包括dataset_idx, variance, entropy, disagreement, occurrence
        # print(uncertainty_record)
        result.append(uncertainty_record)
        count += 1

    # if args.sort_by == "disagreement":
    #     if args.dataset in ("strategyqa", "pubmedqa"):
    #         try:
    #             # sort based on the entropy or the difference between yes and no answers
    #             result.sort(key=lambda x: abs(x['occurrence']['yes'] - x['occurrence']['no']))
    #         except:
    #             # sort by disagreement
    #             result.sort(key=lambda x: -len(x['occurrence'])) # sort默认升序
    #     else:
    #         result.sort(key=lambda x: -len(x['occurrence'])) # 按-len(x['occurrence'])升序排序, 拥有最多不同答案的问题排在前面
    # elif args.sort_by == "variance" and args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith", "gsm8k_tiny"):
    #     # sort by variance
    #     result.sort(key=lambda x: -x['variance'])
    # elif args.sort_by == "entropy" :
    #     result.sort(key=lambda x:-x['entropy'])

    return result

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')
    device = args.device if torch.cuda.is_available() else "cpu"
    Qwen_model = None
    Qwen_tokenizer = None
    if args.model == "Qwen2.5-7B":
        model_path = "/home/wbl/All_projects/SoT/model/Qwen2.5-7B-Instruct"
        Qwen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device,
            attn_implementation="flash_attention_2"
        ).to(device)
        Qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"API_KEY: {API_KEY}")
    set_random_seed(args.random_seed)

    dataloader = create_dataloader(args)

    if args.dataset_size > 1000:
        dataloader = dataloader[:1000] # only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")


    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)

    start =time.time()
    result = create_uncertainty(args, dataloader, Qwen_model, Qwen_tokenizer) # dataloader中每个问题样本按照不确定性升序排列的列表，每个问题样本是一个dict：{question_idx, variance, entropy, disagreement, occurrence}
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

    # output the results
    prosessed_result = []
    for idx in range(len(result)):
        item = dataloader[idx]['complete_data']
        item["uncertainty_scores"] = result[idx]
        prosessed_result.append(item)

    path1 = f"{args.output_dir}/{args.dataset}/{args.model}_{args.method}_k{args.num_trails}_add_uncertainty_scores_{args.num_clusters}.json"
    with open(path1, 'w', encoding='utf-8') as f1:
        json.dump(prosessed_result, f1, ensure_ascii=False, indent=4)
    
    path2 = f"{args.output_dir}/{args.dataset}/{args.model}_{args.method}_k{args.num_trails}_only_uncertainty_scores_{args.num_clusters}.json"
    with open(path2, 'w', encoding='utf-8') as f2:
        try:
            json.dump(result, f2, ensure_ascii=False, indent=4)
        except:
            for item in result:
                try:
                    if args.dataset in ("gsm8k", "svamp", "asdiv"):
                        f2.write(f"{item}, uncertainty: {len(item[-1])}, variance: {item[1]}\n")
                    else:
                        f2.write(f"{item}, uncertainty: {len(item[-1])}\n")
                except:
                    pass

if __name__ == "__main__":
    main()