from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('/home/wbl/All_projects/ICASSP2026/Evaluate')
from sketch_of_thought.sketch_of_thought import SoT
import argparse
import time
from tqdm import tqdm
import re
import random
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer
import joblib

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_clusters", type=int, default=11, help="cluster number, default is 11")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "asdiv", "strategyqa", "csqa", "qasc"], help="dataset to inference"
    )
    parser.add_argument(
        "--evaluate_type", type=str, default="first_cluster_then_uncertainty", choices=["entropy_weight", "first_cluster_then_entropy_weight", "first_cluster_then_uncertainty", "first_cluster_then_typicality", "only_typicality", "only_typicality_type", "only_uncertainty", "random_sample"], help="experiment to evaluate"
    )
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot", "few_shot_sot"], help="method"
    )
    parser.add_argument("--metric", type=str, default="entropy", help="uncertain metric")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--dataset_mode", type=str, default="whole", choices=["whole", "part"], help="use whole dataset or part of the dataset"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    parser.add_argument("--thought_type", type=str, default="sot", help="thought type, default is sot")
    
    args = parser.parse_args()
    
    if args.dataset_mode == "part":
        args.dataset_path = f"./final_datasets/{args.dataset}/{args.method}_final_dataset_{args.num_clusters}.json"
    else:
        args.dataset_path = f"./transformed_whole_datasets/{args.dataset}/transformed_train.json"

    # Set the paradigm based on the dataset
    if args.thought_type == "sot":
        if args.dataset in ("gsm8k", "svamp", "asdiv"):
            args.paradigm = "chunked_symbolism"
        elif args.dataset in ("strategyqa", "csqa", "qasc"):
            args.paradigm = "conceptual_chaining"
        elif args.dataset == "pubmedqa":
            args.paradigm = "expert_lexicons"
    else:
        args.paradigm = "cot"  # Default to chain-of-thought if not specified

    if args.evaluate_type in ("first_cluster_then_entropy_weight", "first_cluster_then_uncertainty", "first_cluster_then_typicality"):
        args.demos_path = f"../../Demos_generate/demos/{args.evaluate_type}/{args.dataset}/{args.method}_demos_{args.num_clusters}_{args.metric}.json"
    else:
        args.demos_path = f"../../Demos_generate/demos/{args.evaluate_type}/{args.dataset}/{args.method}_all_datasets_demos_{args.num_clusters}_{args.metric}.json"
    
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataloader(args):
    questions, answers, question_idxes = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":question_idxes[idx]}) # 每个样本是一个字典

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset

def load_data(args):
    with open(args.dataset_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    question_idxes = [item['question_idx'] for item in data]

    print(f"dataset: {args.dataset}")
    return questions, answers, question_idxes

def extract_answer_from_boxed(output):
    # 定义正则表达式模式，匹配 \boxed{...} 或 \\boxed{...}
    pattern = r'\\boxed\{([^}]*)\}'
    
    # 查找所有匹配项
    matches = re.findall(pattern, output)
    
    if matches:
        # 返回最后一个匹配的内容（通常是最终答案）
        return matches[-1].strip()
    else:
        return ""

def are_equal(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return False  # 如果无法转换为数字，返回 False

def get_accuracy(args, dataloader, model, tokenizer, sot): # 为dataloader中的每个qes样本生成回答，并统计准确率
    total_correct = 0
    total_samples = 0
    token_num = 0
    results = []  # 存储每个样本的结果

    for qes in tqdm(dataloader): # 每个qes样本是一个dict：{question, answer, question_idx}
        output, token_count = get_answer(args, qes, model, tokenizer, sot) # 为每个qes样本生成回答，并返回生成答案的token数目
        output_answer = extract_answer_from_boxed(output) # 提取答案
        # print(f"output_answer: {output_answer}")
        ground_truth = qes['answer']
        # print(f"ground_truth: {ground_truth}")

        # 简单比较生成的答案和标准答案是否一致（精确匹配）
        if args.dataset == "strategyqa":
            is_correct = ground_truth.strip().lower() in output_answer.strip().lower()
        else:
            is_correct = output_answer.strip().lower() == ground_truth.strip().lower() or are_equal(output_answer.strip(), ground_truth.strip())
        
        # 记录结果
        results.append({
            "question_idx": qes["question_idx"],
            "question": qes["question"],
            "ground_truth": ground_truth,
            "response": output,
            "prediction": output_answer,
            "is_correct": is_correct
        })
        
        # 更新统计数据
        if is_correct:
            total_correct += 1
        total_samples += 1

        token_num += token_count  # 累加token数目
    
    # 计算准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # 打印结果
    print(f"Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    
    # 返回详细结果和准确率
    return {
        "accuracy": accuracy,
        "Accuracy (correct/total)": f"{total_correct}/{total_samples}",
        "token_number": token_num,
        "results": results
    }


def get_answer(args, question, model, tokenizer, sot):
    # Prepare the question
    prompt = question['question']
    # print(prompt)

    paradigm = args.paradigm  # Use the predefined paradigm from args

    if args.evaluate_type in ("first_cluster_then_entropy_weight", "first_cluster_then_uncertainty", "first_cluster_then_typicality"):
        label = question['label']
        messages = sot.get_initialized_context(
            paradigm,
            prompt,
            format="llm",
            include_system_prompt=True,
            label=str(label)
        )
    else:
        messages = sot.get_initialized_context(
            paradigm,
            prompt,
            format="llm",
            include_system_prompt=True
        )
    # print(messages)

    # Format for the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=args.temperature
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 获取tokens数目
    token_count = sum(len(ids) for ids in generated_ids)

    # Decode response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response, token_count

def main():
    args = arg_parser()
    set_random_seed(args.random_seed)
    print('*****************************')
    print(args)
    print('*****************************')
    device = args.device if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(f'../Typicality/model/all-MiniLM-L6-v2')
    clustering_model_path = f'../Typicality/model/clustering_model_{args.dataset}_{args.num_clusters}.pkl'
    clustering_model = joblib.load(clustering_model_path)
    print(f"Clustering model loaded from {clustering_model_path}")

    dataloader = create_dataloader(args)
    questions = [item['question'] for item in dataloader]

    questions_embeddings = encoder.encode(questions) # 把问题句子转换为向量

    predicted_labels = clustering_model.predict(questions_embeddings) # 预测问题的标签
    for i, item in enumerate(dataloader):
        item['label'] = predicted_labels[i] # 把标签添加到每个问题样本中

    # Initialize SoT
    sot = SoT(args)

    # Load Qwen model and tokenizer
    model_path = "../model/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Model is on device: {model.device}")

    start =time.time()
    result_dict = get_accuracy(args, dataloader, model, tokenizer, sot) # dataloader中每个问题样本按照不确定性升序排列的列表，每个问题样本是一个dict：{question_idx, variance, entropy, disagreement, occurrence}
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")
    result_dict['execution_time'] = end - start  # 添加执行时间到结果字典

    # Save the results
    if args.thought_type == "sot":
        save_path = f"./evaluated_results/{args.evaluate_type}/{args.dataset}/{args.method}_results_{args.dataset_mode}_{args.num_clusters}_{args.metric}.json"
    else:
        save_path = f"./evaluated_results/cot/{args.dataset}/results.json"
    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()