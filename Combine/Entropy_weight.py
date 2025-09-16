import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import defaultdict
import argparse
import random
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_dataset_indices_and_uncertainty(file_path, metric):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    dataset_indices = [item['dataset_idx'] for item in data]
    dataset_metrics = [item[metric] for item in data]
    return dataset_indices, dataset_metrics

def get_json_line(file_path, index):
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == index:
                return json.loads(line)
        return None  # 索引超出范围

def entropy_weight_method(standardized_data):
    # 确保数据是浮点型
    data = np.array(standardized_data, dtype=float)
    
    # 计算样本数和指标数
    n, m = data.shape
    
    # 避免对数计算中的零值
    eps = 1e-10
    data = np.clip(data, eps, None)  # 将小于eps的值替换为eps
    
    # 计算概率矩阵
    p = data / data.sum(axis=0)
    
    # 计算信息熵
    e = -1 / np.log(n) * np.sum(p * np.log(p), axis=0)
    
    # 计算差异系数
    g = 1 - e
    
    # 计算权重
    w = g / np.sum(g)
    
    return w

def arg_parser():
    parser = argparse.ArgumentParser(description="Demo Generation and Structuring")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_clusters", type=int, default=11, help="cluster number, default is 11")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "asdiv", "strategyqa", "csqa", "qasc", "pubmedqa"], help="dataset to inference"
    )
    parser.add_argument("--metric", type=str, default="entropy", help="uncertain metric")
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot", "few_shot_sot"], help="method used in uncertainty estimation"
    )
    args = parser.parse_args()
    
    args.dataset_path = f"./add_uncertainty_scores/{args.dataset}/{args.method}_final_dataset.json"
    
    return args

if __name__ == "__main__":
    args = arg_parser()
    set_random_seed(args.random_seed)
    # if args.dataset == 'gsm8k':
    #     metric_name = 'disagreement'
    # elif args.dataset in ('strategyqa', 'pubmedqa'):
    #     metric_name = 'entropy'
    metric_name = args.metric

    file_path = f"./add_uncertainty_scores/{args.dataset}/Qwen2.5-7B_{args.method}_k10_add_uncertainty_scores_{args.num_clusters}.json"
    with open(file_path, 'r') as file:
        data = json.load(file)

    uncertainty_scores = [item["uncertainty_scores"][metric_name] for item in data]
    typicality_scores = [item["typicality_score"] for item in data]
    
    scalar = MinMaxScaler()
    normalized_metrics = scalar.fit_transform(np.array(uncertainty_scores).reshape(-1,1)).flatten()
    normalized_typicality_scores = scalar.fit_transform(np.array(typicality_scores).reshape(-1,1)).flatten()
    
    # 计算权重
    standardized_data = np.column_stack((normalized_metrics, normalized_typicality_scores))
    weights = entropy_weight_method(standardized_data)
    # print("指标权重:", weights)

    weighted_scores = standardized_data @ weights
    # print("加权得分:", weighted_scores)

    for idx, item in enumerate(data):
        item["entropy_weight"] = weighted_scores[idx]

    # 保存为JSON文件
    with open(f'./add_entropy_weights/{args.dataset}/{args.method}_final_dataset_{args.num_clusters}_{metric_name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)