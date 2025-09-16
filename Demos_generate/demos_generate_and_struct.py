from utils import *
import argparse
import os
from collections import defaultdict

def arg_parser():
    parser = argparse.ArgumentParser(description="Demo Generation and Structuring")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_clusters", type=int, default=11, help="cluster number, default is 11")
    parser.add_argument("--metric", type=str, default="entropy", help="uncertain metric")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "asdiv", "strategyqa", "csqa", "qasc", "pubmedqa"], help="dataset to inference"
    )
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot", "few_shot_sot"], help="method used in uncertainty estimation"
    )
    args = parser.parse_args()
    
    args.dataset_path = f"./add_entropy_weights/{args.dataset}/{args.method}_final_dataset_{args.num_clusters}_{args.metric}.json"

    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def random_sample_demos_generate(args, questions, responses):
    file_path = f"./demos/random_sample/{args.dataset}/{args.method}_all_datasets_demos_{args.num_clusters}_{args.metric}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            random_sample_data = json.load(f)
    else:
        random_sample_data = {}
    
    random_sample_data[args.dataset] = []
    
    random_list = random.sample(range(0, len(questions)), 5)
    print(f"Randomly selected dataset indices: {random_list}")

    for idx in random_list:
        item = {'question': questions[idx], 'response': responses[idx]}
        random_sample_data[args.dataset].append(item)       

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(random_sample_data, f, ensure_ascii=False, indent=4)

def only_typicality_demos_generate(args, questions, responses, typicality_scores):
    file_path = f"./demos/only_typicality/{args.dataset}/{args.method}_all_datasets_demos_{args.num_clusters}_{args.metric}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            only_typicality_data = json.load(f)
    else:
        only_typicality_data = {}
    
    only_typicality_data[args.dataset] = []

    sorted_data = sorted(
        zip(questions, responses, typicality_scores),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=False  # 升序排列
    )
    questions_sorted, responses_sorted, typicality_scores_sorted = zip(*sorted_data)

    questions_selected = questions_sorted[-5:] # 选择最后5个
    responses_selected = responses_sorted[-5:]

    for idx in range(len(questions_selected)):
        item = {'question': questions_selected[idx], 'response': responses_selected[idx]}
        only_typicality_data[args.dataset].append(item)

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(only_typicality_data, f, ensure_ascii=False, indent=4)

def only_typicality_type_demos_generate(args, questions, responses, typicality_scores, labels):
    file_path = f"./demos/only_typicality_type/{args.dataset}/{args.method}_all_datasets_demos_{args.num_clusters}_{args.metric}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            only_typicality_type_data = json.load(f)
    else:
        only_typicality_type_data = {}
    
    only_typicality_type_data[args.dataset] = []

    sorted_data = sorted(
        zip(questions, responses, typicality_scores, labels),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=True  # 降序排列
    )
    questions_sorted, responses_sorted, typicality_scores_sorted, labels_sorted = zip(*sorted_data)

    appeared_labels = []
    questions_selected = []
    responses_selected = []

    for idx in range(len(questions_sorted)):
        if labels_sorted[idx] in appeared_labels:
            continue

        appeared_labels.append(labels_sorted[idx])
        questions_selected.append(questions_sorted[idx])
        responses_selected.append(responses_sorted[idx])

        if len(appeared_labels) == 5:
            break
    
    for idx in reversed(range(len(questions_selected))):
        item = {'question': questions_selected[idx], 'response': responses_selected[idx]}
        only_typicality_type_data[args.dataset].append(item)

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(only_typicality_type_data, f, ensure_ascii=False, indent=4)

def only_uncertainty_demos_generate(args, questions, responses, uncertainty_scores):
    file_path = f"./demos/only_uncertainty/{args.dataset}/{args.method}_all_datasets_demos_{args.num_clusters}_{args.metric}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            only_uncertainty_data = json.load(f)
    else:
        only_uncertainty_data = {}
    
    only_uncertainty_data[args.dataset] = []

    sorted_data = sorted(
        zip(questions, responses, uncertainty_scores),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=False  # 升序排列
    )
    questions_sorted, responses_sorted, uncertainty_scores_sorted = zip(*sorted_data)

    questions_selected = questions_sorted[-5:] # 选择最后5个
    responses_selected = responses_sorted[-5:]

    for idx in range(len(questions_selected)):
        item = {'question': questions_selected[idx], 'response': responses_selected[idx]}
        only_uncertainty_data[args.dataset].append(item)

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(only_uncertainty_data, f, ensure_ascii=False, indent=4)

def use_entropy_weight_demos_generate(args, questions, responses, entropy_weights):
    file_path = f"./demos/entropy_weight/{args.dataset}/{args.method}_all_datasets_demos_{args.num_clusters}_{args.metric}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            use_entropy_weight_data = json.load(f)
    else:
        use_entropy_weight_data = {}
    
    use_entropy_weight_data[args.dataset] = []

    sorted_data = sorted(
        zip(questions, responses, entropy_weights),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=False  # 升序排列
    )
    questions_sorted, responses_sorted, entropy_weights_sorted = zip(*sorted_data)

    questions_selected = questions_sorted[-5:] # 选择最后5个
    responses_selected = responses_sorted[-5:]

    for idx in range(len(questions_selected)):
        item = {'question': questions_selected[idx], 'response': responses_selected[idx]}
        use_entropy_weight_data[args.dataset].append(item)

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(use_entropy_weight_data, f, ensure_ascii=False, indent=4)

def first_cluster_then_uncertainty_demos_generate(args, questions, responses, uncertainty_scores, labels):
    file_path = f"./demos/first_cluster_then_uncertainty/{args.dataset}/{args.method}_demos_{args.num_clusters}_{args.metric}.json"
    demos = defaultdict(list)
    label_counts = defaultdict(int)
    num_qes = 5 # 每类问题需要的问题数量

    sorted_data = sorted(
        zip(questions, responses, uncertainty_scores, labels),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=True  # 降序排列
    )
    questions_sorted, responses_sorted, uncertainty_scores_sorted, labels_sorted = zip(*sorted_data)

    for idx in range(len(questions_sorted)):
        label = labels_sorted[idx]
        if label_counts[label] < num_qes:
            item = {'question': questions_sorted[idx], 'response': responses_sorted[idx]}
            demos[label].insert(0, item)  # 插入到列表的开头
            label_counts[label] += 1
        if all(count == num_qes for count in label_counts.values()) and len(label_counts) == args.num_clusters:
            break

    print(f"Selected questions from each cluster: {dict(label_counts)}")

    # 按键升序排序并重建字典
    sorted_demos = {k: demos[k] for k in sorted(demos.keys(), key=int)}

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(sorted_demos, f, ensure_ascii=False, indent=4)

def first_cluster_then_typicality_demos_generate(args, questions, responses, typicality_scores, labels):
    file_path = f"./demos/first_cluster_then_typicality/{args.dataset}/{args.method}_demos_{args.num_clusters}_{args.metric}.json"
    demos = defaultdict(list)
    label_counts = defaultdict(int)
    num_qes = 5 # 每类问题需要的问题数量

    sorted_data = sorted(
        zip(questions, responses, typicality_scores, labels),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=True  # 降序排列
    )
    questions_sorted, responses_sorted, typicality_scores_sorted, labels_sorted = zip(*sorted_data)

    for idx in range(len(questions_sorted)):
        label = labels_sorted[idx]
        if label_counts[label] < num_qes:
            item = {'question': questions_sorted[idx], 'response': responses_sorted[idx]}
            demos[label].insert(0, item)  # 插入到列表的开头
            label_counts[label] += 1
        if all(count == num_qes for count in label_counts.values()) and len(label_counts) == args.num_clusters:
            break

    print(f"Selected questions from each cluster: {dict(label_counts)}")

    # 按键升序排序并重建字典
    sorted_demos = {k: demos[k] for k in sorted(demos.keys(), key=int)}

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(sorted_demos, f, ensure_ascii=False, indent=4)

def first_cluster_then_entropy_weight_demos_generate(args, questions, responses, entropy_weights, labels):
    file_path = f"./demos/first_cluster_then_entropy_weight/{args.dataset}/{args.method}_demos_{args.num_clusters}_{args.metric}.json"
    demos = defaultdict(list)
    label_counts = defaultdict(int)
    num_qes = 5 # 每类问题需要的问题数量

    sorted_data = sorted(
        zip(questions, responses, entropy_weights, labels),
        key=lambda x: x[2],  # 以索引2为排序依据
        reverse=True  # 降序排列
    )
    questions_sorted, responses_sorted, entropy_weights_sorted, labels_sorted = zip(*sorted_data)

    for idx in range(len(questions_sorted)):
        label = labels_sorted[idx]
        if label_counts[label] < num_qes:
            item = {'question': questions_sorted[idx], 'response': responses_sorted[idx]}
            demos[label].insert(0, item)  # 插入到列表的开头
            label_counts[label] += 1
        if all(count == num_qes for count in label_counts.values()) and len(label_counts) == args.num_clusters:
            break

    print(f"Selected questions from each cluster: {dict(label_counts)}")
    
    # 按键升序排序并重建字典
    sorted_demos = {k: demos[k] for k in sorted(demos.keys(), key=int)}

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(sorted_demos, f, ensure_ascii=False, indent=4)

def main():
    args = arg_parser()
    set_random_seed(args.random_seed)
    print('*****************************')
    print(args)
    print('*****************************')

    # if args.dataset == 'gsm8k':
    #     metric_name = 'disagreement'
    # elif args.dataset in ('strategyqa', 'pubmedqa'):
    #     metric_name = 'entropy'

    with open(args.dataset_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [item['question'] for item in data]
    responses = [item['response'] for item in data]
    typicality_scores = [item['typicality_score'] for item in data]
    uncertainty_scores = [item['uncertainty_scores'][args.metric] for item in data]
    entropy_weights = [item['entropy_weight'] for item in data]
    labels = [item['typicality_label'] for item in data]

    random_sample_demos_generate(args, questions, responses)

    only_typicality_demos_generate(args, questions, responses, typicality_scores)

    only_typicality_type_demos_generate(args, questions, responses, typicality_scores, labels)

    only_uncertainty_demos_generate(args, questions, responses, uncertainty_scores)

    use_entropy_weight_demos_generate(args, questions, responses, entropy_weights)

    first_cluster_then_uncertainty_demos_generate(args, questions, responses, uncertainty_scores, labels)

    first_cluster_then_typicality_demos_generate(args, questions, responses, typicality_scores, labels)

    first_cluster_then_entropy_weight_demos_generate(args, questions, responses, entropy_weights, labels)

if __name__ == "__main__":
    main()