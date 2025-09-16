import numpy as np
import random
import torch
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from sentence_transformers import SentenceTransformer

plt.rcParams['axes.unicode_minus'] = False		# 显示负号

def parse_arguments():
    parser = argparse.ArgumentParser(description="Determine the best K value for K-Means clustering using the elbow method.")
    parser.add_argument(
        "--task", type=str, default="gsm8k", choices=["gsm8k", "svamp", "asdiv", "strategyqa", "csqa", "qasc", "pubmedqa"], help="dataset used for experiment"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    encoder = SentenceTransformer(f'./model/all-MiniLM-L6-v2')

    task = args.task
    dataset_path = f"annotated_dataset/{task}/annotated_train.json"
    output_path = f"output/{task}"

    with open(dataset_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]

    questions_embeddings = encoder.encode(questions) # 把问题句子转换为向量

    # # 标准化数据
    # scaler = StandardScaler()
    # questions_embeddings_scaled = scaler.fit_transform(questions_embeddings)

    # 使用手肘法确定最佳的K值
    inertia = []
    for k in range(1, 31):
        kmeans = KMeans(n_clusters=k, random_state=args.random_seed)
        kmeans.fit(questions_embeddings)
        inertia.append(kmeans.inertia_)

    # 绘制手肘法图表
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 31), inertia, marker='o', linestyle='--')

    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('elbow method')
    plt.savefig(output_path + '/elbow method.png',dpi=300)
    plt.grid(True)

    plt.show()

    # 使用最佳的K值进行K-Means聚类
    best_k = 14
    kmeans = KMeans(n_clusters=best_k, random_state=args.random_seed)
    kmeans.fit(questions_embeddings)

    # 将簇标签添加到原始数据中
    for idx, item in enumerate(data):
        item['label'] = kmeans.labels_[idx]

    # 统计label种类和数量
    label_counts = {}
    for item in data:
        label = item["label"].item()
        label_counts[label] = label_counts.get(label, 0) + 1
    sorted_label_counts = {k: v for k, v in sorted(label_counts.items(), key=lambda x: x[0])}
    print("Label counts:", sorted_label_counts)

    # # 使用PCA进行降维
    # pca = PCA(n_components=2)
    # questions_embeddings_pca = pca.fit_transform(questions_embeddings)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=100, random_state=args.random_seed)  # 保持随机种子一致
    questions_embeddings_tsne = tsne.fit_transform(questions_embeddings)  # 注意变量名也相应修改

    # PCA绘制降维后的数据及其簇分布
    plt.figure(figsize=(8, 6))
    plt.scatter(questions_embeddings_tsne[:, 0], questions_embeddings_tsne[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.xlabel('Principal 1')
    plt.ylabel('Principal 2')
    plt.title('K-Means result')
    plt.savefig(output_path + '/K-Means result.png',dpi=300)
    plt.show()

if __name__ == "__main__":
    main()