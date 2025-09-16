import random
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description="add typicality scores and labels to the dataset")
    parser.add_argument(
        "--task", type=str, default="gsm8k", choices=["gsm8k", "svamp", "asdiv", "strategyqa", "csqa", "qasc", "pubmedqa"], help="dataset used for experiment"
    )
    parser.add_argument("--num_clusters", type=int, default=11, help="cluster number, default is 11")
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

    questions_embeddings = encoder.encode(questions) # 把问题句子转换为向量

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=args.num_clusters, random_state=args.random_seed)
    clustering_model.fit(questions_embeddings)

    # 保存模型
    joblib.dump(clustering_model, f'./model/clustering_model_{args.task}_{args.num_clusters}.pkl')

    cluster_assignment = clustering_model.labels_ # 每个句子的聚类标签,list

    dist = clustering_model.transform(questions_embeddings) # 每个句子到每个聚类中心的距离
    # print(dist.shape)
    distances_to_center = np.min(dist, axis=1) # 每个句子到最近聚类中心的距禽
    # print(distances_to_center.shape)

    # 定义标准差 \sigma
    sigma = 1.0

    # 高斯函数转换为分数
    scores = np.exp(-(distances_to_center ** 2) / (2 * sigma ** 2))

    # print("到聚类中心距离:", distances_to_center)
    # print("代表性分数:", scores)
    # print(scores.shape)

    # 将分数添加到数据集中
    for i, item in enumerate(data):
        item['typicality_score'] = float(scores[i])  # 确保分数是浮点数
        item['typicality_label'] = int(cluster_assignment[i])

    # Save the processed dataset
    with open(output_path + f"/dataset_with_scores_and_labels_{args.num_clusters}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    y_km = clustering_model.predict(questions_embeddings)

    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(questions_embeddings) # 降维到2维
    centers = pca_model.transform(clustering_model.cluster_centers_)

    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, cmap='viridis', s=20)
    # plt.scatter(centers[:, 0],centers[:, 1],
    #         s=250, marker='*', label='centroids',
    #         edgecolor='black',
    #        c=np.arange(0,args.num_clusters),cmap=plt.cm.Paired,)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path+f"/K-Means result({args.num_clusters}).png", dpi=600)

if __name__ == "__main__":
    main()