import random
import numpy as np
import torch
import json

def create_dataloader(args)->list:
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx}) # 每个样本是一个字典

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset

def load_data(args):
    with open(args.dataset_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [item['question'] for item in data]
    responses = [item['response'] for item in data]

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(questions)}")
    args.dataset_size = len(questions)
    return questions, responses