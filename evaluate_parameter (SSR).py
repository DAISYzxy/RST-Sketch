import json
import requests
from tqdm import tqdm

API_URL = "https://api.nuwaapi.com/v1/embeddings"
API_KEY = "sk-0rZuopSK9jEXMtpo2LYsjB2eTsPZ1K21AeBzEQIupsnLWqZn"
MODEL_NAME = "text-embedding-3-large"


from typing import List
import numpy as np

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def average_similarity(references_emb: List[List[float]], candidates_emb: List[List[float]]) -> float:
    """计算两个embedding列表中对应位置的平均相似度"""
    assert len(references_emb) == len(candidates_emb), "两个list长度必须相同"
    
    sims = [
        cosine_similarity(ref, cand) 
        for ref, cand in zip(references_emb, candidates_emb)
    ]
    return float(np.mean(sims))


def get_text_embedding(text):
    """
    调用 OpenAI API 生成文本 embedding
    :param text: 输入文本 (str)，<=500 tokens
    :return: embedding 向量 (list of floats)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "input": text,
        "model": MODEL_NAME,
        "encoding_format": "float"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 如果有错误抛出异常
        embedding = response.json()["data"][0]["embedding"]
        return embedding
    except requests.exceptions.RequestException as e:
        print("API 请求失败:", e)
        return None
    except KeyError:
        print("API 返回结果解析失败:", response.text)
        return None


with open("wcep_summaries_0dot7_gpt4omini.json", "r", encoding="utf-8") as f:
    ours_answers_list = json.load(f)
    
ours_answers = []
for item in ours_answers_list:
    # for a_item in item:
    ours_answers.append(item["summary"])

ours_candidates_emb = []
for item in tqdm(ours_answers):
    emb = get_text_embedding(item)
    ours_candidates_emb.append(emb)

with open("gt_embedding.json", "r", encoding="utf-8") as f:
    references_emb = json.load(f)

print("----------------------------------ours summaries semantic similarity----------------------------------\n")
print(average_similarity(references_emb, ours_candidates_emb))

