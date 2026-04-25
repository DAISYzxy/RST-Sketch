import sys, json, re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer

def compute_lengths(text_list, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    result = {}
    for idx, text in enumerate(text_list):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        result[str(idx + 1)] = len(tokens)
    
    return result
    

# 形如 "2:Nucleus=Joint:24" / "8:Nucleus=span:17" / "393:Satellite=Background:393"
_PART_RE = re.compile(
    r'^\s*(\d+)\s*:(Nucleus|Satellite)\s*=\s*([A-Za-z\-]+)\s*:(\d+)\s*$'
)

def _parse_side(side: str) -> Dict[str, Any]:
    """
    解析单侧（left/right）字符串为:
    {'span': [start, end], 'role': 'Nucleus'|'Satellite', 'tag': 'Joint'|'Background'|'span'|...}
    其中 tag 用来后续决定 relation（若为 'span' 则仅表示跨度，不作为关系名）
    """
    m = _PART_RE.match(side)
    if not m:
        raise ValueError(f"无法解析片段: {side!r}")
    start_s, role, tag, end_s = m.groups()
    return {
        "span": [int(start_s), int(end_s)],
        "role": role,
        "tag": tag,  # 可能是 'span'，也可能是 'Joint'/'Background' 等关系名
    }

def _decide_relation(left: Dict[str, Any], right: Dict[str, Any]) -> str:
    """
    relation 的决定规则：
    - 若有 Satellite=xxx，则 relation=该 xxx；
    - 否则 relation=Nucleus=后面的单词（通常两侧一致）。
    注意：当 tag == 'span' 时，它不是关系名，只表示跨度，不参与 relation 决策。
    """
    if left["role"] == "Satellite" and left["tag"] != "span":
        return left["tag"]
    if right["role"] == "Satellite" and right["tag"] != "span":
        return right["tag"]
    # 两侧都是 Nucleus，或 Satellite 的 tag 为 'span'，则取任一侧的非 'span' tag
    for side in (left, right):
        if side["tag"] != "span":
            return side["tag"]
    # 理论上不会到这里；兜底返回 'span'
    return "span"

def parse_rst_pairs(s: str) -> List[Dict[str, Any]]:
    """
    输入如：
    '(1:Satellite=Background:1,2:Nucleus=span:1811) (2:Nucleus=Joint:170,171:Nucleus=Joint:1811) ...'
    输出为一个按顺序的 list，每个元素：
    {
      'relation': str,
      'left':  {'span': [a,b], 'role': 'Nucleus'|'Satellite'},
      'right': {'span': [c,d], 'role': 'Nucleus'|'Satellite'}
    }
    """
    # 抓取每一对括号内的片段
    groups = re.findall(r'\((.*?)\)', s)
    results: List[Dict[str, Any]] = []

    for g in groups:
        # 只在第一个逗号处分割，避免把右侧起始数字误切
        if ',' not in g:
            raise ValueError(f"括号内缺少逗号分隔的左右两部分: {g!r}")
        left_raw, right_raw = g.split(',', 1)
        left = _parse_side(left_raw)
        right = _parse_side(right_raw)

        relation = _decide_relation(left, right)

        # 只保留需要的字段
        out = {
            "relation": relation,
            "left":  {"span": left["span"],  "role": left["role"]},
            "right": {"span": right["span"], "role": right["role"]},
        }
        results.append(out)

    return results


def detok(tokens):
    s = "".join(t.replace("▁", " ") for t in tokens).strip()
    for a,b in [(" ,", ","), (" .", "."), (" !","!"), (" ?","?"),
                (" '","'"), (' "','"'), ('" ', '"'), (" (", "("), (" )", ")"),
                (" ;", ";"), (" :", ":")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    return s

def edus_from(tokens, breaks):
    edus, start = [], 0
    for brk in breaks:
        edus.append(detok(tokens[start:brk+1])); start = brk+1
    if start < len(tokens):
        edus.append(detok(tokens[start:]))
    return edus


def collect_roles(data):
    result = {}
    for item in data:
        for side in ["left", "right"]:
            span = item[side]["span"]
            role = item[side]["role"]
            if span[0] == span[1]:
                result[span[0]] = role
    return result



def accumulate_weights(
    parsed: List[Dict[str, Any]],
    llm_relation_docuType_weights: Dict[str, float],
    edus: List[Any],
) -> Dict[str, float]:
    """
    根据 parsed 中的左右 span 与 relation，将权重累计到 weight_acc 上。
    假设 span 使用 1-based 索引（例如 [1, 3] 表示 EDU #1 到 EDU #3）。
    """
    n = len(edus)
    weight_acc = {i: 0.0 for i in range(1, n + 1)}

    def apply_span(span: List[int], weight: float):
        if not span or len(span) != 2:
            return
        start, end = span
        if start > end:
            start, end = end, start  # 容错：若传入反向区间则交换
        # 裁剪到合法范围，以避免越界
        start = max(1, start)
        end = min(n, end)
        length = end - start + 1
        if length <= 0 or weight == 0:
            return
        inc = weight / length
        for i in range(start, end + 1):
            weight_acc[i] += inc

    for rel in parsed:
        r = rel.get("relation")
        w = llm_relation_docuType_weights.get(r, 0.0)

        # 按顺序处理 left 再处理 right
        left = rel.get("left", {})
        right = rel.get("right", {})

        if "span" in left:
            apply_span(left["span"], w)
        if "span" in right:
            apply_span(right["span"], w)
            
    weight_acc = {str(k): v for k, v in weight_acc.items()}
    return weight_acc



def count_EDU_layer(parsed: List[Dict[str, Any]], n_edus: Optional[int] = None) -> Dict[str, int]:
    """
    对每个 EDU（1..n_edus）统计被覆盖的次数。
    覆盖区间 = [left.span[0], right.span[1]]（两端都包含）。
    返回形如 {"1": c1, "2": c2, ..., "n_edus": cn} 的字典。
    """
    if not parsed:
        return {}

    # 自动推断 EDU 总数（最大出现的 span 端点）
    if n_edus is None:
        max_idx = 0
        for item in parsed:
            l0, l1 = item["left"]["span"]
            r0, r1 = item["right"]["span"]
            max_idx = max(max_idx, l0, l1, r0, r1)
        n_edus = max_idx

    # 差分数组，长度 n_edus + 2 用于处理 end+1 下标
    diff = [0] * (n_edus + 2)

    for i, item in enumerate(parsed, start=1):
        start = int(item["left"]["span"][0])
        end   = int(item["right"]["span"][1])

        if not (1 <= start <= end <= n_edus):
            raise ValueError(f"第 {i} 个条目区间非法：[{start}, {end}]，应满足 1 <= start <= end <= {n_edus}")

        diff[start] += 1
        diff[end + 1] -= 1  # end 为闭区间，因此在 end+1 处减回

    # 前缀和得到每个 EDU 的覆盖次数
    cover_counts: Dict[str, int] = {}
    running = 0
    for idx in range(1, n_edus + 1):
        running += diff[idx]
        cover_counts[str(idx)] = running

    return cover_counts


def mark_nucleus_backbone(all_edus, rel_items):
    h = {str(i): 1 for i in sorted(all_edus)}

    # 3. 遍历，遇到 Satellite 就标记为 0
    for it in rel_items:
        for side in ["left", "right"]:
            role = it[side]["role"]
            span = it[side]["span"]
            if role.lower() == "satellite":
                for edu in range(span[0], span[1] + 1):
                    h[str(edu)] = 0
    return h


def calculate_depth_nucleus_prior(h_indicator, EDU_layers, beta=0.2, mu=0.8):
    d_priors = {}
    for key in h_indicator.keys():
        h = h_indicator[key]
        z = EDU_layers[key]
        d = (1 + beta * h) / (1 + mu * z)
        d_priors[key] = d
    return d_priors


def calculate_edu_information_density(edus, alpha_tf=0.2, kappa=2.0):
    """
    计算每个 EDU 的信息密度分数。

    参数:
        edus (list of str): EDU 文本列表
        alpha_tf (float): TF-IDF 权重系数 (推荐 0.15 ~ 0.3)
        kappa (float): z-score 对称截断阈值 (默认 2)

    返回:
        list of float: 每个 EDU 的信息密度分数
    """
    # Step 1: 计算 TF-IDF 矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(edus)

    # Step 2: 计算每个 EDU 的信息密度 (取 TF-IDF 均值或和)
    # 这里用 "均值" 更稳健
    raw_density = tfidf_matrix.mean(axis=1).A.ravel()

    # Step 3: z-score 标准化
    mean_val = raw_density.mean()
    std_val = raw_density.std() if raw_density.std() > 0 else 1e-8
    z_scores = (raw_density - mean_val) / std_val

    # Step 4: 对称截断
    clipped = np.clip(z_scores, -kappa, kappa)

    # Step 5: 转换为信息密度分数
    scores = 1 + alpha_tf * clipped

    return scores.tolist()



def normalize_to_mean_one(weights: dict) -> dict:
    """
    Normalize dictionary values so that their mean = 1.

    参数:
        weights (dict): {key: value} 格式，value 是原始权重

    返回:
        dict: 归一化后的新字典
    """
    if not weights:
        return {}

    values = list(weights.values())
    mean_val = sum(values) / len(values)

    if mean_val == 0:
        raise ValueError("Mean of values is zero, cannot normalize.")

    normalized = {k: v / mean_val for k, v in weights.items()}
    return normalized